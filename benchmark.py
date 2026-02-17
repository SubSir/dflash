import argparse
import time
import random
from itertools import chain
from types import SimpleNamespace
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import os
import io
from collections import defaultdict

try:
    from safetensors import safe_open
except Exception:
    safe_open = None
import distributed as dist

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    total_target_time = 0.0
    total_draft_time = 0.0

    step_avg_nlls = []  # average NLL of draft-proposed tokens per speculative step
    valid_token_stats = []  # list of (nll, is_accepted) for tokens that were actually checked
    step_token_nlls = []  # list of per-step nll arrays, shape: (block_size-1,)
    step_token_ids = []  # list of per-step token id arrays, shape: (block_size-1,)

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            # Draft forward timing
            draft_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)


            draft_time = cuda_time() - draft_start
            total_draft_time += draft_time
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        # Target forward timing
        target_start = cuda_time()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )
        target_time = cuda_time() - target_start
        total_target_time += target_time

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()

        if block_size > 1:
            draft_logprobs = torch.log_softmax(draft_logits, dim=-1)
            proposed = block_output_ids[:, 1:]  # (1, block_size-1)
            token_logp = draft_logprobs.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)  # (1, block_size-1)
            token_nll = (-token_logp).squeeze(0)  # (block_size-1,)
            step_avg_nlls.append(token_nll.mean().item())

            # 记录每个位置的 NLL（用于检查位置单调性/示例展示）
            step_token_nlls.append(token_nll.detach().to("cpu").tolist())
            step_token_ids.append(proposed.squeeze(0).detach().to("cpu").tolist())

            valid_k = min(int(acceptance_length) + 1, block_size - 1)
            if valid_k > 0:
                for j in range(valid_k):
                    is_acc = j < int(acceptance_length)
                    valid_token_stats.append((float(token_nll[j].item()), bool(is_acc)))

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        total_target_time=total_target_time,
        total_draft_time=total_draft_time,
        step_avg_nlls=step_avg_nlls,
        valid_token_stats=valid_token_stats,
        step_token_nlls=step_token_nlls,
        step_token_ids=step_token_ids,
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="swe-bench")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--export-predict-json", type=str, default=None, help="Path to save per-step token NLL + acc_len + thresholds json for offline tuning")
    parser.add_argument(
        "--delta-n-mode",
        type=str,
        default="none",
        choices=["none", "learn", "infer"],
        help="none: disable delta-n pseudo quant; learn: collect stats and save per-layer n; infer: load per-layer n and apply",
    )
    parser.add_argument("--delta-n-config", type=str, default=None, help="Path to save/load per-layer n config (json)")
    parser.add_argument("--delta-n-select", type=str, default="best_count", choices=["best_count", "total_mse"], help="How to select fixed n from stats when saving")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        dtype=torch.bfloat16,
    ).to(device).eval()

    # Replace draft model linears with PseudoLinear for delta-n learn/infer
    if args.delta_n_mode != "none":
        from model.pseudo import PseudoLinear, report_delta_stats, load_fixed_n_config, save_fixed_n_config
        replaced_draft = 0
        for name, module in list(draft_model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            if name.endswith("lm_head"):
                continue
            # We want to learn for all linear layers in the draft model
            parent = draft_model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            
            quant_mode = "delta_learn" if args.delta_n_mode == "learn" else "delta_inference"
            new_m = PseudoLinear(
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                bias=(module.bias is not None),
                quant_mode=quant_mode,
                layer_name=name,
            ).to(device=device, dtype=module.weight.dtype)
            new_m.from_linear(module)
            setattr(parent, parts[-1], new_m)
            replaced_draft += 1
        if args.delta_n_mode == "infer":
            if args.delta_n_config is None:
                raise ValueError("--delta-n-config is required when --delta-n-mode=infer")
            fixed_n_cfg = load_fixed_n_config(args.delta_n_config)
            missing = 0
            for name, m in draft_model.named_modules():
                if isinstance(m, PseudoLinear) and m.quant_mode == "delta_inference":
                    if name in fixed_n_cfg:
                        m.fixed_n = int(fixed_n_cfg[name])
                    else:
                        missing += 1
            if missing > 0 and dist.is_main():
                print(f"[delta-n][infer] warning: {missing} PseudoLinear layers missing fixed_n config (default=1)")

        print(f"[draft pseudolinear] replaced linear layers: {replaced_draft} mode={args.delta_n_mode}")
    else:
        print("[delta-n] disabled, using original linear layers")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for block_size in [1, args.block_size]:
                response[block_size] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )
            
            spec_response = response[args.block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[args.block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[args.block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[args.block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(args.block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    # Per-model timing breakdown
    total_target_time = np.mean([r[args.block_size].total_target_time for r in responses])
    total_draft_time = np.mean([r[args.block_size].total_draft_time for r in responses])
    print(f"Average target model time per sample: {total_target_time:.4f}s")
    print(f"Average draft model time per sample: {total_draft_time:.4f}s")
    print(f"Draft/Target time ratio: {total_draft_time / total_target_time:.3f}")

    if dist.is_main():
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)

        # 绘制：横坐标 acc length，纵坐标 ppl
        # 将每个 speculative step 的 (acc_len, step_avg_nll) 聚合起来
        pairs_by_acc: dict[int, list[float]] = defaultdict(list)
        for r in responses:
            accs = r[args.block_size].acceptance_lengths
            nlls = r[args.block_size].step_avg_nlls
            # 对齐到共同步数（理论上两者长度一致；保险起见取 min）
            for a, n in zip(accs, nlls):
                pairs_by_acc[int(a)].append(float(n))

        acc_x = sorted(pairs_by_acc.keys())
        ppl_y_all = [float(np.exp(np.mean(pairs_by_acc[a]))) for a in acc_x]

        # 画图时过滤 mean_ppl > 100 的点（统计不变，仅显示时排除）
        acc_x_plot = [a for a, p in zip(acc_x, ppl_y_all) if p <= 100.0]
        ppl_y_plot = [p for p in ppl_y_all if p <= 100.0]

        plt.figure(figsize=(10, 6))
        plt.plot(acc_x_plot, ppl_y_plot, marker='o', markersize=4)
        plt.title("Draft Token PPL vs Acceptance Length")
        plt.xlabel("Acceptance Length (acc_len)")
        plt.ylabel("PPL")
        plt.grid(True)
        plt.savefig("plots/ppl_vs_acc_len.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        # 额外给一个散点图，便于看分布
        scatter_x = []
        scatter_y = []
        for a, nll_list in pairs_by_acc.items():
            for n in nll_list:
                scatter_x.append(a)
                scatter_y.append(float(np.exp(n)))
        plt.scatter(scatter_x, scatter_y, alpha=0.25, s=8)
        plt.title("Draft Token PPL vs Acceptance Length (Scatter)")
        plt.xlabel("Acceptance Length (acc_len)")
        plt.ylabel("PPL")
        plt.grid(True)
        plt.savefig("plots/ppl_vs_acc_len_scatter.png")
        plt.close()

        print(f"[stats] Saved ppl_vs_acc_len.png and ppl_vs_acc_len_scatter.png to plots/ directory")

        # 基于有效 token（被接受 + 第一个被拒绝）统计 P(accept | ppl)
        all_valid = list(chain(*[r[args.block_size].valid_token_stats for r in responses]))
        if all_valid:
            nlls = np.array([x[0] for x in all_valid], dtype=np.float64)
            accs = np.array([1.0 if x[1] else 0.0 for x in all_valid], dtype=np.float64)
            ppls = np.exp(nlls)
            # 画图时把 ppl>100 归到 100（避免长尾影响观感）
            ppls_plot = np.minimum(ppls, 100.0)

            # 对 PPL 做 log 分桶（更稳定）
            logppl = np.log(ppls_plot + 1e-12)
            num_bins = 30
            lo, hi = float(np.percentile(logppl, 1)), float(np.percentile(logppl, 99))
            if hi <= lo:
                lo, hi = float(logppl.min()), float(logppl.max() + 1e-6)
            edges = np.linspace(lo, hi, num_bins + 1)
            bin_ids = np.clip(np.digitize(logppl, edges) - 1, 0, num_bins - 1)

            bin_centers_log = 0.5 * (edges[:-1] + edges[1:])
            bin_centers_ppl = np.exp(bin_centers_log)
            p_accept = np.zeros(num_bins, dtype=np.float64)
            counts = np.zeros(num_bins, dtype=np.int64)
            for b in range(num_bins):
                m = bin_ids == b
                c = int(m.sum())
                counts[b] = c
                if c > 0:
                    p_accept[b] = float(accs[m].mean())
                else:
                    p_accept[b] = np.nan

            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers_ppl, p_accept, marker='o', markersize=4)
            plt.xscale('log')
            plt.xlim(1.0, 100.0)
            plt.ylim(-0.05, 1.05)
            plt.title('P(accept) vs PPL (valid tokens)')
            plt.xlabel('PPL (log scale, binned)')
            plt.ylabel('Acceptance Probability')
            plt.grid(True, which='both')
            plt.savefig('plots/p_accept_vs_ppl.png')
            plt.close()

            # 接受/拒绝的 PPL 直方图对比
            ppl_acc = ppls[accs == 1.0]
            ppl_rej = ppls[accs == 0.0]
            plt.figure(figsize=(10, 6))
            bins = np.logspace(np.log10(1.0), np.log10(100.0), 40)
            plt.hist(np.minimum(ppl_acc, 100.0), bins=bins, alpha=0.6, label='accepted', density=True)
            plt.hist(np.minimum(ppl_rej, 100.0), bins=bins, alpha=0.6, label='first_rejected', density=True)
            plt.xscale('log')
            plt.xlim(1.0, 100.0)
            plt.title('PPL distribution of valid tokens')
            plt.xlabel('PPL (log scale)')
            plt.ylabel('Density')
            plt.grid(True, which='both')
            plt.legend()
            plt.savefig('plots/ppl_hist_valid_tokens.png')
            plt.close()

            print('[stats] Saved p_accept_vs_ppl.png and ppl_hist_valid_tokens.png')

        # 位置 NLL 统计（按 acc_len 分组，计算各位置平均 NLL）
        all_step_nlls = list(chain(*[r[args.block_size].step_token_nlls for r in responses]))
        all_step_accs = list(chain(*[r[args.block_size].acceptance_lengths for r in responses]))
        
        if all_step_nlls and all_step_accs:
            nll_by_acc_len = defaultdict(list)
            for nlls, acc_len in zip(all_step_nlls, all_step_accs):
                nll_by_acc_len[int(acc_len)].append(nlls)
            
            print("\n[pos-nll] Grouped by Acceptance Length:")
            for acc_len in sorted(nll_by_acc_len.keys()):
                group_nlls = np.array(nll_by_acc_len[acc_len]) # (num_samples, block_size-1)
                group_mean = group_nlls.mean(axis=0)
                print(f"  acc_len={acc_len} (count={len(group_nlls)}):")
                print(f"    mean NLL: {[round(float(x), 3) for x in group_mean]}")

            # 之前的全局统计保留（可选，或者直接替换为分组统计）
            nll_mat = np.array(all_step_nlls, dtype=np.float64)
            pos_mean = nll_mat.mean(axis=0)
            diffs = np.diff(pos_mean)
            nondecreasing_ratio = float((diffs >= 0).mean()) if diffs.size > 0 else 1.0
            print(f"\n[pos-nll] Global mean across all steps: {[round(float(x), 3) for x in pos_mean]}")
            print(f"[pos-nll] Global mean is non-decreasing at {nondecreasing_ratio*100:.1f}% of adjacent positions")

            # --- 模拟预测单元逻辑 ---
            # 构造阈值表 threshold[pos]
            # 根据用户逻辑：pos1=inf, posk = mean_nll(acc_len=k-1, pos=k)
            # 注意：pos 是从 1 开始计数的索引，nll 数组是从 0 开始的
            num_pos = args.block_size - 1
            thresholds = [float('inf')] * num_pos
            for k in range(2, num_pos + 1):
                acc_idx = k - 1
                if acc_idx in nll_by_acc_len:
                    group_mean = np.array(nll_by_acc_len[acc_idx]).mean(axis=0)
                    thresholds[k-1] = float(group_mean[k-1])
            
            print(f"\n[predict-unit] Constructed thresholds: {[round(x, 3) for x in thresholds]}")

            total_saved = 0
            success_saved = 0
            success_count = 0
            total_steps = len(all_step_nlls)
            num_pos = args.block_size - 1
            max_possible_save = total_steps * num_pos

            print("\n[predict-unit] Step-by-step Log (First 50 steps):")
            for i, (nlls, acc_true) in enumerate(zip(all_step_nlls, all_step_accs)):
                # 预测 acc_len
                acc_pred = num_pos
                for j in range(num_pos):
                    if nlls[j] > thresholds[j]:
                        acc_pred = j
                        break
                
                # 统计
                saved = num_pos - acc_pred
                total_saved += saved
                is_success = acc_pred >= acc_true
                if is_success:
                    success_count += 1
                    success_saved += saved
                
                # 逐 step 打印 (限制前 50 个避免日志过长)
                if i < 50:
                    nlls_str = [round(float(x), 3) for x in nlls]
                    thres_str = [round(float(x), 3) for x in thresholds]
                    print(f"  step {i:03d}: pred={acc_pred} true={acc_true} success={is_success}")
                    print(f"    current nll: {nlls_str}")
                    print(f"    thresholds : {thres_str}")
                
            print(f"\n[predict-unit] Summary Statistics:")
            print(f"  Success Rate P(pred >= true): {success_count/total_steps*100:.2f}%")
            print(f"  Total Verify Saving Ratio: {total_saved/max_possible_save*100:.2f}%")
            if success_count > 0:
                # 使用第 2 种口径：成功子集的分母
                print(f"  Success-case Saving Ratio: {success_saved/(success_count * num_pos)*100:.2f}%")

            # 导出离线调参所需 JSON
            if args.export_predict_json is not None:
                export_obj = {
                    "block_size": int(args.block_size),
                    "num_pos": int(num_pos),
                    "thresholds": [float(x) for x in thresholds],
                    "steps": [
                        {
                            "acc_true": int(a),
                            "token_nll": [float(x) for x in n],
                        }
                        for n, a in zip(all_step_nlls, all_step_accs)
                    ],
                }
                try:
                    import json
                    # 确保父目录存在
                    os.makedirs(os.path.dirname(args.export_predict_json), exist_ok=True)
                    with open(args.export_predict_json, "w", encoding="utf-8") as f:
                        json.dump(export_obj, f, ensure_ascii=False)
                    print(f"[predict-unit] Saved offline json to: {args.export_predict_json}")
                except Exception as e:
                    print(f"[predict-unit] Failed to save json: {e}")
            # -----------------------

        # 打印 5 组示例：每组展示各位置 token 文本 + NLL
        all_step_ids = list(chain(*[r[args.block_size].step_token_ids for r in responses]))
        all_step_acc = list(chain(*[r[args.block_size].acceptance_lengths for r in responses]))
        if all_step_nlls and all_step_ids and all_step_acc:
            num_examples = min(5, len(all_step_nlls), len(all_step_acc), len(all_step_ids))
            for ex_i in range(num_examples):
                ids = all_step_ids[ex_i]
                nlls = all_step_nlls[ex_i]
                acc_len = int(all_step_acc[ex_i])
                toks = tokenizer.convert_ids_to_tokens(ids)
                pairs = [f"{t}:{float(n):.3f}" for t, n in zip(toks, nlls)]
                # 每个 example 打印成两行：第一行头部信息，第二行内容（避免终端自动断行难读）
                print(f"[example {ex_i}] acc_len={acc_len} (tokens@pos1..{len(pairs)})")
                print("  " + " | ".join(pairs))

    if dist.is_main() and args.delta_n_mode != "none":
        report_delta_stats()
        if args.delta_n_mode == "learn" and args.delta_n_config is not None:
            save_fixed_n_config(args.delta_n_config, selection=args.delta_n_select)
            print(f"[delta-n][learn] saved fixed_n config to: {args.delta_n_config}")

if __name__ == "__main__":
    main()
