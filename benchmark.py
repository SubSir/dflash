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
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
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
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
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
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        if block_size > 1:
            draft_logprobs = torch.log_softmax(draft_logits, dim=-1)
            proposed = block_output_ids[:, 1:]  # (1, block_size-1)
            token_logp = draft_logprobs.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)  # (1, block_size-1)
            token_nll = (-token_logp).squeeze(0)  # (block_size-1,)
            step_avg_nlls.append(token_nll.mean().item())

            step_token_nlls.append(token_nll.detach().to("cpu").tolist())
            step_token_ids.append(proposed.squeeze(0).detach().to("cpu").tolist())

            valid_k = min(int(acceptance_length) + 1, block_size - 1)
            if valid_k > 0:
                for j in range(valid_k):
                    is_acc = j < int(acceptance_length)
                    valid_token_stats.append((float(token_nll[j].item()), bool(is_acc)))

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]

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


def _compute_thresholds_from_steps(step_nlls: list[list[float]], step_accs: list[int], num_pos: int) -> list[float]:
    """按单个 sample 的 steps 统计阈值。

    逻辑保持和原先一致：
      thresholds[0] = +inf
      thresholds[k-1] = mean_nll(acc_len=k-1 group, position=k)  (k从2到num_pos)

    注意：step_accs 里的 acc_len 取值范围是 [1, block_size]。
    我们只用到 acc_len in [1, num_pos]，因为 nll 只有 num_pos 个位置。
    """
    nll_by_acc_len: dict[int, list[list[float]]] = defaultdict(list)
    for nlls, a in zip(step_nlls, step_accs):
        nll_by_acc_len[int(a)].append(nlls)

    thresholds = [float("inf")] * num_pos
    for k in range(2, num_pos + 1):
        acc_idx = k - 1
        if acc_idx in nll_by_acc_len and len(nll_by_acc_len[acc_idx]) > 0:
            group_mean = np.array(nll_by_acc_len[acc_idx], dtype=np.float64).mean(axis=0)
            thresholds[k - 1] = float(group_mean[k - 1])
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--export-predict-json",
        type=str,
        default=None,
        help="Path to save per-sample thresholds + steps json for offline tuning",
    )
    parser.add_argument(
        "--delta-n-mode",
        type=str,
        default="none",
        choices=["none", "learn", "infer"],
        help="none: disable delta-n pseudo quant; learn: collect stats and save per-layer n; infer: load per-layer n and apply",
    )
    parser.add_argument("--delta-n-config", type=str, default=None, help="Path to save/load per-layer n config (json)")
    parser.add_argument(
        "--delta-n-select",
        type=str,
        default="best_count",
        choices=["best_count", "total_mse"],
        help="How to select fixed n from stats when saving",
    )
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

    # responses 保存每个 sample 的结果（包含两种 block_size 的输出）
    responses = []

    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []

        # 每个 sample 内部按 turn 跑生成，收集该 sample 的所有 step 数据
        sample_steps_nlls: list[list[float]] = []
        sample_steps_accs: list[int] = []

        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
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
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

            # 把这个 turn 的 steps 追加到 sample 级别
            sample_steps_nlls.extend(spec_response.step_token_nlls)
            sample_steps_accs.extend([int(x) for x in spec_response.acceptance_lengths])

            responses.append(response)

        # 把 sample 级别统计结果挂到最后一个 response 上（只在主进程最终汇总时使用）
        # 由于 responses 是按 turn push 的，这里不太好原地修改；我们在导出 JSON 时重新按 sample 重建。

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

    # ===== per-sample thresholds 导出 =====
    if dist.is_main() and args.export_predict_json is not None:
        num_pos = args.block_size - 1

        # 重新按 dataset sample 级别跑一遍“只聚合统计信息”的逻辑：
        # 注意：上面 responses 的结构是按 turn 展开，不好可靠地反向拼回 sample。
        # 因此这里直接再遍历 dataset（只在主进程），并复用同样的随机种子保证一致性。
        # 代价是会重复一次生成；如果你不想重复，可以进一步重构 responses 的存储结构。

        # 为避免重复生成的开销，先尝试从 responses 中按顺序重建：
        # dataset 每个 sample 的 turn 数是 len(instance['turns'])，responses 也是按 turn append。
        # 因此主进程可以按这个 turn 数切片。
        samples_export = []
        cursor = 0
        for s_idx in range(len(dataset)):
            instance = dataset[s_idx]
            n_turns = len(instance["turns"])
            turn_resps = responses[cursor : cursor + n_turns]
            cursor += n_turns

            # 聚合该 sample 的所有 steps
            step_nlls = list(chain(*[tr[args.block_size].step_token_nlls for tr in turn_resps]))
            step_accs = list(chain(*[tr[args.block_size].acceptance_lengths for tr in turn_resps]))

            thresholds = _compute_thresholds_from_steps(step_nlls, step_accs, num_pos=num_pos)

            samples_export.append(
                {
                    "sample_idx": int(s_idx),
                    "num_pos": int(num_pos),
                    "thresholds": [float(x) for x in thresholds],
                    "steps": [
                        {"acc_true": int(a), "token_nll": [float(x) for x in n]}
                        for n, a in zip(step_nlls, step_accs)
                    ],
                }
            )

        export_obj = {
            "block_size": int(args.block_size),
            "num_pos": int(num_pos),
            "threshold_type": "per_sample",
            "samples": samples_export,
        }

        try:
            import json

            os.makedirs(os.path.dirname(args.export_predict_json), exist_ok=True)
            with open(args.export_predict_json, "w", encoding="utf-8") as f:
                json.dump(export_obj, f, ensure_ascii=False)
            print(f"[predict-unit][per-sample] Saved offline json to: {args.export_predict_json}")
        except Exception as e:
            print(f"[predict-unit][per-sample] Failed to save json: {e}")

    if dist.is_main() and args.delta_n_mode != "none":
        from model.pseudo import report_delta_stats, save_fixed_n_config

        report_delta_stats()
        if args.delta_n_mode == "learn" and args.delta_n_config is not None:
            save_fixed_n_config(args.delta_n_config, selection=args.delta_n_select)
            print(f"[delta-n][learn] saved fixed_n config to: {args.delta_n_config}")


if __name__ == "__main__":
    main()
