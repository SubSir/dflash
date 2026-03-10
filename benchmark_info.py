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

try:
    from safetensors import safe_open
except Exception:
    safe_open = None
import distributed as dist


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def hidden_entropy_over_dim(hidden: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy over hidden dimension using softmax-normalized activations."""
    probs = torch.softmax(hidden.float(), dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)


def target_layer0_output_hidden(
    target: AutoModelForCausalLM,
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Get hidden states after full target model layer0 output (attention + mlp)."""
    hidden_states = target.model.embed_tokens(token_ids)
    layer0 = target.model.layers[0]
    residual = hidden_states
    hidden_states = layer0.input_layernorm(hidden_states)
    position_embeddings = target.model.rotary_emb(hidden_states, position_ids)
    attn_output = layer0.self_attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=position_embeddings,
    )[0]
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = layer0.post_attention_layernorm(hidden_states)
    mlp_output = layer0.mlp(hidden_states)
    return residual + mlp_output


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
    wrong_token_topk_stats = []  # list of stats for the first rejected position in each speculative step
    rejected_prev_same_flags = []  # rejected token == previous token?
    accepted_adj_same_pair_total = 0  # total adjacent accepted pairs across steps
    accepted_adj_same_pair_hits = 0  # adjacent accepted pairs with identical token id
    accepted_adj_pair_eligible_steps = 0  # steps with at least 2 accepted tokens
    accepted_adj_pair_hit_steps = 0  # steps containing >=1 identical adjacent accepted pair
    step_entropy_blocks = []  # list of per-step entropy arrays for draft tokens after target layer0 output

    # 二次提议统计：在首次提议被部分拒绝时，使用“首次提议时的 target_hidden + accept前缀+mask”再提议一次
    second_try_total_steps = 0
    second_try_any_accept_steps = 0
    second_try_total_tokens = 0
    second_try_accepted_tokens = 0
    second_try_accept_lengths = []
    second_try_first_accept_lengths = []

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            # Draft forward timing
            draft_start = cuda_time()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            first_try_noise_kv_by_layer = []
            draft_hidden_all = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                collect_noise_kv_by_layer=first_try_noise_kv_by_layer,
                is_causal=False,
            )
            draft_hidden = draft_hidden_all[:, -block_size + 1 :, :]
            draft_logits = target.lm_head(draft_hidden)
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            draft_token_position_ids = block_position_ids[:, 1:]
            layer0_output_hidden = target_layer0_output_hidden(
                target=target,
                token_ids=block_output_ids[:, 1:],
                position_ids=draft_token_position_ids,
            )
            step_entropy = hidden_entropy_over_dim(layer0_output_hidden).squeeze(0)
            step_entropy_list = step_entropy.detach().to("cpu").tolist()
            step_entropy_blocks.append(step_entropy_list)

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

        if block_size > 1 and int(acceptance_length) < (block_size - 1):
            # 串行二次提议：在“第一次已接受前缀”基础上，继续预测后续一个新 block
            second_try_total_steps += 1
            acc = int(acceptance_length)

            second_start = start + acc
            second_block_position_ids = position_ids[:, second_start : second_start + block_size]

            # 二次提议输入保持全 mask，通过 retry_noise_kv 注入第一次已接受前缀信息
            second_input_ids = torch.full_like(block_output_ids, mask_token_id)
            second_noise_embedding = target.model.embed_tokens(second_input_ids)
            second_hidden_all = model(
                target_hidden=target_hidden,
                noise_embedding=second_noise_embedding,
                position_ids=second_block_position_ids,
                past_key_values=None,
                use_cache=False,
                retry_accept_length=acc + 1,
                retry_noise_kv_by_layer=first_try_noise_kv_by_layer,
                is_causal=False,
            )
            second_draft_logits = target.lm_head(second_hidden_all[:, -block_size + 1 :, :])

            # 串行 verify 序列：第0位锚点用第一次最后一个已接受 token，后续为二次提议 token
            second_block_ids = torch.full_like(block_output_ids, mask_token_id)
            second_block_ids[:, 0] = block_output_ids[:, acc]
            second_block_ids[:, 1:] = sample(second_draft_logits)

            past_key_values_target.crop(start)
            second_output = target(
                second_block_ids,
                position_ids=second_block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=False,
            )
            second_posterior = sample(second_output.logits, temperature)
            second_acceptance_length = (
                (second_block_ids[:, 1:] == second_posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            )

            second_gain = int(second_acceptance_length)
            second_try_total_tokens += int(block_size - 1)
            second_try_accepted_tokens += second_gain
            second_try_first_accept_lengths.append(int(acc))
            second_try_accept_lengths.append(int(second_acceptance_length))
            if second_gain > 0:
                second_try_any_accept_steps += 1

        if block_size > 1:
            step_id = len(step_entropy_blocks)
            print(
                f"[entropy][target_layer0_output][spec_step={step_id}] per_token={['%.6f' % x for x in step_entropy_blocks[-1]]} "
                f"verify_acc_len={int(acceptance_length) + 1}"
            )

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

            # 统计 accepted token 中相邻 pair 的“相同 token”概率
            accepted_cnt = int(acceptance_length)
            if accepted_cnt >= 2:
                accepted_adj_pair_eligible_steps += 1
                acc_tokens = block_output_ids[0, 1 : 1 + accepted_cnt]
                same_adj = acc_tokens[1:] == acc_tokens[:-1]
                pair_total = int(same_adj.numel())
                pair_hits = int(same_adj.sum().item())
                accepted_adj_same_pair_total += pair_total
                accepted_adj_same_pair_hits += pair_hits
                if pair_hits > 0:
                    accepted_adj_pair_hit_steps += 1

            # 统计每个 speculative step 第一个拒绝位置上：正确 token 在 draft top-k 的情况
            if int(acceptance_length) < block_size - 1:
                reject_pos = int(acceptance_length)
                correct_token_id = int(posterior[0, reject_pos].item())

                reject_logits = draft_logits[0, reject_pos]
                reject_probs = torch.softmax(reject_logits, dim=-1)
                k = min(5, reject_probs.shape[-1])
                topk_probs, topk_ids = torch.topk(reject_probs, k=k, dim=-1)

                correct_prob = float(reject_probs[correct_token_id].item())
                in_topk = topk_ids.eq(correct_token_id)
                rank = int(torch.nonzero(in_topk, as_tuple=True)[0].item() + 1) if in_topk.any() else None

                rejected_token_id = int(block_output_ids[0, reject_pos + 1].item())
                prev_token_id = int(block_output_ids[0, reject_pos].item())
                rejected_prev_same_flags.append(bool(rejected_token_id == prev_token_id))

                wrong_token_topk_stats.append(
                    {
                        "reject_pos": reject_pos,
                        "correct_token_id": correct_token_id,
                        "correct_prob": correct_prob,
                        "rejected_token_id": rejected_token_id,
                        "prev_token_id": prev_token_id,
                        "rejected_eq_prev": bool(rejected_token_id == prev_token_id),
                        "topk_token_ids": [int(x) for x in topk_ids.detach().to("cpu").tolist()],
                        "topk_probs": [float(x) for x in topk_probs.detach().to("cpu").tolist()],
                        "correct_rank_in_top5": rank,
                        "correct_in_top2": bool(rank is not None and rank <= 2),
                        "correct_in_top3": bool(rank is not None and rank <= 3),
                        "correct_in_top4": bool(rank is not None and rank <= 4),
                        "correct_in_top5": bool(rank is not None and rank <= 5),
                    }
                )

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
        wrong_token_topk_stats=wrong_token_topk_stats,
        rejected_prev_same_flags=rejected_prev_same_flags,
        accepted_adj_same_pair_total=accepted_adj_same_pair_total,
        accepted_adj_same_pair_hits=accepted_adj_same_pair_hits,
        accepted_adj_pair_eligible_steps=accepted_adj_pair_eligible_steps,
        accepted_adj_pair_hit_steps=accepted_adj_pair_hit_steps,
        step_entropy_blocks=step_entropy_blocks,
        second_try_total_steps=second_try_total_steps,
        second_try_any_accept_steps=second_try_any_accept_steps,
        second_try_total_tokens=second_try_total_tokens,
        second_try_accepted_tokens=second_try_accepted_tokens,
        second_try_accept_lengths=second_try_accept_lengths,
        second_try_first_accept_lengths=second_try_first_accept_lengths,
    )


def _expected_accept_len_from_confidences(confidences: list[float]) -> float:
    """按条件概率计算一个 speculative step 的期望接收长度。

    设第 i 个 draft token 的 confidence 为 p_i（这里直接视为“该 token 被接收”的条件概率）。
    则接收长度 L 的期望：
      E[L] = sum_{k=1..m} P(L >= k)
           = sum_{k=1..m} prod_{i=1..k} p_i
    其中 m = block_size - 1。
    """
    if len(confidences) == 0:
        return 0.0

    p = torch.tensor(confidences, dtype=torch.float32)
    p = p.clamp_(0.0, 1.0)
    expected = torch.cumprod(p, dim=0).sum()
    return float(expected.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="mt-bench")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--export-predict-json",
        type=str,
        default=None,
        help="Path to save per-sample confidence-based acceptance expectation json",
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

    # 错误 token 分析：第一个拒绝位置上，正确 token 在 draft top2~top5 的覆盖率与概率
    wrong_topk_stats_all = list(chain(*[r[args.block_size].wrong_token_topk_stats for r in responses]))
    if len(wrong_topk_stats_all) > 0:
        mean_correct_prob = float(np.mean([x["correct_prob"] for x in wrong_topk_stats_all]))
        hit2 = float(np.mean([1.0 if x["correct_in_top2"] else 0.0 for x in wrong_topk_stats_all]))
        hit3 = float(np.mean([1.0 if x["correct_in_top3"] else 0.0 for x in wrong_topk_stats_all]))
        hit4 = float(np.mean([1.0 if x["correct_in_top4"] else 0.0 for x in wrong_topk_stats_all]))
        hit5 = float(np.mean([1.0 if x["correct_in_top5"] else 0.0 for x in wrong_topk_stats_all]))
        print(f"[wrong-token] checked reject positions: {len(wrong_topk_stats_all)}")
        print(f"[wrong-token] mean prob of correct token: {mean_correct_prob:.6f}")
        print(
            f"[wrong-token] hit@2={hit2 * 100:.2f}% hit@3={hit3 * 100:.2f}% "
            f"hit@4={hit4 * 100:.2f}% hit@5={hit5 * 100:.2f}%"
        )
    else:
        print("[wrong-token] no rejected draft token found for analysis")

    rejected_prev_same_all = list(chain(*[r[args.block_size].rejected_prev_same_flags for r in responses]))
    if len(rejected_prev_same_all) > 0:
        rejected_prev_same_prob = float(np.mean([1.0 if x else 0.0 for x in rejected_prev_same_all]))
        print(
            f"[reject-repeat] P(rejected token == previous token) = {rejected_prev_same_prob * 100:.2f}% "
            f"({sum(rejected_prev_same_all)}/{len(rejected_prev_same_all)})"
        )
    else:
        print("[reject-repeat] no rejected token found")

    accepted_pair_total = int(np.sum([r[args.block_size].accepted_adj_same_pair_total for r in responses]))
    accepted_pair_hits = int(np.sum([r[args.block_size].accepted_adj_same_pair_hits for r in responses]))
    accepted_pair_steps = int(np.sum([r[args.block_size].accepted_adj_pair_eligible_steps for r in responses]))
    accepted_pair_hit_steps = int(np.sum([r[args.block_size].accepted_adj_pair_hit_steps for r in responses]))

    if accepted_pair_total > 0:
        pair_same_prob = float(accepted_pair_hits / accepted_pair_total)
        print(
            f"[accept-repeat] P(adjacent accepted pair has same token) = {pair_same_prob * 100:.2f}% "
            f"({accepted_pair_hits}/{accepted_pair_total})"
        )
    else:
        print("[accept-repeat] no adjacent accepted token pair found")

    if accepted_pair_steps > 0:
        step_has_same_pair_prob = float(accepted_pair_hit_steps / accepted_pair_steps)
        print(
            f"[accept-repeat] P(step has >=1 same adjacent accepted pair | step has >=2 accepted) "
            f"= {step_has_same_pair_prob * 100:.2f}% ({accepted_pair_hit_steps}/{accepted_pair_steps})"
        )

    # 二次提议效果统计
    second_try_total_steps = int(np.sum([r[args.block_size].second_try_total_steps for r in responses]))
    second_try_any_accept_steps = int(np.sum([r[args.block_size].second_try_any_accept_steps for r in responses]))
    second_try_total_tokens = int(np.sum([r[args.block_size].second_try_total_tokens for r in responses]))
    second_try_accepted_tokens = int(np.sum([r[args.block_size].second_try_accepted_tokens for r in responses]))
    second_try_accept_lengths = list(chain(*[r[args.block_size].second_try_accept_lengths for r in responses]))
    second_try_first_accept_lengths = list(chain(*[r[args.block_size].second_try_first_accept_lengths for r in responses]))

    if second_try_total_steps > 0:
        step_gain_prob = float(second_try_any_accept_steps / second_try_total_steps)
        token_accept_rate = (
            float(second_try_accepted_tokens / second_try_total_tokens) if second_try_total_tokens > 0 else 0.0
        )
        print(
            f"[second-try] steps={second_try_total_steps}, improved_steps={second_try_any_accept_steps}, "
            f"P(improved)={step_gain_prob * 100:.2f}%"
        )
        print(
            f"[second-try] serial_tokens={second_try_total_tokens}, accepted_serial_tokens={second_try_accepted_tokens}, "
            f"accept_rate={token_accept_rate * 100:.2f}%"
        )
        if len(second_try_accept_lengths) > 0:
            mean_first = float(np.mean(second_try_first_accept_lengths)) if len(second_try_first_accept_lengths) > 0 else 0.0
            mean_second = float(np.mean(second_try_accept_lengths))
            print(f"[second-try] mean first acceptance length (prefix from first try): {mean_first:.2f}")
            print(f"[second-try] mean serial second acceptance length: {mean_second:.2f}")
            print(f"[second-try] mean total sequential acceptance (first+second): {mean_first + mean_second:.2f}")
    else:
        print("[second-try] no partially rejected speculative steps found")

    # ===== per-sample confidence-based acceptance expectation 导出 =====
    if dist.is_main() and args.export_predict_json is not None:
        num_pos = args.block_size - 1

        # 为避免重复生成的开销，从 responses 中按 sample-turn 顺序重建。
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

            # 将每个 token 的 confidence = exp(-nll) 视作条件接受概率
            step_confidences = [
                torch.exp(-torch.tensor(nlls, dtype=torch.float32)).tolist()
                for nlls in step_nlls
            ]
            step_expected_accept = [
                float(_expected_accept_len_from_confidences(conf))
                for conf in step_confidences
            ]

            samples_export.append(
                {
                    "sample_idx": int(s_idx),
                    "num_pos": int(num_pos),
                    "mean_expected_accept_len": (
                        float(torch.tensor(step_expected_accept, dtype=torch.float32).mean().item())
                        if len(step_expected_accept) > 0
                        else 0.0
                    ),
                    "mean_true_accept_len": (
                        float(torch.tensor(step_accs, dtype=torch.float32).mean().item())
                        if len(step_accs) > 0
                        else 0.0
                    ),
                    "steps": [
                        {
                            "acc_true": int(a),
                            "token_confidence": [float(x) for x in c],
                            "acc_expected": float(e),
                        }
                        for c, e, a in zip(step_confidences, step_expected_accept, step_accs)
                    ],
                }
            )

        export_obj = {
            "block_size": int(args.block_size),
            "num_pos": int(num_pos),
            "predict_type": "confidence_conditional_expectation",
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
