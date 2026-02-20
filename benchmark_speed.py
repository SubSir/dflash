import argparse
import json
import os
import random
import time
from itertools import chain
from types import SimpleNamespace

import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import distributed as dist
from model import DFlashDraftModel, extract_context_feature, load_and_process_dataset, sample


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _predict_acc_len_from_nll(token_nll: torch.Tensor, thresholds: torch.Tensor) -> int:
    # token_nll: (num_pos,), thresholds: (num_pos,)
    gt = token_nll > thresholds
    if bool(gt.any().item()):
        return int(gt.float().argmax().item())
    return int(token_nll.numel())


def _build_thresholds_from_running(
    num_pos: int,
    sum_by_acc: torch.Tensor,
    count_by_acc: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # sum_by_acc: (num_pos + 1, num_pos), count_by_acc: (num_pos + 1,)
    thresholds = torch.full((num_pos,), float("inf"), device=device, dtype=torch.float32)
    acc_idx = torch.arange(num_pos, device=device, dtype=torch.long)
    c = count_by_acc.index_select(0, acc_idx).to(torch.float32)
    has = c > 0
    if bool(has.any().item()):
        mean_vec = sum_by_acc.index_select(0, acc_idx).to(torch.float32) / c.clamp_min(1.0).unsqueeze(1)
        thresholds = torch.where(has, mean_vec[acc_idx, acc_idx], thresholds)
    return thresholds


@torch.inference_mode()
def generate_verify_all(
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

    decode_start = cuda_time()
    start = input_ids.shape[1]
    draft_prefill = True
    total_target_time = 0.0
    total_draft_time = 0.0
    total_target_verify_calls = 0
    total_target_verify_tokens = 0
    total_acc_tokens = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        if block_size > 1:
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
        total_target_verify_calls += 1
        total_target_verify_tokens += int(block_output_ids.shape[1])

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        total_acc_tokens += int(acceptance_length) + 1

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

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
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)

    return SimpleNamespace(
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        total_target_time=total_target_time,
        total_draft_time=total_draft_time,
        total_target_verify_calls=total_target_verify_calls,
        total_target_verify_tokens=total_target_verify_tokens,
        total_acc_tokens=total_acc_tokens,
    )


@torch.inference_mode()
def generate_verify_pred_k_online(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    offset: int,
    warmup_steps: int,
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

    decode_start = cuda_time()
    start = input_ids.shape[1]
    draft_prefill = True
    total_target_time = 0.0
    total_draft_time = 0.0
    total_target_verify_calls = 0
    total_target_verify_tokens = 0
    total_acc_tokens = 0

    # 诊断指标（只统计非 warmup 阶段）：
    # - short_verify: verify_len < block_size 的比例
    # - boundary_hit_when_short: verify_len < block_size 且 acc_true == verify_len 的比例
    short_verify_count = 0
    short_verify_denom = 0
    boundary_hit_when_short_count = 0
    boundary_hit_when_short_denom = 0

    num_pos = block_size - 1

    # 每次调用本函数即对应一个“独立的在线统计过程”。
    sum_by_acc = torch.zeros((num_pos + 1, num_pos), device=model.device, dtype=torch.float32)
    count_by_acc = torch.zeros((num_pos + 1,), device=model.device, dtype=torch.int64)
    global_step = 0
    step_traces = []

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        if block_size > 1:
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

            draft_logprobs = torch.log_softmax(draft_logits, dim=-1)
            proposed = block_output_ids[:, 1:]
            token_logp = draft_logprobs.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
            token_nll = (-token_logp).squeeze(0)  # (num_pos,)

            draft_time = cuda_time() - draft_start
            total_draft_time += draft_time
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()
        else:
            token_nll = None

        draft_pred_start = cuda_time()

        thresholds = _build_thresholds_from_running(
            num_pos=num_pos,
            sum_by_acc=sum_by_acc,
            count_by_acc=count_by_acc,
            device=model.device,
        )

        if global_step < warmup_steps or token_nll is None:
            verify_len = block_size
            acc_pred = num_pos
            k = num_pos
        else:
            acc_pred = _predict_acc_len_from_nll(token_nll, thresholds)
            k = min(int(acc_pred) + int(offset), num_pos)
            verify_len = int(k) + 1

        draft_pred_time = cuda_time() - draft_pred_start
        total_draft_time += draft_pred_time

        block_output_ids_k = block_output_ids[:, :verify_len]
        block_position_ids_k = block_position_ids[:, :verify_len]

        target_start = cuda_time()
        output = target(
            block_output_ids_k,
            position_ids=block_position_ids_k,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )
        target_time = cuda_time() - target_start
        total_target_time += target_time
        total_target_verify_calls += 1
        total_target_verify_tokens += int(block_output_ids_k.shape[1])

        posterior = sample(output.logits, temperature)

        if verify_len > 1:
            acceptance_length = (
                (block_output_ids_k[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            )
        else:
            acceptance_length = 0

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids_k[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acc_true = int(acceptance_length) + 1
        total_acc_tokens += int(acc_true)

        if global_step >= warmup_steps:
            short_verify_denom += 1
            if int(verify_len) < int(block_size):
                short_verify_count += 1
                boundary_hit_when_short_denom += 1
                if int(acc_true) == int(verify_len):
                    boundary_hit_when_short_count += 1

        acc_true_idx = int(min(int(acc_true), int(num_pos)))

        trace = {
            "token_nll": token_nll.detach() if token_nll is not None else None,
            "thresholds": thresholds.detach(),
            "verify_len": int(verify_len),
            "acc_true": int(acc_true),
            "acc_pred": int(acc_pred),
        }
        step_traces.append(trace)

        draft_update_start = cuda_time()
        if token_nll is not None:
            sum_by_acc[acc_true_idx, :] += token_nll.to(torch.float32)
            count_by_acc[acc_true_idx] += 1
        draft_update_time = cuda_time() - draft_update_start
        total_draft_time += draft_update_time

        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]

        global_step += 1

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]

    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)

    pred_stats = {
        "warmup_steps": int(warmup_steps),
        "offset": int(offset),
        "global_step": int(global_step),
        "step_traces": step_traces,
        "short_verify_count": int(short_verify_count),
        "short_verify_denom": int(short_verify_denom),
        "short_verify_prob": float(short_verify_count / max(1, short_verify_denom)),
        "boundary_hit_when_short_count": int(boundary_hit_when_short_count),
        "boundary_hit_when_short_denom": int(boundary_hit_when_short_denom),
        "boundary_hit_when_short_prob": float(boundary_hit_when_short_count / max(1, boundary_hit_when_short_denom)),
    }

    return SimpleNamespace(
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        total_target_time=total_target_time,
        total_draft_time=total_draft_time,
        total_target_verify_calls=total_target_verify_calls,
        total_target_verify_tokens=total_target_verify_tokens,
        total_acc_tokens=total_acc_tokens,
        pred_stats=pred_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="swe-bench", help="Single dataset name. Deprecated: prefer --datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets to run sequentially, e.g. 'gsm8k,math500,humaneval,mt-bench'. If set, overrides --dataset.",
    )
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "baseline", "pred"],
        help="Run baseline verify-all, pred verify-k (online), or both",
    )
    parser.add_argument("--offset", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--out-json", type=str, default="speed_results.json")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16,
    ).to(torch.device(f"cuda:{dist.local_rank()}"))
    target.eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        dtype=torch.bfloat16,
    ).to(torch.device(f"cuda:{dist.local_rank()}"))
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    if args.datasets is not None and str(args.datasets).strip():
        dataset_names = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    else:
        dataset_names = [str(args.dataset).strip()]

    out = {"args": vars(args), "by_dataset": {}}

    for data_name in dataset_names:
        dataset = load_and_process_dataset(data_name)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

        responses = []

        indices = range(dist.rank(), len(dataset), dist.size())
        for idx in tqdm(indices, disable=not dist.is_main()):
            instance = dataset[idx]
            messages = []

            for user_content in instance["turns"]:
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

                if args.mode in ["both", "baseline"]:
                    r_all = generate_verify_all(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=tokenizer.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=args.block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                    responses.append(("baseline", r_all))

                if args.mode in ["both", "pred"]:
                    r_pred = generate_verify_pred_k_online(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=tokenizer.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=args.block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        offset=args.offset,
                        warmup_steps=args.warmup_steps,
                        temperature=args.temperature,
                    )
                    responses.append(("pred", r_pred))

                messages.append({"role": "assistant", "content": ""})

        if dist.size() > 1:
            responses = dist.gather(responses, dst=0)
            if not dist.is_main():
                continue
            responses = list(chain(*responses))

        if not dist.is_main():
            continue

        by_mode = {"baseline": [], "pred": []}
        for mode, r in responses:
            by_mode[mode].append(r)

        def _agg(rs):
            return {
                "time_to_first_token": float(sum(x.time_to_first_token for x in rs) / max(1, len(rs))),
                "time_per_output_token": float(sum(x.time_per_output_token for x in rs) / max(1, len(rs))),
                "total_target_time": float(sum(x.total_target_time for x in rs) / max(1, len(rs))),
                "total_draft_time": float(sum(x.total_draft_time for x in rs) / max(1, len(rs))),
                "total_target_verify_calls": int(sum(getattr(x, "total_target_verify_calls", 0) for x in rs)),
                "total_target_verify_tokens": int(sum(getattr(x, "total_target_verify_tokens", 0) for x in rs)),
                "num_output_tokens": float(sum(x.num_output_tokens for x in rs) / max(1, len(rs))),
                "total_acc_tokens": float(sum(getattr(x, "total_acc_tokens", 0) for x in rs) / max(1, len(rs))),
            }

        data_out = {"dataset": data_name, "n_samples": int(len(dataset))}
        if by_mode["baseline"]:
            data_out["baseline_verify_all"] = _agg(by_mode["baseline"])
        if by_mode["pred"]:
            data_out["pred_verify_k_online"] = _agg(by_mode["pred"])

            # 跨所有 pred 请求聚合诊断指标（只统计非 warmup 阶段）
            total_short_count = 0
            total_short_denom = 0
            total_boundary_hit_count = 0
            total_boundary_hit_denom = 0

            for r in by_mode["pred"]:
                ps = r.pred_stats
                total_short_count += int(ps.get("short_verify_count", 0))
                total_short_denom += int(ps.get("short_verify_denom", 0))
                total_boundary_hit_count += int(ps.get("boundary_hit_when_short_count", 0))
                total_boundary_hit_denom += int(ps.get("boundary_hit_when_short_denom", 0))

            data_out["verify_len_diag"] = {
                "short_verify_count": int(total_short_count),
                "short_verify_denom": int(total_short_denom),
                "short_verify_prob": float(total_short_count / max(1, total_short_denom)),
                "boundary_hit_when_short_count": int(total_boundary_hit_count),
                "boundary_hit_when_short_denom": int(total_boundary_hit_denom),
                "boundary_hit_when_short_prob": float(total_boundary_hit_count / max(1, total_boundary_hit_denom)),
                "num_pred_requests": int(len(by_mode["pred"])),
            }

            # pred_stats/step_traces 可能很大：保留一个示例（第一个请求）用于检查结构
            first_pred_stats = dict(by_mode["pred"][0].pred_stats)
            step_traces_ser = []
            for tr in first_pred_stats.get("step_traces", []) or []:
                if not isinstance(tr, dict):
                    step_traces_ser.append(tr)
                    continue
                tr2 = dict(tr)
                for k_ in ["token_nll", "thresholds"]:
                    v = tr2.get(k_)
                    if isinstance(v, torch.Tensor):
                        tr2[k_] = [float(x) for x in v.detach().to("cpu").flatten().tolist()]
                step_traces_ser.append(tr2)
            first_pred_stats["step_traces"] = step_traces_ser

            data_out["pred_stats_example"] = {
                **first_pred_stats,
            }

        if "baseline_verify_all" in data_out and "pred_verify_k_online" in data_out:
            data_out["speedup_time_per_token"] = float(
                data_out["baseline_verify_all"]["time_per_output_token"]
                / max(1e-9, data_out["pred_verify_k_online"]["time_per_output_token"])
            )
            data_out["target_verify_token_ratio"] = float(
                data_out["pred_verify_k_online"]["total_target_verify_tokens"]
                / max(1e-9, data_out["baseline_verify_all"]["total_target_verify_tokens"])
            )

        out["by_dataset"][data_name] = data_out

    if dist.is_main():
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("\n[speed] results saved to:", args.out_json)


if __name__ == "__main__":
    main()
