import argparse
import time
import random
import heapq
import copy
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _build_verify_tokens_from_topk(
    topk_probs: torch.Tensor,
    topk_ids: torch.Tensor,
    max_verify_tokens: int,
) -> list[list[int]]:
    """Build a greedy verify tree based on top-k probabilities at each position.

    topk_ids/topk_probs: shape=(block_size-1, k)
    Returns:
      - tree_choices: list[list[int]], each element is a path (sequence of token ids)
    """
    num_pos = topk_probs.shape[0]
    if num_pos == 0:
        return []

    max_verify_tokens = min(int(max_verify_tokens), num_pos)
    heap: list[tuple[float, list[int], float]] = []
    # 初始化：放入位置0的所有token id
    for r in range(topk_probs.shape[1]):
        p = float(topk_probs[0, r].item())
        token_id = int(topk_ids[0, r].item())
        heapq.heappush(heap, (-p, [token_id], p))

    tree_choices: list[list[int]] = []
    i = 0

    while heap and i < max_verify_tokens:
        _, path, prob = heapq.heappop(heap)
        i += 1
        tree_choices.append(path)
        pos = len(path) - 1

        next_pos = pos + 1
        if next_pos < num_pos:
            for r in range(topk_probs.shape[1]):
                next_prob = float(topk_probs[next_pos, r].item())
                next_token_id = int(topk_ids[next_pos, r].item())
                joint_prob = prob * next_prob
                heapq.heappush(heap, (-joint_prob, path + [next_token_id], joint_prob))

    if len(tree_choices) <= 1:
        return tree_choices

    filtered_choices: list[list[int]] = []
    for i, path in enumerate(tree_choices):
        is_prefix = False
        for j, other in enumerate(tree_choices):
            if i == j:
                continue
            if len(other) > len(path) and other[: len(path)] == path:
                is_prefix = True
                break
        if not is_prefix:
            filtered_choices.append(path)

    return filtered_choices

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
    mode: str = "vanilla",
    topk: int = 10,
    max_verify_tokens: int | None = None,
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

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
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

            if mode == "vanilla":
                block_output_ids[:, 1:] = sample(draft_logits)
            else:
                k = min(int(topk), draft_logits.shape[-1])
                draft_probs = torch.softmax(draft_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(draft_probs, k=k, dim=-1)
                if max_verify_tokens is None:
                    max_verify = block_size - 1
                else:
                    max_verify = max_verify_tokens
                tree_choices = _build_verify_tokens_from_topk(
                    topk_probs=topk_probs.squeeze(0),
                    topk_ids=topk_ids.squeeze(0),
                    max_verify_tokens=max_verify,
                )

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        if mode == "topk_tree" and tree_choices:
            best_acceptance_length = -1
            best_output = None
            best_posterior = None
            best_block_output_ids = None

            for path in tree_choices:
                path_len = len(path)
                temp_block_output_ids = torch.full_like(block_output_ids, 3)
                temp_block_output_ids[:, 0] = block_output_ids[:, 0]
                for pos, token_id in enumerate(path):
                    temp_block_output_ids[:, pos + 1] = token_id

                path_past_key_values = copy.deepcopy(past_key_values_target)
                output = target(
                    temp_block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=path_past_key_values,
                    attention_mask=None,
                    use_cache=True,
                    output_hidden_states=True if block_size > 1 else False,
                )
                logits_block = output.logits[:, :path_len + 1, :]
                posterior = sample(logits_block, temperature)
                acceptance_length = (
                    (temp_block_output_ids[:, 1 : path_len + 1] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
                )
                if acceptance_length > best_acceptance_length:
                    best_acceptance_length = acceptance_length
                    best_output = output
                    best_posterior = posterior
                    best_block_output_ids = temp_block_output_ids
                    best_past_key_values_target = path_past_key_values

            output = best_output
            posterior = best_posterior
            block_output_ids = best_block_output_ids
            acceptance_length = best_acceptance_length
            past_key_values_target = best_past_key_values_target
        else:
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                attention_mask=None,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )
            posterior = sample(output.logits, temperature)

            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="mt-bench")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="vanilla",
        choices=["vanilla", "topk_tree"],
        help="vanilla: original sampling; topk_tree: greedy tree/heap sampling based on top-k confidence",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--max-verify-tokens",
        type=int,
        default=None,
        help="max tokens to verify per block in topk_tree mode (default=block_size-1)",
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

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

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
            for bs in [block_size]:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=tokenizer.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    mode=args.mode if bs > 1 else "vanilla",
                    topk=args.topk,
                    max_verify_tokens=args.max_verify_tokens,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    # t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    # print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

if __name__ == "__main__":
    main()