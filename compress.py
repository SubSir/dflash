import argparse
import os
from datasets import load_dataset
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel
from model.utils import extract_context_feature


def get_calib_dataset(tokenizer, n_samples=128, block_size=512):
    """
    Load calibration dataset from mit-han-lab/pile-val-backup.
    Returns a list of tensors each of shape (1, block_size).
    """
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def main():
    parser = argparse.ArgumentParser(description="Quantize DFlashDraftModel with ModelOpt")
    parser.add_argument("--model_name_or_path", type=str, default="z-lab/Qwen3-8B-DFlash-b16",
                        help="Path to the pretrained draft model.")
    parser.add_argument("--target_model_name_or_path", type=str, default="Qwen/Qwen3-8B",
                        help="Path to the pretrained target model for prefill.")
    parser.add_argument("--export_dir", type=str, default="./quantized_model/Qwen3-8B-DFlash-b16-nvfp4",
                        help="Directory to export the quantized model.")
    parser.add_argument("--calib_size", type=int, default=128,
                        help="Number of calibration samples.")
    parser.add_argument("--quant_type", type=str, choices=["nvfp4", "fp8"], default="nvfp4",
                        help="Quantization type: nvfp4 (NVFP4) or fp8 (FP8).")
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    # Load models
    draft = DFlashDraftModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16).cuda()
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(next(draft.parameters()).device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path, use_fast=False, trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    # Prepare calibration set
    calib_set = get_calib_dataset(tokenizer=tokenizer, n_samples=args.calib_size)

    # Define forward loop for calibration
    def forward_loop(model):
        model.eval()
        with torch.no_grad():
            for batch in calib_set:
                device = next(model.parameters()).device
                batch = batch.to(device)
                q_len = batch.shape[1]
                ctx_len = q_len  # In calibration, target_hidden has the same seq len as the input batch
                position_ids = torch.arange(ctx_len + q_len, device=device).unsqueeze(0)

                # Target prefill to get target_hidden
                past_key_values_target = DynamicCache()
                target_output = target(
                    batch,
                    position_ids=position_ids[:, :ctx_len],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )
                target_hidden = extract_context_feature(target_output.hidden_states, model.target_layer_ids)

                # Draft forward with noise_embedding from target and target_hidden
                noise_embedding = target.model.embed_tokens(batch)
                _ = model(
                    position_ids=position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                )

    # Quantize with selected type
    if args.quant_type == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
    else:
        quant_cfg = mtq.FP8_DEFAULT_CFG

    draft = mtq.quantize(draft, quant_cfg, forward_loop)

    # Export quantized model
    with torch.inference_mode():
        export_hf_checkpoint(
            draft,
            torch.bfloat16,
            args.export_dir,
        )
    # Post-process exported config.json to adjust quant_method
    config_path = os.path.join(args.export_dir, "config.json")
    if os.path.exists(config_path):
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        qcfg = cfg.get("quantization_config")
        if isinstance(qcfg, dict) and qcfg.get("quant_method") == "modelopt":
            if args.quant_type == "nvfp4":
                qcfg["quant_method"] = "modelopt_fp4"
            else:
                qcfg["quant_method"] = "modelopt_fp8"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f"Quantized model exported to: {args.export_dir}")


if __name__ == "__main__":
    main()
