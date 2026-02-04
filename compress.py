from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from model import DFlashDraftModel
import torch
import json
import os


def _strip_quantization_config(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")


device = torch.device("cuda:0")
draft_model = DFlashDraftModel.from_pretrained(
    "z-lab/Qwen3-4B-DFlash-b16",
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
).to(device)

# Apply W4A16 quantization to Linear layers
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

output_dir = "./quantized_model/Qwen3-4B-DFlash-b16-W4A16"
oneshot(
    model=draft_model,
    recipe=recipe,
    output_dir=output_dir,
)

_strip_quantization_config(os.path.join(output_dir, "config.json"))
