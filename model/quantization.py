import re
from typing import Optional

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

from sglang.srt.layers.quantization.compressed_tensors.schemes import CompressedTensorsWNA16


class QuantLinearW4A16(nn.Module):
    """Quantized Linear layer using W4A16 scheme.

    IMPORTANT: each instance must have its own scheme instance.
    The scheme stores per-layer state (kernel_config/workspace) in `process_weights_after_loading`.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.scheme = CompressedTensorsWNA16(strategy="symmetric", num_bits=4, group_size=128)
        self.scheme.create_weights(
            self,
            self.out_features,
            self.in_features,
            [self.out_features],
            self.in_features,
            dtype,
            lambda x: x,
        )

    @torch.inference_mode()
    def load_quantized_weights(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_shape: torch.Tensor,
    ) -> None:
        self.weight_packed.data.copy_(weight_packed)
        self.weight_scale.data.copy_(weight_scale)
        self.weight_shape.data.copy_(weight_shape.to(device=self.weight_shape.device))
        self.scheme.process_weights_after_loading(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scheme.apply_weights(self, x, bias=self.bias)


def _iter_named_modules_with_parents(root: nn.Module):
    for name, module in root.named_modules():
        if name == "":
            continue
        parent = root
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        yield name, parent, parts[-1], module


@torch.inference_mode()
def quantize_w4a16_inplace(
    model: nn.Module,
    quant_safetensors_path: str,
    strict: bool = True,
    verbose: bool = True,
) -> int:
    """Replace nn.Linear modules with QuantLinearW4A16 wherever quant keys exist.

    Rule: if quant checkpoint has keys for a module name prefix, the model must contain
    that module and it will be replaced. If any quant prefix cannot be mapped to a module,
    raise (strict behavior).
    """

    if safetensors_load_file is None:
        raise ImportError(
            "Missing dependency: safetensors. Install with `pip install safetensors` to load .safetensors files."
        )

    qsd = safetensors_load_file(quant_safetensors_path, device="cpu")

    quant_prefixes: set[str] = set()
    for k in qsd.keys():
        if k.endswith(".weight_packed"):
            quant_prefixes.add(k[: -len(".weight_packed")])

    name_to_module: dict[str, nn.Module] = {name: m for name, m in model.named_modules() if name}

    missing_in_model = sorted([p for p in quant_prefixes if p not in name_to_module])
    if strict and missing_in_model:
        raise KeyError(f"Quant keys not found in model modules: {missing_in_model[:20]}")

    replaced = 0
    for prefix in sorted(quant_prefixes):
        m = name_to_module.get(prefix)
        if m is None:
            continue
        if not isinstance(m, nn.Linear):
            if strict:
                raise TypeError(f"Quant prefix {prefix} maps to non-Linear module: {type(m)}")
            continue

        wpk = qsd.get(f"{prefix}.weight_packed")
        wsc = qsd.get(f"{prefix}.weight_scale")
        wsh = qsd.get(f"{prefix}.weight_shape")
        if wpk is None or wsc is None or wsh is None:
            if strict:
                raise KeyError(
                    f"Missing quant tensors for {prefix}: "
                    f"packed={wpk is not None} scale={wsc is not None} shape={wsh is not None}"
                )
            continue

        qlinear = QuantLinearW4A16(
            in_features=int(m.in_features),
            out_features=int(m.out_features),
            bias=(m.bias is not None),
            dtype=m.weight.dtype,
        ).to(device=m.weight.device)

        qlinear.load_quantized_weights(
            weight_packed=wpk.to(device=m.weight.device, non_blocking=True),
            weight_scale=wsc.to(device=m.weight.device, non_blocking=True),
            weight_shape=wsh,
        )

        if m.bias is not None:
            qlinear.bias.data.copy_(m.bias.data)

        # set into parent
        parent = model
        parts = prefix.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], qlinear)

        replaced += 1
        if verbose:
            print(
                f"[draft quant][replace] {prefix}: in={m.in_features} out={m.out_features} "
                f"packed={tuple(wpk.shape)} scale={tuple(wsc.shape)} wsh={wsh.tolist()}"
            )

    # Final strict check: no quant prefix should remain un-replaced.
    if strict:
        # Rebuild name_to_module after replacement.
        name_to_module2: dict[str, nn.Module] = {name: m for name, m in model.named_modules() if name}
        not_replaced = [p for p in quant_prefixes if isinstance(name_to_module2.get(p), nn.Linear)]
        if not_replaced:
            raise RuntimeError(f"Quantization incomplete, still nn.Linear for prefixes: {not_replaced[:20]}")

    return replaced


@torch.inference_mode()
def audit_w4a16_replacement(model: nn.Module, quant_safetensors_path: str) -> dict:
    if safetensors_load_file is None:
        raise ImportError(
            "Missing dependency: safetensors. Install with `pip install safetensors` to load .safetensors files."
        )

    qsd = safetensors_load_file(quant_safetensors_path, device="cpu")
    quant_prefixes: set[str] = set()
    for k in qsd.keys():
        if k.endswith(".weight_packed"):
            quant_prefixes.add(k[: -len(".weight_packed")])

    name_to_module: dict[str, nn.Module] = {name: m for name, m in model.named_modules() if name}

    total_linear = sum(1 for m in name_to_module.values() if isinstance(m, nn.Linear))
    total_quantlinear = sum(1 for m in name_to_module.values() if isinstance(m, QuantLinearW4A16))

    missing_in_model = sorted([p for p in quant_prefixes if p not in name_to_module])
    not_replaced_but_quant_available = sorted(
        [p for p in quant_prefixes if isinstance(name_to_module.get(p), nn.Linear)]
    )

    return {
        "total_linear": total_linear,
        "total_quantlinear": total_quantlinear,
        "quant_keys_prefixes": len(quant_prefixes),
        "missing_in_model": missing_in_model,
        "not_replaced_but_quant_available": not_replaced_but_quant_available,
    }
