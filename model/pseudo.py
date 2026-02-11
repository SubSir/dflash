import torch.nn as nn
import torch

# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _n_ones(n: int) -> int:
    return (1 << n) - 1


EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L972-L974
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L977-L982
def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# Ref: Based on `_bfloat16_to_float4_e2m1fn_x2` of https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L985-L990
def to_fp4(x: torch.Tensor) -> torch.Tensor:
    x = _f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    x = pack_uint4(x)
    return x


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L13-L22
_kE2M1ToFloatTensor_cache = {}


def _get_kE2M1ToFloatTensor(device):
    if device not in _kE2M1ToFloatTensor_cache:
        _kE2M1ToFloatTensor_cache[device] = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device
        )
    return _kE2M1ToFloatTensor_cache[device]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L25-L32
# Convert FP4 into FP32
def e2m1_to_fp32_vectorized(int4_values):
    """
    Vectorized version of e2m1_to_fp32.
    int4_values: tensor of uint8, each element in [0, 15], on any CUDA device
    """
    device = int4_values.device
    sign_bits = int4_values & 0x8
    abs_indices = (int4_values & 0x7).to(torch.int64)

    lut = _get_kE2M1ToFloatTensor(device)
    float_vals = lut[abs_indices]
    float_vals = torch.where(sign_bits != 0, -float_vals, float_vals)
    return float_vals


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L35-L49
# Unpack float4_e2m1fn_x2 into two separate fp32 values
def unpack_fp4_bytes(a):
    m, n = a.shape
    a = a.view(torch.uint8).flatten()  # shape: (m * n,)

    upper_half_byte = (a & 0xF0) >> 4  # high 4 bits
    lower_half_byte = a & 0x0F  # low 4 bits

    upper_half_float = e2m1_to_fp32_vectorized(upper_half_byte)
    lower_half_float = e2m1_to_fp32_vectorized(lower_half_byte)

    out = torch.stack((lower_half_float, upper_half_float), dim=-1).reshape(m, n * 2)
    return out


def simple_fp4_pseudo_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    Simple pseudo-quantization that converts float/float16/bfloat16 to FP4 and back to float32.
    This function does NOT handle scales - it's a simple quantize-dequantize operation.

    Args:
        x: Input tensor with dtype float, float16, or bfloat16. Must be a CUDA tensor
           with at least 2 dimensions and last dimension divisible by 2.

    Returns:
        torch.Tensor: Dequantized tensor with float32 dtype, same shape as input.
                     The tensor is quantized to FP4 and then dequantized back to float32,
                     simulating the precision loss of FP4 quantization.
    """
    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), f"x.dtype needs to be float, fp16 or bf16 but got {x.dtype}"
    assert x.ndim >= 2, f"x.ndim needs to be >= 1, but got {x.ndim}"
    assert (
        x.shape[-1] % 2 == 0
    ), f"last dim has to be multiple of 2, but got {x.shape[-1]}"

    original_shape = x.shape

    # Convert to FP4
    x_fp4 = to_fp4(x.reshape(-1, original_shape[-1]))

    # Convert back to float32 first
    x_dequantized = unpack_fp4_bytes(x_fp4)

    # Cast back to original dtype
    return x_dequantized.reshape(original_shape)

def quant_to_fp8(weight: torch.Tensor) -> torch.Tensor:
    org_type = weight.dtype
    scale = weight.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-4) / 448.0
    weight = weight / scale
    weight = weight.to(torch.float8_e4m3fn).to(org_type)
    return weight * scale

def quant_to_nvfp4(weight: torch.Tensor) -> torch.Tensor:
    org_shape = weight.shape
    org_dtype = weight.dtype
    weight = weight.reshape(-1, 16)
    global_scale = weight.float().abs().amax() / 448.0 / 6.0
    scale = (weight / global_scale / 6.0).abs().max(dim=-1, keepdim=True).values.clamp(min=1e-4).to(torch.float8_e4m3fn).to(org_dtype)
    weight = (weight / scale / global_scale).to(org_dtype)
    weight = simple_fp4_pseudo_quantize(weight)
    weight = weight * scale * global_scale
    weight = weight.reshape(org_shape).to(org_dtype)
    return weight

import json
from collections import defaultdict

import torch.nn.functional as F

# Global registry to store stats for all layers
PSEUDO_LINEAR_STATS = defaultdict(lambda: {n: {"best_count": 0, "total_mse": 0.0} for n in [1, 2, 4, 8, 16]})

def delta_quantize_dequantize_x(x: torch.Tensor, n: int) -> torch.Tensor:
    """Delta quantize-dequantize x along token dimension (dim=1), fully vectorized.

    Assumptions for fast path (as you stated for hidden_states):
      - x is (bs, seq_len, dim)
      - seq_len == block_size
      - block_size % n == 0

    Grouping rule (within the block):
      - groups: [0..n-1], [n..2n-1], ...
      - token 0 is kept as-is (no quant, no delta)
      - for group0: base is token1, apply delta to tokens [2..n-1]
      - for other groups: base is first token of group (k*n), apply delta to rest

    Note: We quantize both base and delta with `quant_to_nvfp4`.
    """
    if x.dim() != 3:
        return x

    bs, seq_len, dim = x.shape
    if seq_len <= 1:
        return x

    if n == 1:
        return quant_to_nvfp4(x)

    # If not divisible, fallback to original x quant (keeps behavior safe for unexpected shapes)
    if seq_len % n != 0:
        return quant_to_nvfp4(x)

    g = seq_len // n

    x_recon = x.clone()

    # reshape into groups
    xg = x.view(bs, g, n, dim)

    # ---- group 0 ----
    # token 0 unchanged
    if n > 1:
        base0 = xg[:, 0, 1:2, :]  # token1
        base0_hat = quant_to_nvfp4(base0)
        x_recon[:, 1, :] = base0_hat.squeeze(1)
        if n > 2:
            others0 = xg[:, 0, 2:, :]  # tokens [2..n-1]
            delta0 = others0 - base0
            delta0_hat = quant_to_nvfp4(delta0)
            recon0 = base0_hat + delta0_hat
            x_recon[:, 2:n, :] = recon0

    # ---- groups 1..g-1 ----
    if g > 1:
        base = xg[:, 1:, 0:1, :]  # (bs, g-1, 1, dim)
        base_hat = quant_to_nvfp4(base)

        # place base tokens
        base_flat = base_hat.squeeze(2)  # (bs, g-1, dim)
        base_pos = (torch.arange(1, g, device=x.device) * n).view(1, g - 1, 1).expand(bs, g - 1, 1)
        x_recon.scatter_(1, base_pos.expand(bs, g - 1, dim), base_flat)

        if n > 1:
            others = xg[:, 1:, 1:, :]  # (bs, g-1, n-1, dim)
            delta = others - base
            delta_hat = quant_to_nvfp4(delta)
            recon = base_hat + delta_hat  # (bs, g-1, n-1, dim)

            # scatter recon tokens to positions base_pos + [1..n-1]
            offsets = torch.arange(1, n, device=x.device).view(1, 1, n - 1, 1)
            pos = (torch.arange(1, g, device=x.device).view(1, g - 1, 1, 1) * n) + offsets
            pos = pos.expand(bs, g - 1, n - 1, 1)
            x_recon.scatter_(1, pos.expand(bs, g - 1, n - 1, dim).reshape(bs, -1, dim), recon.reshape(bs, -1, dim))

    return x_recon

class PseudoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, quant_mode: str = "nvfp4", layer_name: str = "unknown"):
        """
        Args:
            quant_mode: "nvfp4", "fp8", "delta_learn", "delta_inference", or "none"
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.layer_name = layer_name
        self.fixed_n = 1
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, is_hidden_states: bool = True) -> torch.Tensor:
        w = self.weight

        if self.quant_mode == "delta_learn":
            if is_hidden_states and x.dim() == 3:
                best_n = 1
                min_mse = float("inf")
                for n in [1, 2, 4, 8, 16]:
                    x_hat = delta_quantize_dequantize_x(x, n)
                    mse = F.mse_loss(x, x_hat).item()
                    PSEUDO_LINEAR_STATS[self.layer_name][n]["total_mse"] += mse
                    if mse < min_mse:
                        min_mse = mse
                        best_n = n
                PSEUDO_LINEAR_STATS[self.layer_name][best_n]["best_count"] += 1
            # learning阶段不改变真实前向输入
        elif self.quant_mode == "delta_inference":
            if is_hidden_states and x.dim() == 3:
                x = delta_quantize_dequantize_x(x, self.fixed_n)
            elif not is_hidden_states:
                x = quant_to_nvfp4(x)
        elif self.quant_mode == "nvfp4":
            x = quant_to_nvfp4(x)
        elif self.quant_mode == "fp8":
            x = quant_to_fp8(x)

        if self.bias is not None:
            return x @ w.t() + self.bias
        return x @ w.t()

    def from_linear(self, linear: nn.Linear) -> None:
        self.weight.data = linear.weight.data
        if self.quant_mode == "nvfp4":
            self.weight.data = quant_to_nvfp4(self.weight.data)
        elif self.quant_mode == "fp8":
            self.weight.data = quant_to_fp8(self.weight.data)
        if linear.bias is not None:
            self.bias.data = linear.bias.data
            # Do not delete bias from original linear here if we are just replacing
        # Optimization: cast to original dtype if needed
        self.weight.data = self.weight.data.to(linear.weight.dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(linear.weight.dtype)

def select_fixed_n_config(selection: str = "best_count"):
    """Return dict[layer_name] = fixed_n.

    selection:
      - "best_count": choose n with max best_count
      - "total_mse": choose n with min total_mse
    """
    final_n_config = {}
    for name, stats in PSEUDO_LINEAR_STATS.items():
        if selection == "total_mse":
            best_n = min([1, 2, 4, 8, 16], key=lambda n: stats[n]["total_mse"])
        else:
            best_n = max([1, 2, 4, 8, 16], key=lambda n: stats[n]["best_count"])
        final_n_config[name] = int(best_n)
    return dict(sorted(final_n_config.items()))


def save_fixed_n_config(path: str, selection: str = "best_count") -> dict:
    cfg = select_fixed_n_config(selection=selection)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, sort_keys=True)
    return cfg


def load_fixed_n_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # normalize
    return {str(k): int(v) for k, v in cfg.items()}


def report_delta_stats():
    print("\n" + "="*80)
    print(f"{'Layer Name':<50} | Best n (Count) | Selected n")
    print("-" * 80)
    final_n_config = {}
    for name, stats in sorted(PSEUDO_LINEAR_STATS.items()):
        # Find n with most best_counts
        best_n = 1
        max_count = -1
        counts_str = []
        for n in [1, 2, 4, 8, 16]:
            count = stats[n]["best_count"]
            counts_str.append(f"{n}:{count}")
            if count > max_count:
                max_count = count
                best_n = n
        
        # Alternatively, find n with minimum total_mse
        # min_mse_n = 1
        # min_mse = float('inf')
        # for n in [1, 2, 4, 8, 16]:
        #     if stats[n]["total_mse"] < min_mse:
        #         min_mse = stats[n]["total_mse"]
        #         min_mse_n = n
        
        final_n_config[name] = best_n
        print(f"{name[:50]:<50} | {', '.join(counts_str)} | {best_n}")
    print("="*80 + "\n")
    return final_n_config

