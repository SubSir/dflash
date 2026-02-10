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

class PseudoLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, quant_mode: str = "nvfp4"):
        """
        Args:
            quant_mode: "nvfp4", "fp8", or "none" (no quantization)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_mode = quant_mode
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.quant_mode == "nvfp4":
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
            del linear.bias
        del linear.weight
        torch.cuda.empty_cache()

