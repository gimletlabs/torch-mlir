import torch
from torch.library import Library, impl

# This file is used to register the gml namespace and fused_moe op so that
# code gen succeeds when trying to generate ops for the torch mlir dialect.
# The codegen compares the actual implementation output signatures to the defined
# op, so this module is necessary to provide a fake implementation of the op.
# This op will instead be lowered to an actual fused implementation in the compiler.

# Register the gml namespace and fused_moe op so torch.ops.gml.fused_moe exists
_lib = Library("gml", "DEF")

# Signature sources:
# input: Tensor[num_tokens, hidden_size]
# gate_proj: List of tensors per expert in the layer, each Tensor[inter_size, hidden_size]
# down_proj: List of tensors per expert in the layer, each Tensor[hidden_size, inter_size]
# up_proj: List of tensors per expert in the layer, each Tensor[inter_size, hidden_size]
# expert_indices: Tensor[num_tokens, top_k], int32
# expert_weights: Tensor[num_tokens, top_k], float32
_lib.define(
    "fused_moe(Tensor input, Tensor[] gate_proj, Tensor[] down_proj, Tensor[] up_proj, Tensor expert_indices, Tensor expert_weights, str hidden_act) -> Tensor"
)


def _infer_output_like_input(input: torch.Tensor, *args, **kwargs):
    # Minimal meta kernel: return an empty tensor with same shape/dtype/device
    # so that abstract interp tests can query shape/dtype without real compute.
    return torch.empty_like(input, device="meta")


@impl("gml::fused_moe", "Meta")
def _fused_moe_meta(
    input, gate_proj, down_proj, up_proj, expert_indices, expert_weights, hidden_act
):
    return _infer_output_like_input(input)


@impl("gml::fused_moe", "CPU")
def _fused_moe_cpu(
    input, gate_proj, down_proj, up_proj, expert_indices, expert_weights, hidden_act
):
    # CPU stub for safety in case CPU is used; maintain shape/dtype.
    return torch.empty_like(input)


# Signature:
# input: Tensor (float32, float16, or bfloat16)
# block_size: List[int] - granularity of quantization
# scale: Tensor - quantization scale parameter(s)
# zero_point: Tensor - quantization zero point parameter(s)
# output_dtype: int - requested dtype (e.g. torch.uint8)
_lib.define(
    "quantize_affine(Tensor input, int[] block_size, Tensor scale, Tensor zero_point, int output_dtype) -> Tensor"
)


@impl("gml::quantize_affine", "Meta")
def _quantize_affine_meta(input, block_size, scale, zero_point, output_dtype):
    # Output has same shape as input but with output_dtype
    return torch.empty_like(input, dtype=output_dtype, device="meta")


@impl("gml::quantize_affine", "CPU")
def _quantize_affine_cpu(input, block_size, scale, zero_point, output_dtype):
    # CPU stub for safety in case CPU is used; maintain shape with output_dtype.
    return torch.empty_like(input, dtype=output_dtype)


# Signature:
# input: Tensor (quantized tensor)
# block_size: List[int] - granularity of quantization
# scale: Tensor - quantization scale parameter(s)
# zero_point: Tensor - quantization zero point parameter(s)
# input_dtype: int - dtype of input tensor
# output_dtype: int - desired output dtype (default fp32)
_lib.define(
    "dequantize_affine(Tensor input, int[] block_size, Tensor scale, Tensor zero_point, int input_dtype, int output_dtype) -> Tensor"
)


@impl("gml::dequantize_affine", "Meta")
def _dequantize_affine_meta(input, block_size, scale, zero_point, input_dtype, output_dtype):
    # Output has same shape as input but with output_dtype (fp32/fp16/bf16)
    return torch.empty_like(input, dtype=output_dtype, device="meta")


@impl("gml::dequantize_affine", "CPU")
def _dequantize_affine_cpu(input, block_size, scale, zero_point, input_dtype, output_dtype):
    # CPU stub for safety in case CPU is used; maintain shape with output_dtype.
    return torch.empty_like(input, dtype=output_dtype)

# Dynamic scale op - dynamically computes quantization scale
# Signature:
# x: Tensor - input tensor to compute scale for
# dtype: int - target dtype for quantization (e.g. torch.int8, torch.float8_e4m3fn)
# symmetric: bool - whether to use symmetric quantization (default True)
# strategy: int - quantization strategy level for scale computation (enum as int)
#   Strategy values:
#   0 = TENSOR: per-tensor quantization - returns scalar []
#   1 = CHANNEL: per-channel quantization - returns [channels]
#   2 = GROUP: per-group quantization - returns [num_groups]
#   3 = BLOCK: block-wise quantization - returns block shape
#   4 = TOKEN: per-token quantization - returns [num_tokens]
#   5 = TENSOR_GROUP: tensor-group quantization
#   6 = ATTN_HEAD: per-attention-head quantization - returns [num_heads]
_lib.define(
    "dynamic_scale(Tensor x, int dtype, bool symmetric=True, int strategy=0) -> Tensor"
)


@impl("gml::dynamic_scale", "Meta")
def _dynamic_scale_meta(x, dtype, symmetric=True, strategy=0):
    # Returns a tensor with scale values
    # Scale dtype matches input dtype
    if strategy == 0:  # TENSOR
        return torch.empty([], dtype=x.dtype, device="meta")
    elif strategy == 4:  # TOKEN
        return torch.empty([x.shape[1]], dtype=x.dtype, device="meta")
    else:
        # For other strategies, return scalar for now
        return torch.empty([], dtype=x.dtype, device="meta")


@impl("gml::dynamic_scale", "CPU")
def _dynamic_scale_cpu(x, dtype, symmetric=True, strategy=0):
    # CPU stub
    if strategy == 0:  # TENSOR
        return torch.empty([], dtype=x.dtype)
    elif strategy == 4:  # TOKEN
        return torch.empty([x.shape[1]], dtype=x.dtype)
    else:
        return torch.empty([], dtype=x.dtype)
