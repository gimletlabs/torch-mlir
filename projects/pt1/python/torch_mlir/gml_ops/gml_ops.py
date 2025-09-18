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
# w13: Tensor[num_experts, hidden_size, 2*inter_size]
# w2: Tensor[num_experts, inter_size, hidden_size]
# expert_indices: Tensor[num_tokens, top_k], int32
# expert_weights: Tensor[num_tokens, top_k], float32
_lib.define(
    "fused_moe(Tensor input, Tensor w13, Tensor w2, Tensor expert_indices, Tensor expert_weights) -> Tensor"
)


def _infer_output_like_input(input: torch.Tensor, *args, **kwargs):
    # Minimal meta kernel: return an empty tensor with same shape/dtype/device
    # so that abstract interp tests can query shape/dtype without real compute.
    return torch.empty_like(input, device="meta")


@impl("gml::fused_moe", "Meta")
def _fused_moe_meta(input, w13, w2, expert_indices, expert_weights):
    return _infer_output_like_input(input)


@impl("gml::fused_moe", "CPU")
def _fused_moe_cpu(input, w13, w2, expert_indices, expert_weights):
    # CPU stub for safety in case CPU is used; maintain shape/dtype.
    return torch.empty_like(input)

