"""Side-effect import to register the fake gml namespace and fused_moe op."""

# Importing this module defines the `gml` namespace and `fused_moe` op via
# torch.library so that `torch.ops.gml.fused_moe` exists at import time.
from . import gml_ops as _register_gml_ops  # noqa: F401
