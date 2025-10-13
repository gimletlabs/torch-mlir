"""Side-effect import to register the fake gml namespace and custom ops."""

# Importing this module defines the `gml` namespace and corresponding ops via
# torch.library so that `torch.ops.gml.*` exists at import time.
from . import gml_ops as _register_gml_ops  # noqa: F401
