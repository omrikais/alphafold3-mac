"""MLX neural network modules for AlphaFold 3.

This module provides MLX implementations of Linear, LayerNorm, and GatedLinearUnit
layers that match the Haiku API for numerical parity with the original JAX model.
"""

from __future__ import annotations

from alphafold3_mlx.modules.linear import Linear
from alphafold3_mlx.modules.layer_norm import LayerNorm
from alphafold3_mlx.modules.glu import GatedLinearUnit, get_activation

__all__ = [
    "Linear",
    "LayerNorm",
    "GatedLinearUnit",
    "get_activation",
]
