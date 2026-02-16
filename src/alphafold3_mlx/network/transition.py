"""Transition block for AlphaFold 3 MLX.

This module implements the TransitionBlock used throughout the model.
The transition block applies a feedforward network with gated linear units
to transform representations.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm, GatedLinearUnit


class TransitionBlock(nn.Module):
    """Transition block with gated linear unit.

    The transition block is a two-layer feedforward network:
    1. LayerNorm
    2. GLU (projects to 2x intermediate, applies activation, multiplies)
    3. Linear projection back to input dimension

    This follows the Tokamax GLU pattern from the original JAX implementation.
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_factor: int = 4,
        activation: str = "swish",
    ) -> None:
        """Initialize transition block.

        Args:
            input_dim: Input/output feature dimension.
            intermediate_factor: Expansion factor for intermediate dimension.
            activation: Activation function for GLU (swish, silu, gelu, relu).
        """
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = input_dim * intermediate_factor

        # LayerNorm before GLU
        self.norm = LayerNorm(input_dim)

        # GLU: input_dim → intermediate_dim (internally projects to 2x and splits)
        self.glu = GatedLinearUnit(
            input_dim=input_dim,
            output_dim=self.intermediate_dim,
            activation=activation,
        )

        # Output projection: intermediate_dim → input_dim
        self.output_proj = Linear(input_dim, input_dims=self.intermediate_dim, use_bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transition block.

        Args:
            x: Input tensor. Shape: [..., input_dim]

        Returns:
            Output tensor. Shape: [..., input_dim]
        """
        # Pre-norm
        residual = x
        x = self.norm(x)

        # GLU
        x = self.glu(x)

        # Output projection
        x = self.output_proj(x)

        # Residual connection
        return residual + x


class TransitionBlockNoBias(nn.Module):
    """Transition block variant without bias in output projection.

    Used in certain parts of the model where bias-free projections are preferred.
    """

    def __init__(
        self,
        input_dim: int,
        intermediate_factor: int = 4,
        activation: str = "swish",
    ) -> None:
        """Initialize transition block without output bias.

        Args:
            input_dim: Input/output feature dimension.
            intermediate_factor: Expansion factor for intermediate dimension.
            activation: Activation function for GLU.
        """
        super().__init__()

        self.input_dim = input_dim
        self.intermediate_dim = input_dim * intermediate_factor

        self.norm = LayerNorm(input_dim)
        self.glu = GatedLinearUnit(
            input_dim=input_dim,
            output_dim=self.intermediate_dim,
            activation=activation,
        )
        self.output_proj = Linear(input_dim, input_dims=self.intermediate_dim, use_bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transition block."""
        residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.output_proj(x)
        return residual + x
