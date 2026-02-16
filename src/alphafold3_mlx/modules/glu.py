"""GatedLinearUnit module implementation for MLX.

This module provides a GatedLinearUnit layer with configurable activations
for transition blocks in the AlphaFold 3 architecture.
"""

from __future__ import annotations

from typing import Callable, Literal

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules.linear import Linear

__all__ = [
    "GatedLinearUnit",
    "get_activation",
]


def get_activation(name: Literal["swish", "silu", "gelu", "relu"]) -> Callable[[mx.array], mx.array]:
    """Get activation function by name.

    Args:
        name: Activation name ('swish', 'silu', 'gelu', 'relu').

    Returns:
        Activation function (mx.array -> mx.array).

    Raises:
        ValueError: If activation name is unknown.
    """
    if name == "swish" or name == "silu":
        # swish(x) = x * sigmoid(x)
        return lambda x: x * mx.sigmoid(x)
    elif name == "gelu":
        return nn.gelu
    elif name == "relu":
        return lambda x: mx.maximum(x, 0)
    else:
        raise ValueError(f"Unknown activation: {name}")


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit with configurable activation.

    Projects input to 2x output dimension, splits into value and gate,
    returns value * activation(gate).

    Attributes:
        linear: Internal projection layer (projects to 2 * output_dim).
        activation: Name of activation function.

    Example:
        >>> glu = GatedLinearUnit(256, 512, activation='swish')
        >>> x = mx.random.normal(shape=(2, 10, 256))
        >>> y = glu(x)  # shape: (2, 10, 512)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: Literal["swish", "silu", "gelu", "relu"] = "swish",
        use_bias: bool = False,
        precision: Literal["highest"] | None = None,
    ) -> None:
        """Initialize GatedLinearUnit.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            activation: Gating activation ('swish'/'silu', 'gelu', 'relu').
            use_bias: Whether internal Linear uses bias.
            precision: Precision mode for internal Linear.
        """
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation_name = activation

        # Internal linear projects to 2 * output_dim for split
        self.linear = Linear(
            2 * output_dim,
            input_dims=input_dim,
            use_bias=use_bias,
            precision=precision,
        )

        # Get activation function
        self._activation_fn = get_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply gated linear projection.

        JAX AF3 computes: activation(first_half) * second_half
        NOT: first_half * activation(second_half)

        Args:
            x: Input array of shape [..., input_dim].

        Returns:
            Output array of shape [..., output_dim].
        """
        # Project to 2 * output_dim
        projected = self.linear(x)  # [..., 2 * output_dim]

        # Split into a and b (JAX naming convention)
        a, b = mx.split(projected, 2, axis=-1)

        # Apply gating: activation(a) * b (JAX AF3 style)
        return self._activation_fn(a) * b
