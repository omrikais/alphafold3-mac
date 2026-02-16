"""Linear module implementation for MLX.

This module provides a Linear layer that matches Haiku Linear semantics
with precision control for numerically sensitive operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.geometry.exceptions import ShapeMismatchError

if TYPE_CHECKING:
    from typing import TypeAlias

    Precision: TypeAlias = Literal["highest"] | None
    Initializer: TypeAlias = Literal["linear", "relu", "zeros"]

__all__ = [
    "Linear",
]


class Linear(nn.Module):
    """Linear projection supporting arbitrary input/output ranks via einsum.

    Matches Haiku Linear semantics with precision control for numerically
    sensitive operations.

    Attributes:
        weight: Learnable weight matrix.
        bias: Optional bias vector (None if use_bias=False).
        num_input_dims: Number of trailing input dims to contract.
        output_shape: Output dimensions.
        precision: 'highest' for float32 computation, None for input dtype.
        initializer: Weight initialization scheme.

    Example:
        >>> layer = Linear(64, input_dims=32)
        >>> x = mx.random.normal(shape=(2, 10, 32))
        >>> y = layer(x)  # shape: (2, 10, 64)
    """

    def __init__(
        self,
        num_output: int | Sequence[int],
        *,
        initializer: Literal["linear", "relu", "zeros"] = "linear",
        num_input_dims: int = 1,
        use_bias: bool = False,
        bias_init: float = 0.0,
        precision: Literal["highest"] | None = None,
        transpose_weights: bool = False,
        input_dims: int | Sequence[int] | None = None,
    ) -> None:
        """Initialize Linear layer.

        Args:
            num_output: Output dimension(s). Can be int or sequence.
            initializer: Weight init scheme ('linear', 'relu', 'zeros').
            num_input_dims: Number of input dimensions to contract.
            use_bias: Whether to include bias term.
            bias_init: Initial bias value.
            precision: 'highest' forces float32 computation.
            transpose_weights: If True, weights are [output, input].
            input_dims: Input dimension(s) for weight initialization.
        """
        super().__init__()

        # Normalize to tuples
        if isinstance(num_output, int):
            self._output_shape = (num_output,)
        else:
            self._output_shape = tuple(num_output)

        if input_dims is None:
            raise ValueError("input_dims must be specified")

        if isinstance(input_dims, int):
            self._input_shape = (input_dims,)
        else:
            self._input_shape = tuple(input_dims)

        # Validate num_input_dims matches input_dims length
        if num_input_dims != len(self._input_shape):
            raise ValueError(
                f"num_input_dims ({num_input_dims}) must match length of "
                f"input_dims ({len(self._input_shape)})"
            )

        self._num_input_dims = num_input_dims
        self._precision = precision
        self._transpose_weights = transpose_weights

        # Compute weight shape
        if transpose_weights:
            weight_shape = self._output_shape + self._input_shape
        else:
            weight_shape = self._input_shape + self._output_shape

        # Initialize weights
        self.weight = self._init_weights(weight_shape, initializer)

        # Initialize bias
        if use_bias:
            self.bias = mx.full(self._output_shape, bias_init)
        else:
            self.bias = None

        # Build einsum equation
        self._equation = self._build_equation()

    def _init_weights(
        self, shape: tuple[int, ...], initializer: str
    ) -> mx.array:
        """Initialize weights using specified scheme.

        Args:
            shape: Weight tensor shape.
            initializer: Name of initialization scheme.

        Returns:
            Initialized weight tensor.

        Raises:
            ValueError: If initializer is unknown.
        """
        # Compute fan-in (number of input units)
        if self._transpose_weights:
            fan_in = int(mx.prod(mx.array(shape[len(self._output_shape):])).item())
        else:
            fan_in = int(mx.prod(mx.array(shape[:len(self._input_shape)])).item())

        if initializer == "zeros":
            return mx.zeros(shape)
        elif initializer == "linear":
            # Variance scaling: 1/fan_in
            std = 1.0 / (fan_in ** 0.5)
            return mx.random.normal(shape=shape) * std
        elif initializer == "relu":
            # He initialization: 2/fan_in
            std = (2.0 / fan_in) ** 0.5
            return mx.random.normal(shape=shape) * std
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

    def _build_equation(self) -> str:
        """Build einsum equation for the contraction.

        Returns:
            Einsum equation string.
        """
        # Input: ...abc (where abc are input dims to contract)
        # Weight: abc...def (input dims followed by output dims)
        # Output: ...def

        # Use letters for dimensions
        # First, figure out how many batch dimensions we might have
        # We'll use dynamic batch handling with ... in einsum

        n_in = len(self._input_shape)
        n_out = len(self._output_shape)

        # Input dimensions to contract
        in_dims = "".join(chr(ord("i") + i) for i in range(n_in))
        # Output dimensions
        out_dims = "".join(chr(ord("a") + i) for i in range(n_out))

        if self._transpose_weights:
            # Weight is [output, input]
            weight_dims = out_dims + in_dims
            eq = f"...{in_dims},{weight_dims}->...{out_dims}"
        else:
            # Weight is [input, output]
            weight_dims = in_dims + out_dims
            eq = f"...{in_dims},{weight_dims}->...{out_dims}"

        return eq

    def __call__(self, x: mx.array) -> mx.array:
        """Apply linear projection.

        Args:
            x: Input array of shape [..., *input_dims].

        Returns:
            Output array of shape [..., *output_dims].

        Raises:
            ShapeMismatchError: If input trailing dims don't match input_shape.
        """
        # Validate input shape
        n_in = len(self._input_shape)
        if len(x.shape) < n_in:
            raise ShapeMismatchError(
                f"Input has {len(x.shape)} dims, but need at least {n_in} "
                f"trailing dims to match input_shape {self._input_shape}",
                expected=self._input_shape,
                actual=x.shape,
            )

        actual_trailing = x.shape[-n_in:]
        if actual_trailing != self._input_shape:
            raise ShapeMismatchError(
                f"Input trailing dims {actual_trailing} don't match "
                f"expected input_shape {self._input_shape}",
                expected=self._input_shape,
                actual=actual_trailing,
            )

        input_dtype = x.dtype

        # Handle precision='highest'
        if self._precision == "highest" and input_dtype in (mx.float16, mx.bfloat16):
            x = x.astype(mx.float32)
            w = self.weight.astype(mx.float32)
        else:
            w = self.weight

        # Apply linear transformation
        out = mx.einsum(self._equation, x, w)

        # Add bias if present
        if self.bias is not None:
            if self._precision == "highest" and input_dtype in (mx.float16, mx.bfloat16):
                out = out + self.bias.astype(mx.float32)
            else:
                out = out + self.bias

        # Cast back to original dtype if needed
        if self._precision == "highest" and input_dtype in (mx.float16, mx.bfloat16):
            out = out.astype(input_dtype)

        return out
