"""LayerNorm module implementation for MLX.

This module provides a LayerNorm layer that matches Haiku LayerNorm semantics
with automatic upcast for reduced precision inputs.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "LayerNorm",
]


class LayerNorm(nn.Module):
    """Layer normalization with automatic upcast for reduced precision.

    Automatically casts float16/bfloat16 inputs to float32 for computation,
    then casts back to original dtype. This ensures numerical stability in
    deep networks (48 Evoformer layers).

    Attributes:
        scale: Learnable scale parameter (None if create_scale=False).
        offset: Learnable offset parameter (None if create_offset=False).
        axis: Normalization axis.
        eps: Epsilon for numerical stability.
        upcast: Whether to upcast 16-bit inputs.

    Example:
        >>> norm = LayerNorm(256)
        >>> x = mx.random.normal(shape=(2, 10, 256)).astype(mx.float16)
        >>> y = norm(x)  # Computed in float32, output is float16
    """

    def __init__(
        self,
        dims: int,
        *,
        axis: int = -1,
        create_scale: bool = True,
        create_offset: bool = True,
        eps: float = 1e-5,
        upcast: bool = True,
    ) -> None:
        """Initialize LayerNorm.

        Args:
            dims: Size of the normalized dimension.
            axis: Axis to normalize over.
            create_scale: Whether to create learnable scale.
            create_offset: Whether to create learnable offset.
            eps: Small constant for numerical stability.
            upcast: If True, compute in float32 for 16-bit inputs.
        """
        super().__init__()

        self._dims = dims
        self._axis = axis
        self._eps = eps
        self._upcast = upcast

        if create_scale:
            self.scale = mx.ones((dims,))
        else:
            self.scale = None

        if create_offset:
            self.offset = mx.zeros((dims,))
        else:
            self.offset = None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer normalization.

        Args:
            x: Input array.

        Returns:
            Normalized array with same dtype as input.
        """
        input_dtype = x.dtype
        is_16bit = input_dtype in (mx.float16, mx.bfloat16)

        # Upcast to float32 if needed for stability
        if self._upcast and is_16bit:
            x = x.astype(mx.float32)

        # Compute mean and variance
        mean = mx.mean(x, axis=self._axis, keepdims=True)
        # Use ddof=0 for population variance (matching JAX/Haiku behavior)
        var = mx.var(x, axis=self._axis, keepdims=True)

        # Normalize: (x - mean) / sqrt(var + eps)
        out = (x - mean) * mx.rsqrt(var + self._eps)

        # Apply affine transformation
        # Reshape scale/offset to broadcast along the correct axis
        # (not just the last dimension)
        if self.scale is not None or self.offset is not None:
            broadcast_shape = [1] * x.ndim
            broadcast_shape[self._axis] = self._dims

        if self.scale is not None:
            scale = self.scale
            if self._upcast and is_16bit:
                scale = scale.astype(mx.float32)
            scale = scale.reshape(broadcast_shape)
            out = out * scale

        if self.offset is not None:
            offset = self.offset
            if self._upcast and is_16bit:
                offset = offset.astype(mx.float32)
            offset = offset.reshape(broadcast_shape)
            out = out + offset

        # Cast back to original dtype
        if self._upcast and is_16bit:
            out = out.astype(input_dtype)

        return out
