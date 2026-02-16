"""Utility functions for geometry module.

This module provides helper functions used across geometry primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "unstack",
]


def unstack(array: mx.array, axis: int = -1) -> list[mx.array]:
    """Split array along an axis into a list of arrays.

    Similar to tf.unstack or torch.unbind. Each output array has one fewer
    dimension than the input.

    Args:
        array: Input array to unstack.
        axis: Axis along which to unstack. Default: -1.

    Returns:
        List of arrays, one per slice along the axis.

    Example:
        >>> x = mx.zeros((2, 3))
        >>> parts = unstack(x, axis=1)
        >>> len(parts)
        3
        >>> parts[0].shape
        (2,)
    """
    # Normalize negative axis
    if axis < 0:
        axis = array.ndim + axis

    # Get the size along the axis
    n = array.shape[axis]

    # Use split to create n equal parts, then squeeze the axis
    parts = mx.split(array, n, axis=axis)
    return [mx.squeeze(p, axis=axis) for p in parts]
