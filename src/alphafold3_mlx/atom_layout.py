"""Atom layout utilities for MLX parity with AF3 JAX modules.

This is a minimal subset of alphafold3.model.atom_layout for use in
Diffusion/Confidence heads. It provides GatherInfo and convert() to
translate between atom layouts using precomputed gather indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class GatherInfo:
    """Gather info for converting between atom layouts.

    Attributes:
        gather_idxs: Indices into flattened source layout.
        gather_mask: Mask indicating valid entries.
        input_shape: Shape of the source layout axes.
    """

    gather_idxs: mx.array
    gather_mask: mx.array
    input_shape: mx.array

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the gather output shape (same as gather_idxs shape)."""
        return tuple(int(x) for x in self.gather_idxs.shape)

    @classmethod
    def from_dict(cls, data: dict, key_prefix: str) -> "GatherInfo":
        """Create GatherInfo from a flat dict with key_prefix."""
        idxs_key = f"{key_prefix}/gather_idxs"
        mask_key = f"{key_prefix}/gather_mask"
        shape_key = f"{key_prefix}/input_shape"
        if idxs_key not in data:
            idxs_key = f"{key_prefix}:gather_idxs"
        if mask_key not in data:
            mask_key = f"{key_prefix}:gather_mask"
        if shape_key not in data:
            shape_key = f"{key_prefix}:input_shape"

        gather_idxs = mx.array(data[idxs_key])
        gather_mask = mx.array(data[mask_key])
        input_shape = mx.array(data[shape_key])
        return cls(gather_idxs=gather_idxs, gather_mask=gather_mask, input_shape=input_shape)

    def as_dict(self, key_prefix: str) -> dict[str, np.ndarray]:
        """Serialize GatherInfo to a flat dict."""
        return {
            f"{key_prefix}/gather_idxs": np.asarray(self.gather_idxs),
            f"{key_prefix}/gather_mask": np.asarray(self.gather_mask),
            f"{key_prefix}/input_shape": np.asarray(self.input_shape),
        }


def _to_tuple(x: mx.array | Sequence[int] | np.ndarray) -> tuple[int, ...]:
    if isinstance(x, (tuple, list)):
        return tuple(int(v) for v in x)
    if isinstance(x, np.ndarray):
        return tuple(int(v) for v in x.tolist())
    if isinstance(x, mx.array):
        return tuple(int(v) for v in np.asarray(x).tolist())
    return (int(x),)


def convert(
    gather_info: GatherInfo,
    arr: mx.array,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> mx.array:
    """Convert an array from one atom layout to another.

    Args:
        gather_info: Gather indices and mask.
        arr: Input array.
        layout_axes: Axes corresponding to the layout to be converted.

    Returns:
        Converted array with gather applied and mask zeroed.
    """
    # Translate negative indices to positive.
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)

    # Ensure layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1
    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f"layout_axes must be continuous. Got {layout_axes}.")

    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]
    input_shape = _to_tuple(gather_info.input_shape)

    if len(layout_shape) != len(input_shape):
        raise ValueError(
            f"Input layout rank mismatch: {layout_shape} vs {input_shape}."
        )
    # First axis may be >= input_shape[0], others must match.
    if layout_shape[0] < input_shape[0]:
        raise ValueError(
            f"Input layout axis 0 too small: {layout_shape[0]} < {input_shape[0]}"
        )
    if len(layout_shape) > 1:
        if any(int(a) != int(b) for a, b in zip(layout_shape[1:], input_shape[1:])):
            raise ValueError(
                f"Input layout axes mismatch: {layout_shape} vs {input_shape}"
            )

    # Flatten layout axes.
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    layout_size = int(np.prod(layout_shape))
    arr_flat = arr.reshape(batch_shape + (layout_size,) + features_shape)

    gather_idxs = gather_info.gather_idxs
    if layout_axes_begin == 0:
        out = mx.take(arr_flat, gather_idxs, axis=0)
    elif layout_axes_begin == 1:
        out = mx.take(arr_flat, gather_idxs, axis=1)
    elif layout_axes_begin == 2:
        out = mx.take(arr_flat, gather_idxs, axis=2)
    elif layout_axes_begin == 3:
        out = mx.take(arr_flat, gather_idxs, axis=3)
    elif layout_axes_begin == 4:
        out = mx.take(arr_flat, gather_idxs, axis=4)
    else:
        raise ValueError("Only up to 4 batch axes supported for convert().")

    # Apply gather mask.
    mask_shape = (
        (1,) * len(batch_shape)
        + gather_info.gather_mask.shape
        + (1,) * len(features_shape)
    )
    mask = gather_info.gather_mask.reshape(mask_shape)
    out = out * mask
    return out
