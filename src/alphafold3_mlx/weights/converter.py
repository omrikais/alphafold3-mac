"""Array conversion utilities for AlphaFold 3 MLX.

This module provides utilities for converting JAX/NumPy arrays to MLX format,
with special handling for bfloat16 which NumPy doesn't natively support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp
    import mlx.core as mx


class UnsupportedDtypeError(Exception):
    """Raised when a dtype cannot be converted to MLX format."""

    pass


# Mapping from NumPy/JAX dtype names to MLX dtypes
_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int32": "int32",
    "int64": "int64",
    "int16": "int16",
    "int8": "int8",
    "uint32": "uint32",
    "uint64": "uint64",
    "uint16": "uint16",
    "uint8": "uint8",
    "bool": "bool_",
}


def _get_mlx_dtype(dtype_name: str) -> "mx.Dtype":
    """Get the MLX dtype corresponding to a dtype name.

    Args:
        dtype_name: String name of the dtype (e.g., "float32", "bfloat16").

    Returns:
        The corresponding MLX dtype.

    Raises:
        UnsupportedDtypeError: If the dtype is not supported.
    """
    import mlx.core as mx

    mlx_dtype_name = _DTYPE_MAP.get(dtype_name)
    if mlx_dtype_name is None:
        raise UnsupportedDtypeError(
            f"Cannot convert dtype {dtype_name!r} to MLX. "
            f"Supported dtypes: {list(_DTYPE_MAP.keys())}"
        )

    return getattr(mx, mlx_dtype_name)


def convert_array(
    arr: "jnp.ndarray | np.ndarray",
    dtype: "mx.Dtype | None" = None,
) -> "mx.array":
    """Convert a single JAX or NumPy array to MLX format.

    Args:
        arr: Source array (JAX jnp.ndarray or NumPy ndarray).
        dtype: Target dtype (None preserves source dtype).

    Returns:
        MLX array with same shape and (optionally) dtype.

    Raises:
        UnsupportedDtypeError: If source dtype can't be converted.

    Note:
        bfloat16 requires special handling because NumPy doesn't support it.
        JAX bf16 arrays are converted using the view trick: interpret the
        underlying bytes as uint16, then construct an MLX bf16 array.
    """
    import mlx.core as mx

    # Get source dtype name
    src_dtype_name = str(arr.dtype)

    # Explicitly check dtype is supported BEFORE attempting conversion
    # This ensures we raise UnsupportedDtypeError, not a cryptic MLX error
    if src_dtype_name not in _DTYPE_MAP:
        raise UnsupportedDtypeError(
            f"Cannot convert dtype {src_dtype_name!r} to MLX. "
            f"Supported dtypes: {list(_DTYPE_MAP.keys())}"
        )

    # Handle bfloat16 specially - NumPy doesn't support it
    if src_dtype_name == "bfloat16":
        # Convert to raw bytes via uint16 view
        # JAX bf16 -> view as uint16 -> MLX bf16
        np_arr = np.asarray(arr)
        # bf16 has same bit layout, so view as uint16
        uint16_view = np_arr.view(np.uint16)
        # Create MLX array from uint16 and view as bfloat16
        mlx_arr = mx.array(uint16_view)
        mlx_arr = mlx_arr.view(mx.bfloat16)
    else:
        # For other dtypes, convert via NumPy
        np_arr = np.asarray(arr)
        mlx_arr = mx.array(np_arr)

    # Convert to target dtype if specified
    if dtype is not None:
        mlx_arr = mlx_arr.astype(dtype)
    else:
        # Ensure dtype is correct (NumPy may have changed it)
        if src_dtype_name in _DTYPE_MAP and src_dtype_name != "bfloat16":
            expected_dtype = _get_mlx_dtype(src_dtype_name)
            if mlx_arr.dtype != expected_dtype:
                mlx_arr = mlx_arr.astype(expected_dtype)

    return mlx_arr


def _convert_tree(
    tree: dict,
    dtype: "mx.Dtype | None" = None,
) -> dict:
    """Recursively convert a nested dict of arrays to MLX format.

    Args:
        tree: Nested dict where leaves are arrays (JAX or NumPy).
        dtype: Optional target dtype for all arrays.

    Returns:
        Nested dict with same structure, leaves converted to MLX arrays.

    Raises:
        UnsupportedDtypeError: If any array dtype can't be converted.
    """
    result = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            result[key] = _convert_tree(value, dtype)
        else:
            # Assume it's an array
            result[key] = convert_array(value, dtype)
    return result
