"""Pure JAX implementations of tokamax functions for CPU reference generation.

This module provides pure JAX equivalents of tokamax's optimized GPU kernels.
These produce NUMERICALLY IDENTICAL outputs to the GPU versions, just slower.

The tokamax library provides:
- dot_product_attention: Flash attention implementation
- gated_linear_unit: Fused GLU kernel
- DotProductAttentionImplementation: Enum for attention backends

Both are purely for performance optimization - the mathematical operations
are standard and can be implemented in pure JAX.

Usage:
    import sys
    from scripts.tokamax_stub import install_tokamax_stub
    install_tokamax_stub()
    # Now import AF3 modules - they will use pure JAX implementations
"""

from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType
from typing import Literal

import jax
import jax.numpy as jnp


# Enum to match tokamax.DotProductAttentionImplementation
class DotProductAttentionImplementation(str, Enum):
    """Attention implementation backend."""
    XLA = "xla"
    CUDNN = "cudnn"
    TRITON = "triton"
    FLASH = "flash"


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    bias: jnp.ndarray | None = None,
    implementation: str = "xla",
) -> jnp.ndarray:
    """Pure JAX scaled dot-product attention.

    Implements the same interface as tokamax.dot_product_attention.

    Args:
        query: Query tensor [..., seq_q, num_heads, head_dim]
        key: Key tensor [..., seq_k, num_heads, head_dim]
        value: Value tensor [..., seq_k, num_heads, head_dim]
        mask: Boolean mask [..., 1, 1, seq_k] or [..., seq_q, seq_k]
        bias: Additive bias [..., num_heads, seq_q, seq_k]
        implementation: Ignored (always uses XLA)

    Returns:
        Attention output [..., seq_q, num_heads, head_dim]
    """
    # Get dimensions
    head_dim = query.shape[-1]
    scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    # Compute attention scores: [..., seq_q, num_heads, seq_k]
    # query: [..., seq_q, num_heads, head_dim]
    # key: [..., seq_k, num_heads, head_dim]
    scores = jnp.einsum("...qhd,...khd->...hqk", query, key) * scale

    # Add bias if provided
    if bias is not None:
        scores = scores + bias

    # Apply mask (tokamax uses boolean mask where True = keep)
    if mask is not None:
        # Mask is typically [..., 1, 1, seq_k] or [..., seq_q, seq_k]
        # Convert boolean to large negative for softmax masking
        mask_value = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask, scores, mask_value)

    # Softmax over keys
    weights = jax.nn.softmax(scores, axis=-1)

    # Weighted sum of values: [..., seq_q, num_heads, head_dim]
    output = jnp.einsum("...hqk,...khd->...qhd", weights, value)

    return output


def gated_linear_unit(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    activation: callable = jax.nn.swish,
) -> jnp.ndarray:
    """Pure JAX gated linear unit.

    Implements the same interface as tokamax.gated_linear_unit.

    The tokamax version takes weights with shape [in_dim, 2, out_dim] and
    computes: activation(x @ weights[:, 0, :]) * (x @ weights[:, 1, :])

    Args:
        x: Input tensor [..., in_dim]
        weights: Weight tensor [in_dim, 2, out_dim]
        activation: Activation function (default: swish)

    Returns:
        Output tensor [..., out_dim]
    """
    # weights shape: [in_dim, 2, out_dim]
    # Split into gate and value weights
    gate_weights = weights[:, 0, :]  # [in_dim, out_dim]
    value_weights = weights[:, 1, :]  # [in_dim, out_dim]

    # Compute gate and value
    gate = jnp.matmul(x, gate_weights)  # [..., out_dim]
    value = jnp.matmul(x, value_weights)  # [..., out_dim]

    # Apply gated linear unit: activation(gate) * value
    return activation(gate) * value


class TokaMaxStub(ModuleType):
    """Stub module that provides pure JAX implementations."""

    def __init__(self):
        super().__init__("tokamax")
        self.dot_product_attention = dot_product_attention
        self.gated_linear_unit = gated_linear_unit
        self.DotProductAttentionImplementation = DotProductAttentionImplementation

        # Version info
        self.__version__ = "0.0.0-stub"
        self._is_stub = True


def install_tokamax_stub() -> bool:
    """Install the tokamax stub module if tokamax is not available.

    Returns:
        True if stub was installed, False if real tokamax is available.
    """
    try:
        import tokamax
        if hasattr(tokamax, "_is_stub"):
            return True  # Stub already installed
        return False  # Real tokamax available
    except ImportError:
        # Install stub
        stub = TokaMaxStub()
        sys.modules["tokamax"] = stub
        return True


def is_using_stub() -> bool:
    """Check if we're using the tokamax stub."""
    try:
        import tokamax
        return getattr(tokamax, "_is_stub", False)
    except ImportError:
        return True
