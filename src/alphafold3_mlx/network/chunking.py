"""Chunking utilities for memory-efficient computation.

This module provides chunking wrappers for memory-intensive operations
like attention and triangle multiplication. Chunking processes data in
smaller pieces to reduce peak memory usage.

Key operations that benefit from chunking:
- Triangle multiplication: O(seq²) memory
- Attention: O(seq²) for attention weights
- Outer product mean: O(seq²) output
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx


def chunk_apply(
    fn: Callable[[mx.array], mx.array],
    x: mx.array,
    chunk_size: int,
    axis: int = 1,
) -> mx.array:
    """Apply a function to chunks of an array along an axis.

    Splits the input along the specified axis, applies the function to each
    chunk, and concatenates the results.

    Args:
        fn: Function to apply to each chunk.
        x: Input array.
        chunk_size: Size of each chunk.
        axis: Axis to chunk along.

    Returns:
        Concatenated results from processing each chunk.
    """
    size = x.shape[axis]

    if chunk_size >= size:
        # No chunking needed
        return fn(x)

    # Process chunks
    results = []
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)

        # Create slice tuple
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(start, end)

        chunk = x[tuple(slices)]
        result = fn(chunk)
        results.append(result)

        # Force evaluation to release memory
        mx.eval(result)

    return mx.concatenate(results, axis=axis)


def chunk_einsum(
    equation: str,
    a: mx.array,
    b: mx.array,
    chunk_size: int | None = None,
    chunk_axis: int = 1,
) -> mx.array:
    """Chunked einsum for memory-efficient tensor contraction.

    Particularly useful for triangle multiplication where the contraction
    is over a large dimension.

    Args:
        equation: Einsum equation (e.g., "bikc,bjkc->bijc").
        a: First input array.
        b: Second input array.
        chunk_size: Size of each chunk (None = no chunking).
        chunk_axis: Axis in the output to chunk along.

    Returns:
        Result of the einsum operation.
    """
    if chunk_size is None:
        return mx.einsum(equation, a, b)

    # Parse equation to understand dimensions
    # For triangle: "bikc,bjkc->bijc" or "bkic,bkjc->bijc"
    parts = equation.split("->")
    if len(parts) != 2:
        # Fall back to non-chunked
        return mx.einsum(equation, a, b)

    # Determine output shape from doing a small einsum
    # (more robust than parsing)
    output_shape = mx.einsum(equation, a[:1], b[:1]).shape[1:]
    output_shape = (a.shape[0], *output_shape)

    # Chunk along specified axis (typically i or j in the output)
    size = output_shape[chunk_axis]

    if chunk_size >= size:
        return mx.einsum(equation, a, b)

    results = []
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)

        # Slice input arrays appropriately
        # For "bikc,bjkc->bijc": chunk on i means slice a's i dimension
        # For "bkic,bkjc->bijc": chunk on i means slice a and b's i and j dims
        if "i" in equation.split("->")[0].split(",")[0]:
            # First input has i dimension
            if "j" in equation.split("->")[0].split(",")[1]:
                # outgoing: "bikc,bjkc->bijc"
                a_slice = a[:, start:end, :, :]
                # b is not sliced on this dimension
                b_slice = b
            else:
                # May need different handling
                a_slice = a[:, start:end]
                b_slice = b
        else:
            # incoming: "bkic,bkjc->bijc"
            # Both sliced on different dims
            a_slice = a[:, :, start:end, :]
            b_slice = b

        chunk_result = mx.einsum(equation, a_slice, b_slice)
        results.append(chunk_result)
        mx.eval(chunk_result)

    return mx.concatenate(results, axis=chunk_axis)


def chunk_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk_size: int | None = None,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    """Chunked attention computation for memory efficiency.

    Chunks the query sequence to reduce peak memory from attention weights.

    Args:
        q: Query tensor [batch, heads, seq_q, head_dim].
        k: Key tensor [batch, heads, seq_k, head_dim].
        v: Value tensor [batch, heads, seq_k, head_dim].
        chunk_size: Query chunk size (None = no chunking).
        scale: Attention scale (default: 1/sqrt(head_dim)).
        mask: Optional attention mask [batch, 1, seq_q, seq_k] or [batch, heads, seq_q, seq_k].

    Returns:
        Attention output [batch, heads, seq_q, head_dim].
    """
    # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
    if mask is not None and mask.dtype != q.dtype:
        mask = mask.astype(q.dtype)

    if chunk_size is None or chunk_size >= q.shape[2]:
        # Use MLX SDPA directly
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

    seq_q = q.shape[2]
    results = []

    for start in range(0, seq_q, chunk_size):
        end = min(start + chunk_size, seq_q)

        q_chunk = q[:, :, start:end, :]

        # Slice mask if provided
        mask_chunk = None
        if mask is not None:
            if mask.ndim == 4:
                mask_chunk = mask[:, :, start:end, :]
            else:
                # Broadcast-compatible mask
                mask_chunk = mask

        chunk_result = mx.fast.scaled_dot_product_attention(
            q_chunk, k, v, scale=scale, mask=mask_chunk
        )
        results.append(chunk_result)
        mx.eval(chunk_result)

    return mx.concatenate(results, axis=2)


class ChunkingConfig:
    """Configuration for chunked computation.

    Attributes:
        enabled: Whether chunking is enabled.
        chunk_size: Size of each chunk (number of residues).
        attention_chunk_size: Specific chunk size for attention (overrides chunk_size).
        triangle_chunk_size: Specific chunk size for triangle ops (overrides chunk_size).
    """

    def __init__(
        self,
        enabled: bool = False,
        chunk_size: int = 256,
        attention_chunk_size: int | None = None,
        triangle_chunk_size: int | None = None,
    ):
        self.enabled = enabled
        self.chunk_size = chunk_size
        self.attention_chunk_size = attention_chunk_size or chunk_size
        self.triangle_chunk_size = triangle_chunk_size or chunk_size

    @classmethod
    def from_global_config(cls, global_config) -> "ChunkingConfig":
        """Create ChunkingConfig from GlobalConfig.

        Args:
            global_config: GlobalConfig instance.

        Returns:
            ChunkingConfig with settings from global config.
        """
        chunk_size = getattr(global_config, "chunk_size", None)
        if chunk_size is None:
            return cls(enabled=False)
        return cls(enabled=True, chunk_size=chunk_size)


# Global chunking config (can be set by Model)
_chunking_config: ChunkingConfig | None = None


def get_chunking_config() -> ChunkingConfig:
    """Get the current global chunking configuration."""
    global _chunking_config
    if _chunking_config is None:
        _chunking_config = ChunkingConfig(enabled=False)
    return _chunking_config


def set_chunking_config(config: ChunkingConfig) -> None:
    """Set the global chunking configuration."""
    global _chunking_config
    _chunking_config = config
