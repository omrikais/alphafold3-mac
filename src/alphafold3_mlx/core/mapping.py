"""Chunking and mapping utilities for memory-efficient inference.

This module provides utilities for processing large sequences within memory
limits by chunking operations. This is critical for sequences >1000 residues.

The main function is inference_subbatch, which splits batched arguments into
chunks, applies a function to each chunk, and concatenates the results.
"""

from __future__ import annotations

from typing import Callable, Sequence, TypeVar

import mlx.core as mx

T = TypeVar("T")


def inference_subbatch(
    fn: Callable[..., mx.array | tuple[mx.array, ...]],
    chunk_size: int | None,
    batched_args: Sequence[mx.array],
    nonbatched_args: Sequence[mx.array] | None = None,
    batch_axis: int = 0,
) -> mx.array | tuple[mx.array, ...]:
    """Apply function in chunks for memory efficiency.

    This is the MLX port of mapping.inference_subbatch from the original JAX
    implementation. It enables processing sequences that would otherwise
    exceed memory limits.

    Args:
        fn: Function to apply. Should accept (*batched_args, *nonbatched_args)
            and return an array or tuple of arrays.
        chunk_size: Chunk size along batch axis. None = no chunking.
        batched_args: Arguments to split along batch axis.
        nonbatched_args: Arguments passed unchanged to each call.
        batch_axis: Axis to chunk along (default: 0).

    Returns:
        Concatenated results from all chunks.

    Example:
        >>> def attention(q, k, v, mask):
        ...     return mx.fast.scaled_dot_product_attention(q, k, v, mask=mask)
        >>> # Process large sequence in chunks
        >>> output = inference_subbatch(
        ...     attention,
        ...     chunk_size=256,
        ...     batched_args=[q, k, v],
        ...     nonbatched_args=[mask],
        ... )
    """
    if chunk_size is None or len(batched_args) == 0:
        # No chunking - apply function directly
        if nonbatched_args:
            return fn(*batched_args, *nonbatched_args)
        return fn(*batched_args)

    # Get batch size from first batched argument
    batch_size = batched_args[0].shape[batch_axis]

    if batch_size <= chunk_size:
        # Batch fits in single chunk
        if nonbatched_args:
            return fn(*batched_args, *nonbatched_args)
        return fn(*batched_args)

    # Process in chunks
    outputs: list[mx.array | tuple[mx.array, ...]] = []

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)

        # Extract chunk from batched arguments
        # Build slice object for the batch axis
        slices = [slice(None)] * batched_args[0].ndim
        slices[batch_axis] = slice(start, end)
        chunk_slice = tuple(slices)

        chunk_args = [arg[chunk_slice] for arg in batched_args]

        # Apply function to chunk
        if nonbatched_args:
            chunk_output = fn(*chunk_args, *nonbatched_args)
        else:
            chunk_output = fn(*chunk_args)

        outputs.append(chunk_output)

        # Evaluate to free intermediate memory
        if isinstance(chunk_output, tuple):
            mx.eval(*chunk_output)
        else:
            mx.eval(chunk_output)

    # Concatenate results
    if isinstance(outputs[0], tuple):
        # Multiple outputs - concatenate each
        num_outputs = len(outputs[0])
        result = tuple(
            mx.concatenate([o[i] for o in outputs], axis=batch_axis)
            for i in range(num_outputs)
        )
        return result
    else:
        # Single output
        return mx.concatenate(outputs, axis=batch_axis)


def sharded_map(
    fn: Callable[[mx.array], mx.array],
    inputs: mx.array,
    chunk_size: int,
    axis: int = 0,
) -> mx.array:
    """Apply function to input in shards along an axis.

    Simpler version of inference_subbatch for single input/output cases.

    Args:
        fn: Function to apply to each shard.
        inputs: Input array to shard.
        chunk_size: Size of each shard.
        axis: Axis to shard along.

    Returns:
        Concatenated results.
    """
    size = inputs.shape[axis]

    if size <= chunk_size:
        return fn(inputs)

    outputs = []
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)

        slices = [slice(None)] * inputs.ndim
        slices[axis] = slice(start, end)
        shard = inputs[tuple(slices)]

        output = fn(shard)
        outputs.append(output)
        mx.eval(output)

    return mx.concatenate(outputs, axis=axis)


def chunk_layer(
    layer_fn: Callable[[mx.array], mx.array],
    inputs: mx.array,
    chunk_size: int | None,
    low_memory: bool = True,
) -> mx.array:
    """Apply a layer function with optional chunking.

    Convenience wrapper for applying layers to large inputs.

    Args:
        layer_fn: Layer function to apply.
        inputs: Input tensor.
        chunk_size: Chunk size (None = no chunking).
        low_memory: If True, evaluate each chunk to free memory.

    Returns:
        Layer output.
    """
    if chunk_size is None:
        return layer_fn(inputs)

    batch_size = inputs.shape[0]
    outputs = []

    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk = inputs[start:end]
        output = layer_fn(chunk)
        outputs.append(output)

        if low_memory:
            mx.eval(output)

    return mx.concatenate(outputs, axis=0)
