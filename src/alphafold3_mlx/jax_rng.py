"""JAX-compatible Threefry RNG primitives for MLX inference.

The production MLX path cannot depend on JAX, but AF3's sampling contract is
defined by JAX's partitionable Threefry2x32 implementation. This module keeps
the small required subset in NumPy and returns MLX arrays for floating samples.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


_THREEFRY_PARITY = np.uint32(0x1BD11BDA)
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))


def _rotate_left(value: np.ndarray, distance: int) -> np.ndarray:
    return (value << np.uint32(distance)) | (
        value >> np.uint32(32 - distance)
    )


def _threefry2x32(
    prng_key: np.ndarray,
    count_hi: np.ndarray | np.uint32,
    count_lo: np.ndarray | np.uint32,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Threefry2x32 matching JAX's partitionable implementation."""

    prng_key = np.asarray(prng_key, dtype=np.uint32)
    if prng_key.ndim < 1 or prng_key.shape[0] != 2:
        raise ValueError(
            f"PRNG key must have leading shape (2,), got {prng_key.shape}"
        )

    k0, k1 = prng_key
    keys = (k0, k1, k0 ^ k1 ^ _THREEFRY_PARITY)
    x0 = np.asarray(count_hi, dtype=np.uint32) + k0
    x1 = np.asarray(count_lo, dtype=np.uint32) + k1

    with np.errstate(over="ignore"):
        for block in range(5):
            for rotation in _ROTATIONS[block % 2]:
                x0 = x0 + x1
                x1 = _rotate_left(x1, rotation)
                x1 = x0 ^ x1
            x0 = x0 + keys[(block + 1) % 3]
            x1 = x1 + keys[(block + 2) % 3] + np.uint32(block + 1)
    return x0, x1


def key(seed: int) -> np.ndarray:
    """Create a legacy JAX uint32[2] key from an integer seed."""

    seed_u64 = np.uint64(seed)
    return np.array(
        [np.uint32(seed_u64 >> np.uint64(32)), np.uint32(seed_u64)],
        dtype=np.uint32,
    )


def split(prng_key: mx.array | np.ndarray, num: int = 2) -> np.ndarray:
    """Match ``jax.random.split(key, num)`` for partitionable Threefry."""

    if num < 0:
        raise ValueError(f"num must be non-negative, got {num}")
    count_hi = np.zeros((num,), dtype=np.uint32)
    count_lo = np.arange(num, dtype=np.uint32)
    bits_hi, bits_lo = _threefry2x32(
        np.asarray(prng_key, dtype=np.uint32), count_hi, count_lo
    )
    return np.stack((bits_hi, bits_lo), axis=-1)


def fold_in(prng_key: mx.array | np.ndarray, data: np.ndarray) -> np.ndarray:
    """Match vectorized ``jax.random.fold_in`` for uint32 data."""

    data = np.asarray(data, dtype=np.uint32)
    zeros = np.zeros_like(data)
    high, low = _threefry2x32(
        np.asarray(prng_key, dtype=np.uint32), zeros, data
    )
    return np.stack((high, low), axis=-1)


def random_bits(
    prng_key: mx.array | np.ndarray,
    shape: Sequence[int],
) -> np.ndarray:
    """Generate JAX-compatible uint32 random bits with the requested shape."""

    shape = tuple(int(dim) for dim in shape)
    size = int(np.prod(shape, dtype=np.int64))
    counts = np.arange(size, dtype=np.uint64)
    count_hi = (counts >> np.uint64(32)).astype(np.uint32)
    count_lo = counts.astype(np.uint32)
    bits_hi, bits_lo = _threefry2x32(
        np.asarray(prng_key, dtype=np.uint32), count_hi, count_lo
    )
    return (bits_hi ^ bits_lo).reshape(shape)


def uniform(
    prng_key: mx.array | np.ndarray,
    shape: Sequence[int],
    *,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> mx.array:
    """Generate JAX-compatible float32 uniform values."""

    bits = random_bits(prng_key, shape)
    float_bits = (bits >> np.uint32(9)) | np.uint32(0x3F800000)
    unit = mx.array(float_bits.view(np.float32) - np.float32(1.0))
    lower = mx.array(minval, dtype=mx.float32)
    upper = mx.array(maxval, dtype=mx.float32)
    return mx.maximum(lower, unit * (upper - lower) + lower)


def normal(
    prng_key: mx.array | np.ndarray,
    shape: Sequence[int],
) -> mx.array:
    """Generate JAX-compatible float32 standard normal values."""

    lower = np.nextafter(np.float32(-1.0), np.float32(0.0), dtype=np.float32)
    values = uniform(prng_key, shape, minval=float(lower), maxval=1.0)
    return mx.array(np.sqrt(2.0), dtype=mx.float32) * mx.erfinv(values)


def haiku_next_rng_keys(
    root_key: mx.array | np.ndarray,
    count: int,
) -> tuple[np.ndarray, ...]:
    """Return sequential ``hk.next_rng_key`` values from a Haiku root key."""

    state = np.asarray(root_key, dtype=np.uint32)
    result = []
    for _ in range(count):
        state, subkey = split(state, 2)
        result.append(subkey)
    return tuple(result)


def official_af3_model_and_diffusion_keys(
    root_key: mx.array | np.ndarray,
    *,
    num_recycles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive official AF3 model and post-trunk diffusion Haiku keys.

    The model key is the first ``hk.next_rng_key()`` value. Canonical AF3 trunk
    construction and execution consume additional Haiku keys even though the
    Evoformer also receives an explicit key. With the official trunk structure,
    diffusion receives Haiku key position ``16 + 12 * num_recycles`` (one-based).
    This helper intentionally scopes that offset to the official AF3 trunk RNG
    schedule rather than presenting it as a general Haiku rule.
    """

    if num_recycles < 0:
        raise ValueError(f"num_recycles must be non-negative, got {num_recycles}")
    diffusion_position = 16 + 12 * int(num_recycles)
    keys = haiku_next_rng_keys(root_key, diffusion_position)
    return keys[0], keys[-1]
