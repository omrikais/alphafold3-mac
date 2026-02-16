"""Template modules (distogram features) for AF3 parity."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class DistogramFeaturesConfig:
    min_bin: float = 3.25
    max_bin: float = 50.75
    num_bins: int = 39


def dgram_from_positions(positions: mx.array, config: DistogramFeaturesConfig) -> mx.array:
    """Compute distogram from positions (AF3 JAX parity).

    Args:
        positions: (num_res, 3) positions.
        config: Distogram bin config.
    Returns:
        Distogram [num_res, num_res, num_bins].
    """
    lower_breaks = mx.linspace(config.min_bin, config.max_bin, config.num_bins)
    lower_breaks = lower_breaks ** 2
    upper_breaks = mx.concatenate(
        [lower_breaks[1:], mx.array([1e8], dtype=mx.float32)], axis=-1
    )
    diff = positions[:, None, :] - positions[None, :, :]
    dist2 = mx.sum(diff ** 2, axis=-1, keepdims=True)
    dgram = (dist2 > lower_breaks).astype(mx.float32) * (dist2 < upper_breaks).astype(mx.float32)
    return dgram

