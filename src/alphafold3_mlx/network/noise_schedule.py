"""Noise schedule for diffusion head in AlphaFold 3 MLX.

This module implements the Karras noise schedule and noise level
embeddings used in the diffusion head for coordinate generation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.core.constants import SIGMA_DATA, SIGMA_MAX, SIGMA_MIN, RHO


def karras_schedule(
    num_steps: int,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    rho: float = RHO,
) -> mx.array:
    """Compute Karras noise schedule.

    The schedule follows the formulation from "Elucidating the Design Space
    of Diffusion-Based Generative Models" (Karras et al., 2022):

    sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    Args:
        num_steps: Number of diffusion steps.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Schedule curvature parameter.

    Returns:
        Noise levels array. Shape: [num_steps + 1]
    """
    # Step indices from 0 to num_steps
    step_indices = mx.arange(num_steps + 1, dtype=mx.float32)

    # Compute sigma values using Karras formula
    sigma_max_inv_rho = sigma_max ** (1.0 / rho)
    sigma_min_inv_rho = sigma_min ** (1.0 / rho)

    # Interpolation factor (0 to 1)
    t = step_indices / num_steps

    # Karras schedule: SIGMA_DATA * (smax^(1/rho) + t*(smin^(1/rho) - smax^(1/rho)))^rho
    sigma = SIGMA_DATA * (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho

    return sigma


def get_noise_level_embedding_dim(embed_dim: int) -> int:
    """Get the number of frequencies for noise level embedding.

    Args:
        embed_dim: Target embedding dimension.

    Returns:
        Number of Fourier frequencies (embed_dim // 2).
    """
    return embed_dim // 2


class NoiseLevelEmbedding(nn.Module):
    """Fourier feature embedding for noise levels.

    Encodes the noise level sigma into a high-dimensional representation
    using sinusoidal (Fourier) features, similar to positional encoding
    in transformers.
    """

    def __init__(
        self,
        embed_dim: int,
        sigma_data: float = SIGMA_DATA,
        max_period: float = 10000.0,
    ) -> None:
        """Initialize noise level embedding.

        Args:
            embed_dim: Output embedding dimension.
            sigma_data: Data noise scale for normalization.
            max_period: Maximum period for Fourier features.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.sigma_data = sigma_data
        self.max_period = max_period

        # Number of frequency components
        self.num_freqs = embed_dim // 2

        # Precompute frequency bands (logarithmically spaced)
        freqs = mx.exp(
            -math.log(max_period)
            * mx.arange(self.num_freqs, dtype=mx.float32)
            / self.num_freqs
        )
        self.freqs = freqs  # Will be used as a constant

    def __call__(self, sigma: mx.array) -> mx.array:
        """Compute noise level embedding.

        Args:
            sigma: Noise level(s). Shape: [...] (any shape)

        Returns:
            Noise level embedding. Shape: [..., embed_dim]
        """
        # Normalize by sigma_data (log scale)
        # log(sigma / sigma_data) gives normalized noise level
        sigma_normalized = mx.log(sigma / self.sigma_data)

        # Expand for broadcasting: [...] -> [..., 1]
        sigma_expanded = sigma_normalized[..., None]

        # Compute Fourier features
        # freqs shape: [num_freqs]
        args = sigma_expanded * self.freqs

        # Concatenate sin and cos features
        embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)

        return embedding


class NoiseScheduleSampler:
    """Sampler for diffusion noise schedule.

    Provides utilities for sampling from the noise schedule during
    the diffusion denoising process.
    """

    def __init__(
        self,
        num_steps: int = 200,
        sigma_min: float = SIGMA_MIN,
        sigma_max: float = SIGMA_MAX,
        rho: float = RHO,
        sigma_data: float = SIGMA_DATA,
    ) -> None:
        """Initialize noise schedule sampler.

        Args:
            num_steps: Number of diffusion steps.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            rho: Schedule curvature parameter.
            sigma_data: Data noise scale.
        """
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data

        # Precompute schedule
        self.sigmas = karras_schedule(num_steps, sigma_min, sigma_max, rho)

    def get_sigma(self, step: int) -> float:
        """Get sigma at a specific step.

        Args:
            step: Step index (0 to num_steps).

        Returns:
            Noise level sigma.
        """
        return float(self.sigmas[step].item())

    def get_sigma_next(self, step: int) -> float:
        """Get sigma at the next step.

        Args:
            step: Current step index.

        Returns:
            Noise level sigma at step + 1.
        """
        return float(self.sigmas[step + 1].item())

    def get_c_skip(self, sigma: float) -> float:
        """Compute skip connection scaling.

        Args:
            sigma: Current noise level.

        Returns:
            Skip connection scale factor.
        """
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def get_c_out(self, sigma: float) -> float:
        """Compute output scaling.

        Args:
            sigma: Current noise level.

        Returns:
            Output scale factor.
        """
        return sigma * self.sigma_data / math.sqrt(sigma**2 + self.sigma_data**2)

    def get_c_in(self, sigma: float) -> float:
        """Compute input scaling.

        Args:
            sigma: Current noise level.

        Returns:
            Input scale factor.
        """
        return 1.0 / math.sqrt(sigma**2 + self.sigma_data**2)

    def get_c_noise(self, sigma: float) -> float:
        """Compute noise conditioning value.

        Args:
            sigma: Current noise level.

        Returns:
            Noise conditioning value (0.25 * log(sigma)).
        """
        return 0.25 * math.log(sigma)
