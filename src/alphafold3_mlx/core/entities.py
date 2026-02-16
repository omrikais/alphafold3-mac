"""Core data entities for AlphaFold 3 MLX.

This module defines the core data structures that flow through the model:
- Embeddings: Evoformer output representations
- AtomPositions: Predicted 3D coordinates
- ConfidenceScores: pLDDT, PAE, PDE, pTM, ipTM predictions
- NoiseSchedule: Karras diffusion noise schedule
- GatherInfo: Information for gathering pseudo-beta positions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class Embeddings:
    """Evoformer output embeddings.

    State Transitions:
        - Initial: Zero tensors
        - After each recycling iteration: Updated by Evoformer
        - After final iteration: Cast to float32, passed to Diffusion and Confidence heads

    Validation Rules:
        - single.shape[0] == pair.shape[0] == pair.shape[1] == num_residues
        - No NaN values after each Evoformer iteration
        - dtype matches global_config precision setting
    """

    single: mx.array
    """Per-residue representation. Shape: [num_residues, seq_channel]"""

    pair: mx.array
    """Pairwise residue relationships. Shape: [num_residues, num_residues, pair_channel]"""

    target_feat: mx.array
    """Input feature embedding. Shape: [num_residues, target_feat_dim]"""

    def __post_init__(self) -> None:
        """Validate embedding shapes."""
        if self.single.ndim != 2:
            raise ValueError(
                f"single must be 2D [num_residues, seq_channel], got shape {self.single.shape}"
            )
        if self.pair.ndim != 3:
            raise ValueError(
                f"pair must be 3D [num_residues, num_residues, pair_channel], "
                f"got shape {self.pair.shape}"
            )
        if self.single.shape[0] != self.pair.shape[0]:
            raise ValueError(
                f"single and pair must have matching num_residues: "
                f"{self.single.shape[0]} vs {self.pair.shape[0]}"
            )
        if self.pair.shape[0] != self.pair.shape[1]:
            raise ValueError(
                f"pair must be square in first two dims: {self.pair.shape}"
            )

    @property
    def num_residues(self) -> int:
        """Return number of residues."""
        return self.single.shape[0]

    @property
    def seq_channel(self) -> int:
        """Return single representation channel dimension."""
        return self.single.shape[1]

    @property
    def pair_channel(self) -> int:
        """Return pair representation channel dimension."""
        return self.pair.shape[2]

    @classmethod
    def zeros(
        cls,
        num_residues: int,
        seq_channel: int = 384,
        pair_channel: int = 128,
        target_feat_dim: int = 22,
        dtype: mx.Dtype = mx.float32,
    ) -> "Embeddings":
        """Create zero-initialized embeddings."""
        return cls(
            single=mx.zeros((num_residues, seq_channel), dtype=dtype),
            pair=mx.zeros((num_residues, num_residues, pair_channel), dtype=dtype),
            target_feat=mx.zeros((num_residues, target_feat_dim), dtype=dtype),
        )


@dataclass(frozen=True)
class AtomPositions:
    """Predicted atomic coordinates.

    State Transitions:
        - Initial (diffusion): Random noise scaled by sigma_max
        - During diffusion: Iteratively denoised
        - Final: Predicted coordinates

    Validation Rules:
        - positions.shape[-1] == 3
        - Bond lengths within physical limits (0.8-2.5Å typical)
        - Bond angles within physical limits (90°-180° typical)
    """

    positions: mx.array
    """3D coordinates. Shape: [num_samples, num_residues, max_atoms, 3]"""

    mask: mx.array
    """Validity mask. Shape: [num_samples, num_residues, max_atoms]"""

    def __post_init__(self) -> None:
        """Validate positions and mask shapes."""
        if self.positions.ndim != 4:
            raise ValueError(
                f"positions must be 4D [num_samples, num_residues, max_atoms, 3], "
                f"got shape {self.positions.shape}"
            )
        if self.positions.shape[-1] != 3:
            raise ValueError(
                f"positions last dim must be 3 (xyz), got {self.positions.shape[-1]}"
            )
        if self.mask.ndim != 3:
            raise ValueError(
                f"mask must be 3D [num_samples, num_residues, max_atoms], "
                f"got shape {self.mask.shape}"
            )
        if self.positions.shape[:3] != self.mask.shape:
            raise ValueError(
                f"positions and mask shapes must match: "
                f"{self.positions.shape[:3]} vs {self.mask.shape}"
            )

    @property
    def num_samples(self) -> int:
        """Return number of structure samples."""
        return self.positions.shape[0]

    @property
    def num_residues(self) -> int:
        """Return number of residues."""
        return self.positions.shape[1]

    @property
    def max_atoms(self) -> int:
        """Return maximum atoms per residue."""
        return self.positions.shape[2]

    @classmethod
    def from_noise(
        cls,
        num_samples: int,
        num_residues: int,
        max_atoms: int,
        atom_mask: mx.array,
        sigma: float,
        key: mx.array,
    ) -> "AtomPositions":
        """Create positions from noise for diffusion initialization.

        Args:
            num_samples: Number of structure samples.
            num_residues: Number of residues.
            max_atoms: Maximum atoms per residue.
            atom_mask: Atom validity mask [num_residues, max_atoms].
            sigma: Noise scale (typically sigma_max).
            key: Random key for noise generation.

        Returns:
            AtomPositions with random noise scaled by sigma.
        """
        shape = (num_samples, num_residues, max_atoms, 3)
        noise = mx.random.normal(shape=shape, key=key)
        positions = noise * sigma

        # Expand mask to num_samples
        mask = mx.broadcast_to(
            atom_mask[None, ...], (num_samples, num_residues, max_atoms)
        )

        return cls(positions=positions, mask=mask)


@dataclass(frozen=True)
class ConfidenceScores:
    """Model confidence predictions.

    Validation Rules:
        - plddt values in [0, 100]
        - pae and pde non-negative
        - ptm and iptm in [0, 1]
        - tm_pae_global and tm_pae_interface in [0, 1]
    """

    plddt: mx.array
    """Per-atom predicted lDDT score. Shape: [num_samples, num_residues, max_atoms]"""

    pae: mx.array
    """Predicted aligned error matrix. Shape: [num_samples, num_residues, num_residues]"""

    pde: mx.array
    """Predicted distance error. Shape: [num_samples, num_residues, num_residues]"""

    ptm: mx.array
    """Predicted TM-score (global). Shape: [num_samples]"""

    iptm: mx.array
    """Interface predicted TM-score. Shape: [num_samples]"""

    tm_pae_global: mx.array | None = None
    """TM-score weighted PAE (global). Shape: [num_samples, num_residues, num_residues]"""

    tm_pae_interface: mx.array | None = None
    """TM-score weighted PAE (interface only). Shape: [num_samples, num_residues, num_residues]"""

    def __post_init__(self) -> None:
        """Validate confidence score shapes."""
        if self.plddt.ndim != 3:
            raise ValueError(
                f"plddt must be 3D [num_samples, num_residues, max_atoms], "
                f"got shape {self.plddt.shape}"
            )
        if self.pae.ndim != 3:
            raise ValueError(
                f"pae must be 3D [num_samples, num_residues, num_residues], "
                f"got shape {self.pae.shape}"
            )
        if self.pde.ndim != 3:
            raise ValueError(
                f"pde must be 3D [num_samples, num_residues, num_residues], "
                f"got shape {self.pde.shape}"
            )
        if self.ptm.ndim != 1:
            raise ValueError(f"ptm must be 1D [num_samples], got shape {self.ptm.shape}")
        if self.iptm.ndim != 1:
            raise ValueError(
                f"iptm must be 1D [num_samples], got shape {self.iptm.shape}"
            )
        # Optional TM-adjusted PAE fields
        if self.tm_pae_global is not None and self.tm_pae_global.ndim != 3:
            raise ValueError(
                f"tm_pae_global must be 3D [num_samples, num_residues, num_residues], "
                f"got shape {self.tm_pae_global.shape}"
            )
        if self.tm_pae_interface is not None and self.tm_pae_interface.ndim != 3:
            raise ValueError(
                f"tm_pae_interface must be 3D [num_samples, num_residues, num_residues], "
                f"got shape {self.tm_pae_interface.shape}"
            )

    @property
    def num_samples(self) -> int:
        """Return number of structure samples."""
        return self.ptm.shape[0]


@dataclass
class NoiseSchedule:
    """Karras et al. noise schedule for diffusion.

    The schedule follows the formulation from "Elucidating the Design Space
    of Diffusion-Based Generative Models" (Karras et al., 2022).
    """

    sigma: mx.array
    """Noise levels from sigma_max to sigma_min. Shape: [num_steps + 1]"""

    t: mx.array
    """Normalized time values. Shape: [num_steps + 1]"""

    @classmethod
    def karras(
        cls,
        num_steps: int,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        rho: float = 7.0,
    ) -> "NoiseSchedule":
        """Create Karras noise schedule.

        Args:
            num_steps: Number of diffusion steps.
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            rho: Schedule curvature parameter.

        Returns:
            NoiseSchedule with sigma and t arrays.
        """
        # t values from 0 to 1
        t = mx.linspace(0, 1, num_steps + 1)

        # Karras schedule: sigma = (sigma_max^(1/rho) + t*(sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        sigma_max_inv_rho = sigma_max ** (1.0 / rho)
        sigma_min_inv_rho = sigma_min ** (1.0 / rho)
        sigma = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** rho

        return cls(sigma=sigma, t=t)

    @property
    def num_steps(self) -> int:
        """Return number of diffusion steps."""
        return len(self.sigma) - 1


@dataclass(frozen=True)
class GatherInfo:
    """Information for gathering pseudo-beta positions from atom positions.

    Used to extract CB positions (or CA for glycine) for distogram computation.
    """

    indices: mx.array
    """(residue_idx, atom_idx) pairs. Shape: [num_residues, 2]"""

    mask: mx.array
    """Validity mask. Shape: [num_residues]"""

    def __post_init__(self) -> None:
        """Validate gather info shapes."""
        if self.indices.ndim != 2 or self.indices.shape[1] != 2:
            raise ValueError(
                f"indices must be 2D with shape [N, 2], got {self.indices.shape}"
            )
        if self.mask.ndim != 1:
            raise ValueError(f"mask must be 1D, got shape {self.mask.shape}")
        if self.indices.shape[0] != self.mask.shape[0]:
            raise ValueError(
                f"indices and mask must have same length: "
                f"{self.indices.shape[0]} vs {self.mask.shape[0]}"
            )
