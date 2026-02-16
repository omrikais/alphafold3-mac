"""Output dataclasses for AlphaFold 3 MLX.

This module provides output containers including:
- AttentionOutput: Low-level attention output (Phase 0)
- ModelResult: Complete model output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import mlx.core as mx
import numpy as np

from alphafold3_mlx.core.intermediates import AttentionIntermediates

if TYPE_CHECKING:
    from alphafold3_mlx.core.entities import (
        AtomPositions,
        ConfidenceScores,
        Embeddings,
    )


# Type alias for array-like
Array = Any


def _to_numpy_safe(arr: Any) -> np.ndarray:
    """Convert array to numpy, handling bfloat16 which NumPy doesn't support.

    Args:
        arr: Array-like object (MLX array, NumPy array, etc.)

    Returns:
        NumPy array (bfloat16 is converted to float32)
    """
    # Check if it's an MLX array with bfloat16 dtype
    if hasattr(arr, 'dtype'):
        dtype_str = str(arr.dtype)
        if 'bfloat16' in dtype_str:
            # MLX bfloat16 must be cast to float32 before NumPy conversion
            import mlx.core as mx
            arr = arr.astype(mx.float32)
            mx.eval(arr)
    return np.asarray(arr)


@dataclass
class AttentionOutput:
    """Output from scaled dot-product attention.

    Attributes:
        output: Result tensor [batch, heads, seq_q, head_dim]
        intermediates: Optional captured intermediate activations
    """

    output: Array
    intermediates: AttentionIntermediates | None = None

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert to numpy arrays for validation.

        Note: bfloat16 arrays are automatically converted to float32
        since NumPy doesn't support bfloat16 natively.
        """
        result = {"output": _to_numpy_safe(self.output)}
        if self.intermediates is not None:
            result.update(self.intermediates.to_numpy())
        return result

    @classmethod
    def from_numpy(cls, data: dict[str, np.ndarray]) -> "AttentionOutput":
        """Create from numpy arrays."""
        intermediates = None
        if "logits_pre_mask" in data:
            intermediates = AttentionIntermediates.from_numpy(data)
        return cls(output=data["output"], intermediates=intermediates)


# =============================================================================
# Model Output (Phase 3 )
# =============================================================================


@dataclass
class ModelResult:
    """Complete model output.

    This container holds all outputs from a single model inference run,
    including predicted coordinates, confidence scores, and optional embeddings.
    """

    atom_positions: "AtomPositions"
    """Predicted atomic coordinates. Shape: [num_samples, num_residues, max_atoms, 3]"""

    confidence: "ConfidenceScores"
    """Model confidence predictions (pLDDT, PAE, PDE, pTM, ipTM)."""

    embeddings: "Embeddings | None" = None
    """Optional Evoformer embeddings (returned if config.return_embeddings=True)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata including timing, memory usage, etc."""

    @property
    def num_samples(self) -> int:
        """Return number of structure samples."""
        return self.atom_positions.num_samples

    @property
    def best_sample_index(self) -> int:
        """Return index of sample with highest mean pLDDT.

        The best sample is defined as the one with the highest average pLDDT
        score across all valid atoms.
        """
        # Compute mean pLDDT per sample (masked average)
        plddt = self.confidence.plddt  # [num_samples, num_residues, max_atoms]
        mask = self.atom_positions.mask  # [num_samples, num_residues, max_atoms]

        # Masked sum / count per sample
        plddt_masked = plddt * mask
        sum_per_sample = mx.sum(plddt_masked, axis=(1, 2))  # [num_samples]
        count_per_sample = mx.sum(mask, axis=(1, 2))  # [num_samples]

        # Avoid division by zero
        mean_plddt = sum_per_sample / mx.maximum(count_per_sample, 1.0)

        # Return index of maximum
        return int(mx.argmax(mean_plddt).item())

    @property
    def best_positions(self) -> mx.array:
        """Return coordinates of the best sample.

        Returns:
            Atom positions for the sample with highest mean pLDDT.
            Shape: [num_residues, max_atoms, 3]
        """
        return self.atom_positions.positions[self.best_sample_index]

    def to_numpy(self) -> dict[str, np.ndarray | None]:
        """Convert model outputs to NumPy arrays for post-processing.

        Returns:
            Dictionary containing:
            - atom_positions: [num_samples, num_residues, max_atoms, 3]
            - atom_mask: [num_samples, num_residues, max_atoms]
            - plddt: [num_samples, num_residues, max_atoms]
            - pae: [num_samples, num_residues, num_residues]
            - pde: [num_samples, num_residues, num_residues]
            - ptm: [num_samples]
            - iptm: [num_samples]
            - tm_pae_global: [num_samples, num_residues, num_residues] (if available)
            - tm_pae_interface: [num_samples, num_residues, num_residues] (if available)
            - single_embedding: [num_residues, seq_channel] (if available)
            - pair_embedding: [num_residues, num_residues, pair_channel] (if available)

        Note:
            bfloat16 arrays are automatically converted to float32.
        """
        result = {
            "atom_positions": _to_numpy_safe(self.atom_positions.positions),
            "atom_mask": _to_numpy_safe(self.atom_positions.mask),
            "plddt": _to_numpy_safe(self.confidence.plddt),
            "pae": _to_numpy_safe(self.confidence.pae),
            "pde": _to_numpy_safe(self.confidence.pde),
            "ptm": _to_numpy_safe(self.confidence.ptm),
            "iptm": _to_numpy_safe(self.confidence.iptm),
        }

        # TM-score adjusted PAE outputs
        if self.confidence.tm_pae_global is not None:
            result["tm_pae_global"] = _to_numpy_safe(self.confidence.tm_pae_global)
        else:
            result["tm_pae_global"] = None

        if self.confidence.tm_pae_interface is not None:
            result["tm_pae_interface"] = _to_numpy_safe(self.confidence.tm_pae_interface)
        else:
            result["tm_pae_interface"] = None

        if self.embeddings is not None:
            result["single_embedding"] = _to_numpy_safe(self.embeddings.single)
            result["pair_embedding"] = _to_numpy_safe(self.embeddings.pair)
        else:
            result["single_embedding"] = None
            result["pair_embedding"] = None

        return result
