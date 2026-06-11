"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import numpy as np


def create_test_batch(num_residues: int = 10, seed: int = 42):
    """Create minimal FeatureBatch for testing.

    Args:
        num_residues: Number of residues.
        seed: Random seed for reproducibility.

    Returns:
        FeatureBatch with basic protein features.
    """
    from alphafold3_mlx.core import FeatureBatch

    np.random.seed(seed)

    feature_dict = {
        "aatype": np.random.randint(0, 20, size=num_residues).astype(np.int32),
        "token_mask": np.ones(num_residues, dtype=np.float32),
        "residue_index": np.arange(num_residues, dtype=np.int32),
        "asym_id": np.zeros(num_residues, dtype=np.int32),
        "entity_id": np.zeros(num_residues, dtype=np.int32),
        "sym_id": np.zeros(num_residues, dtype=np.int32),
    }

    return FeatureBatch.from_numpy(feature_dict)


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets.

    Args:
        coords1: First coordinate set [N, 3].
        coords2: Second coordinate set [N, 3].

    Returns:
        RMSD value in Angstroms.
    """
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))
