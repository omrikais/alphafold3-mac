"""Shared fixtures for unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def create_test_batch(num_residues: int = 8, seed: int = 42):
    """Create minimal feature batch for testing."""
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


def create_small_model(use_compile: bool = False):
    """Create a small model for testing.

    Args:
        use_compile: Whether to enable mx.compile. Defaults to False for faster tests.
    """
    from alphafold3_mlx.model import Model
    from alphafold3_mlx.core import ModelConfig
    from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig

    config = ModelConfig(
        evoformer=EvoformerConfig(
            num_pairformer_layers=2,
            use_msa_stack=False,
        ),
        diffusion=DiffusionConfig(
            num_steps=3,
            num_samples=1,
            num_transformer_blocks=4,
        ),
        global_config=GlobalConfig(
            precision="float32",
            use_compile=use_compile,
        ),
        num_recycles=1,
    )
    return Model(config)


def load_golden_data(golden_file: str, generate_cmd: str = "python scripts/generate_model_reference_outputs.py"):
    """Load golden reference NPZ data, skipping if not found.

    Args:
        golden_file: Path to the NPZ file (relative to project root).
        generate_cmd: Command to show in skip message for regenerating the file.

    Returns:
        Loaded NPZ data.
    """
    golden_path = Path(golden_file)
    if not golden_path.exists():
        pytest.skip(
            f"Golden reference not found: {golden_path}. Run: {generate_cmd}"
        )
    return np.load(golden_path)
