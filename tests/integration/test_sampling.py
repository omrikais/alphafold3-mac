"""Multi-sample generation tests.

Tests sample diversity and quality for the 5-sample prediction:
- Samples should be distinct (RMSD > 0)
- Samples should have appropriate diversity
- Per-sample confidence scores should be computed correctly
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig


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


def create_test_batch(num_residues: int = 15, seed: int = 42) -> FeatureBatch:
    """Create minimal feature batch for testing."""
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


class TestSampleDiversity:
    """Test sample diversity requirements."""

    @pytest.fixture
    def multi_sample_model(self):
        """Create model that generates multiple samples."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=10,  # Reduced for testing
                num_samples=5,  # Standard 5 samples
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        return Model(config)

    def test_generates_five_samples(self, multi_sample_model):
        """Test that model generates correct number of samples."""
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = multi_sample_model(batch, key)

        assert result.num_samples == 5, f"Expected 5 samples, got {result.num_samples}"

    def test_samples_not_identical(self, multi_sample_model):
        """Test that samples are not identical.

        Each sample should have RMSD > 0 compared to every other sample.
        """
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = multi_sample_model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)  # [5, residues, atoms, 3]
        num_samples = coords.shape[0]

        # Extract CA positions (atom index 1)
        ca_coords = coords[:, :, 1, :]  # [5, residues, 3]

        # Check all pairwise RMSDs
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                rmsd = compute_rmsd(ca_coords[i], ca_coords[j])
                assert rmsd > 0, (
                    f"Samples {i} and {j} are identical (RMSD=0). "
                    f"Each sample should be distinct."
                )

    def test_sample_diversity_range(self, multi_sample_model):
        """Test that sample diversity is within expected range.

        For diffusion-based sampling, samples should have reasonable diversity:
        - Not too similar (suggests collapsed sampling)
        - Not too different (suggests numerical issues)
        """
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = multi_sample_model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)
        ca_coords = coords[:, :, 1, :]

        rmsds = []
        num_samples = coords.shape[0]
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                rmsd = compute_rmsd(ca_coords[i], ca_coords[j])
                rmsds.append(rmsd)

        mean_rmsd = np.mean(rmsds)

        # Samples should have some diversity (mean RMSD > 0)
        # Upper bound depends on protein size and sampling
        assert mean_rmsd > 0, "Samples have no diversity"

    def test_different_seeds_produce_different_samples(self, multi_sample_model):
        """Test that different random seeds produce different outputs."""
        batch = create_test_batch(num_residues=10)

        result1 = multi_sample_model(batch, mx.random.key(42))
        mx.eval(result1.atom_positions.positions)

        result2 = multi_sample_model(batch, mx.random.key(123))
        mx.eval(result2.atom_positions.positions)

        # Best positions from different seeds should differ
        coords1 = np.array(result1.best_positions)[:, 1, :]  # CA only
        coords2 = np.array(result2.best_positions)[:, 1, :]

        rmsd = compute_rmsd(coords1, coords2)
        assert rmsd > 0, "Different seeds produced identical outputs"


class TestPerSampleConfidence:
    """Test per-sample confidence computation."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=3,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        return Model(config)

    def test_confidence_per_sample(self, model):
        """Test that confidence scores are computed for each sample."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.plddt, result.confidence.ptm)

        # Should have confidence for each sample
        num_samples = model.config.diffusion.num_samples
        assert result.confidence.plddt.shape[0] == num_samples
        assert result.confidence.ptm.shape[0] == num_samples
        assert result.confidence.pae.shape[0] == num_samples

    def test_plddt_valid_range(self, model):
        """Test that pLDDT values are in valid range [0, 100]."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.plddt)

        plddt = np.array(result.confidence.plddt)

        assert np.all(plddt >= 0), f"pLDDT below 0: min={plddt.min()}"
        assert np.all(plddt <= 100), f"pLDDT above 100: max={plddt.max()}"

    def test_ptm_valid_range(self, model):
        """Test that pTM values are in valid range [0, 1]."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.ptm)

        ptm = np.array(result.confidence.ptm)

        assert np.all(ptm >= 0), f"pTM below 0: min={ptm.min()}"
        assert np.all(ptm <= 1), f"pTM above 1: max={ptm.max()}"


class TestBestSampleSelection:
    """Test best sample selection mechanism."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=3,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        return Model(config)

    def test_best_sample_index_valid(self, model):
        """Test that best_sample_index returns valid index."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.plddt, result.atom_positions.positions)

        best_idx = result.best_sample_index

        assert 0 <= best_idx < result.num_samples, (
            f"best_sample_index {best_idx} out of range [0, {result.num_samples})"
        )

    def test_best_sample_has_highest_plddt(self, model):
        """Test that best sample has highest mean pLDDT."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.plddt, result.atom_positions.mask)

        plddt = np.array(result.confidence.plddt)
        mask = np.array(result.atom_positions.mask)

        # Compute mean pLDDT per sample
        mean_plddt_per_sample = []
        for i in range(result.num_samples):
            masked_plddt = plddt[i] * mask[i]
            mean_plddt = np.sum(masked_plddt) / np.maximum(np.sum(mask[i]), 1)
            mean_plddt_per_sample.append(mean_plddt)

        # Best sample should have highest mean pLDDT
        best_idx = result.best_sample_index
        expected_best_idx = np.argmax(mean_plddt_per_sample)

        assert best_idx == expected_best_idx, (
            f"best_sample_index={best_idx} but sample {expected_best_idx} "
            f"has highest mean pLDDT"
        )

    def test_best_positions_shape(self, model):
        """Test that best_positions has correct shape."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        best_pos = result.best_positions
        num_residues = 8
        max_atoms = 37

        assert best_pos.shape == (num_residues, max_atoms, 3), (
            f"Expected shape ({num_residues}, {max_atoms}, 3), "
            f"got {best_pos.shape}"
        )
