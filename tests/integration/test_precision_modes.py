"""Precision mode stability tests.

Tests that the model operates correctly in different precision modes:
- float32: Full precision (default)
- float16: Half precision for memory savings
- bfloat16: Brain floating point for M3+ chips
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig


def create_test_batch(num_residues: int = 10, seed: int = 42) -> FeatureBatch:
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


class TestFloat32Precision:
    """Test float32 precision mode (default)."""

    @pytest.fixture
    def model(self):
        """Create model in float32 mode."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(
                precision="float32",
                use_compile=False,
            ),
            num_recycles=1,
        )
        return Model(config)

    def test_float32_no_nan(self, model):
        """Test that float32 mode produces no NaN values."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions, result.confidence.plddt)

        coords = np.array(result.atom_positions.positions)
        plddt = np.array(result.confidence.plddt)

        assert not np.any(np.isnan(coords)), "float32 produced NaN coordinates"
        assert not np.any(np.isnan(plddt)), "float32 produced NaN pLDDT"

    def test_float32_precision_property(self, model):
        """Test that precision property returns correct value."""
        assert model.precision == "float32"


class TestFloat16Precision:
    """Test float16 precision mode."""

    @pytest.fixture
    def model(self):
        """Create model configured for float16."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(
                precision="float16",
                use_compile=False,
            ),
            num_recycles=1,
        )
        return Model(config)

    def test_float16_inference_runs(self, model):
        """Test that float16 inference completes without error."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        # Should complete without error
        assert result.atom_positions.positions.shape[0] == 1

    def test_float16_no_nan(self, model):
        """Test that float16 mode doesn't produce NaN values."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions, result.confidence.plddt)

        coords = np.array(result.atom_positions.positions)
        plddt = np.array(result.confidence.plddt)

        # float16 should not produce NaN for small models
        assert not np.any(np.isnan(coords)), "float16 produced NaN coordinates"
        assert not np.any(np.isnan(plddt)), "float16 produced NaN pLDDT"

    def test_float16_plddt_in_range(self, model):
        """Test that float16 pLDDT values are in valid range."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.confidence.plddt)

        plddt = np.array(result.confidence.plddt)

        assert np.all(plddt >= 0), f"pLDDT below 0: {plddt.min()}"
        assert np.all(plddt <= 100), f"pLDDT above 100: {plddt.max()}"


class TestBFloat16Precision:
    """Test bfloat16 precision mode.

    Note: bfloat16 is natively supported on M3+ chips.
    On older chips, MLX will emulate bfloat16.
    """

    @pytest.fixture
    def model(self):
        """Create model configured for bfloat16."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(
                precision="bfloat16",
                use_compile=False,
            ),
            num_recycles=1,
        )
        return Model(config)

    def test_bfloat16_inference_runs(self, model):
        """Test that bfloat16 inference completes without error."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        assert result.atom_positions.positions.shape[0] == 1

    def test_bfloat16_no_nan(self, model):
        """Test that bfloat16 mode doesn't produce NaN values."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions, result.confidence.plddt)

        coords = np.array(result.atom_positions.positions)
        plddt = np.array(result.confidence.plddt)

        assert not np.any(np.isnan(coords)), "bfloat16 produced NaN coordinates"
        assert not np.any(np.isnan(plddt)), "bfloat16 produced NaN pLDDT"


class TestBFloat16EvoformerStability:
    """Test bfloat16 stability across Evoformer layers. requirement: bfloat16 mode must be stable across all 48 layers.
    This test uses reduced layers for faster testing.
    """

    @pytest.fixture
    def deep_model(self):
        """Create model with more layers for stability testing."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=8, # More layers for stability test
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4, # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(
                precision="bfloat16",
                use_compile=False,
            ),
            num_recycles=1,
        )
        return Model(config)

    def test_bfloat16_multi_layer_stability(self, deep_model):
        """Test bfloat16 stability with multiple Evoformer layers."""
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = deep_model(batch, key)
        mx.eval(result.atom_positions.positions, result.confidence.plddt)

        coords = np.array(result.atom_positions.positions)
        plddt = np.array(result.confidence.plddt)

        # Check for NaN (indicates numerical instability)
        assert not np.any(np.isnan(coords)), (
            "bfloat16 produced NaN after multiple Evoformer layers"
        )
        assert not np.any(np.isnan(plddt)), (
            "bfloat16 produced NaN in confidence scores"
        )

        # Check for Inf (indicates overflow)
        assert not np.any(np.isinf(coords)), (
            "bfloat16 produced Inf after multiple Evoformer layers"
        )

    def test_bfloat16_values_reasonable(self, deep_model):
        """Test that bfloat16 produces reasonable coordinate values."""
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = deep_model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)

        # Coordinates should be in reasonable range (not exploded)
        max_coord = np.max(np.abs(coords))
        assert max_coord < 1000, (
            f"Coordinates exploded to {max_coord}, indicating numerical instability"
        )


class TestPrecisionModeSwitching:
    """Test switching between precision modes."""

    def test_set_precision_float16(self):
        """Test switching to float16 precision."""
        config = ModelConfig(
            evoformer=EvoformerConfig(num_pairformer_layers=1, use_msa_stack=False),
            diffusion=DiffusionConfig(num_steps=3, num_samples=1, num_transformer_blocks=4), # Must be divisible by super_block_size=4
            global_config=GlobalConfig(precision="float32", use_compile=False),
            num_recycles=1,
        )
        model = Model(config)

        assert model.precision == "float32"

        model.set_precision("float16")
        assert model.precision == "float16"

    def test_set_precision_bfloat16(self):
        """Test switching to bfloat16 precision."""
        config = ModelConfig(
            evoformer=EvoformerConfig(num_pairformer_layers=1, use_msa_stack=False),
            diffusion=DiffusionConfig(num_steps=3, num_samples=1, num_transformer_blocks=4), # Must be divisible by super_block_size=4
            global_config=GlobalConfig(precision="float32", use_compile=False),
            num_recycles=1,
        )
        model = Model(config)

        model.set_precision("bfloat16")
        assert model.precision == "bfloat16"

    def test_set_precision_invalid_raises(self):
        """Test that invalid precision raises error."""
        config = ModelConfig(
            evoformer=EvoformerConfig(num_pairformer_layers=1, use_msa_stack=False),
            diffusion=DiffusionConfig(num_steps=3, num_samples=1, num_transformer_blocks=4), # Must be divisible by super_block_size=4
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        model = Model(config)

        with pytest.raises(ValueError, match="precision must be"):
            model.set_precision("float64")
