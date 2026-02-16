"""MSA stack integration tests.

Tests that the MSA stack works correctly end-to-end when
MSA features are provided to the Evoformer.
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig, MSAStackConfig
from alphafold3_mlx.network.evoformer import Evoformer


def create_test_batch_with_msa(
    num_residues: int = 10,
    num_msa_sequences: int = 16,
    msa_channel: int = 64,
    seed: int = 42,
) -> tuple[FeatureBatch, mx.array, mx.array]:
    """Create feature batch with MSA features for testing.

    Returns:
        Tuple of (batch, msa_features, msa_mask)
    """
    np.random.seed(seed)

    feature_dict = {
        "aatype": np.random.randint(0, 20, size=num_residues).astype(np.int32),
        "token_mask": np.ones(num_residues, dtype=np.float32),
        "residue_index": np.arange(num_residues, dtype=np.int32),
        "asym_id": np.zeros(num_residues, dtype=np.int32),
        "entity_id": np.zeros(num_residues, dtype=np.int32),
        "sym_id": np.zeros(num_residues, dtype=np.int32),
    }

    batch = FeatureBatch.from_numpy(feature_dict)

    # Create MSA features [batch, num_seqs, seq_len, msa_channel]
    msa_features = mx.array(
        np.random.randn(1, num_msa_sequences, num_residues, msa_channel).astype(np.float32)
    )
    msa_mask = mx.ones((1, num_msa_sequences, num_residues))

    return batch, msa_features, msa_mask


class TestMSAStackIntegration:
    """Test MSA stack integration with Evoformer."""

    @pytest.fixture
    def msa_evoformer(self):
        """Create Evoformer with MSA stack enabled."""
        config = EvoformerConfig(
            num_pairformer_layers=2,
            num_msa_layers=2,
            use_msa_stack=True,
            msa_stack=MSAStackConfig(
                num_layers=2,
                msa_channel=64,
                pair_channel=128,
            ),
        )
        global_config = GlobalConfig(use_compile=False)
        return Evoformer(config=config, global_config=global_config)

    @pytest.fixture
    def no_msa_evoformer(self):
        """Create Evoformer with MSA stack disabled."""
        config = EvoformerConfig(
            num_pairformer_layers=2,
            num_msa_layers=0,
            use_msa_stack=False,
        )
        global_config = GlobalConfig(use_compile=False)
        return Evoformer(config=config, global_config=global_config)

    def test_evoformer_with_msa_runs(self, msa_evoformer):
        """Test that Evoformer processes MSA features without error."""
        batch_size = 1
        num_residues = 8
        num_msa_sequences = 16
        seq_channel = 384
        pair_channel = 128
        msa_channel = 64

        single = mx.zeros((batch_size, num_residues, seq_channel))
        pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))
        msa_features = mx.zeros((batch_size, num_msa_sequences, num_residues, msa_channel))
        msa_mask = mx.ones((batch_size, num_msa_sequences, num_residues))

        single_out, pair_out = msa_evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            msa_features=msa_features,
            msa_mask=msa_mask,
        )

        mx.eval(single_out, pair_out)

        assert single_out.shape == single.shape
        assert pair_out.shape == pair.shape

    def test_evoformer_without_msa_runs(self, no_msa_evoformer):
        """Test that Evoformer works without MSA features."""
        batch_size = 1
        num_residues = 8
        seq_channel = 384
        pair_channel = 128

        single = mx.zeros((batch_size, num_residues, seq_channel))
        pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        single_out, pair_out = no_msa_evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
        )

        mx.eval(single_out, pair_out)

        assert single_out.shape == single.shape
        assert pair_out.shape == pair.shape

    def test_msa_features_affect_output(self, msa_evoformer):
        """Test that MSA features change the output (not ignored)."""
        batch_size = 1
        num_residues = 8
        num_msa_sequences = 16
        seq_channel = 384
        pair_channel = 128
        msa_channel = 64

        single = mx.zeros((batch_size, num_residues, seq_channel))
        pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        # Run without MSA
        single_no_msa, pair_no_msa = msa_evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            msa_features=None,
        )

        # Run with random MSA features
        key = mx.random.key(42)
        msa_features = mx.random.normal(
            (batch_size, num_msa_sequences, num_residues, msa_channel),
            key=key,
        )
        msa_mask = mx.ones((batch_size, num_msa_sequences, num_residues))

        single_with_msa, pair_with_msa = msa_evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            msa_features=msa_features,
            msa_mask=msa_mask,
        )

        mx.eval(single_no_msa, pair_no_msa, single_with_msa, pair_with_msa)

        # Pair should differ when MSA is provided (MSA -> OPM -> pair update).
        # Single may or may not differ depending on PairFormer propagation depth.
        pair_diff = np.max(np.abs(np.array(pair_with_msa) - np.array(pair_no_msa)))

        assert pair_diff > 0, "MSA features should affect pair representation"

    def test_msa_no_nan(self, msa_evoformer):
        """Test that MSA processing produces no NaN values."""
        batch_size = 1
        num_residues = 8
        num_msa_sequences = 16
        seq_channel = 384
        pair_channel = 128
        msa_channel = 64

        key = mx.random.key(42)
        k1, k2, k3 = mx.random.split(key, 3)

        single = mx.random.normal((batch_size, num_residues, seq_channel), key=k1) * 0.1
        pair = mx.random.normal((batch_size, num_residues, num_residues, pair_channel), key=k2) * 0.1
        msa_features = mx.random.normal((batch_size, num_msa_sequences, num_residues, msa_channel), key=k3) * 0.1

        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))
        msa_mask = mx.ones((batch_size, num_msa_sequences, num_residues))

        single_out, pair_out = msa_evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            msa_features=msa_features,
            msa_mask=msa_mask,
        )

        mx.eval(single_out, pair_out)

        single_np = np.array(single_out)
        pair_np = np.array(pair_out)

        assert not np.any(np.isnan(single_np)), "Single representation contains NaN"
        assert not np.any(np.isnan(pair_np)), "Pair representation contains NaN"


class TestMSAMasking:
    """Test MSA masking functionality."""

    @pytest.fixture
    def evoformer(self):
        """Create Evoformer with MSA stack."""
        config = EvoformerConfig(
            num_pairformer_layers=2,
            num_msa_layers=2,
            use_msa_stack=True,
        )
        global_config = GlobalConfig(use_compile=False)
        return Evoformer(config=config, global_config=global_config)

    def test_msa_partial_mask(self, evoformer):
        """Test that partial MSA masks are handled correctly."""
        batch_size = 1
        num_residues = 8
        num_msa_sequences = 16
        seq_channel = 384
        pair_channel = 128
        msa_channel = 64

        single = mx.zeros((batch_size, num_residues, seq_channel))
        pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        # Create MSA with partial mask (some sequences masked out)
        msa_features = mx.zeros((batch_size, num_msa_sequences, num_residues, msa_channel))
        msa_mask = mx.ones((batch_size, num_msa_sequences, num_residues))
        # Mask out half the sequences
        msa_mask = mx.concatenate([
            mx.ones((1, num_msa_sequences // 2, num_residues)),
            mx.zeros((1, num_msa_sequences // 2, num_residues)),
        ], axis=1)

        single_out, pair_out = evoformer(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            msa_features=msa_features,
            msa_mask=msa_mask,
        )

        mx.eval(single_out, pair_out)

        # Should complete without error
        assert single_out.shape == single.shape
        assert pair_out.shape == pair.shape


class TestMSAFullPipeline:
    """Test MSA processing through full model pipeline."""

    @pytest.fixture
    def model_with_msa(self):
        """Create model with MSA support enabled."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                num_msa_layers=2,
                use_msa_stack=True,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        return Model(config)

    def test_model_inference_without_msa(self, model_with_msa):
        """Test that model runs without MSA (graceful fallback)."""
        batch, _, _ = create_test_batch_with_msa(num_residues=10)
        key = mx.random.key(42)

        # Run without providing MSA features
        result = model_with_msa(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)
        assert not np.any(np.isnan(coords)), "Coordinates should not contain NaN"
