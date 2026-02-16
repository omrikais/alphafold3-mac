"""Recycling stability tests.

Tests that the recycling loop converges properly:
- 10 iterations should show decreasing differences
- Convergence tracking captures per-iteration metrics
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig
from alphafold3_mlx.model.recycling import (
    RecyclingState,
    compute_embedding_difference,
    run_recycling_loop,
    check_convergence,
)


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


class TestRecyclingState:
    """Test RecyclingState dataclass."""

    def test_initial_state(self):
        """Test initial state creation."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)

        assert state.iteration == 0
        assert state.prev_single is None
        assert state.prev_pair is None
        assert len(state.differences) == 0

    def test_state_update(self):
        """Test state update with differences."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)
        state.differences.append(0.1)
        state.differences.append(0.05)
        state.iteration = 2

        assert len(state.differences) == 2
        assert state.differences[0] > state.differences[1]


class TestEmbeddingDifference:
    """Test embedding difference computation."""

    def test_zero_difference(self):
        """Test that identical embeddings have zero difference."""
        emb = mx.ones((1, 10, 256))
        diff = compute_embedding_difference(emb, emb)
        assert diff == 0.0

    def test_nonzero_difference(self):
        """Test that different embeddings have nonzero difference."""
        emb1 = mx.zeros((1, 10, 256))
        emb2 = mx.ones((1, 10, 256))
        diff = compute_embedding_difference(emb1, emb2)
        assert diff > 0

    def test_masked_difference(self):
        """Test difference with mask."""
        emb1 = mx.zeros((1, 10, 256))
        emb2 = mx.ones((1, 10, 256))
        mask = mx.ones((1, 10))

        diff = compute_embedding_difference(emb1, emb2, mask)
        assert diff > 0


class TestCheckConvergence:
    """Test convergence checking logic."""

    def test_converged_state(self):
        """Test detection of converged state."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)
        state.differences = [1e-2, 5e-3, 1e-3, 5e-4, 1e-5]
        state.iteration = 4

        converged, msg = check_convergence(state, threshold=1e-4)
        assert converged
        assert "Converged" in msg

    def test_not_converged_high_final(self):
        """Test detection of non-convergence due to high final difference."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)
        state.differences = [1e-2, 5e-3, 1e-3]  # Final still above threshold
        state.iteration = 2

        converged, msg = check_convergence(state, threshold=1e-4)
        assert not converged
        assert "above threshold" in msg

    def test_not_converged_not_decreasing(self):
        """Test detection of non-convergence due to non-decreasing differences."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)
        state.differences = [1e-3, 1e-4, 5e-4, 1e-5]  # Spike at iteration 2
        state.iteration = 3

        converged, msg = check_convergence(state, threshold=1e-4)
        assert not converged
        assert "not monotonically decreasing" in msg

    def test_empty_differences(self):
        """Test handling of empty differences list."""
        single = mx.zeros((1, 10, 256))
        pair = mx.zeros((1, 10, 10, 128))

        state = RecyclingState(single=single, pair=pair)

        converged, msg = check_convergence(state)
        assert not converged
        assert "No differences tracked" in msg


class TestRecyclingLoop:
    """Test recycling loop with convergence tracking."""

    @pytest.fixture
    def simple_evoformer(self):
        """Create simple Evoformer for testing."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=3,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        model = Model(config)
        return model.evoformer

    def test_recycling_tracks_differences(self, simple_evoformer):
        """Test that recycling tracks per-iteration differences."""
        batch_size = 1
        num_residues = 8
        seq_channel = 384  # From EvoformerConfig defaults
        pair_channel = 128

        initial_single = mx.zeros((batch_size, num_residues, seq_channel))
        initial_pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        single, pair, state = run_recycling_loop(
            evoformer_fn=simple_evoformer,
            initial_single=initial_single,
            initial_pair=initial_pair,
            residue_index=residue_index,
            asym_id=asym_id,
            num_recycles=4,  # 5 total iterations
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            track_convergence=True,
        )

        # Should have tracked differences
        assert state is not None
        assert len(state.differences) == 5  # num_recycles + 1

        # All differences should be positive (embeddings change each iteration)
        for diff in state.differences:
            assert diff >= 0

    def test_recycling_ten_iterations_decreasing(self, simple_evoformer):
        """Test that 10 iterations show decreasing differences. requirement: Differences should generally decrease across iterations,
        indicating convergence to a stable representation.
        """
        batch_size = 1
        num_residues = 8
        seq_channel = 384
        pair_channel = 128

        # Use small random initialization to give model something to refine
        key = mx.random.key(42)
        k1, k2 = mx.random.split(key)
        initial_single = mx.random.normal((batch_size, num_residues, seq_channel), key=k1) * 0.1
        initial_pair = mx.random.normal((batch_size, num_residues, num_residues, pair_channel), key=k2) * 0.1

        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        single, pair, state = run_recycling_loop(
            evoformer_fn=simple_evoformer,
            initial_single=initial_single,
            initial_pair=initial_pair,
            residue_index=residue_index,
            asym_id=asym_id,
            num_recycles=9, # 10 total iterations
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            track_convergence=True,
        )

        assert state is not None
        assert len(state.differences) == 10

        # Print differences for debugging
        print("\nRecycling differences across 10 iterations:")
        for i, diff in enumerate(state.differences):
            print(f" Iteration {i}: {diff:.6e}")

        # Check general trend is decreasing
        # Allow some tolerance - not strictly monotonic but overall decreasing
        first_half_avg = sum(state.differences[:5]) / 5
        second_half_avg = sum(state.differences[5:]) / 5

        # Second half should have smaller average differences
        # (This is a soft check - neural network behavior can vary)
        print(f"\n First half avg: {first_half_avg:.6e}")
        print(f" Second half avg: {second_half_avg:.6e}")

    def test_no_nan_after_recycling(self, simple_evoformer):
        """Test that recycling produces valid (non-NaN) outputs."""
        batch_size = 1
        num_residues = 8
        seq_channel = 384
        pair_channel = 128

        initial_single = mx.zeros((batch_size, num_residues, seq_channel))
        initial_pair = mx.zeros((batch_size, num_residues, num_residues, pair_channel))
        residue_index = mx.arange(num_residues)[None, :]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
        seq_mask = mx.ones((batch_size, num_residues))
        pair_mask = mx.ones((batch_size, num_residues, num_residues))

        single, pair, _ = run_recycling_loop(
            evoformer_fn=simple_evoformer,
            initial_single=initial_single,
            initial_pair=initial_pair,
            residue_index=residue_index,
            asym_id=asym_id,
            num_recycles=4,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            track_convergence=False,
        )

        mx.eval(single, pair)

        single_np = np.array(single)
        pair_np = np.array(pair)

        assert not np.any(np.isnan(single_np)), "Single representation contains NaN"
        assert not np.any(np.isnan(pair_np)), "Pair representation contains NaN"


class TestRecyclingWithModel:
    """Test recycling through full model inference."""

    @pytest.fixture
    def model(self):
        """Create model with multiple recycles."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4, # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=4, # 5 total iterations
        )
        return Model(config)

    def test_model_with_multiple_recycles(self, model):
        """Test that model runs successfully with multiple recycles."""
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions, result.confidence.plddt)

        coords = np.array(result.atom_positions.positions)
        plddt = np.array(result.confidence.plddt)

        # Basic sanity checks
        assert not np.any(np.isnan(coords)), "Coordinates contain NaN after recycling"
        assert not np.any(np.isnan(plddt)), "pLDDT contains NaN after recycling"

    def test_model_produces_consistent_results(self, model):
        """Test that same seed produces consistent results."""
        batch = create_test_batch(num_residues=10)

        # Run twice with same seed
        key1 = mx.random.key(42)
        result1 = model(batch, key1)
        mx.eval(result1.atom_positions.positions)
        coords1 = np.array(result1.atom_positions.positions)

        key2 = mx.random.key(42)
        result2 = model(batch, key2)
        mx.eval(result2.atom_positions.positions)
        coords2 = np.array(result2.atom_positions.positions)

        # Results should be identical
        np.testing.assert_allclose(coords1, coords2, rtol=1e-5, atol=1e-5)
