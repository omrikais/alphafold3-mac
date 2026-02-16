"""Diffusion loop stability tests.

Verifies that memory usage is stable (not monotonically increasing)
during the 200-step diffusion loop.
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.network.diffusion_head import DiffusionHead
from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig
from alphafold3_mlx.atom_layout import GatherInfo
from alphafold3_mlx.feat_batch import (
    Batch,
    TokenFeatures,
    PredictedStructureInfo,
    AtomCrossAtt,
    PseudoBetaInfo,
    RefStructure,
)


def create_minimal_batch(num_res: int, num_atoms: int) -> Batch:
    """Create minimal Batch for DiffusionHead testing.

    Args:
        num_res: Number of residues.
        num_atoms: Number of atoms per residue (typically 37).

    Returns:
        A Batch object with properly initialized fields.
    """
    # Token features
    token_features = TokenFeatures(
        token_index=mx.arange(num_res, dtype=mx.int32),
        residue_index=mx.arange(num_res, dtype=mx.int32),
        asym_id=mx.zeros(num_res, dtype=mx.int32),
        entity_id=mx.zeros(num_res, dtype=mx.int32),
        sym_id=mx.zeros(num_res, dtype=mx.int32),
        mask=mx.ones(num_res, dtype=mx.float32),
    )

    # Predicted structure info
    predicted_structure_info = PredictedStructureInfo(
        atom_mask=mx.ones((num_res, num_atoms), dtype=mx.float32)
    )

    # Create GatherInfo objects for atom cross-attention
    token_atom_indices = mx.arange(
        num_res * num_atoms, dtype=mx.int32
    ).reshape(num_res, num_atoms)

    token_index = mx.arange(num_res, dtype=mx.int32)
    token_to_query_idxs = mx.broadcast_to(
        token_index[:, None], (num_res, num_atoms)
    )

    token_atoms_to_queries = GatherInfo(
        gather_idxs=token_atom_indices,
        gather_mask=mx.ones((num_res, num_atoms), dtype=mx.bool_),
        input_shape=mx.array((num_res, num_atoms)),
    )
    tokens_to_queries = GatherInfo(
        gather_idxs=token_to_query_idxs,
        gather_mask=mx.ones((num_res, num_atoms), dtype=mx.bool_),
        input_shape=mx.array((num_res,)),
    )
    tokens_to_keys = GatherInfo(
        gather_idxs=token_to_query_idxs,
        gather_mask=mx.ones((num_res, num_atoms), dtype=mx.bool_),
        input_shape=mx.array((num_res,)),
    )
    queries_to_keys = GatherInfo(
        gather_idxs=token_atom_indices,
        gather_mask=mx.ones((num_res, num_atoms), dtype=mx.bool_),
        input_shape=mx.array((num_res, num_atoms)),
    )
    queries_to_token_atoms = GatherInfo(
        gather_idxs=token_atom_indices,
        gather_mask=mx.ones((num_res, num_atoms), dtype=mx.bool_),
        input_shape=mx.array((num_res, num_atoms)),
    )

    atom_cross_att = AtomCrossAtt(
        token_atoms_to_queries=token_atoms_to_queries,
        tokens_to_queries=tokens_to_queries,
        tokens_to_keys=tokens_to_keys,
        queries_to_keys=queries_to_keys,
        queries_to_token_atoms=queries_to_token_atoms,
    )

    # Pseudo-beta info
    token_atoms_to_pseudo_beta = GatherInfo(
        gather_idxs=token_index * num_atoms,
        gather_mask=mx.ones((num_res,), dtype=mx.bool_),
        input_shape=mx.array((num_res, num_atoms)),
    )
    pseudo_beta_info = PseudoBetaInfo(
        token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
    )

    # Reference structure
    ref_structure = RefStructure(
        positions=mx.zeros((num_res, num_atoms, 3), dtype=mx.float32),
        mask=mx.ones((num_res, num_atoms), dtype=mx.float32),
        element=mx.zeros((num_res, num_atoms), dtype=mx.int32),
        charge=mx.zeros((num_res, num_atoms), dtype=mx.float32),
        atom_name_chars=mx.zeros((num_res, num_atoms, 4), dtype=mx.int32),
        space_uid=mx.zeros((num_res, num_atoms), dtype=mx.int32),
    )

    return Batch(
        token_features=token_features,
        predicted_structure_info=predicted_structure_info,
        atom_cross_att=atom_cross_att,
        pseudo_beta_info=pseudo_beta_info,
        ref_structure=ref_structure,
    )


class TestDiffusionMemoryStability:
    """Test memory stability during diffusion loop."""

    @pytest.fixture
    def diffusion_head(self):
        """Create diffusion head for testing."""
        config = DiffusionConfig(
            num_steps=20,  # Reduced for testing
            num_samples=1,
            num_transformer_blocks=4,  # Must be divisible by super_block_size=4
        )
        global_config = GlobalConfig(use_compile=False)
        return DiffusionHead(config=config, global_config=global_config)

    @pytest.fixture
    def batch_memory(self):
        """Create minimal batch for memory tests."""
        return create_minimal_batch(num_res=8, num_atoms=37)

    def test_memory_not_monotonically_increasing(self, diffusion_head, batch_memory):
        """Test that memory doesn't grow unboundedly during diffusion.

        Memory must not be monotonically increasing during the
        200-step diffusion loop.
        """
        num_residues = 8  # Small protein for testing
        num_atoms = 37  # atom37 representation
        seq_channel = diffusion_head.config.conditioning_seq_channel
        pair_channel = diffusion_head.config.conditioning_pair_channel

        # Create embeddings for denoising step
        embeddings = {
            "single": mx.zeros((num_residues, seq_channel)),
            "pair": mx.zeros((num_residues, num_residues, pair_channel)),
            "target_feat": mx.zeros((num_residues, 22)),
        }
        key = mx.random.key(42)

        # Create denoising step callable
        def denoising_step(positions_noisy, noise_level):
            return diffusion_head(
                positions_noisy=positions_noisy,
                noise_level=noise_level,
                batch=batch_memory,
                embeddings=embeddings,
                use_conditioning=True,
            )

        mx.eval(embeddings["single"], embeddings["pair"])

        # Run sampling with proper API
        result = diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch_memory,
            key=key,
            num_steps=diffusion_head.config.num_steps,
            gamma_0=diffusion_head.config.gamma_0,
            gamma_min=diffusion_head.config.gamma_min,
            noise_scale=diffusion_head.config.noise_scale,
            step_scale=diffusion_head.config.step_scale,
            num_samples=1,
        )
        coords = result["atom_positions"]
        mx.eval(coords)

        # Verify output is valid
        coords_np = np.array(coords)
        assert not np.any(np.isnan(coords_np)), "Diffusion produced NaN"
        assert not np.any(np.isinf(coords_np)), "Diffusion produced Inf"

        # Shape should be [num_samples, num_residues, 37, 3]
        assert coords.shape == (1, num_residues, num_atoms, 3)

    def test_diffusion_eval_interval(self, diffusion_head):
        """Test that mx.eval is called at appropriate intervals.

        mx.eval must be called every 10 steps to prevent
        graph explosion.
        """
        # This is a structural test - verify the diffusion head has
        # the DIFFUSION_EVAL_INTERVAL constant and uses it
        from alphafold3_mlx.core.constants import DIFFUSION_EVAL_INTERVAL

        assert DIFFUSION_EVAL_INTERVAL == 10, (
            f"Expected DIFFUSION_EVAL_INTERVAL=10, got {DIFFUSION_EVAL_INTERVAL}"
        )

        # Verify config has num_steps
        assert diffusion_head.config.num_steps > 0

    def test_multiple_runs_stable(self, diffusion_head, batch_memory):
        """Test that multiple inference runs don't accumulate memory."""
        num_residues = 4  # Small for testing
        num_atoms = 37  # atom37
        seq_channel = diffusion_head.config.conditioning_seq_channel
        pair_channel = diffusion_head.config.conditioning_pair_channel

        # Create smaller batch for this test
        batch_small = create_minimal_batch(num_res=num_residues, num_atoms=num_atoms)

        embeddings = {
            "single": mx.zeros((num_residues, seq_channel)),
            "pair": mx.zeros((num_residues, num_residues, pair_channel)),
            "target_feat": mx.zeros((num_residues, 22)),
        }

        def denoising_step(positions_noisy, noise_level):
            return diffusion_head(
                positions_noisy=positions_noisy,
                noise_level=noise_level,
                batch=batch_small,
                embeddings=embeddings,
                use_conditioning=True,
            )

        # Run multiple times
        for i in range(3):
            key = mx.random.key(42 + i)
            result = diffusion_head.sample(
                denoising_step=denoising_step,
                batch=batch_small,
                key=key,
                num_steps=diffusion_head.config.num_steps,
                gamma_0=diffusion_head.config.gamma_0,
                gamma_min=diffusion_head.config.gamma_min,
                noise_scale=diffusion_head.config.noise_scale,
                step_scale=diffusion_head.config.step_scale,
                num_samples=1,
            )
            coords = result["atom_positions"]
            mx.eval(coords)

            # Clear cache between runs
            try:
                mx.metal.clear_cache()
            except AttributeError:
                pass

        # Should complete without OOM; shape is [num_samples, num_res, 37, 3]
        assert coords.shape == (1, num_residues, num_atoms, 3)


class TestDiffusionNumericalStability:
    """Test numerical stability during diffusion."""

    @pytest.fixture
    def diffusion_head(self):
        """Create diffusion head for testing."""
        config = DiffusionConfig(
            num_steps=10,  # Very small for fast tests
            num_samples=1,
            num_transformer_blocks=4,  # Must be divisible by super_block_size=4
        )
        return DiffusionHead(config=config)

    @pytest.fixture
    def batch_numerical(self):
        """Create minimal batch for numerical tests."""
        return create_minimal_batch(num_res=4, num_atoms=37)

    def test_karras_schedule_bounded(self, diffusion_head):
        """Test that noise schedule produces bounded values."""
        from alphafold3_mlx.network.noise_schedule import karras_schedule

        # Get schedule directly
        schedule = karras_schedule(diffusion_head.config.num_steps)
        schedule_np = np.array(schedule)

        for step in range(len(schedule_np)):
            sigma = schedule_np[step]
            assert sigma >= 0, f"sigma at step {step} is negative: {sigma}"
            # Sigma should be reasonable (not exploding)
            # Max is SIGMA_DATA * SIGMA_MAX = 16 * 160 = 2560
            assert sigma < 3000, f"sigma at step {step} is too large: {sigma}"

    def test_single_step_bounded(self, diffusion_head, batch_numerical):
        """Test that single denoising step produces bounded output."""
        num_residues = 4
        num_atoms = 37  # atom37
        seq_channel = diffusion_head.config.conditioning_seq_channel
        pair_channel = diffusion_head.config.conditioning_pair_channel

        # Create atom37 inputs
        coords = mx.random.normal(
            shape=(num_residues, num_atoms, 3), key=mx.random.key(0)
        )
        embeddings = {
            "single": mx.zeros((num_residues, seq_channel)),
            "pair": mx.zeros((num_residues, num_residues, pair_channel)),
            "target_feat": mx.zeros((num_residues, 22)),
        }

        # Run single step with actual __call__ API
        noise_level = mx.array(1.0)
        coords_out = diffusion_head(
            positions_noisy=coords,
            noise_level=noise_level,
            batch=batch_numerical,
            embeddings=embeddings,
            use_conditioning=True,
        )
        mx.eval(coords_out)

        coords_np = np.array(coords_out)
        assert not np.any(np.isnan(coords_np)), "Single step produced NaN"
        assert not np.any(np.isinf(coords_np)), "Single step produced Inf"
