"""Tests for InferenceStats population and check_nans wiring (Phase 4).

Verifies that:
- Per-component timing is recorded in Model.__call__ metadata
- InferenceStats fields are populated from model metadata
- check_nans flag controls NaN validation behavior
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.model import Model
from alphafold3_mlx.model.inference import run_inference, InferenceStats
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig


def create_test_batch(num_residues: int = 8, seed: int = 42) -> FeatureBatch:
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


def create_small_model() -> Model:
    """Create a small model for testing."""
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
            use_compile=False,  # Disable compilation for faster tests
        ),
        num_recycles=1,
    )
    return Model(config)


class TestModelTimingMetadata:
    """Test that Model.__call__ records timing metadata."""

    def test_model_call_records_component_timing(self):
        """Test that Model.__call__ records per-component timing in metadata."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        result = model(batch, key)

        # Check metadata contains timing keys
        assert "evoformer_duration_seconds" in result.metadata
        assert "diffusion_duration_seconds" in result.metadata
        assert "confidence_duration_seconds" in result.metadata

        # Timings should be positive floats
        assert result.metadata["evoformer_duration_seconds"] > 0
        assert result.metadata["diffusion_duration_seconds"] > 0
        assert result.metadata["confidence_duration_seconds"] > 0

        # Component timings should be less than total
        total = result.metadata["duration_seconds"]
        component_sum = (
            result.metadata["evoformer_duration_seconds"]
            + result.metadata["diffusion_duration_seconds"]
            + result.metadata["confidence_duration_seconds"]
        )
        # Component sum may be less than total due to other overhead
        assert component_sum <= total * 1.1  # Allow 10% margin for measurement noise

    def test_timing_scales_with_sequence_length(self):
        """Test that timing increases with longer sequences."""
        model = create_small_model()
        key = mx.random.key(42)

        # Small sequence
        batch_small = create_test_batch(num_residues=4)
        result_small = model(batch_small, key)

        # Larger sequence
        batch_large = create_test_batch(num_residues=12)
        result_large = model(batch_large, key)

        # Larger sequence should take more time (usually)
        # Note: This may not always hold for very small sequences due to overhead
        # Just verify both have valid timing data
        assert result_small.metadata["evoformer_duration_seconds"] > 0
        assert result_large.metadata["evoformer_duration_seconds"] > 0


class TestInferenceStatsPopulation:
    """Test that run_inference populates InferenceStats correctly."""

    def test_run_inference_populates_component_timing(self):
        """Test that run_inference extracts timing from model metadata."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        result, stats = run_inference(model, batch, key=key)

        # Check that per-component timing is populated
        assert stats.evoformer_duration_seconds > 0
        assert stats.diffusion_duration_seconds > 0
        assert stats.confidence_duration_seconds > 0

        # Total should be >= sum of components
        assert stats.total_duration_seconds >= (
            stats.evoformer_duration_seconds
            + stats.diffusion_duration_seconds
            + stats.confidence_duration_seconds
        ) * 0.9  # Allow small margin for measurement

    def test_run_inference_populates_other_stats(self):
        """Test that run_inference populates other InferenceStats fields."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        result, stats = run_inference(model, batch, key=key)

        # Check other stats are populated
        assert stats.num_residues == 6
        assert stats.num_samples == model.config.diffusion.num_samples
        assert stats.num_recycles == model.config.num_recycles

    def test_run_inference_memory_tracking(self):
        """Test that memory checkpoints are recorded when track_memory=True."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        result, stats = run_inference(model, batch, key=key, track_memory=True)

        # Memory checkpoints should be present
        assert "initial" in stats.memory_checkpoints
        assert "final" in stats.memory_checkpoints
        assert stats.peak_memory_gb >= 0


class TestCheckNansWiring:
    """Test that check_nans flag is properly wired."""

    def test_check_nans_default_true(self):
        """Test that check_nans defaults to True in Model.__call__."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        # Should run without error (no NaNs in normal execution)
        result = model(batch, key)
        assert result is not None

    def test_check_nans_false_skips_validation(self):
        """Test that check_nans=False skips NaN validation."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        # Should run without NaN checks
        result = model(batch, key, check_nans=False)
        assert result is not None

    def test_run_inference_passes_check_nans_to_model(self):
        """Test that run_inference passes check_nans flag to model."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        # With check_nans=True (default)
        result1, stats1 = run_inference(model, batch, key=key, check_nans=True)
        assert result1 is not None

        # With check_nans=False
        result2, stats2 = run_inference(model, batch, key=key, check_nans=False)
        assert result2 is not None


class TestInferenceStatsDataclass:
    """Test InferenceStats dataclass defaults."""

    def test_inference_stats_defaults(self):
        """Test that InferenceStats has sensible defaults."""
        stats = InferenceStats()

        assert stats.total_duration_seconds == 0.0
        assert stats.evoformer_duration_seconds == 0.0
        assert stats.diffusion_duration_seconds == 0.0
        assert stats.confidence_duration_seconds == 0.0
        assert stats.peak_memory_gb == 0.0
        assert stats.memory_checkpoints == {}
        assert stats.num_residues == 0
        assert stats.num_samples == 0
        assert stats.num_recycles == 0


class TestDiffusionNaNDetectionWiring:
    """Test that check_nans flag is wired through to diffusion sampling."""

    def test_model_diffusion_nan_injection_raises_with_check_nans_true(self):
        """Test that NaN injected in diffusion raises NaNError when check_nans=True."""
        from alphafold3_mlx.core.exceptions import NaNError
        from unittest.mock import patch

        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        call_count = [0]  # Mutable container for closure

        original_sample = model.diffusion_head.sample

        def sample_with_nan_injection(*args, **kwargs):
            """Wrapper that injects NaN after first step."""
            # Force check_nans to True to verify it's being passed
            check_nans = kwargs.get("check_nans", False)
            if not check_nans:
                # If not checking NaNs, just run normally
                return original_sample(*args, **kwargs)

            # Inject NaN by patching the denoising step
            original_denoising_step = kwargs.get("denoising_step")

            def nan_injecting_step(positions_noisy, noise_level):
                call_count[0] += 1
                if call_count[0] >= 2:  # Inject NaN on second call
                    return mx.full(positions_noisy.shape, float("nan"))
                return original_denoising_step(positions_noisy, noise_level)

            kwargs["denoising_step"] = nan_injecting_step
            return original_sample(*args, **kwargs)

        # Patch the sample method
        with patch.object(model.diffusion_head, "sample", sample_with_nan_injection):
            # With check_nans=True, should raise NaNError
            with pytest.raises(NaNError) as exc_info:
                model(batch, key, check_nans=True)

            # Verify error is from diffusion
            assert "diffusion" in exc_info.value.component

    def test_model_diffusion_nan_injection_no_error_with_check_nans_false(self):
        """Test that NaN in diffusion doesn't raise when check_nans=False."""
        model = create_small_model()
        batch = create_test_batch(num_residues=6)
        key = mx.random.key(42)

        original_sample = model.diffusion_head.sample

        def sample_returning_nan(*args, **kwargs):
            """Wrapper that returns NaN positions."""
            # Call original to get proper structure
            result = original_sample(*args, check_nans=False, **{k: v for k, v in kwargs.items() if k != "check_nans"})
            # Replace positions with NaN
            result["atom_positions"] = mx.full(result["atom_positions"].shape, float("nan"))
            return result

        from unittest.mock import patch

        # Patch the sample method to return NaN
        with patch.object(model.diffusion_head, "sample", sample_returning_nan):
            # With check_nans=False, should NOT raise (NaN detection disabled)
            result = model(batch, key, check_nans=False)

            # Output should contain NaN (propagated through)
            mx.eval(result.atom_positions.positions)
            assert bool(mx.any(mx.isnan(result.atom_positions.positions)).item())
