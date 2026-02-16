"""Cross-validation tests comparing MLX attention spike against JAX golden outputs."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alphafold3_mlx.core.golden import GoldenOutputs
from alphafold3_mlx.spike.attention import MLXAttentionSpike
from alphafold3_mlx.validation.validator import AttentionValidator


# Path to golden outputs
GOLDEN_DIR = Path(__file__).parent.parent / "golden"


def load_golden(case_id: str) -> GoldenOutputs:
    """Load golden outputs for a test case."""
    npz_path = GOLDEN_DIR / f"{case_id}.npz"
    meta_path = GOLDEN_DIR / f"{case_id}.meta.json"
    return GoldenOutputs.load(npz_path, meta_path)


class TestNoMaskNoBias:
    """Test MLX vs JAX with no mask, no bias."""

    @pytest.mark.parametrize("seq", [256, 512, 1024])
    def test_no_mask_no_bias(self, seq: int):
        """MLX matches JAX golden for basic attention."""
        case_id = f"no_mask_no_bias_seq{seq}"
        golden = load_golden(case_id)

        # Convert inputs to MLX arrays
        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        # Run MLX attention
        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        # Compare with golden
        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
            config=golden.config,
        )

        assert validation.passed, (
            f"seq={seq}: max_diff={validation.max_abs_diff:.2e}, "
            f"tolerance=(rtol={golden.rtol}, atol={golden.atol})"
        )


class TestMaskOnly:
    """Test MLX vs JAX with boolean mask only."""

    @pytest.mark.parametrize("seq", [256, 512, 1024])
    def test_mask_only(self, seq: int):
        """MLX matches JAX golden with boolean mask."""
        case_id = f"mask_only_seq{seq}"
        golden = load_golden(case_id)

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        mask = mx.array(golden.inputs.boolean_mask)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, boolean_mask=mask)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed, (
            f"seq={seq}: max_diff={validation.max_abs_diff:.2e}"
        )


class TestBiasOnly:
    """Test MLX vs JAX with additive bias only."""

    @pytest.mark.parametrize("seq", [256, 512, 1024])
    def test_bias_only(self, seq: int):
        """MLX matches JAX golden with additive bias."""
        case_id = f"bias_only_seq{seq}"
        golden = load_golden(case_id)

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        bias = mx.array(golden.inputs.additive_bias)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, additive_bias=bias)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed, (
            f"seq={seq}: max_diff={validation.max_abs_diff:.2e}"
        )


class TestMaskAndBias:
    """Test MLX vs JAX with both mask and bias - AF3 pattern."""

    @pytest.mark.parametrize("seq", [256, 512, 1024])
    def test_mask_and_bias(self, seq: int):
        """MLX matches JAX golden with mask and bias (AF3 pattern)."""
        case_id = f"mask_and_bias_seq{seq}"
        golden = load_golden(case_id)

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        mask = mx.array(golden.inputs.boolean_mask)
        bias = mx.array(golden.inputs.additive_bias)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed, (
            f"seq={seq}: max_diff={validation.max_abs_diff:.2e}"
        )


class TestIntermediateCapture:
    """Test that intermediate capture works correctly."""

    def test_intermediates_captured(self):
        """Verify intermediates are captured when requested."""
        golden = load_golden("no_mask_no_bias_seq256")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, capture_intermediates=True)
        mx.eval(result.output)

        assert result.intermediates is not None
        assert result.intermediates.logits_pre_mask is not None
        assert result.intermediates.logits_masked is not None
        assert result.intermediates.weights is not None

    def test_intermediates_match_golden(self):
        """Verify captured intermediates match JAX reference."""
        golden = load_golden("no_mask_no_bias_seq256")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, capture_intermediates=True)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            golden.intermediates | {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        # Check each intermediate
        for key, passed in validation.tensor_results.items():
            assert passed, f"Intermediate {key} mismatch"
