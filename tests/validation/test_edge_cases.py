"""Edge case tests for MLX attention spike."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from alphafold3_mlx.core.golden import GoldenOutputs
from alphafold3_mlx.spike.attention import MLXAttentionSpike
from alphafold3_mlx.validation.validator import AttentionValidator


GOLDEN_DIR = Path(__file__).parent.parent / "golden"


def load_golden(case_id: str) -> GoldenOutputs:
    """Load golden outputs for a test case."""
    npz_path = GOLDEN_DIR / f"{case_id}.npz"
    meta_path = GOLDEN_DIR / f"{case_id}.meta.json"
    return GoldenOutputs.load(npz_path, meta_path)


class TestAllMasked:
    """Test fully masked rows."""

    def test_all_masked_no_nan(self):
        """All masked positions should not produce NaN."""
        golden = load_golden("all_masked")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        mask = mx.array(golden.inputs.boolean_mask)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, boolean_mask=mask)
        mx.eval(result.output)

        output_np = np.array(result.output)

        # Output should not contain NaN
        assert not np.any(np.isnan(output_np)), "Output contains NaN"

    def test_all_masked_returns_zeros(self):
        """All masked rows MUST return zero vectors (spec requirement)."""
        golden = load_golden("all_masked")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        mask = mx.array(golden.inputs.boolean_mask)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, boolean_mask=mask)
        mx.eval(result.output)

        output_np = np.array(result.output)

        # The spec explicitly requires zeros for fully masked rows
        assert np.allclose(output_np, 0.0), (
            f"Violation: fully masked output must be zeros, "
            f"got min={output_np.min()}, max={output_np.max()}"
        )

    def test_all_masked_matches_golden(self):
        """All masked output matches JAX reference behavior."""
        golden = load_golden("all_masked")

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
            f"all_masked: max_diff={validation.max_abs_diff:.2e}"
        )


class TestSeq1:
    """Test seq_q=seq_k=1."""

    def test_seq_1_produces_valid_output(self):
        """seq_q=seq_k=1 produces valid output."""
        golden = load_golden("seq_1")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        output_np = np.array(result.output)

        assert output_np.shape == (1, 4, 1, 64)
        assert not np.any(np.isnan(output_np))

    def test_seq_1_matches_golden(self):
        """seq_1 output matches JAX reference."""
        golden = load_golden("seq_1")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed


class TestCrossAttention:
    """Test cross-attention seq_q != seq_k."""

    def test_cross_attention_shape(self):
        """Cross-attention produces correct output shape."""
        golden = load_golden("cross_attention")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        # seq_q=256, seq_k=128 → output shape [1, 4, 256, 64]
        assert result.output.shape == (1, 4, 256, 64)

    def test_cross_attention_matches_golden(self):
        """Cross-attention output matches JAX reference."""
        golden = load_golden("cross_attention")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed


class TestLargeBiasValues:
    """Test large bias values ±1e9."""

    def test_large_positive_bias(self):
        """Large positive bias (+1e9) produces valid output without NaN."""
        golden = load_golden("large_bias_positive")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        bias = mx.array(golden.inputs.additive_bias)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, additive_bias=bias)
        mx.eval(result.output)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_large_negative_bias(self):
        """Large negative bias (-1e9) produces valid output without NaN."""
        golden = load_golden("large_bias_negative")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)
        bias = mx.array(golden.inputs.additive_bias)

        spike = MLXAttentionSpike()
        result = spike(q, k, v, additive_bias=bias)
        mx.eval(result.output)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN"
        assert not np.any(np.isinf(output_np)), "Output contains Inf"

    def test_large_bias_matches_golden(self):
        """Large bias outputs match JAX reference."""
        for case_id in ["large_bias_positive", "large_bias_negative"]:
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
                f"{case_id}: max_diff={validation.max_abs_diff:.2e}"
            )


class TestNonPower2HeadDim:
    """Test non-power-of-2 head_dim."""

    def test_head_dim_48(self):
        """head_dim=48 produces correct outputs."""
        golden = load_golden("non_power2_head_dim")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        # head_dim=48
        assert result.output.shape == (1, 4, 256, 48)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np))

    def test_head_dim_48_matches_golden(self):
        """head_dim=48 output matches JAX reference."""
        golden = load_golden("non_power2_head_dim")

        q = mx.array(golden.inputs.q)
        k = mx.array(golden.inputs.k)
        v = mx.array(golden.inputs.v)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        validator = AttentionValidator()
        validation = validator.compare(
            result,
            {"output": golden.output},
            rtol=golden.rtol,
            atol=golden.atol,
        )

        assert validation.passed
