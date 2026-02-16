"""Tests for JAX reference harness."""

import numpy as np
import pytest

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.reference.jax_attention import (
    JAXReferenceHarness,
    jax_scaled_dot_product_attention,
)


class TestJAXAttention:
    """Test pure JAX attention function."""

    def test_basic_attention_shape(self):
        """Test that basic attention produces correct output shape."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 256, 64
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))

        output, intermediates = jax_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq, head_dim)
        assert intermediates is not None
        assert "logits_pre_mask" in intermediates
        assert "logits_masked" in intermediates
        assert "weights" in intermediates

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        k = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        v = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))

        _, intermediates = jax_scaled_dot_product_attention(q, k, v)

        weights = np.array(intermediates["weights"])
        weight_sums = weights.sum(axis=-1)

        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5, atol=1e-5)


class TestJAXHarnessDeterminism:
    """Test that JAX harness produces deterministic outputs."""

    def test_same_seed_produces_identical_outputs(self):
        """Same seed produces identical outputs across 3 runs."""
        config = AttentionConfig(seq_q=256, seq_k=256, seed=42)

        outputs_list = []
        for _ in range(3):
            harness = JAXReferenceHarness(seed=42)
            q, k, v, _, _ = harness.generate_inputs(config)
            outputs = harness.run_attention(q, k, v)
            outputs_list.append(outputs)

        # All runs should be identical
        for i in range(1, 3):
            for key in ["output", "logits_pre_mask", "logits_masked", "weights"]:
                np.testing.assert_array_equal(
                    outputs_list[0][key],
                    outputs_list[i][key],
                    err_msg=f"Run {i} differs from run 0 for {key}",
                )

    def test_different_seeds_produce_different_outputs(self):
        """Different seeds produce different outputs."""
        config = AttentionConfig(seq_q=256, seq_k=256)

        harness1 = JAXReferenceHarness(seed=42)
        harness2 = JAXReferenceHarness(seed=123)

        q1, k1, v1, _, _ = harness1.generate_inputs(config)
        q2, k2, v2, _, _ = harness2.generate_inputs(config)

        # Inputs should differ
        assert not np.allclose(q1, q2)


class TestIntermediateCapture:
    """Test intermediate capture functionality."""

    def test_logits_pre_mask_captured(self):
        """Verify logits_pre_mask is captured correctly."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))

        _, intermediates = jax_scaled_dot_product_attention(q, k, v)

        logits = intermediates["logits_pre_mask"]
        assert logits.shape == (batch, heads, seq, seq)

        # With all-ones inputs, logits should be: head_dim * scale = head_dim / sqrt(head_dim)
        expected_logit = head_dim ** 0.5  # sqrt(32) ≈ 5.66
        np.testing.assert_allclose(logits[0, 0, 0, 0], expected_logit, rtol=1e-5)

    def test_logits_masked_differs_with_mask(self):
        """Verify logits_masked reflects mask application."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))

        # Create mask: first half attend, second half masked
        boolean_mask = jnp.array([[True] * 32 + [False] * 32])

        _, intermediates = jax_scaled_dot_product_attention(
            q, k, v, boolean_mask=boolean_mask
        )

        logits_pre = intermediates["logits_pre_mask"]
        logits_masked = intermediates["logits_masked"]

        # Pre-mask logits should be uniform
        assert np.allclose(logits_pre[0, 0, 0, 0], logits_pre[0, 0, 0, 63])

        # Post-mask logits should differ: masked positions have large negative value
        assert logits_masked[0, 0, 0, 0] > logits_masked[0, 0, 0, 63]

    def test_weights_captured(self):
        """Verify attention weights are captured."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        k = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        v = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))

        _, intermediates = jax_scaled_dot_product_attention(q, k, v)

        weights = intermediates["weights"]
        assert weights.shape == (batch, heads, seq, seq)
        assert np.all(weights >= 0), "Weights should be non-negative"
        assert np.all(weights <= 1), "Weights should be <= 1"


class TestFullyMaskedRows:
    """Test handling of fully masked rows."""

    def test_fully_masked_no_nan_in_output(self):
        """Fully masked rows should not produce NaN in output.

        Note: When all positions have equal (very large negative) logits,
        softmax produces uniform weights, not NaN. This is numerically
        correct behavior. NaN only occurs with -inf inputs.
        """
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))

        # All positions masked with boolean mask
        boolean_mask = jnp.zeros((batch, seq), dtype=bool)

        output, intermediates = jax_scaled_dot_product_attention(
            q, k, v, boolean_mask=boolean_mask
        )

        # Output should not contain NaN
        assert not np.any(np.isnan(output)), "Output contains NaN"

        # Weights should not be NaN
        weights = intermediates["weights"]
        assert not np.any(np.isnan(weights)), "Weights contain NaN"

    def test_fully_masked_returns_zeros(self):
        """All masked rows MUST return zero vectors (spec requirement)."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim)) * 3.14  # Non-trivial values

        # All positions masked with boolean mask
        boolean_mask = jnp.zeros((batch, seq), dtype=bool)

        output, _ = jax_scaled_dot_product_attention(
            q, k, v, boolean_mask=boolean_mask
        )

        # The spec explicitly requires zeros for fully masked rows
        np.testing.assert_allclose(
            output, 0.0,
            err_msg="Violation: fully masked output must be zeros"
        )

    def test_nan_weights_converted_to_zero(self):
        """Test that NaN weights (from -inf logits) are converted to zero."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 4, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))

        # Use -inf bias to force NaN weights
        inf_bias = jnp.full((batch, heads, seq, seq), float('-inf'))

        output, intermediates = jax_scaled_dot_product_attention(
            q, k, v, additive_bias=inf_bias
        )

        # Weights should be zeros (NaN converted)
        weights = intermediates["weights"]
        np.testing.assert_array_equal(weights, 0.0)

        # Output should also be zeros (0 * v = 0)
        np.testing.assert_array_equal(output, 0.0)


class TestEdgeCases:
    """Test edge cases."""

    def test_seq_1(self):
        """seq_q=seq_k=1 produces valid output."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 1, 64
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        k = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        v = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))

        output, _ = jax_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq, head_dim)
        assert not np.any(np.isnan(output))

    def test_cross_attention(self):
        """seq_q != seq_k produces correct shape."""
        import jax.numpy as jnp

        batch, heads, seq_q, seq_k, head_dim = 1, 4, 256, 128, 64
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal((batch, heads, seq_q, head_dim)))
        k = jnp.array(rng.standard_normal((batch, heads, seq_k, head_dim)))
        v = jnp.array(rng.standard_normal((batch, heads, seq_k, head_dim)))

        output, intermediates = jax_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq_q, head_dim)
        assert intermediates["logits_pre_mask"].shape == (batch, heads, seq_q, seq_k)

    def test_non_power2_head_dim(self):
        """head_dim=48 (not power of 2) works correctly."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 256, 48
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        k = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))
        v = jnp.array(rng.standard_normal((batch, heads, seq, head_dim)))

        output, _ = jax_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch, heads, seq, head_dim)
        assert not np.any(np.isnan(output))

    def test_large_positive_bias(self):
        """Large positive bias (+1e9) doesn't cause NaN."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))
        bias = jnp.full((batch, heads, seq, seq), 1e9)

        output, _ = jax_scaled_dot_product_attention(q, k, v, additive_bias=bias)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_large_negative_bias(self):
        """Large negative bias (-1e9) doesn't cause NaN."""
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = jnp.ones((batch, heads, seq, head_dim))
        k = jnp.ones((batch, heads, seq, head_dim))
        v = jnp.ones((batch, heads, seq, head_dim))
        bias = jnp.full((batch, heads, seq, seq), -1e9)

        output, _ = jax_scaled_dot_product_attention(q, k, v, additive_bias=bias)

        # With all large negative bias, softmax will distribute evenly → output is v
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
