"""Unit tests for MLX attention spike implementation."""

import logging

import mlx.core as mx
import numpy as np
import pytest

from alphafold3_mlx.spike.attention import MLXAttentionSpike, mlx_scaled_dot_product_attention


class TestMLXAttentionBasic:
    """Basic functionality tests for MLX attention."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch, heads, seq, head_dim = 1, 4, 256, 64
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        v = mx.random.normal((batch, heads, seq, head_dim))

        result = mlx_scaled_dot_product_attention(q, k, v)
        mx.eval(result.output)

        assert result.output.shape == (batch, heads, seq, head_dim)

    def test_no_nan_in_output(self):
        """Test that output does not contain NaN."""
        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        v = mx.random.normal((batch, heads, seq, head_dim))

        result = mlx_scaled_dot_product_attention(q, k, v)
        mx.eval(result.output)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np))

    def test_deterministic_with_same_inputs(self):
        """Same inputs produce same outputs."""
        mx.random.seed(42)
        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        v = mx.random.normal((batch, heads, seq, head_dim))

        result1 = mlx_scaled_dot_product_attention(q, k, v)
        result2 = mlx_scaled_dot_product_attention(q, k, v)
        mx.eval(result1.output, result2.output)

        np.testing.assert_array_equal(
            np.array(result1.output),
            np.array(result2.output),
        )


class TestMLXAttentionWithMask:
    """Test MLX attention with boolean mask."""

    def test_mask_affects_output(self):
        """Boolean mask should affect the output."""
        batch, heads, seq, head_dim = 1, 4, 64, 32
        mx.random.seed(42)
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        # Use varying V values so mask actually affects output
        v = mx.arange(seq * head_dim).reshape(1, 1, seq, head_dim).astype(mx.float32)
        v = mx.broadcast_to(v, (batch, heads, seq, head_dim))

        # No mask
        result_no_mask = mlx_scaled_dot_product_attention(q, k, v)

        # Partial mask - mask first half
        mask = mx.array([[False] * 32 + [True] * 32])
        result_with_mask = mlx_scaled_dot_product_attention(q, k, v, boolean_mask=mask)

        mx.eval(result_no_mask.output, result_with_mask.output)

        # Outputs should differ since mask changes which V values contribute
        assert not np.allclose(
            np.array(result_no_mask.output),
            np.array(result_with_mask.output),
        )

    def test_mask_conversion(self):
        """Boolean mask is correctly converted to additive mask."""
        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = mx.ones((batch, heads, seq, head_dim))
        k = mx.ones((batch, heads, seq, head_dim))
        v = mx.arange(seq * head_dim).reshape(1, 1, seq, head_dim).astype(mx.float32)
        v = mx.broadcast_to(v, (batch, heads, seq, head_dim))

        # Mask first half
        mask = mx.array([[False] * 32 + [True] * 32])
        result = mlx_scaled_dot_product_attention(q, k, v, boolean_mask=mask, capture_intermediates=True)
        mx.eval(result.output)

        # With first 32 positions masked, attention should focus on last 32
        weights = np.array(result.intermediates.weights)
        # First 32 columns should have near-zero weight
        assert np.allclose(weights[:, :, :, :32], 0.0, atol=1e-6)


class TestMLXAttentionWithBias:
    """Test MLX attention with additive bias."""

    def test_bias_affects_output(self):
        """Additive bias should affect the output."""
        batch, heads, seq, head_dim = 1, 4, 64, 32
        mx.random.seed(42)
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        # Use varying V values so bias-induced weight changes affect output
        v = mx.arange(seq * head_dim).reshape(1, 1, seq, head_dim).astype(mx.float32)
        v = mx.broadcast_to(v, (batch, heads, seq, head_dim))

        # No bias
        result_no_bias = mlx_scaled_dot_product_attention(q, k, v)

        # With large bias to clearly shift attention
        bias = mx.random.normal((batch, heads, seq, seq)) * 10.0
        result_with_bias = mlx_scaled_dot_product_attention(q, k, v, additive_bias=bias)

        mx.eval(result_no_bias.output, result_with_bias.output)

        # Outputs should differ since bias changes attention weights
        assert not np.allclose(
            np.array(result_no_bias.output),
            np.array(result_with_bias.output),
        )


class TestMLXAttentionSpike:
    """Test MLXAttentionSpike class."""

    def test_class_produces_same_as_function(self):
        """Class __call__ produces same output as function."""
        batch, heads, seq, head_dim = 1, 4, 64, 32
        q = mx.random.normal((batch, heads, seq, head_dim))
        k = mx.random.normal((batch, heads, seq, head_dim))
        v = mx.random.normal((batch, heads, seq, head_dim))

        func_result = mlx_scaled_dot_product_attention(q, k, v)
        spike = MLXAttentionSpike()
        class_result = spike(q, k, v)

        mx.eval(func_result.output, class_result.output)

        np.testing.assert_array_equal(
            np.array(func_result.output),
            np.array(class_result.output),
        )

    def test_shape_validation_q(self):
        """Class validates Q tensor shape."""
        spike = MLXAttentionSpike()

        q = mx.ones((4, 64, 32))  # 3D instead of 4D
        k = mx.ones((1, 4, 64, 32))
        v = mx.ones((1, 4, 64, 32))

        with pytest.raises(ValueError, match="Q must be 4D"):
            spike(q, k, v)

    def test_shape_validation_kv_match(self):
        """Class validates K and V shapes match."""
        spike = MLXAttentionSpike()

        q = mx.ones((1, 4, 64, 32))
        k = mx.ones((1, 4, 64, 32))
        v = mx.ones((1, 4, 128, 32))  # Different seq length

        with pytest.raises(ValueError, match="K and V shapes must match"):
            spike(q, k, v)

    def test_mask_shape_validation(self):
        """Class validates mask shape."""
        spike = MLXAttentionSpike()

        q = mx.ones((1, 4, 64, 32))
        k = mx.ones((1, 4, 64, 32))
        v = mx.ones((1, 4, 64, 32))
        mask = mx.ones((1, 32), dtype=mx.bool_)  # Wrong seq length

        with pytest.raises(ValueError, match="boolean_mask shape mismatch"):
            spike(q, k, v, boolean_mask=mask)

    def test_bias_shape_validation(self):
        """Class validates bias shape."""
        spike = MLXAttentionSpike()

        q = mx.ones((1, 4, 64, 32))
        k = mx.ones((1, 4, 64, 32))
        v = mx.ones((1, 4, 64, 32))
        bias = mx.ones((1, 4, 64, 128))  # Wrong seq_k

        with pytest.raises(ValueError, match="additive_bias shape mismatch"):
            spike(q, k, v, additive_bias=bias)


class TestMixedDtypeWarning:
    """Test warning for mixed dtypes."""

    def test_mixed_dtype_logs_warning(self, caplog):
        """Mixed dtypes should log a warning."""
        batch, heads, seq, head_dim = 1, 4, 64, 32

        q = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)
        k = mx.ones((batch, heads, seq, head_dim), dtype=mx.float16)
        v = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)

        with caplog.at_level(logging.WARNING):
            result = mlx_scaled_dot_product_attention(q, k, v)
            mx.eval(result.output)

        assert "Mixed dtypes detected" in caplog.text

    def test_same_dtype_no_warning(self, caplog):
        """Same dtypes should not log a warning."""
        batch, heads, seq, head_dim = 1, 4, 64, 32

        q = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)
        k = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)
        v = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)

        with caplog.at_level(logging.WARNING):
            result = mlx_scaled_dot_product_attention(q, k, v)
            mx.eval(result.output)

        assert "Mixed dtypes" not in caplog.text
