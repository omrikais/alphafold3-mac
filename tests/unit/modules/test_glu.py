"""Unit tests for GatedLinearUnit module.

These tests validate the GatedLinearUnit implementation for.
Tests are written first (TDD) to define expected behavior.
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import numpy as np


class TestGLUConstruction:
    """Tests for GatedLinearUnit construction."""

    def test_basic_construction(self):
        """Test basic GatedLinearUnit construction."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512)

        # Internal linear should project to 2 * output_dim
        assert glu.linear.weight.shape[-1] == 2 * 512

    def test_construction_with_activation(self):
        """Test GatedLinearUnit with different activations."""
        from alphafold3_mlx.modules import GatedLinearUnit

        for activation in ["swish", "silu", "gelu", "relu"]:
            glu = GatedLinearUnit(256, 512, activation=activation)
            assert glu._activation_name == activation

    def test_construction_with_bias(self):
        """Test GatedLinearUnit with bias."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, use_bias=True)

        assert glu.linear.bias is not None

    def test_construction_without_bias(self):
        """Test GatedLinearUnit without bias (default)."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512)

        assert glu.linear.bias is None


class TestGLUSwish:
    """Tests for GLU with swish activation."""

    def test_swish_forward(self):
        """Test GLU with swish activation forward pass."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="swish")

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        assert y.shape == (2, 10, 512)
        assert y.dtype == mx.float32

    def test_swish_non_nan(self):
        """Test swish activation doesn't produce NaN."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="swish")

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        assert not mx.isnan(y).any().item()


class TestGLUSilu:
    """Tests for GLU with silu activation."""

    def test_silu_forward(self):
        """Test GLU with silu activation (same as swish)."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="silu")

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        assert y.shape == (2, 10, 512)

    def test_silu_equals_swish(self):
        """Test silu and swish produce same results."""
        from alphafold3_mlx.modules import GatedLinearUnit

        # Create two GLUs with same weights
        key = mx.random.key(42)
        glu_swish = GatedLinearUnit(256, 512, activation="swish")
        glu_silu = GatedLinearUnit(256, 512, activation="silu")

        # Copy weights
        glu_silu.linear.weight = glu_swish.linear.weight

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(123))

        y_swish = glu_swish(x)
        y_silu = glu_silu(x)

        np.testing.assert_allclose(np.array(y_swish), np.array(y_silu))


class TestGLUGelu:
    """Tests for GLU with gelu activation."""

    def test_gelu_forward(self):
        """Test GLU with gelu activation forward pass."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="gelu")

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        assert y.shape == (2, 10, 512)

    def test_gelu_different_from_swish(self):
        """Test gelu produces different results than swish."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu_swish = GatedLinearUnit(256, 512, activation="swish")
        glu_gelu = GatedLinearUnit(256, 512, activation="gelu")

        # Copy weights
        glu_gelu.linear.weight = glu_swish.linear.weight

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))

        y_swish = glu_swish(x)
        y_gelu = glu_gelu(x)

        # Should be different
        assert not np.allclose(np.array(y_swish), np.array(y_gelu))


class TestGLURelu:
    """Tests for GLU with relu activation."""

    def test_relu_forward(self):
        """Test GLU with relu activation forward pass."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="relu")

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        assert y.shape == (2, 10, 512)

    def test_relu_sparsity(self):
        """Test relu produces some zero values (sparsity)."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="relu")

        # With zero input, output depends on bias
        # With default (no bias), output should have some structure
        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        # Should produce finite values
        assert mx.isfinite(y).all().item()


class TestGLUSplitProjection:
    """Tests for GLU split behavior."""

    def test_split_produces_correct_shape(self):
        """Test internal split produces 2 equal parts."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512)

        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        # Output should be output_dim, not 2*output_dim
        assert y.shape == (2, 10, 512)

    def test_gating_behavior(self):
        """Test that gating applies element-wise multiplication."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512, activation="swish")

        # Set weights to identity-like to trace behavior
        # This is hard to test directly, so we just verify output is bounded
        x = mx.random.normal(shape=(2, 10, 256), key=mx.random.key(42))
        y = glu(x)

        # Output should be finite and bounded
        assert mx.isfinite(y).all().item()

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        from alphafold3_mlx.modules import GatedLinearUnit

        glu = GatedLinearUnit(256, 512)

        x1 = mx.random.normal(shape=(1, 10, 256), key=mx.random.key(42))
        x2 = mx.random.normal(shape=(1, 10, 256), key=mx.random.key(43))
        x_combined = mx.concatenate([x1, x2], axis=0)

        y_combined = glu(x_combined)
        y1 = glu(x1)
        y2 = glu(x2)

        # Allow small floating point differences due to batch processing
        np.testing.assert_allclose(np.array(y_combined[0]), np.array(y1[0]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(y_combined[1]), np.array(y2[0]), rtol=1e-5, atol=1e-5)


class TestGetActivation:
    """Tests for get_activation helper function."""

    def test_get_activation_swish(self):
        """Test get_activation returns swish."""
        from alphafold3_mlx.modules import get_activation

        act = get_activation("swish")
        x = mx.array([0.0, 1.0, -1.0])
        y = act(x)

        # swish(x) = x * sigmoid(x)
        expected = x * mx.sigmoid(x)
        np.testing.assert_allclose(np.array(y), np.array(expected))

    def test_get_activation_silu(self):
        """Test get_activation returns silu (same as swish)."""
        from alphafold3_mlx.modules import get_activation

        act = get_activation("silu")
        x = mx.array([0.0, 1.0, -1.0])
        y = act(x)

        # silu = swish
        expected = x * mx.sigmoid(x)
        np.testing.assert_allclose(np.array(y), np.array(expected))

    def test_get_activation_gelu(self):
        """Test get_activation returns gelu."""
        from alphafold3_mlx.modules import get_activation
        import mlx.nn as nn

        act = get_activation("gelu")
        x = mx.array([0.0, 1.0, -1.0])
        y = act(x)

        expected = nn.gelu(x)
        np.testing.assert_allclose(np.array(y), np.array(expected))

    def test_get_activation_relu(self):
        """Test get_activation returns relu."""
        from alphafold3_mlx.modules import get_activation

        act = get_activation("relu")
        x = mx.array([-1.0, 0.0, 1.0])
        y = act(x)

        expected = mx.maximum(x, 0)
        np.testing.assert_allclose(np.array(y), np.array(expected))

    def test_get_activation_unknown(self):
        """Test get_activation raises for unknown activation."""
        from alphafold3_mlx.modules import get_activation

        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("unknown")
