"""Unit tests for Linear module.

These tests validate the Linear implementation for.
Tests are written first (TDD) to define expected behavior.
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import numpy as np


class TestLinearConstruction:
    """Tests for Linear construction and weight shapes."""

    def test_basic_construction(self):
        """Test basic Linear construction."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32)

        assert layer.weight.shape == (32, 64)
        assert layer.bias is None  # Default: no bias

    def test_construction_with_bias(self):
        """Test Linear construction with bias."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, use_bias=True)

        assert layer.weight.shape == (32, 64)
        assert layer.bias is not None
        assert layer.bias.shape == (64,)

    def test_construction_tuple_output(self):
        """Test Linear with tuple output dimensions."""
        from alphafold3_mlx.modules import Linear

        layer = Linear((8, 16), input_dims=32)

        # Weight shape should be [input_dims, *output_dims]
        assert layer.weight.shape == (32, 8, 16)

    def test_construction_multi_input_dims(self):
        """Test Linear with multiple input dimensions."""
        from alphafold3_mlx.modules import Linear

        # Two input dimensions to contract
        layer = Linear(64, input_dims=(4, 8), num_input_dims=2)

        # Weight shape should be [*input_dims, *output_dims]
        assert layer.weight.shape == (4, 8, 64)

    def test_construction_transpose_weights(self):
        """Test Linear with transposed weight layout."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, transpose_weights=True)

        # With transpose_weights=True, weights are [output, input]
        assert layer.weight.shape == (64, 32)

    def test_default_dtype(self):
        """Test Linear default dtype is float32."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32)

        assert layer.weight.dtype == mx.float32


class TestLinearForwardFloat32:
    """Tests for Linear forward pass with float32."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32)

        x = mx.random.normal(shape=(2, 10, 32), key=mx.random.key(42))
        y = layer(x)

        assert y.shape == (2, 10, 64)
        assert y.dtype == mx.float32

    def test_forward_with_bias(self):
        """Test forward pass with bias."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, use_bias=True, bias_init=1.0)

        x = mx.zeros((2, 10, 32))
        y = layer(x)

        # With zero input and bias_init=1.0, output should be 1.0
        np.testing.assert_allclose(np.array(y), 1.0)

    def test_forward_batch_dims(self):
        """Test forward preserves batch dimensions."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32)

        x = mx.random.normal(shape=(2, 3, 4, 32), key=mx.random.key(42))
        y = layer(x)

        assert y.shape == (2, 3, 4, 64)


class TestLinearPrecisionHighest:
    """Tests for Linear precision='highest' mode."""

    def test_precision_highest_float16(self):
        """Test precision='highest' with float16 input."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, precision="highest")

        x = mx.random.normal(shape=(2, 10, 32), key=mx.random.key(42)).astype(mx.float16)
        y = layer(x)

        # Output should be float16 (matches input dtype)
        assert y.dtype == mx.float16

        # But computation should have been in float32 (we can't verify internally,
        # but we can check output is reasonable)
        assert not mx.isnan(y).any().item()
        assert not mx.isinf(y).any().item()

    def test_precision_highest_bfloat16(self):
        """Test precision='highest' with bfloat16 input."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, precision="highest")

        x = mx.random.normal(shape=(2, 10, 32), key=mx.random.key(42)).astype(mx.bfloat16)
        y = layer(x)

        assert y.dtype == mx.bfloat16
        assert not mx.isnan(y).any().item()

    def test_precision_none_allows_promotion(self):
        """Test precision=None allows MLX to promote dtype (einsum behavior)."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, precision=None)

        x = mx.random.normal(shape=(2, 10, 32), key=mx.random.key(42)).astype(mx.float16)
        y = layer(x)

        # MLX einsum promotes mixed dtypes to float32 by default
        # This is expected behavior when precision=None
        # (precision='highest' explicitly handles dtype preservation)
        assert y.dtype in (mx.float16, mx.float32)


class TestLinearMultiInputDims:
    """Tests for Linear num_input_dims > 1."""

    def test_multi_input_dims_2(self):
        """Test Linear with 2 input dimensions."""
        from alphafold3_mlx.modules import Linear

        # Contract last 2 dimensions (4, 8) -> 64
        layer = Linear(64, input_dims=(4, 8), num_input_dims=2)

        x = mx.random.normal(shape=(2, 10, 4, 8), key=mx.random.key(42))
        y = layer(x)

        assert y.shape == (2, 10, 64)

    def test_multi_input_dims_multi_output(self):
        """Test Linear with multiple input and output dimensions."""
        from alphafold3_mlx.modules import Linear

        # Contract (4, 8) -> (16, 2)
        layer = Linear((16, 2), input_dims=(4, 8), num_input_dims=2)

        x = mx.random.normal(shape=(2, 10, 4, 8), key=mx.random.key(42))
        y = layer(x)

        assert y.shape == (2, 10, 16, 2)


class TestLinearInitializers:
    """Tests for Linear weight initializers."""

    def test_initializer_linear(self):
        """Test 'linear' initializer (fan-in scaling)."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, initializer="linear")

        # Weights should be scaled by 1/sqrt(fan_in)
        # Fan-in is 32, so std should be ~1/sqrt(32) ≈ 0.177
        std = float(mx.std(layer.weight).item())
        expected_std = 1.0 / (32 ** 0.5)

        # Allow 50% tolerance for random initialization
        assert 0.5 * expected_std < std < 2.0 * expected_std

    def test_initializer_relu(self):
        """Test 'relu' initializer (He initialization)."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, initializer="relu")

        # He initialization: std should be ~sqrt(2/fan_in) ≈ 0.25
        std = float(mx.std(layer.weight).item())
        expected_std = (2.0 / 32) ** 0.5

        assert 0.5 * expected_std < std < 2.0 * expected_std

    def test_initializer_zeros(self):
        """Test 'zeros' initializer."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32, initializer="zeros")

        np.testing.assert_allclose(np.array(layer.weight), 0.0)

    def test_invalid_initializer(self):
        """Test that invalid initializer raises error."""
        from alphafold3_mlx.modules import Linear

        with pytest.raises(ValueError, match="Unknown initializer"):
            Linear(64, input_dims=32, initializer="invalid")

    def test_num_input_dims_mismatch(self):
        """Test that mismatched num_input_dims raises error."""
        from alphafold3_mlx.modules import Linear

        # input_dims has 2 dims but num_input_dims says 1
        with pytest.raises(ValueError, match="num_input_dims.*must match"):
            Linear(64, input_dims=(4, 8), num_input_dims=1)


class TestLinearInputValidation:
    """Tests for Linear input shape validation."""

    def test_wrong_trailing_dims_raises_error(self):
        """Test that wrong input dimensions raise ShapeMismatchError."""
        from alphafold3_mlx.modules import Linear
        from alphafold3_mlx.geometry.exceptions import ShapeMismatchError

        layer = Linear(64, input_dims=32)

        # Input with wrong trailing dimension (64 instead of 32)
        x = mx.random.normal(shape=(2, 10, 64), key=mx.random.key(42))

        with pytest.raises(ShapeMismatchError) as exc_info:
            layer(x)

        assert exc_info.value.expected == (32,)
        assert exc_info.value.actual == (64,)

    def test_too_few_dims_raises_error(self):
        """Test that input with too few dimensions raises ShapeMismatchError."""
        from alphafold3_mlx.modules import Linear
        from alphafold3_mlx.geometry.exceptions import ShapeMismatchError

        # Layer expects 2 input dimensions
        layer = Linear(64, input_dims=(4, 8), num_input_dims=2)

        # Input with only 1 dimension
        x = mx.random.normal(shape=(10,), key=mx.random.key(42))

        with pytest.raises(ShapeMismatchError) as exc_info:
            layer(x)

        assert exc_info.value.expected == (4, 8)

    def test_multi_input_dims_wrong_shape(self):
        """Test multi-dim input validation."""
        from alphafold3_mlx.modules import Linear
        from alphafold3_mlx.geometry.exceptions import ShapeMismatchError

        layer = Linear(64, input_dims=(4, 8), num_input_dims=2)

        # Wrong shape: (8, 4) instead of (4, 8)
        x = mx.random.normal(shape=(2, 10, 8, 4), key=mx.random.key(42))

        with pytest.raises(ShapeMismatchError) as exc_info:
            layer(x)

        assert exc_info.value.expected == (4, 8)
        assert exc_info.value.actual == (8, 4)


class TestLinearHaikuWeightLoading:
    """Tests for Linear weight loading from Haiku checkpoint."""

    def test_weight_assignment(self):
        """Test that weights can be assigned for loading."""
        from alphafold3_mlx.modules import Linear

        layer = Linear(64, input_dims=32)

        # Create test weights
        test_weights = mx.ones((32, 64)) * 0.5

        # MLX nn.Module allows direct weight assignment
        layer.weight = test_weights

        np.testing.assert_allclose(np.array(layer.weight), 0.5)

    def test_weight_shape_matches_haiku(self):
        """Test weight shapes match Haiku Linear convention."""
        from alphafold3_mlx.modules import Linear

        # Haiku Linear: weight shape is [in_features, out_features]
        # (not transposed like PyTorch)
        layer = Linear(64, input_dims=32)

        # Should match Haiku convention
        assert layer.weight.shape == (32, 64)
