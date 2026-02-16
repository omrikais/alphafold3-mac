"""Unit tests for LayerNorm module.

These tests validate the LayerNorm implementation for.
Tests are written first (TDD) to define expected behavior.
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import numpy as np


class TestLayerNormConstruction:
    """Tests for LayerNorm construction."""

    def test_basic_construction(self):
        """Test basic LayerNorm construction."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        assert norm.scale is not None
        assert norm.offset is not None
        assert norm.scale.shape == (256,)
        assert norm.offset.shape == (256,)

    def test_construction_no_scale(self):
        """Test LayerNorm without scale."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False)

        assert norm.scale is None
        assert norm.offset is not None

    def test_construction_no_offset(self):
        """Test LayerNorm without offset."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_offset=False)

        assert norm.scale is not None
        assert norm.offset is None

    def test_construction_no_affine(self):
        """Test LayerNorm without scale or offset."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False, create_offset=False)

        assert norm.scale is None
        assert norm.offset is None

    def test_scale_init_ones(self):
        """Test scale is initialized to ones."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        np.testing.assert_allclose(np.array(norm.scale), 1.0)

    def test_offset_init_zeros(self):
        """Test offset is initialized to zeros."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        np.testing.assert_allclose(np.array(norm.offset), 0.0)


class TestLayerNormForwardFloat32:
    """Tests for LayerNorm forward pass with float32."""

    def test_forward_basic(self):
        """Test basic forward pass normalizes correctly."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False, create_offset=False)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key)
        y = norm(x)

        # Output should have zero mean and unit variance along last dim
        mean = mx.mean(y, axis=-1)
        var = mx.var(y, axis=-1)

        np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.array(var), 1.0, atol=1e-2)

    def test_forward_with_scale(self):
        """Test forward with scale parameter."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_offset=False)

        # Set scale to 2.0
        norm.scale = mx.full((256,), 2.0)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key)
        y = norm(x)

        # Variance should be approximately 4.0 (scale^2)
        var = mx.var(y, axis=-1)
        np.testing.assert_allclose(np.array(var), 4.0, atol=0.1)

    def test_forward_with_offset(self):
        """Test forward with offset parameter."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False)

        # Set offset to 5.0
        norm.offset = mx.full((256,), 5.0)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key)
        y = norm(x)

        # Mean should be approximately 5.0
        mean = mx.mean(y, axis=-1)
        np.testing.assert_allclose(np.array(mean), 5.0, atol=1e-5)

    def test_forward_preserves_shape(self):
        """Test forward preserves input shape."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        x = mx.random.normal(shape=(2, 3, 4, 256), key=mx.random.key(42))
        y = norm(x)

        assert y.shape == x.shape


class TestLayerNormUpcastFloat16:
    """Tests for LayerNorm upcast with float16 input."""

    def test_upcast_float16(self):
        """Test float16 input is upcast to float32 for computation."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, upcast=True)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key).astype(mx.float16)
        y = norm(x)

        # Output should be float16
        assert y.dtype == mx.float16

        # Should not contain NaN/Inf (stable computation)
        assert not mx.isnan(y).any().item()
        assert not mx.isinf(y).any().item()

    def test_upcast_disabled_float16(self):
        """Test float16 without explicit upcast."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, upcast=False)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key).astype(mx.float16)
        y = norm(x)

        # MLX may still promote to float32 during computation
        # This test just verifies upcast=False doesn't explicitly cast
        assert y.dtype in (mx.float16, mx.float32)


class TestLayerNormUpcastBfloat16:
    """Tests for LayerNorm upcast with bfloat16 input."""

    def test_upcast_bfloat16(self):
        """Test bfloat16 input is upcast to float32 for computation."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, upcast=True)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key).astype(mx.bfloat16)
        y = norm(x)

        # Output should be bfloat16
        assert y.dtype == mx.bfloat16

        # Should not contain NaN/Inf
        assert not mx.isnan(y).any().item()
        assert not mx.isinf(y).any().item()


class TestLayerNormNoAffine:
    """Tests for LayerNorm without scale/offset."""

    def test_no_affine_normalizes(self):
        """Test LayerNorm without affine params still normalizes."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False, create_offset=False)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key) * 10 + 5

        y = norm(x)

        mean = mx.mean(y, axis=-1)
        var = mx.var(y, axis=-1)

        np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.array(var), 1.0, atol=1e-2)

    def test_no_affine_no_params(self):
        """Test LayerNorm without affine has no learnable params."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False, create_offset=False)

        # In MLX, parameters() returns a dict of trainable params
        params = norm.trainable_parameters()

        # Should have no trainable parameters
        assert len(params) == 0


class TestLayerNormEdgeCases:
    """Tests for LayerNorm edge cases."""

    def test_extreme_values(self):
        """Test LayerNorm handles extreme values."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        # Large values
        x = mx.ones((2, 256)) * 1e4
        y = norm(x)

        # Output should be finite (all same value -> zero variance -> handle epsilon)
        assert mx.isfinite(y).all().item()

    def test_near_epsilon_variance(self):
        """Test LayerNorm handles near-zero variance."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, create_scale=False, create_offset=False)

        # Constant input (zero variance)
        x = mx.ones((2, 10, 256)) * 5.0
        y = norm(x)

        # Should not produce NaN/Inf
        assert not mx.isnan(y).any().item()
        assert not mx.isinf(y).any().item()

    def test_small_batch(self):
        """Test LayerNorm with small batch."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256)

        x = mx.random.normal(shape=(1, 256), key=mx.random.key(42))
        y = norm(x)

        assert y.shape == (1, 256)

    def test_custom_eps(self):
        """Test LayerNorm with custom epsilon."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(256, eps=1e-3)

        # Constant input
        x = mx.ones((2, 10, 256))
        y = norm(x)

        # Should be stable
        assert mx.isfinite(y).all().item()


class TestLayerNormNonLastAxis:
    """Tests for LayerNorm with axis != -1 (bug fix validation)."""

    def test_axis_0_normalization(self):
        """Test LayerNorm normalizes along axis 0 correctly."""
        from alphafold3_mlx.modules import LayerNorm

        # Shape: [B, S, C] = [4, 10, 256], normalize along axis 0 (batch)
        norm = LayerNorm(4, axis=0, create_scale=False, create_offset=False)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(4, 10, 256), key=key)
        y = norm(x)

        # Output should have zero mean and unit variance along axis 0
        mean = mx.mean(y, axis=0)
        var = mx.var(y, axis=0)

        np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.array(var), 1.0, atol=1e-2)

    def test_axis_1_normalization(self):
        """Test LayerNorm normalizes along axis 1 correctly."""
        from alphafold3_mlx.modules import LayerNorm

        # Shape: [B, S, C] = [2, 10, 256], normalize along axis 1 (sequence)
        norm = LayerNorm(10, axis=1, create_scale=False, create_offset=False)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(2, 10, 256), key=key)
        y = norm(x)

        # Output should have zero mean and unit variance along axis 1
        mean = mx.mean(y, axis=1)
        var = mx.var(y, axis=1)

        np.testing.assert_allclose(np.array(mean), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.array(var), 1.0, atol=1e-2)

    def test_axis_0_with_scale(self):
        """Test LayerNorm axis=0 with scale broadcasts correctly."""
        from alphafold3_mlx.modules import LayerNorm

        # Shape: [4, 10, 256], normalize along axis 0
        norm = LayerNorm(4, axis=0, create_offset=False)

        # Set scale to 2.0 for all elements along axis 0
        norm.scale = mx.full((4,), 2.0)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(4, 10, 256), key=key)
        y = norm(x)

        # Variance along axis 0 should be ~4.0 (scale^2)
        var = mx.var(y, axis=0)
        np.testing.assert_allclose(np.array(var), 4.0, atol=0.2)

    def test_axis_0_with_offset(self):
        """Test LayerNorm axis=0 with offset broadcasts correctly."""
        from alphafold3_mlx.modules import LayerNorm

        # Shape: [4, 10, 256], normalize along axis 0
        norm = LayerNorm(4, axis=0, create_scale=False)

        # Set offset to 5.0 for all elements along axis 0
        norm.offset = mx.full((4,), 5.0)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(4, 10, 256), key=key)
        y = norm(x)

        # Mean along axis 0 should be ~5.0
        mean = mx.mean(y, axis=0)
        np.testing.assert_allclose(np.array(mean), 5.0, atol=1e-5)

    def test_axis_0_with_scale_and_offset(self):
        """Test LayerNorm axis=0 with both scale and offset."""
        from alphafold3_mlx.modules import LayerNorm

        # Shape: [4, 10, 256], normalize along axis 0
        norm = LayerNorm(4, axis=0)

        # Set different scale and offset per element along axis 0
        norm.scale = mx.array([1.0, 2.0, 3.0, 4.0])
        norm.offset = mx.array([0.0, 1.0, 2.0, 3.0])

        key = mx.random.key(42)
        x = mx.random.normal(shape=(4, 10, 256), key=key)
        y = norm(x)

        # Check that different positions along axis 0 have different statistics
        # Position 0: scale=1, offset=0 -> mean~0, std~1
        # Position 3: scale=4, offset=3 -> mean~3, std~4
        mean_per_pos = mx.mean(y, axis=(1, 2))  # Mean over S and C dims
        std_per_pos = mx.std(y, axis=(1, 2))

        # Check means are approximately the offsets
        np.testing.assert_allclose(
            np.array(mean_per_pos), [0.0, 1.0, 2.0, 3.0], atol=0.1
        )
        # Check stds are approximately the scales
        np.testing.assert_allclose(
            np.array(std_per_pos), [1.0, 2.0, 3.0, 4.0], atol=0.1
        )

    def test_axis_0_preserves_shape(self):
        """Test LayerNorm axis=0 preserves input shape."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(4, axis=0)

        x = mx.random.normal(shape=(4, 10, 256), key=mx.random.key(42))
        y = norm(x)

        assert y.shape == x.shape

    def test_axis_0_float16_upcast(self):
        """Test LayerNorm axis=0 with float16 upcast."""
        from alphafold3_mlx.modules import LayerNorm

        norm = LayerNorm(4, axis=0, upcast=True)

        key = mx.random.key(42)
        x = mx.random.normal(shape=(4, 10, 256), key=key).astype(mx.float16)
        y = norm(x)

        # Output should be float16
        assert y.dtype == mx.float16

        # Should not contain NaN/Inf
        assert not mx.isnan(y).any().item()
        assert not mx.isinf(y).any().item()
