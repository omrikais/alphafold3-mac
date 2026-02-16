"""Unit tests for Vec3Array struct-of-arrays.

These tests validate the Vec3Array implementation for.
Tests are written first (TDD) to define expected behavior.
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import numpy as np

from alphafold3_mlx.geometry import (
    GeometryError,
    ShapeMismatchError,
    DtypeMismatchError,
)


class TestVec3ArrayConstruction:
    """Tests for Vec3Array construction and validation."""

    def test_basic_construction(self):
        """Test basic Vec3Array construction with valid inputs."""
        from alphafold3_mlx.geometry import Vec3Array

        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        z = mx.array([7.0, 8.0, 9.0])

        v = Vec3Array(x=x, y=y, z=z)

        assert v.shape == (3,)
        assert v.dtype == mx.float32

    def test_batch_construction(self):
        """Test Vec3Array with batch dimensions."""
        from alphafold3_mlx.geometry import Vec3Array

        x = mx.zeros((2, 3, 4))
        y = mx.zeros((2, 3, 4))
        z = mx.zeros((2, 3, 4))

        v = Vec3Array(x=x, y=y, z=z)

        assert v.shape == (2, 3, 4)

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ShapeMismatchError."""
        from alphafold3_mlx.geometry import Vec3Array

        x = mx.array([1.0, 2.0])
        y = mx.array([1.0, 2.0, 3.0])  # Different shape
        z = mx.array([1.0, 2.0])

        with pytest.raises(ShapeMismatchError) as exc_info:
            Vec3Array(x=x, y=y, z=z)

        assert exc_info.value.expected == (2,)
        assert exc_info.value.actual == (3,)

    def test_dtype_mismatch_error(self):
        """Test that mismatched dtypes raise DtypeMismatchError."""
        from alphafold3_mlx.geometry import Vec3Array

        x = mx.array([1.0, 2.0], dtype=mx.float32)
        y = mx.array([1.0, 2.0], dtype=mx.float16)  # Different dtype
        z = mx.array([1.0, 2.0], dtype=mx.float32)

        with pytest.raises(DtypeMismatchError) as exc_info:
            Vec3Array(x=x, y=y, z=z)

        assert mx.float16 in exc_info.value.dtypes

    def test_supported_dtypes(self):
        """Test construction with supported dtypes: float32, float16, bfloat16."""
        from alphafold3_mlx.geometry import Vec3Array

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            x = mx.array([1.0, 2.0], dtype=dtype)
            y = mx.array([3.0, 4.0], dtype=dtype)
            z = mx.array([5.0, 6.0], dtype=dtype)

            v = Vec3Array(x=x, y=y, z=z)
            assert v.dtype == dtype

    def test_frozen_dataclass(self):
        """Test that Vec3Array is immutable (frozen dataclass)."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([2.0]),
            z=mx.array([3.0]),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            v.x = mx.array([0.0])


class TestVec3ArrayArithmetic:
    """Tests for Vec3Array arithmetic operations."""

    def test_addition(self):
        """Test Vec3Array addition."""
        from alphafold3_mlx.geometry import Vec3Array

        v1 = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )
        v2 = Vec3Array(
            x=mx.array([10.0, 20.0]),
            y=mx.array([30.0, 40.0]),
            z=mx.array([50.0, 60.0]),
        )

        result = v1 + v2

        np.testing.assert_allclose(np.array(result.x), [11.0, 22.0])
        np.testing.assert_allclose(np.array(result.y), [33.0, 44.0])
        np.testing.assert_allclose(np.array(result.z), [55.0, 66.0])

    def test_subtraction(self):
        """Test Vec3Array subtraction."""
        from alphafold3_mlx.geometry import Vec3Array

        v1 = Vec3Array(
            x=mx.array([10.0, 20.0]),
            y=mx.array([30.0, 40.0]),
            z=mx.array([50.0, 60.0]),
        )
        v2 = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )

        result = v1 - v2

        np.testing.assert_allclose(np.array(result.x), [9.0, 18.0])
        np.testing.assert_allclose(np.array(result.y), [27.0, 36.0])
        np.testing.assert_allclose(np.array(result.z), [45.0, 54.0])

    def test_scalar_multiplication(self):
        """Test Vec3Array * scalar."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )

        result = v * 2.0

        np.testing.assert_allclose(np.array(result.x), [2.0, 4.0])
        np.testing.assert_allclose(np.array(result.y), [6.0, 8.0])
        np.testing.assert_allclose(np.array(result.z), [10.0, 12.0])

    def test_reverse_scalar_multiplication(self):
        """Test scalar * Vec3Array."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )

        result = 3.0 * v

        np.testing.assert_allclose(np.array(result.x), [3.0, 6.0])
        np.testing.assert_allclose(np.array(result.y), [9.0, 12.0])
        np.testing.assert_allclose(np.array(result.z), [15.0, 18.0])

    def test_division(self):
        """Test Vec3Array / scalar."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([2.0, 4.0]),
            y=mx.array([6.0, 8.0]),
            z=mx.array([10.0, 12.0]),
        )

        result = v / 2.0

        np.testing.assert_allclose(np.array(result.x), [1.0, 2.0])
        np.testing.assert_allclose(np.array(result.y), [3.0, 4.0])
        np.testing.assert_allclose(np.array(result.z), [5.0, 6.0])

    def test_negation(self):
        """Test -Vec3Array."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0, -2.0]),
            y=mx.array([3.0, -4.0]),
            z=mx.array([5.0, -6.0]),
        )

        result = -v

        np.testing.assert_allclose(np.array(result.x), [-1.0, 2.0])
        np.testing.assert_allclose(np.array(result.y), [-3.0, 4.0])
        np.testing.assert_allclose(np.array(result.z), [-5.0, 6.0])

    def test_positive(self):
        """Test +Vec3Array."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0, -2.0]),
            y=mx.array([3.0, -4.0]),
            z=mx.array([5.0, -6.0]),
        )

        result = +v

        np.testing.assert_allclose(np.array(result.x), [1.0, -2.0])
        np.testing.assert_allclose(np.array(result.y), [3.0, -4.0])
        np.testing.assert_allclose(np.array(result.z), [5.0, -6.0])

    def test_array_multiplication(self):
        """Test Vec3Array * mx.array (broadcasting)."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )
        scale = mx.array([2.0, 3.0])

        result = v * scale

        np.testing.assert_allclose(np.array(result.x), [2.0, 6.0])
        np.testing.assert_allclose(np.array(result.y), [6.0, 12.0])
        np.testing.assert_allclose(np.array(result.z), [10.0, 18.0])


class TestVec3ArrayGeometricOps:
    """Tests for Vec3Array geometric operations."""

    def test_dot_product(self):
        """Test Vec3Array.dot() method."""
        from alphafold3_mlx.geometry import Vec3Array

        v1 = Vec3Array(
            x=mx.array([1.0, 0.0]),
            y=mx.array([0.0, 1.0]),
            z=mx.array([0.0, 0.0]),
        )
        v2 = Vec3Array(
            x=mx.array([1.0, 0.0]),
            y=mx.array([0.0, 1.0]),
            z=mx.array([0.0, 0.0]),
        )

        result = v1.dot(v2)

        np.testing.assert_allclose(np.array(result), [1.0, 1.0])

    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors is zero."""
        from alphafold3_mlx.geometry import Vec3Array

        v1 = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        v2 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )

        result = v1.dot(v2)

        np.testing.assert_allclose(np.array(result), [0.0])

    def test_cross_product(self):
        """Test Vec3Array.cross() method."""
        from alphafold3_mlx.geometry import Vec3Array

        # x × y = z
        v1 = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        v2 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )

        result = v1.cross(v2)

        np.testing.assert_allclose(np.array(result.x), [0.0])
        np.testing.assert_allclose(np.array(result.y), [0.0])
        np.testing.assert_allclose(np.array(result.z), [1.0])

    def test_cross_product_anticommutative(self):
        """Test that a × b = -(b × a)."""
        from alphafold3_mlx.geometry import Vec3Array

        v1 = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )
        v2 = Vec3Array(
            x=mx.array([7.0, 8.0]),
            y=mx.array([9.0, 10.0]),
            z=mx.array([11.0, 12.0]),
        )

        cross1 = v1.cross(v2)
        cross2 = v2.cross(v1)

        np.testing.assert_allclose(np.array(cross1.x), -np.array(cross2.x))
        np.testing.assert_allclose(np.array(cross1.y), -np.array(cross2.y))
        np.testing.assert_allclose(np.array(cross1.z), -np.array(cross2.z))

    def test_norm(self):
        """Test Vec3Array.norm() method."""
        from alphafold3_mlx.geometry import Vec3Array

        # 3-4-5 triangle: sqrt(9+16+0) = 5
        v = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result = v.norm()

        np.testing.assert_allclose(np.array(result), [5.0])

    def test_norm_with_epsilon(self):
        """Test Vec3Array.norm() with near-zero vector uses epsilon.

        JAX semantics: epsilon is a norm threshold, so norm2 is clipped to
        epsilon**2, resulting in output being clipped to epsilon.
        """
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )

        result = v.norm(epsilon=1e-6)

        # JAX clips norm2 to epsilon**2, so output is epsilon (not sqrt(epsilon))
        np.testing.assert_allclose(np.array(result), [1e-6], rtol=1e-5)

    def test_norm2(self):
        """Test Vec3Array.norm2() (squared norm)."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result = v.norm2()

        np.testing.assert_allclose(np.array(result), [25.0])

    def test_normalized(self):
        """Test Vec3Array.normalized() method."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result = v.normalized()

        # Should have unit length
        np.testing.assert_allclose(np.array(result.x), [0.6])
        np.testing.assert_allclose(np.array(result.y), [0.8])
        np.testing.assert_allclose(np.array(result.z), [0.0])

        # Verify unit length
        norm = result.norm()
        np.testing.assert_allclose(np.array(norm), [1.0], rtol=1e-5)

    def test_normalized_with_epsilon(self):
        """Test normalized() with near-zero vector doesn't produce NaN."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([1e-10]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )

        result = v.normalized(epsilon=1e-6)

        # Should not contain NaN
        assert not mx.isnan(result.x).any().item()
        assert not mx.isnan(result.y).any().item()
        assert not mx.isnan(result.z).any().item()

    def test_normalized_zero_vector_returns_zero(self):
        """Test normalized() returns zero for exact zero vectors."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )

        result = v.normalized(epsilon=1e-6)

        # Zero vector should return zero (not NaN/inf)
        np.testing.assert_allclose(np.array(result.x), [0.0])
        np.testing.assert_allclose(np.array(result.y), [0.0])
        np.testing.assert_allclose(np.array(result.z), [0.0])
        assert not mx.isnan(result.x).any().item()

    def test_normalized_epsilon_zero_produces_nan_for_zero_vectors(self):
        """Test normalized(epsilon=0) produces NaN for zero vectors (JAX parity).

        JAX semantics: epsilon=0 means no clamping, so zero vectors produce
        0/0 = NaN. This matches the reference implementation behavior.
        """
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([0.0, 3.0]),
            y=mx.array([0.0, 4.0]),
            z=mx.array([0.0, 0.0]),
        )

        result = v.normalized(epsilon=0.0)

        # Second (non-zero) vector should be normalized to unit length
        np.testing.assert_allclose(np.array(result.x[1]), 0.6, rtol=1e-5)
        np.testing.assert_allclose(np.array(result.y[1]), 0.8, rtol=1e-5)

        # First (zero) vector produces NaN with epsilon=0 (JAX parity)
        assert mx.isnan(result.x[0]).item()
        assert mx.isnan(result.y[0]).item()
        assert mx.isnan(result.z[0]).item()


class TestVec3ArrayConversions:
    """Tests for Vec3Array conversion methods."""

    def test_to_array(self):
        """Test Vec3Array.to_array() method."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array(
            x=mx.array([[1.0, 2.0], [3.0, 4.0]]),
            y=mx.array([[5.0, 6.0], [7.0, 8.0]]),
            z=mx.array([[9.0, 10.0], [11.0, 12.0]]),
        )

        result = v.to_array()

        assert result.shape == (2, 2, 3)
        # Check first element has correct [x, y, z]
        np.testing.assert_allclose(np.array(result[0, 0, :]), [1.0, 5.0, 9.0])

    def test_from_array(self):
        """Test Vec3Array.from_array() method."""
        from alphafold3_mlx.geometry import Vec3Array

        arr = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        v = Vec3Array.from_array(arr)

        np.testing.assert_allclose(np.array(v.x), [1.0, 4.0])
        np.testing.assert_allclose(np.array(v.y), [2.0, 5.0])
        np.testing.assert_allclose(np.array(v.z), [3.0, 6.0])

    def test_roundtrip(self):
        """Test to_array -> from_array roundtrip preserves data."""
        from alphafold3_mlx.geometry import Vec3Array

        original = Vec3Array(
            x=mx.array([1.0, 2.0, 3.0]),
            y=mx.array([4.0, 5.0, 6.0]),
            z=mx.array([7.0, 8.0, 9.0]),
        )

        roundtrip = Vec3Array.from_array(original.to_array())

        np.testing.assert_allclose(np.array(roundtrip.x), np.array(original.x))
        np.testing.assert_allclose(np.array(roundtrip.y), np.array(original.y))
        np.testing.assert_allclose(np.array(roundtrip.z), np.array(original.z))

    def test_from_array_wrong_shape(self):
        """Test from_array with wrong last dimension raises error."""
        from alphafold3_mlx.geometry import Vec3Array

        arr = mx.array([[1.0, 2.0]])  # Last dim is 2, not 3

        with pytest.raises((ShapeMismatchError, ValueError)):
            Vec3Array.from_array(arr)


class TestVec3ArrayFactoryMethods:
    """Tests for Vec3Array factory methods."""

    def test_zeros_basic(self):
        """Test Vec3Array.zeros() with basic shape."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array.zeros((4,))

        assert v.shape == (4,)
        np.testing.assert_allclose(np.array(v.x), [0.0] * 4)
        np.testing.assert_allclose(np.array(v.y), [0.0] * 4)
        np.testing.assert_allclose(np.array(v.z), [0.0] * 4)

    def test_zeros_multidim(self):
        """Test Vec3Array.zeros() with multi-dimensional shape."""
        from alphafold3_mlx.geometry import Vec3Array

        v = Vec3Array.zeros((2, 3, 4))

        assert v.shape == (2, 3, 4)
        assert v.x.shape == (2, 3, 4)

    def test_zeros_with_dtype(self):
        """Test Vec3Array.zeros() with explicit dtype."""
        from alphafold3_mlx.geometry import Vec3Array

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            v = Vec3Array.zeros((4,), dtype=dtype)
            assert v.dtype == dtype

    def test_zeros_large_batch(self):
        """Test Vec3Array.zeros with large batch for scale."""
        from alphafold3_mlx.geometry import Vec3Array

        # scale: [1, 5000]
        v = Vec3Array.zeros((1, 5000), dtype=mx.float32)

        assert v.shape == (1, 5000)
        mx.eval(v.x, v.y, v.z)  # Force evaluation
