"""Unit tests for standalone geometry functions.

These tests validate euclidean_distance, dihedral_angle, random_gaussian_vector,
and convenience aliases.
"""

from __future__ import annotations

import pytest
import math

import mlx.core as mx
import numpy as np


class TestEuclideanDistance:
    """Tests for euclidean_distance and square_euclidean_distance."""

    def test_square_euclidean_distance(self):
        """Test square_euclidean_distance function."""
        from alphafold3_mlx.geometry import Vec3Array, square_euclidean_distance

        v1 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        v2 = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result = square_euclidean_distance(v1, v2)

        np.testing.assert_allclose(np.array(result), [25.0])

    def test_euclidean_distance(self):
        """Test euclidean_distance function."""
        from alphafold3_mlx.geometry import Vec3Array, euclidean_distance

        v1 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        v2 = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result = euclidean_distance(v1, v2)

        np.testing.assert_allclose(np.array(result), [5.0])

    def test_euclidean_distance_batched(self):
        """Test euclidean_distance with batched inputs."""
        from alphafold3_mlx.geometry import Vec3Array, euclidean_distance

        v1 = Vec3Array(
            x=mx.array([0.0, 1.0]),
            y=mx.array([0.0, 1.0]),
            z=mx.array([0.0, 1.0]),
        )
        v2 = Vec3Array(
            x=mx.array([3.0, 1.0]),
            y=mx.array([4.0, 2.0]),
            z=mx.array([0.0, 1.0]),
        )

        result = euclidean_distance(v1, v2)

        np.testing.assert_allclose(np.array(result), [5.0, 1.0])

    def test_euclidean_distance_zero(self):
        """Test euclidean_distance with identical points uses epsilon."""
        from alphafold3_mlx.geometry import Vec3Array, euclidean_distance

        v = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([2.0]),
            z=mx.array([3.0]),
        )

        result = euclidean_distance(v, v, epsilon=1e-6)

        # Should return sqrt(epsilon), not 0 or NaN
        assert result.item() >= 0
        assert not math.isnan(result.item())


class TestDihedralAngle:
    """Tests for dihedral_angle function."""

    def test_dihedral_angle_zero(self):
        """Test dihedral angle of coplanar points is 0 or pi."""
        from alphafold3_mlx.geometry import Vec3Array, dihedral_angle

        # Points in the XY plane
        a = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        b = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        c = Vec3Array(
            x=mx.array([2.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )
        d = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )

        result = dihedral_angle(a, b, c, d)

        # Coplanar points should have dihedral near 0 or pi
        angle = result.item()
        assert abs(angle) < 0.1 or abs(abs(angle) - math.pi) < 0.1

    def test_dihedral_angle_90_degrees(self):
        """Test dihedral angle of perpendicular planes is ~pi/2."""
        from alphafold3_mlx.geometry import Vec3Array, dihedral_angle

        # Points forming perpendicular planes
        a = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        b = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        c = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )
        d = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([1.0]),
        )

        result = dihedral_angle(a, b, c, d)

        # Should be approximately pi/2 (90 degrees)
        np.testing.assert_allclose(abs(result.item()), math.pi / 2, rtol=0.1)

    def test_dihedral_angle_range(self):
        """Test that dihedral angle is in [-pi, pi]."""
        from alphafold3_mlx.geometry import Vec3Array, dihedral_angle

        # Random points using MLX random API
        key = mx.random.key(42)
        k1, k2, k3, k4 = mx.random.split(key, 4)

        # Use separate keys for x, y, z to get different values
        ka1, ka2, ka3 = mx.random.split(k1, 3)
        kb1, kb2, kb3 = mx.random.split(k2, 3)
        kc1, kc2, kc3 = mx.random.split(k3, 3)
        kd1, kd2, kd3 = mx.random.split(k4, 3)

        a = Vec3Array(
            x=mx.random.normal(shape=(10,), key=ka1),
            y=mx.random.normal(shape=(10,), key=ka2),
            z=mx.random.normal(shape=(10,), key=ka3),
        )
        b = Vec3Array(
            x=mx.random.normal(shape=(10,), key=kb1),
            y=mx.random.normal(shape=(10,), key=kb2),
            z=mx.random.normal(shape=(10,), key=kb3),
        )
        c = Vec3Array(
            x=mx.random.normal(shape=(10,), key=kc1),
            y=mx.random.normal(shape=(10,), key=kc2),
            z=mx.random.normal(shape=(10,), key=kc3),
        )
        d = Vec3Array(
            x=mx.random.normal(shape=(10,), key=kd1),
            y=mx.random.normal(shape=(10,), key=kd2),
            z=mx.random.normal(shape=(10,), key=kd3),
        )

        result = dihedral_angle(a, b, c, d)
        angles = np.array(result)

        assert np.all(angles >= -math.pi)
        assert np.all(angles <= math.pi)


class TestRandomGaussianVector:
    """Tests for random_gaussian_vector function."""

    def test_basic_generation(self):
        """Test random_gaussian_vector generates correct shape."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key = mx.random.key(42)
        shape = (4,)

        v = random_gaussian_vector(shape, key)

        assert v.shape == shape

    def test_batched_generation(self):
        """Test random_gaussian_vector with batch dimensions."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key = mx.random.key(42)
        shape = (2, 3, 4)

        v = random_gaussian_vector(shape, key)

        assert v.shape == shape

    def test_dtype(self):
        """Test random_gaussian_vector respects dtype."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key = mx.random.key(42)

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            v = random_gaussian_vector((4,), key, dtype=dtype)
            assert v.dtype == dtype

    def test_distribution_statistics(self):
        """Test that random samples have approximately standard normal distribution."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key = mx.random.key(42)
        shape = (10000,)

        v = random_gaussian_vector(shape, key)

        # Combine all components
        all_vals = np.concatenate([
            np.array(v.x),
            np.array(v.y),
            np.array(v.z),
        ])

        # Check mean is approximately 0 and std is approximately 1
        np.testing.assert_allclose(np.mean(all_vals), 0.0, atol=0.1)
        np.testing.assert_allclose(np.std(all_vals), 1.0, atol=0.1)

    def test_deterministic_with_same_key(self):
        """Test that same key produces same result."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key = mx.random.key(42)
        shape = (4,)

        v1 = random_gaussian_vector(shape, key)
        v2 = random_gaussian_vector(shape, key)

        np.testing.assert_array_equal(np.array(v1.x), np.array(v2.x))
        np.testing.assert_array_equal(np.array(v1.y), np.array(v2.y))
        np.testing.assert_array_equal(np.array(v1.z), np.array(v2.z))

    def test_different_with_different_key(self):
        """Test that different keys produce different results."""
        from alphafold3_mlx.geometry import random_gaussian_vector

        key1 = mx.random.key(42)
        key2 = mx.random.key(43)
        shape = (4,)

        v1 = random_gaussian_vector(shape, key1)
        v2 = random_gaussian_vector(shape, key2)

        # Should be different
        assert not np.allclose(np.array(v1.x), np.array(v2.x))


class TestConvenienceAliases:
    """Tests for convenience function aliases."""

    def test_dot_alias(self):
        """Test dot() standalone function."""
        from alphafold3_mlx.geometry import Vec3Array, dot

        v1 = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([2.0]),
            z=mx.array([3.0]),
        )
        v2 = Vec3Array(
            x=mx.array([4.0]),
            y=mx.array([5.0]),
            z=mx.array([6.0]),
        )

        # Should match method result
        result_func = dot(v1, v2)
        result_method = v1.dot(v2)

        np.testing.assert_allclose(np.array(result_func), np.array(result_method))

    def test_cross_alias(self):
        """Test cross() standalone function."""
        from alphafold3_mlx.geometry import Vec3Array, cross

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

        result_func = cross(v1, v2)
        result_method = v1.cross(v2)

        np.testing.assert_allclose(np.array(result_func.x), np.array(result_method.x))
        np.testing.assert_allclose(np.array(result_func.y), np.array(result_method.y))
        np.testing.assert_allclose(np.array(result_func.z), np.array(result_method.z))

    def test_norm_alias(self):
        """Test norm() standalone function."""
        from alphafold3_mlx.geometry import Vec3Array, norm

        v = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result_func = norm(v)
        result_method = v.norm()

        np.testing.assert_allclose(np.array(result_func), np.array(result_method))

    def test_normalized_alias(self):
        """Test normalized() standalone function."""
        from alphafold3_mlx.geometry import Vec3Array, normalized

        v = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        result_func = normalized(v)
        result_method = v.normalized()

        np.testing.assert_allclose(np.array(result_func.x), np.array(result_method.x))
        np.testing.assert_allclose(np.array(result_func.y), np.array(result_method.y))
        np.testing.assert_allclose(np.array(result_func.z), np.array(result_method.z))
