"""Unit tests for Rot3Array struct-of-arrays.

These tests validate the Rot3Array implementation.
Tests are written first (TDD) to define expected behavior.
"""

from __future__ import annotations

import math

import pytest

import mlx.core as mx
import numpy as np

from alphafold3_mlx.geometry import (
    GeometryError,
    ShapeMismatchError,
    DtypeMismatchError,
    Vec3Array,
)


class TestRot3ArrayConstruction:
    """Tests for Rot3Array construction and validation."""

    def test_basic_construction(self):
        """Test basic Rot3Array construction with valid inputs."""
        from alphafold3_mlx.geometry import Rot3Array

        # Create identity-like rotation
        ones = mx.array([1.0, 1.0])
        zeros = mx.array([0.0, 0.0])

        r = Rot3Array(
            xx=ones, xy=zeros, xz=zeros,
            yx=zeros, yy=ones, yz=zeros,
            zx=zeros, zy=zeros, zz=ones,
        )

        assert r.shape == (2,)
        assert r.dtype == mx.float32

    def test_batch_construction(self):
        """Test Rot3Array with batch dimensions."""
        from alphafold3_mlx.geometry import Rot3Array

        shape = (2, 3, 4)
        ones = mx.ones(shape)
        zeros = mx.zeros(shape)

        r = Rot3Array(
            xx=ones, xy=zeros, xz=zeros,
            yx=zeros, yy=ones, yz=zeros,
            zx=zeros, zy=zeros, zz=ones,
        )

        assert r.shape == shape

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ShapeMismatchError."""
        from alphafold3_mlx.geometry import Rot3Array

        ones = mx.array([1.0, 1.0])
        zeros = mx.array([0.0, 0.0])
        wrong_shape = mx.array([0.0])  # Different shape

        with pytest.raises(ShapeMismatchError):
            Rot3Array(
                xx=ones, xy=zeros, xz=zeros,
                yx=zeros, yy=wrong_shape, yz=zeros,  # yy has wrong shape
                zx=zeros, zy=zeros, zz=ones,
            )

    def test_dtype_mismatch_error(self):
        """Test that mismatched dtypes raise DtypeMismatchError."""
        from alphafold3_mlx.geometry import Rot3Array

        ones_f32 = mx.array([1.0, 1.0], dtype=mx.float32)
        zeros_f32 = mx.array([0.0, 0.0], dtype=mx.float32)
        ones_f16 = mx.array([1.0, 1.0], dtype=mx.float16)

        with pytest.raises(DtypeMismatchError):
            Rot3Array(
                xx=ones_f32, xy=zeros_f32, xz=zeros_f32,
                yx=zeros_f32, yy=ones_f16, yz=zeros_f32,  # yy has wrong dtype
                zx=zeros_f32, zy=zeros_f32, zz=ones_f32,
            )

    def test_supported_dtypes(self):
        """Test construction with supported dtypes."""
        from alphafold3_mlx.geometry import Rot3Array

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            ones = mx.array([1.0], dtype=dtype)
            zeros = mx.array([0.0], dtype=dtype)

            r = Rot3Array(
                xx=ones, xy=zeros, xz=zeros,
                yx=zeros, yy=ones, yz=zeros,
                zx=zeros, zy=zeros, zz=ones,
            )

            assert r.dtype == dtype


class TestRot3ArrayIdentity:
    """Tests for Rot3Array.identity() factory."""

    def test_identity_basic(self):
        """Test identity rotation creation."""
        from alphafold3_mlx.geometry import Rot3Array

        r = Rot3Array.identity((4,))

        assert r.shape == (4,)

        # Check diagonal is 1
        np.testing.assert_allclose(np.array(r.xx), [1.0] * 4)
        np.testing.assert_allclose(np.array(r.yy), [1.0] * 4)
        np.testing.assert_allclose(np.array(r.zz), [1.0] * 4)

        # Check off-diagonal is 0
        np.testing.assert_allclose(np.array(r.xy), [0.0] * 4)
        np.testing.assert_allclose(np.array(r.xz), [0.0] * 4)
        np.testing.assert_allclose(np.array(r.yx), [0.0] * 4)
        np.testing.assert_allclose(np.array(r.yz), [0.0] * 4)
        np.testing.assert_allclose(np.array(r.zx), [0.0] * 4)
        np.testing.assert_allclose(np.array(r.zy), [0.0] * 4)

    def test_identity_multidim(self):
        """Test identity with multi-dimensional shape."""
        from alphafold3_mlx.geometry import Rot3Array

        r = Rot3Array.identity((2, 3))

        assert r.shape == (2, 3)
        assert r.xx.shape == (2, 3)

    def test_identity_dtype(self):
        """Test identity with explicit dtype."""
        from alphafold3_mlx.geometry import Rot3Array

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            r = Rot3Array.identity((4,), dtype=dtype)
            assert r.dtype == dtype


class TestRot3ArrayFromQuaternion:
    """Tests for Rot3Array.from_quaternion()."""

    def test_from_quaternion_identity(self):
        """Test quaternion (1, 0, 0, 0) gives identity rotation."""
        from alphafold3_mlx.geometry import Rot3Array

        w = mx.array([1.0])
        x = mx.array([0.0])
        y = mx.array([0.0])
        z = mx.array([0.0])

        r = Rot3Array.from_quaternion(w, x, y, z)

        # Should be identity
        np.testing.assert_allclose(np.array(r.xx), [1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.yy), [1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.zz), [1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.xy), [0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.xz), [0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.yx), [0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.yz), [0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.zx), [0.0], atol=1e-6)
        np.testing.assert_allclose(np.array(r.zy), [0.0], atol=1e-6)

    def test_from_quaternion_90_deg_z(self):
        """Test 90-degree rotation around z-axis."""
        from alphafold3_mlx.geometry import Rot3Array

        # Quaternion for 90-degree rotation around z: (cos(45°), 0, 0, sin(45°))
        angle = math.pi / 2
        w = mx.array([math.cos(angle / 2)])
        x = mx.array([0.0])
        y = mx.array([0.0])
        z = mx.array([math.sin(angle / 2)])

        r = Rot3Array.from_quaternion(w, x, y, z)

        # Expected: rotates x to y, y to -x
        # [0, -1, 0]
        # [1,  0, 0]
        # [0,  0, 1]
        np.testing.assert_allclose(np.array(r.xx), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.xy), [-1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.xz), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.yx), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.yy), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.yz), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.zx), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.zy), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.zz), [1.0], atol=1e-5)

    def test_from_quaternion_normalize(self):
        """Test that non-unit quaternions are normalized."""
        from alphafold3_mlx.geometry import Rot3Array

        # Non-unit quaternion (will be normalized)
        w = mx.array([2.0])
        x = mx.array([0.0])
        y = mx.array([0.0])
        z = mx.array([0.0])

        r = Rot3Array.from_quaternion(w, x, y, z, normalize=True)

        # Should still be identity
        np.testing.assert_allclose(np.array(r.xx), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.yy), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.zz), [1.0], atol=1e-5)

    def test_from_quaternion_batched(self):
        """Test from_quaternion with batched inputs."""
        from alphafold3_mlx.geometry import Rot3Array

        w = mx.array([1.0, 1.0])
        x = mx.array([0.0, 0.0])
        y = mx.array([0.0, 0.0])
        z = mx.array([0.0, 0.0])

        r = Rot3Array.from_quaternion(w, x, y, z)

        assert r.shape == (2,)


class TestRot3ArrayFromSvd:
    """Tests for Rot3Array.from_svd()."""

    def test_from_svd_identity(self):
        """Test from_svd with identity matrix returns identity."""
        from alphafold3_mlx.geometry import Rot3Array

        # Identity matrix flattened: [1,0,0, 0,1,0, 0,0,1]
        mat = mx.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])

        r = Rot3Array.from_svd(mat)

        np.testing.assert_allclose(np.array(r.xx), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.yy), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(r.zz), [1.0], atol=1e-5)

    def test_from_svd_rotation(self):
        """Test from_svd produces valid rotation from valid input."""
        from alphafold3_mlx.geometry import Rot3Array

        # 90-degree rotation around z-axis: [0,-1,0, 1,0,0, 0,0,1]
        mat = mx.array([[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        r = Rot3Array.from_svd(mat)

        # Result should be orthogonal (valid rotation: R @ R^T = I)
        arr = r.to_array()  # Shape: (1, 3, 3)
        rrt = mx.matmul(arr, mx.transpose(arr, axes=(0, 2, 1)))
        np.testing.assert_allclose(np.array(rrt[0]), np.eye(3), atol=1e-4)

        # Result should be a rotation (det = 1)
        det = (
            r.xx * (r.yy * r.zz - r.yz * r.zy) -
            r.xy * (r.yx * r.zz - r.yz * r.zx) +
            r.xz * (r.yx * r.zy - r.yy * r.zx)
        )
        np.testing.assert_allclose(np.array(det), [1.0], atol=1e-4)

    def test_from_svd_projects_nonrotation(self):
        """Test from_svd projects non-rotation to valid rotation."""
        from alphafold3_mlx.geometry import Rot3Array

        # Arbitrary non-rotation matrix
        mat = mx.array([[1.5, 0.1, -0.2, 0.3, 1.1, 0.4, -0.1, 0.2, 0.9]])

        r = Rot3Array.from_svd(mat)

        # Result should be orthogonal (R @ R^T = I)
        arr = r.to_array()  # Shape: (1, 3, 3)
        rrt = mx.matmul(arr, mx.transpose(arr, axes=(0, 2, 1)))

        # Should be close to identity
        np.testing.assert_allclose(np.array(rrt[0]), np.eye(3), atol=1e-4)

    def test_from_svd_batched(self):
        """Test from_svd with batched inputs."""
        from alphafold3_mlx.geometry import Rot3Array

        # Two identity matrices
        mat = mx.array([
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ])

        r = Rot3Array.from_svd(mat)

        assert r.shape == (2,)


class TestRot3ArrayApplyToPoint:
    """Tests for Rot3Array.apply_to_point()."""

    def test_apply_to_point_identity(self):
        """Test identity rotation doesn't change points."""
        from alphafold3_mlx.geometry import Rot3Array

        r = Rot3Array.identity((2,))
        p = Vec3Array(
            x=mx.array([1.0, 2.0]),
            y=mx.array([3.0, 4.0]),
            z=mx.array([5.0, 6.0]),
        )

        result = r.apply_to_point(p)

        np.testing.assert_allclose(np.array(result.x), np.array(p.x))
        np.testing.assert_allclose(np.array(result.y), np.array(p.y))
        np.testing.assert_allclose(np.array(result.z), np.array(p.z))

    def test_apply_to_point_90_z(self):
        """Test 90-degree z rotation transforms x to y."""
        from alphafold3_mlx.geometry import Rot3Array

        # Create 90-degree rotation around z
        angle = math.pi / 2
        w = mx.array([math.cos(angle / 2)])
        x = mx.array([0.0])
        y = mx.array([0.0])
        z = mx.array([math.sin(angle / 2)])

        r = Rot3Array.from_quaternion(w, x, y, z)

        # Point on x-axis
        p = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )

        result = r.apply_to_point(p)

        # Should be on y-axis
        np.testing.assert_allclose(np.array(result.x), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.y), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.z), [0.0], atol=1e-5)

    def test_apply_inverse_to_point(self):
        """Test apply_inverse_to_point reverses apply_to_point."""
        from alphafold3_mlx.geometry import Rot3Array

        # Arbitrary rotation
        angle = 1.23
        w = mx.array([math.cos(angle / 2)])
        x = mx.array([0.1])
        y = mx.array([0.2])
        z = mx.array([0.3])

        r = Rot3Array.from_quaternion(w, x, y, z, normalize=True)

        p = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([2.0]),
            z=mx.array([3.0]),
        )

        # Apply then inverse should return original
        rotated = r.apply_to_point(p)
        recovered = r.apply_inverse_to_point(rotated)

        np.testing.assert_allclose(np.array(recovered.x), np.array(p.x), atol=1e-5)
        np.testing.assert_allclose(np.array(recovered.y), np.array(p.y), atol=1e-5)
        np.testing.assert_allclose(np.array(recovered.z), np.array(p.z), atol=1e-5)


class TestRot3ArrayInverseAndComposition:
    """Tests for Rot3Array.inverse() and __matmul__."""

    def test_inverse_identity(self):
        """Test inverse of identity is identity."""
        from alphafold3_mlx.geometry import Rot3Array

        r = Rot3Array.identity((2,))
        inv = r.inverse()

        np.testing.assert_allclose(np.array(inv.xx), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(inv.yy), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(np.array(inv.zz), [1.0, 1.0], atol=1e-6)

    def test_inverse_is_transpose(self):
        """Test that inverse equals transpose for rotation matrices."""
        from alphafold3_mlx.geometry import Rot3Array

        # Create arbitrary rotation from quaternion
        w = mx.array([0.5])
        x = mx.array([0.5])
        y = mx.array([0.5])
        z = mx.array([0.5])

        r = Rot3Array.from_quaternion(w, x, y, z)
        inv = r.inverse()

        # Inverse should be transpose
        np.testing.assert_allclose(np.array(inv.xx), np.array(r.xx), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.xy), np.array(r.yx), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.xz), np.array(r.zx), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.yx), np.array(r.xy), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.yy), np.array(r.yy), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.yz), np.array(r.zy), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.zx), np.array(r.xz), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.zy), np.array(r.yz), atol=1e-6)
        np.testing.assert_allclose(np.array(inv.zz), np.array(r.zz), atol=1e-6)

    def test_composition_identity(self):
        """Test composition with identity is unchanged."""
        from alphafold3_mlx.geometry import Rot3Array

        # Create arbitrary rotation
        w = mx.array([0.5])
        x = mx.array([0.5])
        y = mx.array([0.5])
        z = mx.array([0.5])

        r = Rot3Array.from_quaternion(w, x, y, z)
        identity = Rot3Array.identity((1,))

        result = r @ identity

        np.testing.assert_allclose(np.array(result.xx), np.array(r.xx), atol=1e-5)
        np.testing.assert_allclose(np.array(result.yy), np.array(r.yy), atol=1e-5)
        np.testing.assert_allclose(np.array(result.zz), np.array(r.zz), atol=1e-5)

    def test_composition_inverse_is_identity(self):
        """Test r @ r.inverse() is identity."""
        from alphafold3_mlx.geometry import Rot3Array

        # Create arbitrary rotation
        w = mx.array([0.5])
        x = mx.array([0.5])
        y = mx.array([0.5])
        z = mx.array([0.5])

        r = Rot3Array.from_quaternion(w, x, y, z)
        result = r @ r.inverse()

        # Should be identity
        np.testing.assert_allclose(np.array(result.xx), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.yy), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.zz), [1.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.xy), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.xz), [0.0], atol=1e-5)
        np.testing.assert_allclose(np.array(result.yx), [0.0], atol=1e-5)


class TestRot3ArrayFromTwoVectors:
    """Tests for Rot3Array.from_two_vectors()."""

    def test_from_two_vectors_basic(self):
        """Test from_two_vectors aligns e0 to x-axis."""
        from alphafold3_mlx.geometry import Rot3Array

        # e0 pointing in x direction, e1 in y direction
        e0 = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        e1 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )

        r = Rot3Array.from_two_vectors(e0, e1)

        # Applying this rotation should map e0 to x-axis
        # (but inverse maps x-axis to e0)
        # The convention varies - check that result is valid rotation
        arr = r.to_array()
        rrt = mx.matmul(arr, mx.transpose(arr, axes=(0, 2, 1)))
        np.testing.assert_allclose(np.array(rrt[0]), np.eye(3), atol=1e-5)

    def test_from_two_vectors_orthogonal(self):
        """Test from_two_vectors with orthogonal vectors."""
        from alphafold3_mlx.geometry import Rot3Array

        # Standard basis
        e0 = Vec3Array(
            x=mx.array([1.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        e1 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([1.0]),
            z=mx.array([0.0]),
        )

        r = Rot3Array.from_two_vectors(e0, e1)

        # Result should be a valid rotation
        assert r.shape == (1,)


class TestRot3ArrayRandomUniform:
    """Tests for Rot3Array.random_uniform()."""

    def test_random_uniform_basic(self):
        """Test random_uniform produces valid rotations."""
        from alphafold3_mlx.geometry import Rot3Array

        key = mx.random.key(42)
        r = Rot3Array.random_uniform(key, (10,))

        assert r.shape == (10,)

        # Check all are valid rotations (R @ R^T = I)
        arr = r.to_array()  # Shape: (10, 3, 3)
        for i in range(10):
            rrt = np.array(arr[i]) @ np.array(arr[i]).T
            np.testing.assert_allclose(rrt, np.eye(3), atol=1e-4)

    def test_random_uniform_dtype(self):
        """Test random_uniform respects dtype."""
        from alphafold3_mlx.geometry import Rot3Array

        key = mx.random.key(42)

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            r = Rot3Array.random_uniform(key, (4,), dtype=dtype)
            assert r.dtype == dtype

    def test_random_uniform_different_keys(self):
        """Test different keys produce different rotations."""
        from alphafold3_mlx.geometry import Rot3Array

        r1 = Rot3Array.random_uniform(mx.random.key(1), (4,))
        r2 = Rot3Array.random_uniform(mx.random.key(2), (4,))

        # Should be different
        assert not np.allclose(np.array(r1.xx), np.array(r2.xx))
