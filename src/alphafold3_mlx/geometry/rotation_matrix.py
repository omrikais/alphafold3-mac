"""Rot3Array struct-of-arrays implementation for MLX.

This module provides the Rot3Array class representing batched 3x3 rotation
matrices in a struct-of-arrays format, matching the JAX geometry API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import mlx.core as mx

from alphafold3_mlx.geometry.exceptions import ShapeMismatchError, DtypeMismatchError
from alphafold3_mlx.geometry.vector import Vec3Array

if TYPE_CHECKING:
    from typing import TypeAlias

    Shape: TypeAlias = tuple[int, ...]

__all__ = [
    "Rot3Array",
]

def _make_matrix_svd_factors() -> mx.array:
    """Generate factors for converting 3x3 matrix to symmetric 4x4 matrix.

    This matches the JAX reference implementation exactly - creates a 16x9
    matrix where each row corresponds to one cell of the 4x4 symmetric matrix
    (in row-major order).
    """
    import numpy as np

    factors = np.zeros((16, 9), dtype=np.float32)

    factors[0, [0, 4, 8]] = 1.0

    factors[[1, 4], 5] = 1.0
    factors[[1, 4], 7] = -1.0

    factors[[2, 8], 6] = 1.0
    factors[[2, 8], 2] = -1.0

    factors[[3, 12], 1] = 1.0
    factors[[3, 12], 3] = -1.0

    factors[5, 0] = 1.0
    factors[5, [4, 8]] = -1.0

    factors[[6, 9], 1] = 1.0
    factors[[6, 9], 3] = 1.0

    factors[[7, 13], 2] = 1.0
    factors[[7, 13], 6] = 1.0

    factors[10, 4] = 1.0
    factors[10, [0, 8]] = -1.0

    factors[[11, 14], 5] = 1.0
    factors[[11, 14], 7] = 1.0

    factors[15, 8] = 1.0
    factors[15, [0, 4]] = -1.0

    return mx.array(factors)


# Constants for quaternion-based SVD (from JAX geometry)
# This is a 16x9 matrix where each row corresponds to one cell of the 4x4
# symmetric matrix (in row-major order). The matrix is generated to match
# the JAX reference implementation exactly.
MATRIX_SVD_QUAT_FACTORS = _make_matrix_svd_factors()


@dataclass(frozen=True)
class Rot3Array:
    """Struct-of-arrays representing batched 3x3 rotation matrices.

    Components are named as (row, column): xx=R[0,0], xy=R[0,1], etc.
    All components must have identical shape and dtype.

    Attributes:
        xx, xy, xz: First row components.
        yx, yy, yz: Second row components.
        zx, zy, zz: Third row components.

    Raises:
        ShapeMismatchError: If component shapes differ.
        DtypeMismatchError: If component dtypes differ.

    Example:
        >>> r = Rot3Array.identity((2,))
        >>> r.shape
        (2,)
    """

    xx: mx.array
    xy: mx.array
    xz: mx.array
    yx: mx.array
    yy: mx.array
    yz: mx.array
    zx: mx.array
    zy: mx.array
    zz: mx.array

    def __post_init__(self) -> None:
        """Validate that all components have same shape and dtype."""
        components = [
            self.xx, self.xy, self.xz,
            self.yx, self.yy, self.yz,
            self.zx, self.zy, self.zz,
        ]
        names = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]

        expected_shape = self.xx.shape
        expected_dtype = self.xx.dtype

        # Check shapes
        for arr, name in zip(components[1:], names[1:]):
            if arr.shape != expected_shape:
                raise ShapeMismatchError(
                    f"Rot3Array component '{name}' has shape {arr.shape}, "
                    f"expected {expected_shape} (matching xx)",
                    expected=expected_shape,
                    actual=arr.shape,
                )

        # Check dtypes
        dtypes = [arr.dtype for arr in components]
        if not all(d == expected_dtype for d in dtypes):
            raise DtypeMismatchError(
                f"Rot3Array components have inconsistent dtypes: {dtypes}",
                dtypes=dtypes,
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Common shape of all component arrays."""
        return self.xx.shape

    @property
    def dtype(self) -> mx.Dtype:
        """Common dtype of all component arrays."""
        return self.xx.dtype

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def inverse(self) -> Self:
        """Return inverse rotation (transpose for orthogonal matrix).

        Returns:
            Rot3Array representing the inverse rotation.
        """
        # For orthogonal matrices, inverse = transpose
        return Rot3Array(
            xx=self.xx, xy=self.yx, xz=self.zx,
            yx=self.xy, yy=self.yy, yz=self.zy,
            zx=self.xz, zy=self.yz, zz=self.zz,
        )

    def apply_to_point(self, point: Vec3Array) -> Vec3Array:
        """Apply rotation to a Vec3Array of points.

        Args:
            point: Points to rotate.

        Returns:
            Rotated points.
        """
        return Vec3Array(
            x=self.xx * point.x + self.xy * point.y + self.xz * point.z,
            y=self.yx * point.x + self.yy * point.y + self.yz * point.z,
            z=self.zx * point.x + self.zy * point.y + self.zz * point.z,
        )

    def apply_inverse_to_point(self, point: Vec3Array) -> Vec3Array:
        """Apply inverse rotation to a Vec3Array of points.

        Args:
            point: Points to rotate.

        Returns:
            Rotated points (by inverse).
        """
        return self.inverse().apply_to_point(point)

    def __matmul__(self, other: Self) -> Self:
        """Compose two rotations: self @ other.

        Args:
            other: Rotation to compose with.

        Returns:
            Composed rotation (self applied after other).
        """
        # Matrix multiplication: result[i,j] = sum_k self[i,k] * other[k,j]
        return Rot3Array(
            xx=self.xx * other.xx + self.xy * other.yx + self.xz * other.zx,
            xy=self.xx * other.xy + self.xy * other.yy + self.xz * other.zy,
            xz=self.xx * other.xz + self.xy * other.yz + self.xz * other.zz,
            yx=self.yx * other.xx + self.yy * other.yx + self.yz * other.zx,
            yy=self.yx * other.xy + self.yy * other.yy + self.yz * other.zy,
            yz=self.yx * other.xz + self.yy * other.yz + self.yz * other.zz,
            zx=self.zx * other.xx + self.zy * other.yx + self.zz * other.zx,
            zy=self.zx * other.xy + self.zy * other.yy + self.zz * other.zy,
            zz=self.zx * other.xz + self.zy * other.yz + self.zz * other.zz,
        )

    # -------------------------------------------------------------------------
    # Conversion
    # -------------------------------------------------------------------------

    def to_array(self) -> mx.array:
        """Convert to dense array of shape [..., 3, 3].

        Returns:
            Array with last two dimensions being the 3x3 rotation matrix.
        """
        # Stack row by row
        row0 = mx.stack([self.xx, self.xy, self.xz], axis=-1)
        row1 = mx.stack([self.yx, self.yy, self.yz], axis=-1)
        row2 = mx.stack([self.zx, self.zy, self.zz], axis=-1)
        return mx.stack([row0, row1, row2], axis=-2)

    @classmethod
    def from_array(cls, array: mx.array) -> Self:
        """Construct from dense array of shape [..., 3, 3].

        Args:
            array: Input array with last two dimensions 3x3.

        Returns:
            Rot3Array with components extracted from matrix.
        """
        if array.shape[-2:] != (3, 3):
            raise ShapeMismatchError(
                f"Expected array with last dimensions (3, 3), got {array.shape[-2:]}",
                expected=(3, 3),
                actual=array.shape[-2:],
            )

        return cls(
            xx=array[..., 0, 0], xy=array[..., 0, 1], xz=array[..., 0, 2],
            yx=array[..., 1, 0], yy=array[..., 1, 1], yz=array[..., 1, 2],
            zx=array[..., 2, 0], zy=array[..., 2, 1], zz=array[..., 2, 2],
        )

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def identity(cls, shape: tuple[int, ...], dtype: mx.Dtype = mx.float32) -> Self:
        """Create identity rotation of given shape.

        Args:
            shape: Shape of each component array.
            dtype: Data type for arrays.

        Returns:
            Rot3Array with all identity rotations.
        """
        ones = mx.ones(shape, dtype=dtype)
        zeros = mx.zeros(shape, dtype=dtype)
        return cls(
            xx=ones, xy=zeros, xz=zeros,
            yx=zeros, yy=ones, yz=zeros,
            zx=zeros, zy=zeros, zz=ones,
        )

    @classmethod
    def from_two_vectors(cls, e0: Vec3Array, e1: Vec3Array) -> Self:
        """Construct rotation aligning e0 to x-axis, e1 in xy-plane.

        This creates a frame where:
        - e0 becomes the x-axis
        - e1 projected onto the xy-plane (orthogonal to e0) becomes y-axis
        - z is e0 cross (orthogonalized e1)

        Args:
            e0: Vector to align to x-axis.
            e1: Vector to align to xy-plane.

        Returns:
            Rot3Array that transforms from the new frame to the original.
        """
        # Normalize e0
        e0_normalized = e0.normalized()

        # Orthogonalize e1 against e0
        dot_e0_e1 = e0_normalized.dot(e1)
        e1_orthogonal = Vec3Array(
            x=e1.x - dot_e0_e1 * e0_normalized.x,
            y=e1.y - dot_e0_e1 * e0_normalized.y,
            z=e1.z - dot_e0_e1 * e0_normalized.z,
        )
        e1_normalized = e1_orthogonal.normalized()

        # e2 = e0 x e1
        e2 = e0_normalized.cross(e1_normalized)

        # The rotation matrix has columns (e0, e1, e2)
        # Matching JAX reference: cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)
        return cls(
            xx=e0_normalized.x, xy=e1_normalized.x, xz=e2.x,
            yx=e0_normalized.y, yy=e1_normalized.y, yz=e2.y,
            zx=e0_normalized.z, zy=e1_normalized.z, zz=e2.z,
        )

    @classmethod
    def from_quaternion(
        cls,
        w: mx.array,
        x: mx.array,
        y: mx.array,
        z: mx.array,
        normalize: bool = True,
        epsilon: float = 1e-6,
    ) -> Self:
        """Construct from quaternion components.

        The quaternion (w, x, y, z) represents a rotation where w is the
        scalar part and (x, y, z) is the vector part.

        Args:
            w: Scalar component.
            x: X component of vector part.
            y: Y component of vector part.
            z: Z component of vector part.
            normalize: Whether to normalize the quaternion.
            epsilon: Small value for normalization stability.

        Returns:
            Rot3Array representing the rotation.
        """
        if normalize:
            inv_norm = mx.rsqrt(mx.maximum(w * w + x * x + y * y + z * z, epsilon))
            w = w * inv_norm
            x = x * inv_norm
            y = y * inv_norm
            z = z * inv_norm

        # Quaternion to rotation matrix formula
        # R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
        #      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
        #      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z

        return cls(
            xx=1.0 - tyy - tzz,
            xy=txy - twz,
            xz=txz + twy,
            yx=txy + twz,
            yy=1.0 - txx - tzz,
            yz=tyz - twx,
            zx=txz - twy,
            zy=tyz + twx,
            zz=1.0 - txx - tyy,
        )

    @classmethod
    def from_svd(cls, mat: mx.array) -> Self:
        """Project arbitrary [..., 9] matrix to valid rotation.

        Uses quaternion-based closed-form algorithm.
        The input matrix should be flattened with shape [..., 9] where
        the 9 elements represent the 3x3 matrix in row-major order.

        This implementation matches the JAX reference exactly by:
        1. Using the same 16x9 factor matrix
        2. Using the same einsum contraction
        3. Reshaping to 4x4 symmetric matrix
        4. Finding largest eigenvector and converting to quaternion

        Args:
            mat: Input matrix with shape [..., 9].

        Returns:
            Rot3Array representing the closest valid rotation matrix.
        """
        if mat.shape[-1] != 9:
            raise ShapeMismatchError(
                f"Expected array with last dimension 9, got {mat.shape[-1]}",
                expected=(9,),
                actual=(mat.shape[-1],),
            )

        batch_shape = mat.shape[:-1]
        dtype = mat.dtype

        # Cast to float32 for eigendecomposition
        mat_f32 = mat.astype(mx.float32)

        # Compute symmetric 4x4 matrix using the same einsum as JAX:
        # 'ji, ...i -> ...j' contracts the last dimension of mat with
        # the second dimension of factors (transposed)
        factors = MATRIX_SVD_QUAT_FACTORS.astype(mx.float32)

        # This produces [..., 16] output - one value per cell of 4x4 matrix
        symmetric_flat = mx.einsum("ji,...i->...j", factors, mat_f32)

        # Reshape to [..., 4, 4]
        sym_matrix = mx.reshape(symmetric_flat, batch_shape + (4, 4))

        # Find largest eigenvector (eigh is CPU-only in MLX)
        eigenvalues, eigenvectors = mx.linalg.eigh(sym_matrix, stream=mx.cpu)

        # The largest eigenvalue is at the end (eigh returns in ascending order)
        # Get the corresponding eigenvector (quaternion)
        # eigenvectors shape: [..., 4, 4], last column is largest eigenvector
        quat = eigenvectors[..., -1]  # Shape: [..., 4]

        # Extract quaternion components (w, x, y, z)
        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]

        # Convert to rotation matrix
        result = cls.from_quaternion(w, x, y, z, normalize=True)

        # For consistency with JAX, return the inverse
        result = result.inverse()

        # Cast back to original dtype
        if dtype != mx.float32:
            result = cls(
                xx=result.xx.astype(dtype),
                xy=result.xy.astype(dtype),
                xz=result.xz.astype(dtype),
                yx=result.yx.astype(dtype),
                yy=result.yy.astype(dtype),
                yz=result.yz.astype(dtype),
                zx=result.zx.astype(dtype),
                zy=result.zy.astype(dtype),
                zz=result.zz.astype(dtype),
            )

        return result

    @classmethod
    def random_uniform(
        cls, key: mx.array, shape: tuple[int, ...], dtype: mx.Dtype = mx.float32
    ) -> Self:
        """Sample uniform random rotation according to Haar measure.

        Uses the quaternion uniform sampling method: sample 4 Gaussians
        and normalize to get a uniform quaternion on the unit sphere.

        Args:
            key: MLX random key.
            shape: Shape of each component array.
            dtype: Data type for arrays.

        Returns:
            Rot3Array with uniformly distributed rotations.
        """
        # Sample 4 independent Gaussians
        k1, k2, k3, k4 = mx.random.split(key, 4)

        w = mx.random.normal(shape=shape, key=k1, dtype=mx.float32)
        x = mx.random.normal(shape=shape, key=k2, dtype=mx.float32)
        y = mx.random.normal(shape=shape, key=k3, dtype=mx.float32)
        z = mx.random.normal(shape=shape, key=k4, dtype=mx.float32)

        # Convert to rotation (normalizes automatically)
        result = cls.from_quaternion(w, x, y, z, normalize=True)

        # Cast to requested dtype
        if dtype != mx.float32:
            result = cls(
                xx=result.xx.astype(dtype),
                xy=result.xy.astype(dtype),
                xz=result.xz.astype(dtype),
                yx=result.yx.astype(dtype),
                yy=result.yy.astype(dtype),
                yz=result.yz.astype(dtype),
                zx=result.zx.astype(dtype),
                zy=result.zy.astype(dtype),
                zz=result.zz.astype(dtype),
            )

        return result
