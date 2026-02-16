"""Vec3Array struct-of-arrays implementation for MLX.

This module provides the Vec3Array class representing batched 3D vectors
in a struct-of-arrays format, matching the JAX geometry API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import mlx.core as mx

# Import exceptions from the exceptions module to avoid circular import
# The exceptions are defined in __init__.py and re-exported here
from alphafold3_mlx.geometry.exceptions import ShapeMismatchError, DtypeMismatchError

if TYPE_CHECKING:
    from typing import TypeAlias

    Float: TypeAlias = float | mx.array
    Shape: TypeAlias = tuple[int, ...]

__all__ = [
    "Vec3Array",
    "square_euclidean_distance",
    "euclidean_distance",
    "dihedral_angle",
    "random_gaussian_vector",
    "dot",
    "cross",
    "norm",
    "normalized",
]


@dataclass(frozen=True)
class Vec3Array:
    """Struct-of-arrays representing batched 3D vectors.

    All components (x, y, z) must have identical shape and dtype.
    Uses frozen dataclass for immutability.

    Attributes:
        x: X component array.
        y: Y component array.
        z: Z component array.

    Raises:
        ShapeMismatchError: If component shapes differ.
        DtypeMismatchError: If component dtypes differ.

    Example:
        >>> v = Vec3Array(
        ...     x=mx.array([1.0, 2.0]),
        ...     y=mx.array([3.0, 4.0]),
        ...     z=mx.array([5.0, 6.0]),
        ... )
        >>> v.shape
        (2,)
    """

    x: mx.array
    y: mx.array
    z: mx.array

    def __post_init__(self) -> None:
        """Validate that all components have same shape and dtype."""
        # Check shapes
        if not (self.x.shape == self.y.shape == self.z.shape):
            # Determine which one differs
            expected = self.x.shape
            for arr, name in [(self.y, "y"), (self.z, "z")]:
                if arr.shape != expected:
                    raise ShapeMismatchError(
                        f"Vec3Array component '{name}' has shape {arr.shape}, "
                        f"expected {expected} (matching x)",
                        expected=expected,
                        actual=arr.shape,
                    )

        # Check dtypes
        if not (self.x.dtype == self.y.dtype == self.z.dtype):
            raise DtypeMismatchError(
                f"Vec3Array components have inconsistent dtypes: "
                f"x={self.x.dtype}, y={self.y.dtype}, z={self.z.dtype}",
                dtypes=[self.x.dtype, self.y.dtype, self.z.dtype],
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Common shape of all component arrays."""
        return self.x.shape

    @property
    def dtype(self) -> mx.Dtype:
        """Common dtype of all component arrays."""
        return self.x.dtype

    # -------------------------------------------------------------------------
    # Arithmetic operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Self) -> Self:
        """Component-wise addition: self + other."""
        return Vec3Array(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other: Self) -> Self:
        """Component-wise subtraction: self - other."""
        return Vec3Array(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def __mul__(self, other: float | mx.array) -> Self:
        """Scalar or array multiplication: self * other."""
        return Vec3Array(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
        )

    def __rmul__(self, other: float | mx.array) -> Self:
        """Reverse scalar multiplication: other * self."""
        return self.__mul__(other)

    def __truediv__(self, other: float | mx.array) -> Self:
        """Scalar or array division: self / other."""
        return Vec3Array(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
        )

    def __neg__(self) -> Self:
        """Negation: -self."""
        return Vec3Array(
            x=-self.x,
            y=-self.y,
            z=-self.z,
        )

    def __pos__(self) -> Self:
        """Positive: +self (returns self, immutable so no copy needed)."""
        # MLX arrays don't support unary +, but since Vec3Array is frozen/immutable,
        # we can just return self
        return self

    # -------------------------------------------------------------------------
    # Geometric operations
    # -------------------------------------------------------------------------

    def dot(self, other: Self) -> mx.array:
        """Compute dot product between self and other.

        Args:
            other: Another Vec3Array with matching shape.

        Returns:
            Array of dot products with same shape as input.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Self) -> Self:
        """Compute cross product between self and other.

        Args:
            other: Another Vec3Array with matching shape.

        Returns:
            Vec3Array of cross products.
        """
        return Vec3Array(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def norm2(self) -> mx.array:
        """Compute squared L2 norm.

        Returns:
            Array of squared norms with same shape as input.
        """
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self, epsilon: float = 1e-6) -> mx.array:
        """Compute L2 norm, clipped to epsilon to avoid NaN.

        Args:
            epsilon: Minimum norm value (output is clipped to be >= epsilon).

        Returns:
            Array of norms with same shape as input.
        """
        # JAX semantics: epsilon is a norm threshold, so compare norm2 to epsilon**2
        norm2 = self.norm2()
        if epsilon:
            norm2 = mx.maximum(norm2, epsilon**2)
        return mx.sqrt(norm2)

    def normalized(self, epsilon: float = 1e-6) -> Self:
        """Return unit vector with optional epsilon clipping.

        Matches JAX semantics: simply divides by norm(epsilon).
        For zero vectors (norm < epsilon), the norm is clipped to epsilon,
        so the result is the original vector scaled by 1/epsilon.

        Args:
            epsilon: Minimum norm value to avoid division by zero.

        Returns:
            Vec3Array of unit vectors.
        """
        return self / self.norm(epsilon)

    # -------------------------------------------------------------------------
    # Conversion methods
    # -------------------------------------------------------------------------

    def to_array(self) -> mx.array:
        """Convert to dense array of shape [..., 3].

        Returns:
            Array with last dimension 3 containing [x, y, z].
        """
        return mx.stack([self.x, self.y, self.z], axis=-1)

    @classmethod
    def from_array(cls, array: mx.array) -> Self:
        """Construct from dense array of shape [..., 3].

        Args:
            array: Input array with last dimension 3.

        Returns:
            Vec3Array with components split from last dimension.

        Raises:
            ShapeMismatchError: If last dimension is not 3.
        """
        if array.shape[-1] != 3:
            raise ShapeMismatchError(
                f"Expected array with last dimension 3, got {array.shape[-1]}",
                expected=(3,),
                actual=(array.shape[-1],),
            )

        # Split along last axis
        x = array[..., 0]
        y = array[..., 1]
        z = array[..., 2]

        return cls(x=x, y=y, z=z)

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: mx.Dtype = mx.float32) -> Self:
        """Create zero vectors of given shape and dtype.

        Args:
            shape: Shape of each component array.
            dtype: Data type for arrays.

        Returns:
            Vec3Array with all components zero.
        """
        return cls(
            x=mx.zeros(shape, dtype=dtype),
            y=mx.zeros(shape, dtype=dtype),
            z=mx.zeros(shape, dtype=dtype),
        )


# =============================================================================
# Standalone functions
# =============================================================================


def square_euclidean_distance(
    vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> mx.array:
    """Compute squared Euclidean distance between two Vec3Arrays.

    Args:
        vec1: First Vec3Array.
        vec2: Second Vec3Array.
        epsilon: Minimum value (currently unused for squared distance).

    Returns:
        Array of squared distances.
    """
    diff = vec1 - vec2
    return diff.norm2()


def euclidean_distance(
    vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> mx.array:
    """Compute Euclidean distance between two Vec3Arrays.

    Args:
        vec1: First Vec3Array.
        vec2: Second Vec3Array.
        epsilon: Minimum squared distance to avoid sqrt(0).

    Returns:
        Array of distances.
    """
    diff = vec1 - vec2
    return diff.norm(epsilon=epsilon)


def dihedral_angle(
    a: Vec3Array, b: Vec3Array, c: Vec3Array, d: Vec3Array
) -> mx.array:
    """Compute dihedral (torsion) angle for four points.

    The dihedral angle is the angle between the plane containing (a, b, c)
    and the plane containing (b, c, d).

    Args:
        a: First point.
        b: Second point (axis start).
        c: Third point (axis end).
        d: Fourth point.

    Returns:
        Array of angles in radians, range [-pi, pi].
    """
    # Vectors along the backbone
    b1 = b - a  # a -> b
    b2 = c - b  # b -> c
    b3 = d - c  # c -> d

    # Normal vectors to the planes
    n1 = b1.cross(b2)
    n2 = b2.cross(b3)

    # Normalize b2 for the frame
    b2_normalized = b2.normalized()

    # m1 is perpendicular to n1 in the plane containing n1 and b2
    m1 = n1.cross(b2_normalized)

    # Compute atan2(y, x) where:
    # x = n1 . n2
    # y = m1 . n2
    x = n1.dot(n2)
    y = m1.dot(n2)

    return mx.arctan2(y, x)


def random_gaussian_vector(
    shape: tuple[int, ...], key: mx.array, dtype: mx.Dtype = mx.float32
) -> Vec3Array:
    """Sample random Vec3Array from standard normal distribution.

    Args:
        shape: Shape of each component array.
        key: MLX random key.
        dtype: Data type for arrays.

    Returns:
        Vec3Array with components sampled from N(0, 1).
    """
    # Split key for each component
    k1, k2, k3 = mx.random.split(key, 3)

    # MLX random API uses keyword arguments
    return Vec3Array(
        x=mx.random.normal(shape=shape, dtype=dtype, key=k1),
        y=mx.random.normal(shape=shape, dtype=dtype, key=k2),
        z=mx.random.normal(shape=shape, dtype=dtype, key=k3),
    )


# =============================================================================
# Convenience aliases
# =============================================================================


def dot(vector1: Vec3Array, vector2: Vec3Array) -> mx.array:
    """Compute dot product between two Vec3Arrays.

    Convenience function that calls vector1.dot(vector2).
    """
    return vector1.dot(vector2)


def cross(vector1: Vec3Array, vector2: Vec3Array) -> Vec3Array:
    """Compute cross product between two Vec3Arrays.

    Convenience function that calls vector1.cross(vector2).
    """
    return vector1.cross(vector2)


def norm(vector: Vec3Array, epsilon: float = 1e-6) -> mx.array:
    """Compute L2 norm of a Vec3Array.

    Convenience function that calls vector.norm(epsilon).
    """
    return vector.norm(epsilon=epsilon)


def normalized(vector: Vec3Array, epsilon: float = 1e-6) -> Vec3Array:
    """Compute unit vector of a Vec3Array.

    Convenience function that calls vector.normalized(epsilon).
    """
    return vector.normalized(epsilon=epsilon)
