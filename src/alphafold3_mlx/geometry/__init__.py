"""MLX geometry primitives for AlphaFold 3.

This module provides struct-of-arrays implementations of 3D geometric primitives
(Vec3Array, Rot3Array) that match the JAX geometry API for numerical parity.
"""

from __future__ import annotations

# Import exceptions first (no circular import issues)
from alphafold3_mlx.geometry.exceptions import (
    GeometryError,
    ShapeMismatchError,
    DtypeMismatchError,
)

# Import Vec3Array and functions
from alphafold3_mlx.geometry.vector import (
    Vec3Array,
    square_euclidean_distance,
    euclidean_distance,
    dihedral_angle,
    random_gaussian_vector,
    dot,
    cross,
    norm,
    normalized,
)

from alphafold3_mlx.geometry.rotation_matrix import Rot3Array

__all__ = [
    # Exceptions
    "GeometryError",
    "ShapeMismatchError",
    "DtypeMismatchError",
    # Vec3Array and functions
    "Vec3Array",
    "square_euclidean_distance",
    "euclidean_distance",
    "dihedral_angle",
    "random_gaussian_vector",
    "dot",
    "cross",
    "norm",
    "normalized",
    # Rot3Array
    "Rot3Array",
]
