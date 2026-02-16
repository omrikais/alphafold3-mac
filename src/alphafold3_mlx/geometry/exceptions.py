"""Exception classes for geometry module.

This module defines the exception hierarchy used by geometry primitives.
Separated from __init__.py to avoid circular imports.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "GeometryError",
    "ShapeMismatchError",
    "DtypeMismatchError",
]


class GeometryError(Exception):
    """Base exception for geometry module errors."""

    pass


class ShapeMismatchError(GeometryError):
    """Raised when array shapes don't match expected dimensions."""

    def __init__(self, message: str, expected: tuple[int, ...], actual: tuple[int, ...]) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class DtypeMismatchError(GeometryError):
    """Raised when array dtypes are inconsistent."""

    def __init__(self, message: str, dtypes: list[mx.Dtype]) -> None:
        super().__init__(message)
        self.dtypes = dtypes
