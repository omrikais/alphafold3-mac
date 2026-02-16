"""AlphaFold 3 MLX Weight Loading API.

This module provides utilities for loading AlphaFold 3 model weights
and converting them to MLX format for inference on Apple Silicon.

Public API:
    - load_mlx_params: Load and convert weights to MLX format
    - get_platform_info: Get platform compatibility information
    - convert_array: Convert a single array to MLX format
    - PlatformInfo: Platform information dataclass
    - WeightConversionResult: Weight conversion result dataclass

Exceptions:
    - UnsupportedDtypeError: Raised when dtype can't be converted
    - WeightsNotFoundError: Raised when weight files are not found
    - CorruptedWeightsError: Raised when weight files are corrupted
    - ShapeMismatchError: Raised when weight shapes don't match
    - PlatformError: Raised when running on unsupported platform
"""

from alphafold3_mlx import PlatformError
from alphafold3_mlx.weights.converter import UnsupportedDtypeError, convert_array
from alphafold3_mlx.weights.loader import (
    CorruptedWeightsError,
    ShapeMismatchError,
    WeightConversionResult,
    WeightsNotFoundError,
    load_mlx_params,
)
from alphafold3_mlx.weights.platform import (
    PlatformInfo,
    get_platform_info,
    validate_platform_for_cli,
)

__all__ = [
    # Functions
    "load_mlx_params",
    "get_platform_info",
    "validate_platform_for_cli",
    "convert_array",
    # Data classes
    "PlatformInfo",
    "WeightConversionResult",
    # Exceptions
    "UnsupportedDtypeError",
    "WeightsNotFoundError",
    "CorruptedWeightsError",
    "ShapeMismatchError",
    "PlatformError",
]
