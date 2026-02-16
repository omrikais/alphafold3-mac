"""Weight loading utilities for AlphaFold 3 MLX.

This module provides utilities for loading AlphaFold 3 model weights
from the official format (zstd-compressed records) and converting
them to MLX arrays for inference on Apple Silicon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


class WeightsNotFoundError(FileNotFoundError):
    """Raised when weight files are not found.

    Subclasses built-in FileNotFoundError for compatibility.
    """

    pass


class CorruptedWeightsError(Exception):
    """Raised when weight files are corrupted or invalid."""

    pass


class ShapeMismatchError(Exception):
    """Raised when weight shapes don't match expected model architecture.

    Attributes:
        mismatches: List of (path, expected_shape, actual_shape) tuples.
    """

    def __init__(self, message: str, mismatches: list[tuple[str, tuple, tuple]] | None = None):
        super().__init__(message)
        self.mismatches = mismatches or []


@dataclass
class WeightConversionResult:
    """Result of converting Haiku params to MLX format with metadata.

    Attributes:
        params: Converted weights as nested dict of MLX arrays.
        source_path: Path to original weight file location.
        num_parameters: Total number of parameter tensors.
        total_bytes: Total memory footprint in bytes.
        dtype_distribution: Count of arrays per dtype.
        conversion_time_ms: Time taken to convert in milliseconds.
    """

    params: dict[str, dict[str, "mx.array"]]
    source_path: Path
    num_parameters: int
    total_bytes: int
    dtype_distribution: dict[str, int]
    conversion_time_ms: float


def _count_parameters(params: dict) -> tuple[int, int, dict[str, int]]:
    """Count parameters and compute statistics.

    Args:
        params: Nested dict of MLX arrays.

    Returns:
        Tuple of (num_parameters, total_bytes, dtype_distribution).
    """
    num_params = 0
    total_bytes = 0
    dtype_dist: dict[str, int] = {}

    def _visit(tree: dict) -> None:
        nonlocal num_params, total_bytes
        for value in tree.values():
            if isinstance(value, dict):
                _visit(value)
            else:
                # It's an MLX array
                num_params += 1
                total_bytes += value.nbytes
                dtype_name = str(value.dtype)
                dtype_dist[dtype_name] = dtype_dist.get(dtype_name, 0) + 1

    _visit(params)
    return num_params, total_bytes, dtype_dist


def load_mlx_params(
    model_dir: Path | str,
    *,
    dtype: "mx.Dtype | None" = None,
    validate: bool = True,
) -> WeightConversionResult:
    """Load AlphaFold 3 weights from official format and convert to MLX.

    This function loads weights using the existing params.py module from
    the original AlphaFold 3 codebase, then converts the JAX arrays to
    MLX format for inference on Apple Silicon.

    Args:
        model_dir: Path to directory containing weight files (.bin.zst).
        dtype: Optional dtype to convert all weights to (None preserves original).
        validate: Whether to validate shapes match expected model architecture.

    Returns:
        WeightConversionResult with converted params and metadata.

    Raises:
        WeightsNotFoundError: If model_dir doesn't exist or has no weights.
        CorruptedWeightsError: If weight files are truncated or invalid.
        ImportError: If MLX is not installed.
        ShapeMismatchError: If validate=True and shapes don't match.
        PlatformError: If not running on macOS ARM64.

    Example:
        >>> from pathlib import Path
        >>> from alphafold3_mlx.weights import load_mlx_params
        >>>
        >>> result = load_mlx_params(Path("~/af3_weights"))
        >>> print(f"Loaded {result.num_parameters} parameters")
        >>> print(f"Memory: {result.total_bytes / 1e9:.2f} GB")
    """
    # Check MLX is available
    try:
        import mlx.core as mx  # noqa: F401
    except ImportError:
        raise ImportError(
            "MLX required for weight conversion. "
            "Install with: pip install 'alphafold3[mlx]'"
        )

    from alphafold3_mlx.weights.converter import _convert_tree

    model_dir = Path(model_dir).expanduser()

    if not model_dir.exists():
        raise WeightsNotFoundError(f"Model directory not found: {model_dir}")

    # Check for weight files - match all patterns that params.select_model_files accepts
    # Patterns: *.bin.zst, *.bin, *.N.bin.zst, *.bin.zst.N, *.N.bin
    weight_patterns = ["*.bin.zst", "*.bin", "*.bin.zst.*", "*.*.bin.zst", "*.*.bin"]
    weight_files = []
    for pattern in weight_patterns:
        weight_files.extend(model_dir.glob(pattern))
    if not weight_files:
        raise WeightsNotFoundError(
            f"No weight files found in {model_dir}. "
            "Expected: *.bin.zst, *.bin, or sharded variants (*.N.bin.zst, *.bin.zst.N)"
        )

    start_time = time.perf_counter()

    try:
        # Import params module from original AlphaFold 3
        from alphafold3.model import params

        # Load Haiku params using the original loading code
        haiku_params = params.get_model_haiku_params(model_dir)
    except Exception as e:
        # Wrap any loading error as CorruptedWeightsError
        if "truncated" in str(e).lower() or "corrupt" in str(e).lower():
            raise CorruptedWeightsError(f"Weight files appear corrupted: {e}")
        raise

    # Convert to MLX arrays
    mlx_params = _convert_tree(haiku_params, dtype)

    end_time = time.perf_counter()
    conversion_time_ms = (end_time - start_time) * 1000

    # Compute statistics
    num_parameters, total_bytes, dtype_distribution = _count_parameters(mlx_params)

    # Validate shapes if requested - compare source vs converted
    if validate:
        _validate_shapes(haiku_params, mlx_params)

    return WeightConversionResult(
        params=mlx_params,
        source_path=model_dir,
        num_parameters=num_parameters,
        total_bytes=total_bytes,
        dtype_distribution=dtype_distribution,
        conversion_time_ms=conversion_time_ms,
    )


def _validate_shapes(source_params: dict, converted_params: dict) -> None:
    """Validate that converted weight shapes exactly match source shapes.

    This enforces: 0% shape mismatch tolerance.

    Args:
        source_params: Original Haiku params (nested dict of JAX arrays).
        converted_params: Converted MLX params (nested dict of MLX arrays).

    Raises:
        ShapeMismatchError: If any shapes don't match exactly.
    """
    import numpy as np

    mismatches: list[tuple[str, tuple, tuple]] = []

    def _compare(source: dict, converted: dict, path: str = "") -> None:
        # Check for missing keys
        source_keys = set(source.keys())
        converted_keys = set(converted.keys())

        for key in source_keys - converted_keys:
            current_path = f"{path}/{key}" if path else key
            mismatches.append((current_path, "present", "missing"))

        for key in converted_keys - source_keys:
            current_path = f"{path}/{key}" if path else key
            mismatches.append((current_path, "missing", "present"))

        # Compare matching keys
        for key in source_keys & converted_keys:
            current_path = f"{path}/{key}" if path else key
            src_val = source[key]
            conv_val = converted[key]

            if isinstance(src_val, dict):
                if isinstance(conv_val, dict):
                    _compare(src_val, conv_val, current_path)
                else:
                    mismatches.append((current_path, "dict", "array"))
            else:
                if isinstance(conv_val, dict):
                    mismatches.append((current_path, "array", "dict"))
                else:
                    # Both are arrays - compare shapes
                    src_shape = tuple(np.asarray(src_val).shape)
                    conv_shape = tuple(conv_val.shape)
                    if src_shape != conv_shape:
                        mismatches.append((current_path, src_shape, conv_shape))

    _compare(source_params, converted_params)

    if mismatches:
        mismatch_details = "\n".join(
            f"  {path}: expected {expected}, got {actual}"
            for path, expected, actual in mismatches[:10]  # Show first 10
        )
        total = len(mismatches)
        msg = f"Shape validation failed with {total} mismatch(es):\n{mismatch_details}"
        if total > 10:
            msg += f"\n  ... and {total - 10} more"
        raise ShapeMismatchError(msg, mismatches=mismatches)
