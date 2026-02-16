"""AlphaFold 3 MLX - Apple Silicon port of AlphaFold 3.

This package provides MLX implementations for AlphaFold 3 inference on Apple Silicon.
"""

import os
import platform
import sys

__version__ = "3.0.1"


class PlatformError(Exception):
    """Raised when running on an unsupported platform.

    This error is raised if the package is used on a non-macOS or
    non-ARM64 system without setting the skip-check env var.
    """

    pass


def _check_platform() -> None:
    """Check that we're running on a supported platform (macOS ARM64).

    Skipped when ``AF3_SKIP_PLATFORM_CHECK=1`` is set.

    Raises:
        PlatformError: If not running on macOS or not on ARM64 architecture.
    """
    if os.environ.get("AF3_SKIP_PLATFORM_CHECK") == "1":
        return
    if sys.platform != "darwin":
        raise PlatformError(
            f"alphafold3_mlx requires macOS. Current platform: {sys.platform}. "
            "Set AF3_SKIP_PLATFORM_CHECK=1 to override."
        )
    if platform.machine() != "arm64":
        raise PlatformError(
            "alphafold3_mlx requires Apple Silicon (ARM64). "
            f"Current architecture: {platform.machine()}. "
            "Set AF3_SKIP_PLATFORM_CHECK=1 to override."
        )


# Only check platform when not explicitly skipped
_check_platform()

# Import subpackages — guarded so non-macOS imports with skip flag don't crash
try:
    import mlx  # noqa: F401 — verify MLX is importable
except ImportError:
    if os.environ.get("AF3_SKIP_PLATFORM_CHECK") != "1":
        raise ImportError(
            "MLX is required but not installed. "
            "Install with: pip install 'mlx>=0.10.0'"
        )
    # Allow import on non-macOS for testing/CI with skip flag
    mlx = None  # type: ignore[assignment]

_MLX_AVAILABLE = mlx is not None

if _MLX_AVAILABLE:
    from alphafold3_mlx import geometry  # noqa: E402
    from alphafold3_mlx import modules  # noqa: E402
    from alphafold3_mlx import network  # noqa: E402
    from alphafold3_mlx import model  # noqa: E402
    from alphafold3_mlx import core  # noqa: E402

    # Convenience exports for common usage
    from alphafold3_mlx.model import Model  # noqa: E402
    from alphafold3_mlx.core import ModelConfig, FeatureBatch, ModelResult  # noqa: E402

__all__ = [
    "__version__",
    "PlatformError",
    "_MLX_AVAILABLE",
    # Subpackages (only available when MLX is installed)
    "geometry",
    "modules",
    "network",
    "model",
    "core",
    # Main exports
    "Model",
    "ModelConfig",
    "FeatureBatch",
    "ModelResult",
]
