"""Platform detection utilities for AlphaFold 3 MLX.

This module provides utilities for detecting Apple Silicon chip information
and determining platform capabilities like bfloat16 support.
"""

from __future__ import annotations

import os
import platform
import subprocess
import warnings
from dataclasses import dataclass

# Import PlatformError from parent package for consistency
from alphafold3_mlx import PlatformError


@dataclass(frozen=True)
class PlatformInfo:
    """System platform information for compatibility checking.

    Attributes:
        system: Operating system name ("Darwin" for macOS).
        machine: CPU architecture ("arm64" for Apple Silicon).
        chip_family: Apple Silicon generation: "M1", "M2", "M3", "M4", or "Unknown".
        supports_bfloat16: Whether chip supports efficient bfloat16 (M3+).
        memory_gb: Unified memory in GB.
    """

    system: str
    machine: str
    chip_family: str
    supports_bfloat16: bool
    memory_gb: int


def _get_chip_family() -> str:
    """Detect Apple Silicon chip family from system information.

    Returns:
        Chip family string: "M1", "M2", "M3", "M4", or "Unknown".
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        brand_string = result.stdout.strip()

        # Parse chip family from brand string
        # Examples: "Apple M2 Max", "Apple M3 Pro", "Apple M4"
        if "M4" in brand_string:
            return "M4"
        elif "M3" in brand_string:
            return "M3"
        elif "M2" in brand_string:
            return "M2"
        elif "M1" in brand_string:
            return "M1"
        else:
            return "Unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # M-03: Fallback via platform.processor() when sysctl fails (sandbox)
        try:
            proc = platform.processor()
            if proc and "Apple" in proc:
                for family in ("M4", "M3", "M2", "M1"):
                    if family in proc:
                        return family
        except Exception:
            pass
        return "Unknown"


def _get_memory_gb() -> int:
    """Get system unified memory in GB.

    Returns:
        Memory in GB (integer).
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        mem_bytes = int(result.stdout.strip())
        return mem_bytes // (1024**3)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def get_platform_info() -> PlatformInfo:
    """Get information about the current platform for compatibility checking.

    Returns:
        PlatformInfo with system details.

    Raises:
        PlatformError: If not running on macOS ARM64.
    """
    import sys

    system = platform.system()
    machine = platform.machine()

    # Validate platform
    if sys.platform != "darwin":
        raise PlatformError(
            f"alphafold3_mlx requires macOS. Current platform: {sys.platform}"
        )
    if machine != "arm64":
        raise PlatformError(
            "alphafold3_mlx requires Apple Silicon (ARM64). "
            f"Current architecture: {machine}"
        )

    chip_family = _get_chip_family()
    supports_bfloat16 = chip_family in ("M3", "M4")
    memory_gb = _get_memory_gb()

    return PlatformInfo(
        system=system,
        machine=machine,
        chip_family=chip_family,
        supports_bfloat16=supports_bfloat16,
        memory_gb=memory_gb,
    )


def validate_platform_for_cli() -> PlatformInfo:
    """Validate platform for CLI usage.

    M2/M3/M4 recommended. M1 and unknown chips emit a warning instead
    of a hard error (H-02). Set ``AF3_SKIP_PLATFORM_CHECK=1`` to
    suppress all platform checks.

    Returns:
        PlatformInfo if platform is valid.

    Raises:
        PlatformError: If not running on macOS ARM64 (unless bypassed).
    """
    import sys

    # H-02: Allow env-var bypass
    if os.environ.get("AF3_SKIP_PLATFORM_CHECK") == "1":
        chip_family = _get_chip_family() if sys.platform == "darwin" else "Unknown"
        return PlatformInfo(
            system=platform.system(),
            machine=platform.machine(),
            chip_family=chip_family,
            supports_bfloat16=chip_family in ("M3", "M4"),
            memory_gb=_get_memory_gb() if sys.platform == "darwin" else 0,
        )

    system = platform.system()
    machine = platform.machine()

    # First check: must be macOS
    if sys.platform != "darwin":
        raise PlatformError(
            f"This tool requires macOS with Apple Silicon. Detected: {sys.platform}. "
            "Set AF3_SKIP_PLATFORM_CHECK=1 to override."
        )

    # Second check: must be ARM64
    if machine != "arm64":
        raise PlatformError(
            f"This tool requires Apple Silicon (ARM64). Detected: {machine}. "
            "Set AF3_SKIP_PLATFORM_CHECK=1 to override."
        )

    # Third check: M2/M3/M4 recommended, but M1 and Unknown are warnings
    chip_family = _get_chip_family()
    if chip_family not in ("M2", "M3", "M4"):
        if chip_family == "M1":
            warnings.warn(
                "Detected M1. M2/M3/M4 recommended for best performance. "
                "Set AF3_SKIP_PLATFORM_CHECK=1 to suppress this warning.",
                stacklevel=2,
            )
        elif chip_family == "Unknown":
            warnings.warn(
                "Could not detect Apple Silicon chip family. Proceeding anyway. "
                "Set AF3_SKIP_PLATFORM_CHECK=1 to suppress this warning.",
                stacklevel=2,
            )

    supports_bfloat16 = chip_family in ("M3", "M4")
    memory_gb = _get_memory_gb()

    return PlatformInfo(
        system=system,
        machine=machine,
        chip_family=chip_family,
        supports_bfloat16=supports_bfloat16,
        memory_gb=memory_gb,
    )
