"""Validation utilities for AlphaFold 3 MLX.

This module provides:
- ValidationResult: Cross-backend comparison result (Phase 0)
- estimate_peak_memory_gb: Memory estimation for OOM prevention
- check_memory_requirements: Pre-execution memory check
- check_nan: NaN detection at validation checkpoints
- NaNCheckpoint: Context manager for NaN detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from alphafold3_mlx.core.config import AttentionConfig


@dataclass
class ValidationResult:
    """Result of numerical validation between MLX and JAX outputs.

    Attributes:
        passed: True if all comparisons passed within tolerance
        rtol: Relative tolerance used
        atol: Absolute tolerance used
        max_abs_diff: Maximum absolute difference (populated on failure or for reporting)
        mean_abs_diff: Mean absolute difference (populated on failure or for reporting)
        tensor_results: Per-tensor pass/fail breakdown
        config: Optional configuration used for the test
    """

    passed: bool
    rtol: float
    atol: float
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    tensor_results: dict[str, bool] = field(default_factory=dict)
    config: AttentionConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        result = {
            "passed": self.passed,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "tensor_results": self.tensor_results,
        }
        if self.config is not None:
            result["config"] = self.config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        config = None
        if "config" in data and data["config"] is not None:
            config = AttentionConfig.from_dict(data["config"])
        return cls(
            passed=data["passed"],
            rtol=data["rtol"],
            atol=data["atol"],
            max_abs_diff=data.get("max_abs_diff"),
            mean_abs_diff=data.get("mean_abs_diff"),
            tensor_results=data.get("tensor_results", {}),
            config=config,
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "PASS" if self.passed else "FAIL"
        msg = f"ValidationResult({status}, rtol={self.rtol:.0e}, atol={self.atol:.0e})"
        if not self.passed and self.max_abs_diff is not None:
            msg += f" max_diff={self.max_abs_diff:.2e}"
        return msg


# =============================================================================
# Memory Estimation (Phase 3)
# =============================================================================


def estimate_peak_memory_gb(
    num_residues: int,
    num_samples: int = 5,
    pair_channel: int = 128,
    seq_channel: int = 384,
    num_heads: int = 4,
    max_atoms: int = 37,
    diffusion_heads: int = 16,
) -> float:
    """Estimate peak memory usage in GB.

    This provides a conservative estimate for pre-execution OOM prevention.
    The estimate accounts for:
    - Evoformer:
      - Pair representation: O(N² × pair_channel × 4 bytes)
      - Single representation: O(N × seq_channel × 4 bytes)
      - Attention logits: O(N² × heads × 4 bytes)
    - Diffusion (atom37 coordinates):
      - Coordinate samples: O(samples × N × 37 × 3 × 4 bytes)
      - Atom-level attention: O((N×37)² × diffusion_heads × 4 bytes) - DOMINANT
      - Pair bias expansion: O((N×37)² × diffusion_heads × 4 bytes)
    - Safety factor for activations (2x)

    Args:
        num_residues: Number of residues in the input.
        num_samples: Number of structure samples (default: 5).
        pair_channel: Pair representation channels (default: 128).
        seq_channel: Single representation channels (default: 384).
        num_heads: Number of Evoformer attention heads (default: 4).
        max_atoms: Maximum atoms per residue (default: 37).
        diffusion_heads: Number of diffusion transformer heads (default: 16).

    Returns:
        Estimated peak memory usage in GB.
    """
    # =========================================================================
    # Evoformer memory (residue-level)
    # =========================================================================
    # Pair representation: N² × pair_channel × 4 bytes
    pair_mem = num_residues ** 2 * pair_channel * 4 / 1e9

    # Single representation: N × seq_channel × 4 bytes
    single_mem = num_residues * seq_channel * 4 / 1e9

    # Evoformer attention logits: N² × heads × 4 bytes
    evoformer_attention_mem = num_residues ** 2 * num_heads * 4 / 1e9

    # =========================================================================
    # Diffusion memory (atom-level)
    # =========================================================================
    num_atoms = num_residues * max_atoms  # N × 37

    # Coordinate samples: samples × N × max_atoms × 3 × 4 bytes
    coords_mem = num_samples * num_residues * max_atoms * 3 * 4 / 1e9

    # Diffusion transformer hidden states: N×37 × hidden_dim (384) × 4 bytes × 2 (in+out)
    diffusion_hidden_mem = num_atoms * 384 * 4 * 2 / 1e9

    # MLX uses SDPA (scaled dot-product attention) which is memory-efficient
    # It doesn't materialize the full (N×37)² attention matrix
    # Instead, it processes in blocks, so memory is O(N×37 × heads × head_dim)
    # QKV projections: 3 × N×37 × heads × head_dim × 4 bytes
    head_dim = 24  # Typical head dimension
    diffusion_qkv_mem = 3 * num_atoms * diffusion_heads * head_dim * 4 / 1e9

    # Note: Pair bias removed from diffusion to avoid O((N×37)²) memory explosion
    # Conditioning comes from single_cond (AdaLN) which already encodes pair info

    # =========================================================================
    # Total with safety factor
    # =========================================================================
    evoformer_total = pair_mem + single_mem + evoformer_attention_mem
    diffusion_total = coords_mem + diffusion_hidden_mem + diffusion_qkv_mem

    # Safety factor for activations, intermediates, gradients (inference: 2x)
    total = (evoformer_total + diffusion_total) * 2.0

    return total


def check_memory_requirements(
    num_residues: int,
    available_gb: float,
    num_samples: int = 5,
    safety_factor: float = 0.8,
) -> None:
    """Check if sufficient memory is available for inference.

    Raises MemoryError if estimated memory exceeds safe limit.

    Args:
        num_residues: Number of residues in the input.
        available_gb: Available system memory in GB.
        num_samples: Number of structure samples.
        safety_factor: Fraction of available memory to use (default: 0.8).

    Raises:
        MemoryError: If estimated memory exceeds safe limit.
    """
    from alphafold3_mlx.core.exceptions import MemoryError

    estimated = estimate_peak_memory_gb(num_residues, num_samples)
    safe_limit = available_gb * safety_factor

    if estimated > safe_limit:
        raise MemoryError(
            estimated_gb=estimated,
            available_gb=available_gb,
            num_residues=num_residues,
            safety_factor=safety_factor,
        )


def get_available_memory_gb() -> float:
    """Get available system memory in GB.

    Returns:
        Available memory in GB (unified memory on Apple Silicon).
    """
    import subprocess
    import re

    try:
        # Get total physical memory from sysctl on macOS
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        memsize_bytes = int(result.stdout.strip())
        return memsize_bytes / (1024 ** 3)
    except (subprocess.SubprocessError, ValueError):
        # Fallback: assume 64GB (conservative)
        return 64.0


# =============================================================================
# NaN Detection (Phase 3 )
# =============================================================================


def check_nan(
    tensor: mx.array,
    component: str,
    step: int | None = None,
    raise_on_nan: bool = True,
) -> bool:
    """Check tensor for NaN values.

    This is a critical validation checkpoint for debugging numerical
    instability in the 48-layer Evoformer and 200-step diffusion loop.

    Args:
        tensor: MLX array to check.
        component: Name of the component (e.g., "evoformer", "diffusion").
        step: Optional iteration/step number.
        raise_on_nan: Whether to raise NaNError on detection (default: True).

    Returns:
        True if NaN values detected, False otherwise.

    Raises:
        NaNError: If NaN detected and raise_on_nan=True.
    """
    from alphafold3_mlx.core.exceptions import NaNError

    has_nan = bool(mx.any(mx.isnan(tensor)).item())

    if has_nan and raise_on_nan:
        raise NaNError(component=component, step=step)

    return has_nan


def check_nan_dict(
    tensors: dict[str, mx.array],
    component: str,
    step: int | None = None,
    raise_on_nan: bool = True,
) -> dict[str, bool]:
    """Check multiple tensors for NaN values.

    Args:
        tensors: Dictionary of tensors to check.
        component: Name of the component.
        step: Optional iteration/step number.
        raise_on_nan: Whether to raise NaNError on detection.

    Returns:
        Dictionary mapping tensor names to NaN detection results.

    Raises:
        NaNError: If NaN detected and raise_on_nan=True.
    """
    results = {}
    for name, tensor in tensors.items():
        full_name = f"{component}.{name}"
        results[name] = check_nan(
            tensor,
            component=full_name,
            step=step,
            raise_on_nan=raise_on_nan,
        )
    return results


# =============================================================================
# Structure Validity
# =============================================================================


# Ideal bond lengths in Angstroms (from standard amino acid geometry)
IDEAL_BOND_LENGTHS = {
    ("N", "CA"): 1.458,
    ("CA", "C"): 1.525,
    ("C", "O"): 1.229,
    ("C", "N"): 1.329,  # peptide bond
    ("CA", "CB"): 1.530,
}

# Ideal bond angles in degrees
IDEAL_BOND_ANGLES = {
    ("N", "CA", "C"): 111.2,
    ("CA", "C", "O"): 120.8,
    ("CA", "C", "N"): 116.2,  # to next residue
    ("C", "N", "CA"): 121.7,  # from previous residue
    ("N", "CA", "CB"): 110.5,
    ("C", "CA", "CB"): 110.1,
}


def validate_bond_lengths(
    coords: "np.ndarray",
    mask: "np.ndarray",
    tolerance: float = 0.05,
    target_pass_rate: float = 0.95,
) -> tuple[bool, float, dict]:
    """Validate bond lengths against ideal values.

    Checks that 95%+ of bond lengths are within tolerance of ideal values.

    Args:
        coords: Atom coordinates. Shape: [num_residues, max_atoms, 3]
        mask: Atom mask. Shape: [num_residues, max_atoms]
        tolerance: Allowed deviation in Angstroms (default: 0.05).
        target_pass_rate: Required fraction of bonds within tolerance (default: 0.95).

    Returns:
        Tuple of (passed, pass_rate, details_dict).
    """
    import numpy as np

    # Atom indices in atom37 format: N=0, CA=1, C=2, O=3, CB=4
    ATOM_N = 0
    ATOM_CA = 1
    ATOM_C = 2
    ATOM_O = 3
    ATOM_CB = 4

    num_residues = coords.shape[0]
    bond_errors = []
    bond_details = []

    def compute_distance(coords, res_i, atom_a, atom_b):
        """Compute distance between two atoms."""
        if mask[res_i, atom_a] > 0 and mask[res_i, atom_b] > 0:
            diff = coords[res_i, atom_a] - coords[res_i, atom_b]
            return float(np.sqrt(np.sum(diff ** 2)))
        return None

    for res_idx in range(num_residues):
        # N-CA bond
        dist = compute_distance(coords, res_idx, ATOM_N, ATOM_CA)
        if dist is not None:
            ideal = IDEAL_BOND_LENGTHS[("N", "CA")]
            error = abs(dist - ideal)
            bond_errors.append(error)
            bond_details.append(("N-CA", res_idx, dist, ideal, error))

        # CA-C bond
        dist = compute_distance(coords, res_idx, ATOM_CA, ATOM_C)
        if dist is not None:
            ideal = IDEAL_BOND_LENGTHS[("CA", "C")]
            error = abs(dist - ideal)
            bond_errors.append(error)
            bond_details.append(("CA-C", res_idx, dist, ideal, error))

        # C-O bond
        dist = compute_distance(coords, res_idx, ATOM_C, ATOM_O)
        if dist is not None:
            ideal = IDEAL_BOND_LENGTHS[("C", "O")]
            error = abs(dist - ideal)
            bond_errors.append(error)
            bond_details.append(("C-O", res_idx, dist, ideal, error))

        # CA-CB bond (if CB present)
        if mask[res_idx, ATOM_CB] > 0:
            dist = compute_distance(coords, res_idx, ATOM_CA, ATOM_CB)
            if dist is not None:
                ideal = IDEAL_BOND_LENGTHS[("CA", "CB")]
                error = abs(dist - ideal)
                bond_errors.append(error)
                bond_details.append(("CA-CB", res_idx, dist, ideal, error))

        # Peptide bond (C to next N)
        if res_idx < num_residues - 1:
            if mask[res_idx, ATOM_C] > 0 and mask[res_idx + 1, ATOM_N] > 0:
                diff = coords[res_idx, ATOM_C] - coords[res_idx + 1, ATOM_N]
                dist = float(np.sqrt(np.sum(diff ** 2)))
                ideal = IDEAL_BOND_LENGTHS[("C", "N")]
                error = abs(dist - ideal)
                bond_errors.append(error)
                bond_details.append(("C-N", res_idx, dist, ideal, error))

    if len(bond_errors) == 0:
        return True, 1.0, {"num_bonds": 0}

    bond_errors = np.array(bond_errors)
    pass_count = np.sum(bond_errors <= tolerance)
    pass_rate = pass_count / len(bond_errors)
    passed = pass_rate >= target_pass_rate

    return passed, float(pass_rate), {
        "num_bonds": len(bond_errors),
        "pass_count": int(pass_count),
        "mean_error": float(np.mean(bond_errors)),
        "max_error": float(np.max(bond_errors)),
    }


def validate_bond_angles(
    coords: "np.ndarray",
    mask: "np.ndarray",
    tolerance: float = 5.0,
    target_pass_rate: float = 0.95,
) -> tuple[bool, float, dict]:
    """Validate bond angles against ideal values.

    Checks that 95%+ of bond angles are within tolerance of ideal values.

    Args:
        coords: Atom coordinates. Shape: [num_residues, max_atoms, 3]
        mask: Atom mask. Shape: [num_residues, max_atoms]
        tolerance: Allowed deviation in degrees (default: 5.0).
        target_pass_rate: Required fraction of angles within tolerance (default: 0.95).

    Returns:
        Tuple of (passed, pass_rate, details_dict).
    """
    import numpy as np

    ATOM_N = 0
    ATOM_CA = 1
    ATOM_C = 2
    ATOM_O = 3
    ATOM_CB = 4

    num_residues = coords.shape[0]
    angle_errors = []

    def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in degrees."""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        return float(np.degrees(np.arccos(cos_angle)))

    def check_angle(res_i, atom_a, atom_b, atom_c, ideal_name):
        """Check if angle ABC is within tolerance of ideal."""
        if mask[res_i, atom_a] > 0 and mask[res_i, atom_b] > 0 and mask[res_i, atom_c] > 0:
            v1 = coords[res_i, atom_a] - coords[res_i, atom_b]
            v2 = coords[res_i, atom_c] - coords[res_i, atom_b]
            angle = compute_angle(v1, v2)
            ideal = IDEAL_BOND_ANGLES.get(ideal_name, 109.5)  # Default tetrahedral
            error = abs(angle - ideal)
            angle_errors.append(error)
            return angle, ideal, error
        return None

    for res_idx in range(num_residues):
        # N-CA-C angle
        check_angle(res_idx, ATOM_N, ATOM_CA, ATOM_C, ("N", "CA", "C"))

        # CA-C-O angle
        check_angle(res_idx, ATOM_CA, ATOM_C, ATOM_O, ("CA", "C", "O"))

        # N-CA-CB angle (if CB present)
        if mask[res_idx, ATOM_CB] > 0:
            check_angle(res_idx, ATOM_N, ATOM_CA, ATOM_CB, ("N", "CA", "CB"))

    if len(angle_errors) == 0:
        return True, 1.0, {"num_angles": 0}

    angle_errors = np.array(angle_errors)
    pass_count = np.sum(angle_errors <= tolerance)
    pass_rate = pass_count / len(angle_errors)
    passed = pass_rate >= target_pass_rate

    return passed, float(pass_rate), {
        "num_angles": len(angle_errors),
        "pass_count": int(pass_count),
        "mean_error": float(np.mean(angle_errors)),
        "max_error": float(np.max(angle_errors)),
    }


def validate_structure(
    coords: "np.ndarray",
    mask: "np.ndarray",
    bond_tolerance: float = 0.05,
    angle_tolerance: float = 5.0,
    target_pass_rate: float = 0.95,
) -> tuple[bool, dict]:
    """Validate predicted structure against geometric constraints.

    Checks both bond lengths and angles against ideal values.

    Args:
        coords: Atom coordinates. Shape: [num_residues, max_atoms, 3]
        mask: Atom mask. Shape: [num_residues, max_atoms]
        bond_tolerance: Bond length tolerance in Angstroms.
        angle_tolerance: Bond angle tolerance in degrees.
        target_pass_rate: Required pass rate for both checks.

    Returns:
        Tuple of (overall_passed, details_dict).
    """
    bond_passed, bond_rate, bond_details = validate_bond_lengths(
        coords, mask, bond_tolerance, target_pass_rate
    )

    angle_passed, angle_rate, angle_details = validate_bond_angles(
        coords, mask, angle_tolerance, target_pass_rate
    )

    overall_passed = bool(bond_passed and angle_passed)

    return overall_passed, {
        "bond_length": {"passed": bool(bond_passed), "pass_rate": bond_rate, **bond_details},
        "bond_angle": {"passed": bool(angle_passed), "pass_rate": angle_rate, **angle_details},
    }


class NaNCheckpoint:
    """Context manager for NaN detection at checkpoints.

    Usage:
        with NaNCheckpoint("evoformer", step=i) as checkpoint:
            output = evoformer(input)
            checkpoint.check(output, "output")

    This provides structured NaN detection with automatic reporting.
    """

    def __init__(
        self,
        component: str,
        step: int | None = None,
        raise_on_nan: bool = True,
    ) -> None:
        """Initialize NaN checkpoint.

        Args:
            component: Component name.
            step: Optional step number.
            raise_on_nan: Whether to raise on NaN detection.
        """
        self.component = component
        self.step = step
        self.raise_on_nan = raise_on_nan
        self.checked: dict[str, bool] = {}

    def __enter__(self) -> "NaNCheckpoint":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def check(self, tensor: mx.array, name: str) -> bool:
        """Check a tensor for NaN values.

        Args:
            tensor: Tensor to check.
            name: Name for this tensor.

        Returns:
            True if NaN detected.
        """
        full_name = f"{self.component}.{name}"
        has_nan = check_nan(
            tensor,
            component=full_name,
            step=self.step,
            raise_on_nan=self.raise_on_nan,
        )
        self.checked[name] = has_nan
        return has_nan

    @property
    def any_nan(self) -> bool:
        """Return True if any checked tensor had NaN."""
        return any(self.checked.values())
