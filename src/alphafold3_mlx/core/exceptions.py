"""Custom exceptions for AlphaFold 3 MLX.

This module defines custom exception types for error handling:
- NaNError: Raised when NaN values are detected
- MemoryError: Override for memory exhaustion
- ShapeMismatchError: Raised when tensor shapes don't match
- WeightsNotFoundError: Raised when weight file is missing
"""

from __future__ import annotations


class NaNError(RuntimeError):
    """Raised when NaN values are detected at validation checkpoints.

    NaN detection is critical for debugging numerical instability in the
    48-layer Evoformer stack and 200-step diffusion loop.
    """

    def __init__(
        self,
        component: str,
        step: int | None = None,
        details: str | None = None,
    ) -> None:
        """Initialize NaN error.

        Args:
            component: Name of the component where NaN was detected
                (e.g., "evoformer", "diffusion", "confidence").
            step: Optional step/iteration number where NaN occurred.
            details: Optional additional details about the error.
        """
        self.component = component
        self.step = step
        self.details = details

        msg = f"NaN values detected in {component}"
        if step is not None:
            msg += f" at step {step}"
        if details:
            msg += f": {details}"
        super().__init__(msg)


class MemoryError(RuntimeError):
    """Raised when estimated memory exceeds hardware limits.

    This exception is raised before execution begins when the memory
    estimation formula predicts OOM for the given input size.
    """

    def __init__(
        self,
        estimated_gb: float,
        available_gb: float,
        num_residues: int,
        safety_factor: float = 0.8,
    ) -> None:
        """Initialize memory error.

        Args:
            estimated_gb: Estimated peak memory usage in GB.
            available_gb: Available system memory in GB.
            num_residues: Number of residues in the input.
            safety_factor: Safety factor used for threshold (default: 0.8).
        """
        self.estimated_gb = estimated_gb
        self.available_gb = available_gb
        self.num_residues = num_residues
        self.safety_factor = safety_factor

        safe_limit = available_gb * safety_factor
        msg = (
            f"Estimated memory: {estimated_gb:.1f} GB, Available: {available_gb:.1f} GB "
            f"({int(safety_factor * 100)}% safety threshold: {safe_limit:.1f} GB). "
            f"Reduce num_samples, diffusion_steps, or use a smaller protein."
        )
        super().__init__(msg)


class ShapeMismatchError(ValueError):
    """Raised when tensor shapes don't match expected values.

    This is used during weight loading to ensure loaded parameters
    match the model architecture.

    Supports two modes:
    - Single parameter: ShapeMismatchError(name, expected, actual)
    - Multiple parameters: ShapeMismatchError(message, mismatches=[(name, expected, actual), ...])
    """

    def __init__(
        self,
        param_name_or_message: str,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
        *,
        mismatches: list[tuple[str, tuple, tuple]] | None = None,
    ) -> None:
        """Initialize shape mismatch error.

        Args:
            param_name_or_message: Either a parameter name (single mode) or
                a full error message (multi mode with mismatches kwarg).
            expected_shape: Expected tensor shape (single mode only).
            actual_shape: Actual tensor shape (single mode only).
            mismatches: List of (param_name, expected, actual) tuples (multi mode).
        """
        if mismatches is not None:
            # Multi-parameter mode
            self.param_name = None
            self.expected_shape = None
            self.actual_shape = None
            self.mismatches = mismatches
            msg = param_name_or_message
        else:
            # Single-parameter mode
            self.param_name = param_name_or_message
            self.expected_shape = expected_shape
            self.actual_shape = actual_shape
            self.mismatches = [(param_name_or_message, expected_shape, actual_shape)]
            msg = (
                f"Shape mismatch for parameter '{param_name_or_message}': "
                f"expected {expected_shape}, got {actual_shape}"
            )

        super().__init__(msg)


class WeightsNotFoundError(FileNotFoundError):
    """Raised when required weight file is not found.

    This is a subclass of FileNotFoundError for compatibility with
    standard Python file handling patterns.
    """

    def __init__(self, weights_path: str) -> None:
        """Initialize weights not found error.

        Args:
            weights_path: Path to the missing weights file.
        """
        self.weights_path = weights_path
        msg = f"Weights file not found: {weights_path}"
        super().__init__(msg)


class ValidationError(RuntimeError):
    """Raised when model output validation fails.

    This includes structure validity checks (bond lengths, angles)
    and numerical validation (tolerance checks against reference).
    """

    def __init__(
        self,
        check_name: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            check_name: Name of the failed validation check.
            message: Human-readable error message.
            details: Optional dictionary with validation details.
        """
        self.check_name = check_name
        self.details = details or {}

        full_msg = f"Validation failed for '{check_name}': {message}"
        super().__init__(full_msg)
