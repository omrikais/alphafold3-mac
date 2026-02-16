"""User-facing error handling for AlphaFold 3 MLX pipeline.

This module defines custom exceptions and failure logging for the CLI.

Exit codes:
- 0: Success
- 1: General error (input, inference, output)
- 130: Interrupted (128 + SIGINT=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class InputError(Exception):
    """User input errors (file not found, invalid JSON, schema violation).

    Attributes:
        exit_code: Unix exit code for this error type.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        """Initialize InputError with message.

        Args:
            message: Human-readable error description.
        """
        super().__init__(message)


class ResourceError(Exception):
    """Resource errors (insufficient memory, disk space, missing weights).

    Attributes:
        exit_code: Unix exit code for this error type.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        """Initialize ResourceError with message.

        Args:
            message: Human-readable error description.
        """
        super().__init__(message)


class InferenceError(Exception):
    """Model inference errors (NaN detected, model failure).

    Attributes:
        exit_code: Unix exit code for this error type.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        """Initialize InferenceError with message.

        Args:
            message: Human-readable error description.
        """
        super().__init__(message)


class InterruptError(Exception):
    """User interrupt (Ctrl+C / SIGINT).

    Attributes:
        exit_code: Unix exit code (130 = 128 + SIGINT).
    """

    exit_code: int = 130

    def __init__(self, message: str = "Interrupted by user") -> None:
        """Initialize InterruptError with message.

        Args:
            message: Human-readable error description. Defaults to "Interrupted by user".
        """
        super().__init__(message)


@dataclass
class FailureLog:
    """Failure information for failure_log.json.

    Records error details when inference fails, including timing snapshot
    and optional traceback for debugging.

    Attributes:
        error_type: Type of error (InputError, ResourceError, InferenceError, etc.).
        error_message: Human-readable error message.
        stage_reached: Last stage completed before failure.
        timing_snapshot: Timing data up to failure point.
        traceback: Optional stack trace for debugging.
    """

    error_type: str
    error_message: str
    stage_reached: str
    timing_snapshot: dict[str, float] = field(default_factory=dict)
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stage_reached": self.stage_reached,
            "timing_snapshot": self.timing_snapshot,
            "traceback": self.traceback,
        }

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        stage_reached: str,
        timing_snapshot: dict[str, float] | None = None,
        include_traceback: bool = True,
    ) -> "FailureLog":
        """Create FailureLog from an exception.

        Args:
            exc: The exception that caused the failure.
            stage_reached: Last completed stage before failure.
            timing_snapshot: Optional timing data up to failure.
            include_traceback: Whether to include stack trace.

        Returns:
            FailureLog instance with error details.
        """
        import traceback as tb

        error_type = type(exc).__name__
        error_message = str(exc)

        traceback_str = None
        if include_traceback:
            traceback_str = "".join(tb.format_exception(type(exc), exc, exc.__traceback__))

        return cls(
            error_type=error_type,
            error_message=error_message,
            stage_reached=stage_reached,
            timing_snapshot=timing_snapshot or {},
            traceback=traceback_str,
        )


def format_error_message(error: Exception) -> str:
    """Format an exception into a user-friendly error message.

    Args:
        error: The exception to format.

    Returns:
        User-friendly error message string.
    """
    error_type = type(error).__name__
    message = str(error)

    # Map error types to friendly prefixes
    prefix_map = {
        "InputError": "Input Error",
        "ResourceError": "Resource Error",
        "InferenceError": "Inference Error",
        "InterruptError": "Interrupted",
        "FileNotFoundError": "File Not Found",
        "MemoryError": "Memory Error",
        "JSONDecodeError": "Invalid JSON",
    }

    prefix = prefix_map.get(error_type, "Error")
    return f"{prefix}: {message}"
