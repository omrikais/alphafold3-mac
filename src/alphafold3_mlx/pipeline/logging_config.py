"""Logging configuration for AlphaFold 3 MLX pipeline.

This module provides centralized logging setup for the pipeline,
allowing consistent logging behavior across all pipeline modules.

Example:
    from alphafold3_mlx.pipeline.logging_config import get_logger, setup_logging

    # Set up logging at application start
    setup_logging(verbose=True)

    # Get a module-specific logger
    logger = get_logger(__name__)
    logger.info("Starting inference...")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO


# Default format strings
DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
VERBOSE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
SIMPLE_FORMAT = "[%(levelname)s] %(message)s"

# Module prefix for pipeline loggers
LOGGER_PREFIX = "alphafold3_mlx.pipeline"


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure logging for the AlphaFold 3 MLX pipeline.

    Sets up console logging and optionally file logging.

    Args:
        verbose: If True, use debug level with detailed format.
                 If False, use info level with simple format.
        log_file: Optional path to write logs to file.
        stream: Output stream for console logging (defaults to stderr).
    """
    # Get the root pipeline logger
    logger = logging.getLogger(LOGGER_PREFIX)

    # Clear existing handlers
    logger.handlers.clear()

    # Set level based on verbose flag
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Choose format based on verbose flag
    if verbose:
        fmt = VERBOSE_FORMAT
    else:
        fmt = SIMPLE_FORMAT

    formatter = logging.Formatter(fmt)

    # Add console handler
    console_handler = logging.StreamHandler(stream or sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(VERBOSE_FORMAT))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def set_log_level(level: int | str) -> None:
    """Set the logging level for all pipeline loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, "INFO", etc.).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(LOGGER_PREFIX)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


class LoggingContext:
    """Context manager for temporarily changing log level.

    Example:
        with LoggingContext(logging.DEBUG):
            # Detailed logging here
            logger.debug("This will be logged")
        # Back to normal logging
    """

    def __init__(self, level: int | str) -> None:
        """Initialize with target level.

        Args:
            level: Temporary logging level.
        """
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.previous_level: int | None = None

    def __enter__(self) -> "LoggingContext":
        """Save current level and set new level."""
        logger = logging.getLogger(LOGGER_PREFIX)
        self.previous_level = logger.level
        set_log_level(self.level)
        return self

    def __exit__(self, *args) -> None:
        """Restore previous level."""
        if self.previous_level is not None:
            set_log_level(self.previous_level)
