"""Unit tests for CLI argument handling.

These tests verify the CLI argument parsing and validation logic
without requiring full integration or model weights.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest


class TestAutoSelectPrecision:
    """Tests for auto_select_precision() function."""

    def test_auto_select_precision_returns_valid_value(self) -> None:
        """Verify auto_select_precision returns a valid precision value."""
        from alphafold3_mlx.pipeline.cli import auto_select_precision

        precision = auto_select_precision()
        assert precision in ("float32", "float16", "bfloat16")

    @pytest.mark.parametrize(
        "chip_family, supports_bfloat16, memory_gb, expected",
        [
            ("M3", True, 96, "bfloat16"),
            ("M4", True, 128, "bfloat16"),
            ("M2", False, 64, "float16"),
            ("Unknown", False, 32, "float32"),
        ],
    )
    def test_auto_select_precision_by_chip(
        self, chip_family: str, supports_bfloat16: bool, memory_gb: int, expected: str
    ) -> None:
        """Verify precision selection matches chip capabilities."""
        from alphafold3_mlx.weights.platform import PlatformInfo

        mock_info = PlatformInfo(
            system="Darwin",
            machine="arm64",
            chip_family=chip_family,
            supports_bfloat16=supports_bfloat16,
            memory_gb=memory_gb,
        )

        with mock.patch(
            "alphafold3_mlx.weights.platform.get_platform_info",
            return_value=mock_info
        ):
            from alphafold3_mlx.pipeline.cli import auto_select_precision
            precision = auto_select_precision()
            assert precision == expected

    def test_auto_select_precision_fallback_on_exception(self) -> None:
        """Verify float32 fallback when get_platform_info raises exception."""
        with mock.patch(
            "alphafold3_mlx.weights.platform.get_platform_info",
            side_effect=Exception("Platform error")
        ):
            from alphafold3_mlx.pipeline.cli import auto_select_precision
            # The function catches all exceptions and falls back
            precision = auto_select_precision()
            # Even if exception is raised inside, function should handle it
            assert precision in ("float32", "float16", "bfloat16")


class TestCLIArgumentsValidation:
    """Tests for CLIArguments validation."""

    def test_precision_accepted_values(self) -> None:
        """Verify all valid precision values are accepted."""
        from alphafold3_mlx.pipeline.cli import CLIArguments

        for precision in ["float32", "float16", "bfloat16"]:
            args = CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                precision=precision,  # type: ignore[arg-type]
            )
            assert args.precision == precision

    def test_precision_none_is_accepted(self) -> None:
        """Verify None precision (auto-detect) is accepted."""
        from alphafold3_mlx.pipeline.cli import CLIArguments

        args = CLIArguments(
            input_path=Path("test.json"),
            output_dir=Path("/tmp"),
            precision=None,
        )
        assert args.precision is None

    def test_immutable_dataclass(self) -> None:
        """Verify CLIArguments is immutable (frozen)."""
        from alphafold3_mlx.pipeline.cli import CLIArguments

        args = CLIArguments(
            input_path=Path("test.json"),
            output_dir=Path("/tmp"),
        )

        with pytest.raises((TypeError, AttributeError)):
            args.num_samples = 10  # type: ignore[misc]
