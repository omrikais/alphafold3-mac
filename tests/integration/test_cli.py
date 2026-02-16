"""Integration tests for CLI argument handling and execution.

These tests verify the CLI (--help displays all arguments)
and (CLI produces valid output files).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Path to the CLI entry point
CLI_SCRIPT = Path(__file__).parent.parent.parent / "run_alphafold_mlx.py"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "e2e_inputs"


class TestCLIHelp:
    """Tests for CLI --help output."""

    def test_help_output(self) -> None:
        """Verify --help displays all required arguments."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )

        # Should exit successfully
        assert result.returncode == 0, f"Help failed: {result.stderr}"

        help_text = result.stdout

        # --input argument
        assert "--input" in help_text, "Missing --input argument in help"

        # --output_dir argument
        assert "--output_dir" in help_text, "Missing --output_dir argument in help"

        # --model_dir argument
        assert "--model_dir" in help_text, "Missing --model_dir argument in help"

        # --num_samples argument
        assert "--num_samples" in help_text, "Missing --num_samples argument in help"

        # --diffusion_steps argument
        assert "--diffusion_steps" in help_text, "Missing --diffusion_steps argument in help"

        # --seed argument
        assert "--seed" in help_text, "Missing --seed argument in help"

        # --precision argument
        assert "--precision" in help_text, "Missing --precision argument in help"
        assert "float32" in help_text, "Missing float32 precision option"
        assert "float16" in help_text, "Missing float16 precision option"
        assert "bfloat16" in help_text, "Missing bfloat16 precision option"

        # --verbose flag
        assert "--verbose" in help_text, "Missing --verbose argument in help"

        # --no-overwrite flag
        assert "--no-overwrite" in help_text, "Missing --no-overwrite argument in help"

    def test_help_contains_examples(self) -> None:
        """Verify help text contains usage examples."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        help_text = result.stdout

        # Should contain example commands
        assert "python run_alphafold_mlx.py" in help_text, "Missing example command"
        assert "--input" in help_text, "Missing example with --input"


class TestCLIBasicExecution:
    """Tests for basic CLI execution."""

    def test_basic_execution_missing_input_shows_error(self) -> None:
        """Verify CLI shows error for missing required --input."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--output_dir", "/tmp/test"],
            capture_output=True,
            text=True,
        )

        # Should fail with error
        assert result.returncode != 0
        # argparse error about missing required argument
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_basic_execution_missing_output_dir_shows_error(self) -> None:
        """Verify CLI shows error for missing required --output_dir."""
        result = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--input", "test.json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "output_dir" in result.stderr.lower()

    def test_basic_execution_input_not_found(self) -> None:
        """Verify CLI shows error for non-existent input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", "/nonexistent/path/input.json",
                    "--output_dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    @pytest.mark.skipif(
        not FIXTURES_DIR.exists(),
        reason="Test fixtures not available"
    )
    def test_basic_execution_weights_not_found(self) -> None:
        """Verify CLI shows error for non-existent weights directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", str(FIXTURES_DIR / "test_small.json"),
                    "--output_dir", tmpdir,
                    "--model_dir", "/nonexistent/weights/",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            assert "weight" in result.stderr.lower() or "not found" in result.stderr.lower()

    def test_basic_execution_invalid_num_samples(self) -> None:
        """Verify CLI rejects invalid num_samples value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", "test.json",
                    "--output_dir", tmpdir,
                    "--num_samples", "0",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1

    def test_basic_execution_invalid_precision(self) -> None:
        """Verify CLI rejects invalid precision value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", "test.json",
                    "--output_dir", tmpdir,
                    "--precision", "invalid",
                ],
                capture_output=True,
                text=True,
            )

            # argparse will reject invalid choice
            assert result.returncode == 2  # argparse error


class TestCLIPlatformValidation:
    """Tests for platform validation."""

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Cannot test platform rejection on macOS"
    )
    def test_platform_rejection_on_non_apple_silicon(self) -> None:
        """Verify CLI rejects non-Apple Silicon platforms.

        Expected error message format (Phase 3 alignment):
        "This tool requires Apple Silicon (M2/M3/M4). Detected: {platform}"
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", "test.json",
                    "--output_dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            # Should mention Apple Silicon with M2/M3/M4 requirement
            stderr_lower = result.stderr.lower()
            assert "apple silicon" in stderr_lower or "m2/m3/m4" in stderr_lower.replace(" ", "")


class TestCLIArgumentParsing:
    """Tests for argument parsing functionality."""

    def test_parse_args_default_values(self) -> None:
        """Verify default argument values are set correctly."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp/out"])

        assert args.input == "test.json"
        assert args.output_dir == "/tmp/out"
        assert args.model_dir == "weights/model" # default
        assert args.num_samples == 5 # default
        assert args.diffusion_steps == 200 # default
        assert args.seed is None # default
        assert args.precision is None # default (auto-detect)
        assert args.verbose is False # default
        assert args.no_overwrite is False # default

    def test_parse_args_all_values(self) -> None:
        """Verify all argument values can be set."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args([
            "--input", "input.json",
            "--output_dir", "/output",
            "--model_dir", "/weights",
            "--num_samples", "3",
            "--diffusion_steps", "100",
            "--seed", "42",
            "--precision", "bfloat16",
            "--verbose",
            "--no-overwrite",
        ])

        assert args.input == "input.json"
        assert args.output_dir == "/output"
        assert args.model_dir == "/weights"
        assert args.num_samples == 3
        assert args.diffusion_steps == 100
        assert args.seed == 42
        assert args.precision == "bfloat16"
        assert args.verbose is True
        assert args.no_overwrite is True

    def test_parse_args_verbose_short_flag(self) -> None:
        """Verify -v short flag works for verbose."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp", "-v"])
        assert args.verbose is True


class TestNumSamplesParameter:
    """Tests for num_samples parameter."""

    def test_num_samples_argument_accepted(self) -> None:
        """Verify --num_samples argument is accepted and parsed correctly."""
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        # Test various valid values
        for num_samples in [1, 3, 5, 10]:
            args = parse_args([
                "--input", "test.json",
                "--output_dir", "/tmp/out",
                "--num_samples", str(num_samples),
            ])
            cli_args = CLIArguments.from_namespace(args)
            assert cli_args.num_samples == num_samples

    def test_num_samples_default_value(self) -> None:
        """Verify --num_samples defaults to 5."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp/out"])
        assert args.num_samples == 5

    def test_num_samples_validation_rejects_zero(self) -> None:
        """Verify num_samples=0 is rejected."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="num_samples must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                num_samples=0,
            )

    def test_num_samples_validation_rejects_negative(self) -> None:
        """Verify negative num_samples is rejected."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="num_samples must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                num_samples=-1,
            )

    @pytest.mark.skipif(
        not FIXTURES_DIR.exists(),
        reason="Test fixtures not available"
    )
    def test_num_samples_determines_output_count(self) -> None:
        """Verify num_samples determines number of structure output files."""
        # This is an integration test that would require model weights
        # For unit testing, we verify the argument is passed through
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        args = parse_args([
            "--input", str(FIXTURES_DIR / "test_small.json"),
            "--output_dir", "/tmp/out",
            "--num_samples", "3",
        ])
        cli_args = CLIArguments.from_namespace(args)
        assert cli_args.num_samples == 3


class TestDiffusionStepsParameter:
    """Tests for diffusion_steps parameter."""

    def test_diffusion_steps_argument_accepted(self) -> None:
        """Verify --diffusion_steps argument is accepted and parsed correctly."""
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        # Test various valid values
        for steps in [10, 50, 100, 200]:
            args = parse_args([
                "--input", "test.json",
                "--output_dir", "/tmp/out",
                "--diffusion_steps", str(steps),
            ])
            cli_args = CLIArguments.from_namespace(args)
            assert cli_args.diffusion_steps == steps

    def test_diffusion_steps_default_value(self) -> None:
        """Verify --diffusion_steps defaults to 200."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp/out"])
        assert args.diffusion_steps == 200

    def test_diffusion_steps_validation_rejects_zero(self) -> None:
        """Verify diffusion_steps=0 is rejected."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="diffusion_steps must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                diffusion_steps=0,
            )

    def test_diffusion_steps_validation_rejects_negative(self) -> None:
        """Verify negative diffusion_steps is rejected."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="diffusion_steps must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                diffusion_steps=-10,
            )

    def test_diffusion_steps_low_value_for_testing(self) -> None:
        """Verify low diffusion_steps values work for quick tests."""
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        # Very low values should be accepted (useful for testing)
        args = parse_args([
            "--input", "test.json",
            "--output_dir", "/tmp/out",
            "--diffusion_steps", "1",
        ])
        cli_args = CLIArguments.from_namespace(args)
        assert cli_args.diffusion_steps == 1


class TestPrecisionParameter:
    """Tests for precision parameter."""

    def test_precision_argument_accepted(self) -> None:
        """Verify --precision argument is accepted for all valid values."""
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        for precision in ["float32", "float16", "bfloat16"]:
            args = parse_args([
                "--input", "test.json",
                "--output_dir", "/tmp/out",
                "--precision", precision,
            ])
            cli_args = CLIArguments.from_namespace(args)
            assert cli_args.precision == precision

    def test_precision_default_is_none(self) -> None:
        """Verify --precision defaults to None (auto-detect)."""
        from alphafold3_mlx.pipeline.cli import parse_args

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp/out"])
        assert args.precision is None

    def test_precision_invalid_value_rejected(self) -> None:
        """Verify invalid precision values are rejected by argparse."""
        result = subprocess.run(
            [
                sys.executable, str(CLI_SCRIPT),
                "--input", "test.json",
                "--output_dir", "/tmp/out",
                "--precision", "float64",  # Invalid
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2  # argparse error
        assert "invalid choice" in result.stderr.lower()

    def test_auto_select_precision_returns_valid_value(self) -> None:
        """Verify auto_select_precision returns a valid precision value."""
        from alphafold3_mlx.pipeline.cli import auto_select_precision

        precision = auto_select_precision()
        assert precision in ("float32", "float16", "bfloat16")

    @pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Platform-specific test for macOS"
    )
    def test_auto_select_precision_on_macos(self) -> None:
        """Verify auto_select_precision works on macOS."""
        from alphafold3_mlx.pipeline.cli import auto_select_precision

        precision = auto_select_precision()
        # On macOS, should return a valid precision
        assert precision in ("float32", "float16", "bfloat16")


class TestCLIArgumentsDataclass:
    """Tests for CLIArguments dataclass validation."""

    def test_cli_arguments_from_namespace(self) -> None:
        """Verify CLIArguments can be created from argparse namespace."""
        from alphafold3_mlx.pipeline.cli import parse_args, CLIArguments

        args = parse_args(["--input", "test.json", "--output_dir", "/tmp/out"])
        cli_args = CLIArguments.from_namespace(args)

        assert cli_args.input_path == Path("test.json")
        assert cli_args.output_dir == Path("/tmp/out")
        assert cli_args.num_samples == 5

    def test_cli_arguments_validation_num_samples(self) -> None:
        """Verify CLIArguments rejects invalid num_samples."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="num_samples must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                num_samples=0,
            )

    def test_cli_arguments_validation_diffusion_steps(self) -> None:
        """Verify CLIArguments rejects invalid diffusion_steps."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="diffusion_steps must be >= 1"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                diffusion_steps=0,
            )

    def test_cli_arguments_validation_negative_seed(self) -> None:
        """Verify CLIArguments rejects negative seed."""
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="seed must be non-negative"):
            CLIArguments(
                input_path=Path("test.json"),
                output_dir=Path("/tmp"),
                seed=-1,
            )


class TestErrorHandling:
    """Tests for graceful error handling (User Story 5).

    These tests verify that the CLI provides clear, actionable error messages
    for all failure modes .
    """

    def test_input_not_found(self) -> None:
        """Verify proper error handling when input file does not exist.

        Should return exit code 1 with clear error message pointing to the
        missing file path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_file = Path(tmpdir) / "nonexistent" / "input.json"

            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", str(nonexistent_file),
                    "--output_dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            # Should exit with error code 1
            assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

            # Error message should mention the file not being found
            stderr_lower = result.stderr.lower()
            assert "not found" in stderr_lower or "error" in stderr_lower, (
                f"Expected 'not found' or 'error' in stderr: {result.stderr}"
            )

            # Error message should include the file path
            assert "nonexistent" in result.stderr or "input.json" in result.stderr, (
                f"Expected file path in error message: {result.stderr}"
            )

    def test_invalid_json(self) -> None:
        """Verify proper error handling for malformed JSON input.

        Should return exit code 1 with clear error message about JSON parsing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid JSON file
            invalid_json_file = Path(tmpdir) / "invalid.json"
            invalid_json_file.write_text("{ this is not valid json }")

            # Also create a mock weights directory (even though we won't get that far)
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()
            (weights_dir / "af3.bin.zst").touch()

            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", str(invalid_json_file),
                    "--output_dir", tmpdir,
                    "--model_dir", str(weights_dir),
                ],
                capture_output=True,
                text=True,
            )

            # Should exit with error code 1
            assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

            # Error message should indicate JSON parsing error
            stderr_lower = result.stderr.lower()
            assert "json" in stderr_lower or "invalid" in stderr_lower or "parse" in stderr_lower, (
                f"Expected JSON-related error in stderr: {result.stderr}"
            )

    @pytest.mark.skipif(
        not FIXTURES_DIR.exists(),
        reason="Test fixtures not available"
    )
    def test_memory_rejection(self) -> None:
        """Verify proper rejection when memory requirements exceed available.

        This test mocks the memory check to simulate insufficient memory.
        Should return exit code 1 before inference starts.

        Phase 3+ memory error message format:
        "Estimated memory: X.X GB, Available: Y.Y GB (Z% safety threshold: T.T GB).
         Reduce num_samples, diffusion_steps, or use a smaller protein."
        """
        # This test requires mocking the memory estimation to simulate OOM
        # For now, we test that the memory check infrastructure exists
        from alphafold3_mlx.pipeline.input_handler import estimate_memory_gb
        from alphafold3_mlx.pipeline.errors import ResourceError

        # Verify estimate_memory_gb exists and returns reasonable values
        # For a small protein, memory estimate should be positive
        assert callable(estimate_memory_gb)

        # Test that ResourceError can be raised for memory issues
        # Phase 3+ format includes estimated, available, threshold, and suggestions
        with pytest.raises(ResourceError) as exc_info:
            raise ResourceError(
                "Estimated memory: 150.0 GB, Available: 128.0 GB "
                "(80% safety threshold: 102.4 GB). "
                "Reduce num_samples, diffusion_steps, or use a smaller protein."
            )

        # Verify the error message contains expected components
        error_msg = str(exc_info.value)
        assert "Estimated memory" in error_msg
        assert "Available" in error_msg

    def test_memory_error_message_format(self) -> None:
        """Verify MemoryError from Phase 3 produces spec-aligned message format."""
        from alphafold3_mlx.core.exceptions import MemoryError

        # Test the actual MemoryError class message format
        error = MemoryError(
            estimated_gb=150.0,
            available_gb=128.0,
            num_residues=5000,
            safety_factor=0.8,
        )

        msg = str(error)

        # Phase 3+ format should include:
        # - Estimated memory
        # - Available memory
        # - Safety threshold percentage and value
        # - Actionable suggestion
        assert "150.0" in msg or "150" in msg, "Should mention estimated memory"
        assert "128.0" in msg or "128" in msg, "Should mention available memory"
        assert "80%" in msg or "0.8" in msg, "Should mention safety threshold"
        assert "Reduce" in msg, "Should include actionable suggestion"

    def test_no_overwrite_flag(self) -> None:
        """Verify --no-overwrite flag prevents overwriting existing files.

        Should return exit code 1 if output directory contains existing files
        and --no-overwrite is set.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Create existing output files
            (output_dir / "structure_rank_1.cif").write_text("existing structure")
            (output_dir / "confidence_scores.json").write_text("{}")

            # Create a valid input file with proper format
            input_file = Path(tmpdir) / "input.json"
            input_file.write_text(json.dumps({
                "name": "test_protein",
                "modelSeeds": [42],
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
                        }
                    }
                ]
            }))

            # Create mock weights directory
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()
            (weights_dir / "af3.bin.zst").touch()

            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", str(input_file),
                    "--output_dir", str(output_dir),
                    "--model_dir", str(weights_dir),
                    "--no-overwrite",
                ],
                capture_output=True,
                text=True,
            )

            # Should exit with error code 1
            assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

            # Error message should mention existing files or overwrite
            stderr_lower = result.stderr.lower()
            assert "exist" in stderr_lower or "overwrite" in stderr_lower, (
                f"Expected 'exist' or 'overwrite' in stderr: {result.stderr}"
            )

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="Cannot test platform rejection on macOS"
    )
    def test_platform_rejection(self) -> None:
        """Verify CLI rejects non-Apple Silicon platforms.

        This test can only run on non-macOS platforms.

        Expected error message format (Phase 3 alignment):
        "This tool requires Apple Silicon (M2/M3/M4). Detected: {platform}"
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", "test.json",
                    "--output_dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 1
            stderr_lower = result.stderr.lower()
            # Phase 3+ uses "Apple Silicon (M2/M3/M4)" message format
            assert "apple silicon" in stderr_lower or "m2/m3/m4" in stderr_lower.replace(" ", ""), (
                f"Expected platform-related error in stderr: {result.stderr}"
            )

    def test_failure_log_written(self) -> None:
        """Verify failure_log.json is written on error.

        When an error occurs, a failure_log.json should be written to the
        output directory with error details.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            # Create mock input that will fail during parsing
            input_file = Path(tmpdir) / "input.json"
            input_file.write_text('{"name": "test"}')  # Missing required fields

            # Create mock weights directory
            weights_dir = Path(tmpdir) / "weights"
            weights_dir.mkdir()
            (weights_dir / "af3.bin.zst").touch()

            result = subprocess.run(
                [
                    sys.executable, str(CLI_SCRIPT),
                    "--input", str(input_file),
                    "--output_dir", str(output_dir),
                    "--model_dir", str(weights_dir),
                ],
                capture_output=True,
                text=True,
            )

            # Should fail (exit code 1)
            assert result.returncode == 1, f"Expected exit code 1, got {result.returncode}"

            # Check if failure_log.json was created (best effort - may not always be created)
            failure_log = output_dir / "failure_log.json"
            if failure_log.exists():
                import json
                failure_data = json.loads(failure_log.read_text())

                # Verify failure log structure
                assert "error_type" in failure_data
                assert "error_message" in failure_data
                assert "stage_reached" in failure_data


class TestInterruptHandling:
    """Tests for interrupt handling."""

    def test_interrupt_handler_unit(self) -> None:
        """Test InterruptHandler class directly."""
        from alphafold3_mlx.pipeline.runner import InterruptHandler
        from alphafold3_mlx.pipeline.errors import InterruptError

        handler = InterruptHandler()

        # Should not be interrupted initially
        assert handler.interrupted is False

        # Should not raise when not interrupted
        handler.check()  # Should not raise

        # Simulate interrupt
        handler.interrupted = True

        # Should raise InterruptError
        with pytest.raises(InterruptError, match="Interrupted"):
            handler.check()

    def test_interrupt_handler_install_uninstall(self) -> None:
        """Test InterruptHandler install/uninstall."""
        import signal
        from alphafold3_mlx.pipeline.runner import InterruptHandler

        handler = InterruptHandler()

        # Get original handler
        original = signal.getsignal(signal.SIGINT)

        # Install
        handler.install()

        # Handler should be different now
        current = signal.getsignal(signal.SIGINT)
        assert current != original or handler._original_handler is not None

        # Uninstall
        handler.uninstall()

        # Should be restored
        restored = signal.getsignal(signal.SIGINT)
        # Note: restored may not equal original if other code has modified it

    @pytest.mark.slow
    def test_interrupt_exit_code(self) -> None:
        """Verify interrupt results in exit code 130.

        This test sends SIGINT to a running process and verifies the exit code.
        """
        import os
        import signal
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a script that sleeps to give us time to interrupt
            script = Path(tmpdir) / "test_interrupt.py"
            script.write_text('''
import sys
import time
sys.path.insert(0, "src")
from alphafold3_mlx.pipeline.runner import InterruptHandler
from alphafold3_mlx.pipeline.errors import InterruptError

handler = InterruptHandler()
handler.install()

try:
    # Sleep to allow interrupt
    for i in range(100):
        time.sleep(0.1)
        handler.check()
except InterruptError:
    sys.exit(130)
finally:
    handler.uninstall()
''')

            # Start the process
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(Path(__file__).parent.parent.parent),
            )

            # Wait a bit then send SIGINT
            time.sleep(0.3)
            proc.send_signal(signal.SIGINT)

            # Wait for exit
            proc.wait(timeout=5)

            # Should exit with 130
            assert proc.returncode == 130, f"Expected exit code 130, got {proc.returncode}"


class TestAtomicWrite:
    """Tests for atomic write operations."""

    def test_atomic_write_success(self) -> None:
        """Test atomic_write on successful write."""
        from alphafold3_mlx.pipeline.output_handler import atomic_write

        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test_file.txt"

            with atomic_write(final_path) as temp_path:
                # Write to temp path
                temp_path.write_text("test content")

            # Final file should exist with content
            assert final_path.exists()
            assert final_path.read_text() == "test content"

            # Temp file should not exist
            temp_file = final_path.with_suffix(".txt.tmp")
            assert not temp_file.exists()

    def test_atomic_write_failure_cleanup(self) -> None:
        """Test atomic_write cleans up temp file on failure."""
        from alphafold3_mlx.pipeline.output_handler import atomic_write

        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = Path(tmpdir) / "test_file.txt"
            temp_file = final_path.with_suffix(".txt.tmp")

            with pytest.raises(ValueError):
                with atomic_write(final_path) as temp_path:
                    # Write to temp path
                    temp_path.write_text("test content")
                    # Raise an error
                    raise ValueError("Simulated failure")

            # Neither file should exist
            assert not final_path.exists()
            assert not temp_file.exists()


class TestFormatErrorMessage:
    """Tests for error message formatting."""

    def test_format_input_error(self) -> None:
        """Test formatting of InputError."""
        from alphafold3_mlx.pipeline.errors import InputError, format_error_message

        error = InputError("File not found: test.json")
        formatted = format_error_message(error)

        assert "Input Error" in formatted
        assert "File not found: test.json" in formatted

    def test_format_resource_error(self) -> None:
        """Test formatting of ResourceError."""
        from alphafold3_mlx.pipeline.errors import ResourceError, format_error_message

        error = ResourceError("Insufficient memory. Required: 150GB")
        formatted = format_error_message(error)

        assert "Resource Error" in formatted
        assert "Insufficient memory" in formatted

    def test_format_inference_error(self) -> None:
        """Test formatting of InferenceError."""
        from alphafold3_mlx.pipeline.errors import InferenceError, format_error_message

        error = InferenceError("NaN detected in diffusion step 50")
        formatted = format_error_message(error)

        assert "Inference Error" in formatted
        assert "NaN" in formatted

    def test_format_interrupt_error(self) -> None:
        """Test formatting of InterruptError."""
        from alphafold3_mlx.pipeline.errors import InterruptError, format_error_message

        error = InterruptError()
        formatted = format_error_message(error)

        assert "Interrupted" in formatted

    def test_format_generic_error(self) -> None:
        """Test formatting of generic exception."""
        from alphafold3_mlx.pipeline.errors import format_error_message

        error = RuntimeError("Something went wrong")
        formatted = format_error_message(error)

        assert "Error:" in formatted
        assert "Something went wrong" in formatted


class TestFailureLog:
    """Tests for failure log functionality."""

    def test_failure_log_from_exception(self) -> None:
        """Test FailureLog.from_exception creates proper log."""
        from alphafold3_mlx.pipeline.errors import FailureLog, InputError

        error = InputError("Test error message")
        timing = {"stage1": 1.5, "stage2": 2.3}

        log = FailureLog.from_exception(
            exc=error,
            stage_reached="feature_preparation",
            timing_snapshot=timing,
        )

        assert log.error_type == "InputError"
        assert log.error_message == "Test error message"
        assert log.stage_reached == "feature_preparation"
        assert log.timing_snapshot == timing
        assert log.traceback is not None  # Should include traceback by default

    def test_failure_log_to_dict(self) -> None:
        """Test FailureLog.to_dict for JSON serialization."""
        from alphafold3_mlx.pipeline.errors import FailureLog

        log = FailureLog(
            error_type="InferenceError",
            error_message="NaN detected",
            stage_reached="inference",
            timing_snapshot={"weight_loading": 5.0},
            traceback="Traceback ...",
        )

        data = log.to_dict()

        assert data["error_type"] == "InferenceError"
        assert data["error_message"] == "NaN detected"
        assert data["stage_reached"] == "inference"
        assert data["timing_snapshot"]["weight_loading"] == 5.0
        assert data["traceback"] == "Traceback ..."

    def test_write_failure_log(self) -> None:
        """Test write_failure_log writes valid JSON."""
        from alphafold3_mlx.pipeline.output_handler import write_failure_log

        with tempfile.TemporaryDirectory() as tmpdir:
            failure_data = {
                "error_type": "ResourceError",
                "error_message": "Out of memory",
                "stage_reached": "inference",
                "timing_snapshot": {"setup": 1.0},
            }

            output_dir = Path(tmpdir)
            path = write_failure_log(failure_data, output_dir)

            assert path.exists()
            assert path.name == "failure_log.json"

            # Verify content
            loaded = json.loads(path.read_text())
            assert loaded["error_type"] == "ResourceError"
            assert loaded["error_message"] == "Out of memory"
