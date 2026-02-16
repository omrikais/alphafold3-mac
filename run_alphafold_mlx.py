#!/usr/bin/env python3
"""AlphaFold 3 MLX - Entry point for protein structure prediction on Apple Silicon.

This script provides a command-line interface for running AlphaFold 3
inference using the MLX implementation on Apple Silicon Macs.

Usage:
    python run_alphafold_mlx.py --input input.json --output_dir output/

Example:
    python run_alphafold_mlx.py \\
        --input examples/monomer.json \\
        --output_dir predictions/ \\
        --num_samples 5

Exit Codes:
    0: Success
    1: Error (input, inference, resource)
    130: Interrupted (Ctrl+C / SIGINT)
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any


# Global flag for SIGINT handling before runner is created
_interrupted = False


def _sigint_handler(signum: int, frame: Any) -> None:
    """Signal handler for early SIGINT (before InferenceRunner)."""
    global _interrupted
    _interrupted = True
    print("\nInterrupt received, cleaning up...", file=sys.stderr)


def validate_platform() -> None:
    """Validate platform at startup.

    Raises:
        InputError: If not running on Apple Silicon M2/M3/M4 macOS.
    """
    from alphafold3_mlx.pipeline.errors import InputError
    from alphafold3_mlx.weights.platform import validate_platform_for_cli
    from alphafold3_mlx import PlatformError

    try:
        validate_platform_for_cli()
    except PlatformError as e:
        raise InputError(str(e))


def validate_input_file(input_path: Path) -> None:
    """Validate input file exists.

    Args:
        input_path: Path to input JSON file.

    Raises:
        InputError: If input file not found.
    """
    from alphafold3_mlx.pipeline.errors import InputError

    if not input_path.exists():
        raise InputError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise InputError(f"Input path is not a file: {input_path}")


def validate_weights_directory(model_dir: Path) -> None:
    """Validate weights directory exists.

    Args:
        model_dir: Path to model weights directory.

    Raises:
        ResourceError: If weights not found.
    """
    from alphafold3_mlx.pipeline.errors import ResourceError

    if not model_dir.exists():
        raise ResourceError(
            f"Model weights directory not found: {model_dir}\n"
            "  Download weights from: https://github.com/google-deepmind/alphafold3\n"
            "  Place in: ~/.alphafold3/weights/model/ or set AF3_WEIGHTS_DIR"
        )

    # C-07: Check for any supported weight format
    patterns = ["af3.bin.zst", "af3.bin", "af3.0.bin.zst"]
    has_weights = (
        any((model_dir / p).exists() for p in patterns)
        or any(model_dir.glob("af3.*.bin.zst"))
        or any(model_dir.glob("af3.bin.zst.*"))
    )
    if not has_weights:
        raise ResourceError(
            f"No weight files found in: {model_dir}\n"
            "  Expected: af3.bin.zst (or af3.bin, sharded af3.*.bin.zst"
            " or af3.bin.zst.*)\n"
            "  Download weights from: https://github.com/google-deepmind/alphafold3\n"
            "  Place in: ~/.alphafold3/weights/model/ or set AF3_WEIGHTS_DIR"
        )


def main() -> int:
    """Main entry point for AlphaFold 3 MLX CLI.

    Returns:
        Exit code: 0 for success, 1 for error, 130 for interrupt.
    """
    global _interrupted

    # Install signal handler early for SIGINT (exit code 130)
    original_handler = signal.signal(signal.SIGINT, _sigint_handler)

    # Import pipeline module here to show --help quickly (M-01)
    try:
        from alphafold3_mlx.pipeline import (
            parse_args,
            CLIArguments,
            parse_input_json,
            validate_input,
            ProgressReporter,
            InferenceRunner,
            create_failure_log,
            write_failure_log,
            format_error_message,
            InputError,
            ResourceError,
            InferenceError,
            InterruptError,
            setup_logging,
        )
    except ModuleNotFoundError as exc:
        print(
            f"Error: {exc}\n"
            "alphafold3_mlx is not installed or dependencies are missing.\n"
            "Install with: pip install -e .",
            file=sys.stderr,
        )
        return 1

    # Parse arguments
    args = parse_args()

    # Set up logging based on verbose flag
    setup_logging(verbose=args.verbose)

    # Convert to CLIArguments dataclass for validation
    try:
        cli_args = CLIArguments.from_namespace(args)
    except InputError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create progress reporter early for failure logging
    progress = ProgressReporter(verbose=cli_args.verbose)

    try:
        # Check for early interrupt
        if _interrupted:
            raise InterruptError("Interrupted by user before validation")

        # Platform validation at startup
        progress.on_stage_start("platform_validation")
        validate_platform()
        progress.on_stage_end("platform_validation")

        # Check for interrupt after platform validation
        if _interrupted:
            raise InterruptError("Interrupted by user")

        # Input file existence validation
        progress.on_stage_start("input_validation")
        validate_input_file(cli_args.input_path)

        # Weights directory validation
        validate_weights_directory(cli_args.model_dir)
        progress.on_stage_end("input_validation")

        # Check for interrupt after input validation
        if _interrupted:
            raise InterruptError("Interrupted by user")

        # Parse and validate input JSON
        progress.on_stage_start("input_parsing")
        input_json = parse_input_json(cli_args.input_path)

        # : Apply --restraints file if provided
        if cli_args.restraints_file is not None:
            from alphafold3_mlx.pipeline.input_handler import apply_restraints_file
            input_json = apply_restraints_file(input_json, cli_args.restraints_file)

        validation_errors = validate_input(input_json)
        if validation_errors:
            raise InputError(f"Invalid input: {'; '.join(validation_errors)}")
        progress.on_stage_end("input_parsing")

        if cli_args.verbose:
            print(f"Input: {input_json.name}")
            print(f"  Sequences: {len(input_json.sequences)}")
            print(f"  Total residues: {input_json.total_residues}")
            print(f"  Is complex: {input_json.is_complex}")

        # Create and run inference runner
        runner = InferenceRunner(
            args=cli_args,
            input_json=input_json,
            progress=progress,
        )
        output_bundle = runner.run()

        # Report success
        print(f"\nSuccess! Output files written to: {cli_args.output_dir}")
        print(f"  Best structure: {output_bundle.structure_files[0].name}")
        print(f"  Confidence scores: {output_bundle.confidence_scores_file.name}")
        print(f"  Timing: {output_bundle.timing_file.name}")
        print(f"  Ranking debug: {output_bundle.ranking_debug_file.name}")

        return 0

    except InterruptError as e:
        # Clean interrupt with exit code 130
        print(f"\n{format_error_message(e)}", file=sys.stderr)
        _write_failure_log_safe(e, progress, cli_args.output_dir)
        return 130

    except InputError as e:
        # Input error with exit code 1
        print(f"\n{format_error_message(e)}", file=sys.stderr)
        _write_failure_log_safe(e, progress, cli_args.output_dir)
        return 1

    except ResourceError as e:
        # Resource error with exit code 1
        print(f"\n{format_error_message(e)}", file=sys.stderr)
        _write_failure_log_safe(e, progress, cli_args.output_dir)
        return 1

    except InferenceError as e:
        # Inference error with exit code 1
        print(f"\n{format_error_message(e)}", file=sys.stderr)
        _write_failure_log_safe(e, progress, cli_args.output_dir)
        return 1

    except Exception as e:
        # Unexpected error
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        _write_failure_log_safe(e, progress, cli_args.output_dir)
        return 1

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def _write_failure_log_safe(
    exc: Exception,
    progress: "ProgressReporter",
    output_dir: Path,
) -> None:
    """Write failure log with best-effort error handling.

    Args:
        exc: The exception that caused the failure.
        progress: Progress reporter for timing data.
        output_dir: Output directory for failure_log.json.
    """
    try:
        from alphafold3_mlx.pipeline import create_failure_log
        create_failure_log(exc, progress, output_dir)
    except Exception:
        pass  # Best effort - don't fail on failure log


if __name__ == "__main__":
    sys.exit(main())
