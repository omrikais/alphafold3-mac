#!/usr/bin/env python3
"""Performance benchmark for AlphaFold 3 MLX pipeline.

This script benchmarks the full inference pipeline on proteins of various sizes
to validate the target: <10 min for 200 residues with 5 samples on M4 Max.

Usage:
    python benchmarks/pipeline_benchmark.py --size 200 --num_samples 5
    python benchmarks/pipeline_benchmark.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    size: int
    num_samples: int
    diffusion_steps: int
    total_seconds: float
    stage_timings: dict[str, float]
    peak_memory_gb: float | None
    precision: str
    passed_threshold: bool


def get_platform_info() -> dict[str, Any]:
    """Get platform information."""
    import platform

    try:
        from alphafold3_mlx.weights.platform import get_platform_info as _get_platform_info

        info = _get_platform_info()
        return {
            "chip_family": info.chip_family,
            "total_memory_gb": info.memory_gb,
            "supports_bfloat16": info.supports_bfloat16,
        }
    except ImportError:
        return {
            "chip_family": "Unknown",
            "total_memory_gb": 0.0,
            "supports_bfloat16": False,
        }


def run_benchmark(
    size: int,
    num_samples: int = 5,
    diffusion_steps: int = 200,
    precision: str | None = None,
    output_dir: Path | None = None,
) -> BenchmarkResult:
    """Run a benchmark with specified parameters.

    Args:
        size: Protein size in residues.
        num_samples: Number of structure samples.
        diffusion_steps: Number of diffusion steps.
        precision: Compute precision (None = auto-select).
        output_dir: Directory for output files.

    Returns:
        BenchmarkResult with timing information.
    """
    import tempfile

    from alphafold3_mlx.pipeline import (
        CLIArguments,
        InferenceRunner,
        ProgressReporter,
        auto_select_precision,
        parse_input_json,
        validate_input,
    )

    # Find the appropriate test file
    test_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "e2e_inputs"
    test_file = test_dir / f"test_{size}.json"

    if not test_file.exists():
        # Generate a test file for this size
        test_file = _create_test_input(size, test_dir)

    # Parse input
    input_json = parse_input_json(test_file)
    errors = validate_input(input_json)
    if errors:
        raise ValueError(f"Invalid input: {'; '.join(errors)}")

    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="af3_benchmark_"))

    # Determine precision
    if precision is None:
        precision = auto_select_precision()

    # Create CLI arguments
    cli_args = CLIArguments(
        input_path=test_file,
        output_dir=output_dir,
        model_dir=Path("weights/model"),
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        precision=precision,
        verbose=True,
    )

    # Create progress reporter
    progress = ProgressReporter(verbose=True)

    # Create runner and execute
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {size} residues, {num_samples} samples, {diffusion_steps} steps")
    print(f"Precision: {precision}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    runner = InferenceRunner(
        args=cli_args,
        input_json=input_json,
        progress=progress,
    )

    try:
        runner.run()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise

    total_time = time.time() - start_time

    # Get timing data
    timing_data = progress.get_timing_data()

    # Get memory info
    peak_memory_gb = None
    try:
        import mlx.core as mx

        # Use new API, fall back to deprecated if not available
        try:
            peak_memory_gb = mx.get_peak_memory() / (1024**3)
        except AttributeError:
            peak_memory_gb = mx.metal.get_peak_memory() / (1024**3)
    except (ImportError, AttributeError):
        pass

    # Check threshold for 200 residues
    threshold_seconds = 600.0  # 10 minutes
    passed = True
    if size == 200 and num_samples == 5:
        passed = total_time < threshold_seconds

    return BenchmarkResult(
        size=size,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        total_seconds=total_time,
        stage_timings=timing_data.stages,
        peak_memory_gb=peak_memory_gb,
        precision=precision,
        passed_threshold=passed,
    )


def _create_test_input(size: int, output_dir: Path) -> Path:
    """Create a test input file for a given size.

    Args:
        size: Number of residues.
        output_dir: Directory to write the file.

    Returns:
        Path to created file.
    """
    # Generate a sequence of the given size
    base_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"
    full_seq = (base_seq * ((size // len(base_seq)) + 1))[:size]

    data = {
        "name": f"test_{size}",
        "modelSeeds": [42],
        "sequences": [{"protein": {"id": "A", "sequence": full_seq}}],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"test_{size}.json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    headers = ["Size", "Samples", "Steps", "Time (s)", "Time (min)", "Status"]
    widths = [8, 8, 8, 12, 12, 10]

    # Print header
    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Print results
    for r in results:
        time_min = r.total_seconds / 60
        status = "PASS" if r.passed_threshold else "FAIL"

        row = [
            str(r.size).center(widths[0]),
            str(r.num_samples).center(widths[1]),
            str(r.diffusion_steps).center(widths[2]),
            f"{r.total_seconds:.1f}".center(widths[3]),
            f"{time_min:.2f}".center(widths[4]),
            status.center(widths[5]),
        ]
        print(" | ".join(row))

    print("=" * 80)

    # Print detailed stage timings for each result
    print("\nSTAGE TIMINGS:")
    for r in results:
        print(f"\n  {r.size} residues:")
        for stage, time in r.stage_timings.items():
            print(f"    {stage}: {time:.1f}s")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark AlphaFold 3 MLX pipeline performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Protein size in residues (default: 200)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of structure samples (default: 5)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=200,
        help="Number of diffusion steps (default: 200)",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Compute precision (default: auto-select)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run benchmarks for all standard sizes (50, 100, 200, 500)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 sample, 50 steps",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Print platform info
    platform_info = get_platform_info()
    print("\nPlatform Information:")
    print(f"  Chip: {platform_info['chip_family']}")
    print(f"  Memory: {platform_info['total_memory_gb']:.1f} GB")
    print(f"  bfloat16 support: {platform_info['supports_bfloat16']}")

    # Determine benchmark parameters
    if args.quick:
        num_samples = 1
        diffusion_steps = 50
    else:
        num_samples = args.num_samples
        diffusion_steps = args.diffusion_steps

    # Run benchmarks
    results = []

    if args.all:
        sizes = [50, 100, 200]  # Skip 500 for typical runs
        for size in sizes:
            try:
                result = run_benchmark(
                    size=size,
                    num_samples=num_samples,
                    diffusion_steps=diffusion_steps,
                    precision=args.precision,
                    output_dir=args.output_dir,
                )
                results.append(result)
            except Exception as e:
                print(f"Benchmark failed for size {size}: {e}")
    else:
        result = run_benchmark(
            size=args.size,
            num_samples=num_samples,
            diffusion_steps=diffusion_steps,
            precision=args.precision,
            output_dir=args.output_dir,
        )
        results.append(result)

    # Print results
    print_results(results)

    # Return exit code based on threshold
    for r in results:
        if r.size == 200 and r.num_samples == 5 and not r.passed_threshold:
            print(f"\nFAILED: 200 residues took {r.total_seconds:.1f}s (> 600s threshold)")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
