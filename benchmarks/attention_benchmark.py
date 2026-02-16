"""Performance benchmark for MLX attention spike.

Measures execution time and memory usage for go/no-go decision.
Memory ratio <= 2.0 is BLOCKING; timing is ADVISORY.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphafold3_mlx.core.benchmark import BenchmarkResult, theoretical_minimum_bytes
from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import AF3_SHAPES, MEMORY_RATIO_THRESHOLD
from alphafold3_mlx.spike.attention import MLXAttentionSpike


def run_benchmark(
    config: AttentionConfig,
    num_warmup: int = 3,
    num_runs: int = 10,
    use_mask: bool = False,
    use_bias: bool = False,
) -> BenchmarkResult:
    """Run benchmark for a single configuration.

    Args:
        config: Attention configuration
        num_warmup: Number of warmup runs (not timed)
        num_runs: Number of timed runs
        use_mask: Include boolean mask
        use_bias: Include additive bias

    Returns:
        BenchmarkResult with timing and memory metrics
    """
    # Generate inputs
    mx.random.seed(config.seed)
    q = mx.random.normal(config.q_shape)
    k = mx.random.normal(config.k_shape)
    v = mx.random.normal(config.v_shape)

    mask = None
    if use_mask:
        mask = mx.random.uniform(shape=config.mask_shape) > 0.1

    bias = None
    if use_bias:
        bias = mx.random.normal(config.bias_shape) * 0.1

    spike = MLXAttentionSpike()

    # Warmup runs
    for _ in range(num_warmup):
        result = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
        mx.eval(result.output)

    # Clear memory tracking
    mx.clear_cache()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
        mx.eval(result.output)
        end = time.perf_counter()
        times.append(end - start)

    # Get peak memory
    peak_memory = mx.get_peak_memory()

    # Calculate average time
    avg_time = sum(times) / len(times)

    return BenchmarkResult.create(
        config=config,
        execution_time_s=avg_time,
        peak_memory_bytes=peak_memory,
        use_bias=use_bias,
    )


def run_all_benchmarks(
    shapes: list[dict] = None,
    output_path: Path = None,
) -> list[BenchmarkResult]:
    """Run benchmarks for all AF3 shapes and configurations.

    Args:
        shapes: List of shape dicts (defaults to AF3_SHAPES)
        output_path: Optional path to save JSON results

    Returns:
        List of BenchmarkResult objects
    """
    if shapes is None:
        shapes = AF3_SHAPES

    results = []
    configurations = [
        ("no_mask_no_bias", False, False),
        ("mask_only", True, False),
        ("bias_only", False, True),
        ("mask_and_bias", True, True),
    ]

    print(f"Running benchmarks for {len(shapes)} shapes, {len(configurations)} configs...")
    print(f"Memory threshold: {MEMORY_RATIO_THRESHOLD}x (BLOCKING)")
    print()

    for shape in shapes:
        config = AttentionConfig(
            batch_size=shape["batch"],
            num_heads=shape["heads"],
            seq_q=shape["seq"],
            seq_k=shape["seq"],
            head_dim=shape["head_dim"],
        )

        for config_name, use_mask, use_bias in configurations:
            result = run_benchmark(config, use_mask=use_mask, use_bias=use_bias)
            results.append(result)

            status = "PASS" if result.memory_within_threshold else "FAIL"
            print(
                f"[{status}] seq={shape['seq']:4d} | {config_name:16s} | "
                f"time={result.execution_time_s:.4f}s | "
                f"memory={result.memory_ratio:.2f}x"
            )

    # Summary
    print()
    print("=" * 60)
    all_passed = all(r.memory_within_threshold for r in results)
    if all_passed:
        print("GO: All configurations pass memory threshold")
    else:
        failed = [r for r in results if not r.memory_within_threshold]
        print(f"NO-GO: {len(failed)} configurations exceed memory threshold")
        for r in failed:
            print(f"  - seq={r.config.seq_q}, memory_ratio={r.memory_ratio:.2f}x")

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "all_passed": all_passed,
                        "threshold": MEMORY_RATIO_THRESHOLD,
                        "num_configurations": len(results),
                    },
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {output_path}")

    return results


def check_go_no_go(results: list[BenchmarkResult]) -> bool:
    """Check if all benchmarks pass the memory threshold.

    Args:
        results: List of benchmark results

    Returns:
        True if all pass (GO), False otherwise (NO-GO)
    """
    return all(r.memory_within_threshold for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLX attention performance benchmarks"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--seq",
        type=int,
        nargs="+",
        default=None,
        help="Specific sequence lengths to test (default: 256, 512, 1024)",
    )
    args = parser.parse_args()

    # Build shapes from args
    if args.seq:
        shapes = [
            {"batch": 1, "heads": 4, "seq": s, "head_dim": 64}
            for s in args.seq
        ]
    else:
        shapes = None

    results = run_all_benchmarks(shapes=shapes, output_path=args.output)

    # Exit with appropriate code
    sys.exit(0 if check_go_no_go(results) else 1)
