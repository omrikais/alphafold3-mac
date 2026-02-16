"""PairFormer benchmark.

Benchmarks the PairFormer block performance to identify optimization
opportunities and validate performance targets.

Usage:
    python benchmarks/pairformer_benchmark.py
    python benchmarks/pairformer_benchmark.py --num-residues 500 --num-iterations 10
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from alphafold3_mlx.network.pairformer import PairFormerStack
from alphafold3_mlx.core.config import GlobalConfig


@dataclass
class BenchmarkResult:
    """Benchmark result."""

    name: str
    num_residues: int
    num_layers: int
    mean_time_ms: float
    std_time_ms: float
    throughput_residues_per_sec: float
    memory_gb: float


def benchmark_pairformer(
    num_residues: int = 200,
    num_layers: int = 48,
    num_iterations: int = 10,
    warmup: int = 3,
) -> BenchmarkResult:
    """Benchmark PairFormer stack.

    Args:
        num_residues: Number of residues.
        num_layers: Number of PairFormer layers.
        num_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        BenchmarkResult with timing statistics.
    """
    config = GlobalConfig(use_compile=False)

    # Create model
    pairformer = PairFormerStack(
        num_layers=num_layers,
        seq_channel=384,
        pair_channel=128,
        share_weights=False,
    )

    # Create inputs
    batch_size = 1
    single = mx.random.normal(
        shape=(batch_size, num_residues, 384),
        key=mx.random.key(0),
    )
    pair = mx.random.normal(
        shape=(batch_size, num_residues, num_residues, 128),
        key=mx.random.key(1),
    )
    seq_mask = mx.ones((batch_size, num_residues))
    pair_mask = mx.ones((batch_size, num_residues, num_residues))

    # Warmup
    for _ in range(warmup):
        out_single, out_pair = pairformer(single, pair, seq_mask, pair_mask)
        mx.eval(out_single, out_pair)

    # Clear cache
    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        out_single, out_pair = pairformer(single, pair, seq_mask, pair_mask)
        mx.eval(out_single, out_pair)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    # Memory estimate
    try:
        memory_bytes = mx.metal.get_peak_memory()
        memory_gb = memory_bytes / (1024**3)
    except AttributeError:
        memory_gb = 0.0

    times_np = np.array(times)
    mean_time = float(np.mean(times_np))
    std_time = float(np.std(times_np))
    throughput = num_residues / (mean_time / 1000)

    return BenchmarkResult(
        name="PairFormerStack",
        num_residues=num_residues,
        num_layers=num_layers,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput_residues_per_sec=throughput,
        memory_gb=memory_gb,
    )


def benchmark_single_layer(num_residues: int = 200) -> BenchmarkResult:
    """Benchmark single PairFormer layer."""
    return benchmark_pairformer(
        num_residues=num_residues,
        num_layers=1,
        num_iterations=20,
        warmup=5,
    )


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="PairFormer benchmark")
    parser.add_argument("--num-residues", type=int, default=200)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sweep", action="store_true", help="Run size sweep")

    args = parser.parse_args()

    print("=" * 60)
    print("PairFormer Benchmark")
    print("=" * 60)

    if args.sweep:
        sizes = [50, 100, 200, 500]
        for size in sizes:
            result = benchmark_pairformer(
                num_residues=size,
                num_layers=args.num_layers,
                num_iterations=5,
                warmup=2,
            )
            print(
                f"N={result.num_residues:4d}: "
                f"{result.mean_time_ms:8.1f} ± {result.std_time_ms:6.1f} ms, "
                f"{result.throughput_residues_per_sec:8.0f} res/s"
            )
    else:
        result = benchmark_pairformer(
            num_residues=args.num_residues,
            num_layers=args.num_layers,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
        )

        print(f"Configuration:")
        print(f"  Residues: {result.num_residues}")
        print(f"  Layers: {result.num_layers}")
        print()
        print(f"Results:")
        print(f"  Time: {result.mean_time_ms:.1f} ± {result.std_time_ms:.1f} ms")
        print(f"  Throughput: {result.throughput_residues_per_sec:.0f} residues/sec")
        if result.memory_gb > 0:
            print(f"  Peak Memory: {result.memory_gb:.2f} GB")

    print("=" * 60)


if __name__ == "__main__":
    main()
