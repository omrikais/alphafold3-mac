"""Evoformer benchmark.

Benchmarks the full Evoformer stack including recycling.

Usage:
    python benchmarks/evoformer_benchmark.py
    python benchmarks/evoformer_benchmark.py --num-residues 200 --num-recycles 3
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from alphafold3_mlx.network.evoformer import Evoformer
from alphafold3_mlx.core.config import EvoformerConfig, GlobalConfig


@dataclass
class EvoformerBenchmarkResult:
    """Evoformer benchmark result."""

    num_residues: int
    num_layers: int
    num_recycles: int
    mean_time_ms: float
    std_time_ms: float
    time_per_layer_ms: float
    time_per_recycle_ms: float
    throughput_residues_per_sec: float


def benchmark_evoformer(
    num_residues: int = 200,
    num_layers: int = 48,
    num_recycles: int = 3,
    num_iterations: int = 5,
    warmup: int = 2,
) -> EvoformerBenchmarkResult:
    """Benchmark Evoformer stack.

    Args:
        num_residues: Number of residues.
        num_layers: Number of PairFormer layers.
        num_recycles: Number of recycling iterations.
        num_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        EvoformerBenchmarkResult with timing statistics.
    """
    config = EvoformerConfig(
        num_pairformer_layers=num_layers,
        use_msa_stack=False,  # Skip MSA for faster benchmark
    )
    global_config = GlobalConfig(use_compile=False)

    # Create model
    evoformer = Evoformer(config=config, global_config=global_config)

    # Create inputs
    batch_size = 1
    single = mx.zeros((batch_size, num_residues, config.seq_channel))
    pair = mx.zeros((batch_size, num_residues, num_residues, config.pair_channel))
    residue_index = mx.arange(num_residues)[None, :]
    asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)
    seq_mask = mx.ones((batch_size, num_residues))
    pair_mask = mx.ones((batch_size, num_residues, num_residues))

    # Warmup
    for _ in range(warmup):
        out_single, out_pair, _ = evoformer(
            single, pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
        )
        mx.eval(out_single, out_pair)

    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        out_single, out_pair, _ = evoformer(
            single, pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
        )
        mx.eval(out_single, out_pair)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times_np = np.array(times)
    mean_time = float(np.mean(times_np))
    std_time = float(np.std(times_np))
    time_per_layer = mean_time / num_layers
    time_per_recycle = mean_time  # Single Evoformer pass
    throughput = num_residues / (mean_time / 1000)

    return EvoformerBenchmarkResult(
        num_residues=num_residues,
        num_layers=num_layers,
        num_recycles=1,  # Per-evoformer call
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        time_per_layer_ms=time_per_layer,
        time_per_recycle_ms=time_per_recycle,
        throughput_residues_per_sec=throughput,
    )


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="Evoformer benchmark")
    parser.add_argument("--num-residues", type=int, default=200)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("Evoformer Benchmark")
    print("=" * 60)

    if args.sweep:
        sizes = [50, 100, 200, 500]
        for size in sizes:
            result = benchmark_evoformer(
                num_residues=size,
                num_layers=args.num_layers,
                num_iterations=3,
                warmup=1,
            )
            print(
                f"N={result.num_residues:4d}: "
                f"{result.mean_time_ms:8.1f} ± {result.std_time_ms:6.1f} ms "
                f"({result.time_per_layer_ms:.2f} ms/layer)"
            )
    else:
        result = benchmark_evoformer(
            num_residues=args.num_residues,
            num_layers=args.num_layers,
            num_recycles=args.num_recycles,
            num_iterations=args.num_iterations,
        )

        print(f"Configuration:")
        print(f"  Residues: {result.num_residues}")
        print(f"  Layers: {result.num_layers}")
        print()
        print(f"Results (single Evoformer pass):")
        print(f"  Total Time: {result.mean_time_ms:.1f} ± {result.std_time_ms:.1f} ms")
        print(f"  Per Layer: {result.time_per_layer_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_residues_per_sec:.0f} residues/sec")
        print()
        # Estimate full recycle time
        estimated_recycle_time = result.mean_time_ms * args.num_recycles
        print(f"Estimated {args.num_recycles}-recycle time: {estimated_recycle_time:.0f} ms")

    print("=" * 60)


if __name__ == "__main__":
    main()
