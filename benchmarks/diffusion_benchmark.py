"""Diffusion loop benchmark.

Benchmarks the diffusion head and 200-step denoising loop.

Usage:
    python benchmarks/diffusion_benchmark.py
    python benchmarks/diffusion_benchmark.py --num-residues 200 --num-steps 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from alphafold3_mlx.network.diffusion_head import DiffusionHead
from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig


@dataclass
class DiffusionBenchmarkResult:
    """Diffusion benchmark result."""

    num_residues: int
    num_steps: int
    num_samples: int
    mean_time_ms: float
    std_time_ms: float
    time_per_step_ms: float
    time_per_sample_ms: float
    samples_per_second: float


def benchmark_diffusion(
    num_residues: int = 200,
    num_steps: int = 200,
    num_samples: int = 5,
    num_iterations: int = 3,
    warmup: int = 1,
) -> DiffusionBenchmarkResult:
    """Benchmark diffusion head.

    Args:
        num_residues: Number of residues/atoms.
        num_steps: Number of denoising steps.
        num_samples: Number of samples to generate.
        num_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.

    Returns:
        DiffusionBenchmarkResult with timing statistics.
    """
    config = DiffusionConfig(
        num_steps=num_steps,
        num_samples=num_samples,
        num_transformer_blocks=24,
    )
    global_config = GlobalConfig(use_compile=False)

    diffusion_head = DiffusionHead(config=config, global_config=global_config)

    # Create inputs
    batch_size = 1
    single = mx.random.normal(
        shape=(batch_size, num_residues, config.conditioning_seq_channel),
        key=mx.random.key(0),
    )
    pair = mx.random.normal(
        shape=(batch_size, num_residues, num_residues, config.conditioning_pair_channel),
        key=mx.random.key(1),
    )
    atom_types = mx.zeros((batch_size, num_residues), dtype=mx.int32)

    # Warmup (use fewer steps)
    warmup_config = DiffusionConfig(
        num_steps=min(10, num_steps),
        num_samples=1,
        num_transformer_blocks=config.num_transformer_blocks,
    )
    warmup_head = DiffusionHead(config=warmup_config, global_config=global_config)

    for i in range(warmup):
        key = mx.random.key(100 + i)
        coords = warmup_head.sample(
            single_cond=single,
            pair_cond=pair,
            atom_types=atom_types,
            key=key,
            num_samples=1,
        )
        mx.eval(coords)

    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass

    # Benchmark
    times = []
    for i in range(num_iterations):
        key = mx.random.key(42 + i)
        start = time.perf_counter()
        coords = diffusion_head.sample(
            single_cond=single,
            pair_cond=pair,
            atom_types=atom_types,
            key=key,
            num_samples=num_samples,
        )
        mx.eval(coords)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times_np = np.array(times)
    mean_time = float(np.mean(times_np))
    std_time = float(np.std(times_np))
    time_per_step = mean_time / num_steps
    time_per_sample = mean_time / num_samples
    samples_per_sec = num_samples / (mean_time / 1000)

    return DiffusionBenchmarkResult(
        num_residues=num_residues,
        num_steps=num_steps,
        num_samples=num_samples,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        time_per_step_ms=time_per_step,
        time_per_sample_ms=time_per_sample,
        samples_per_second=samples_per_sec,
    )


def benchmark_single_step(num_residues: int = 200) -> float:
    """Benchmark single denoising step.

    Returns:
        Time per step in milliseconds.
    """
    config = DiffusionConfig(
        num_steps=10,
        num_samples=1,
        num_transformer_blocks=24,
    )

    diffusion_head = DiffusionHead(config=config)

    batch_size = 1
    coords = mx.random.normal(
        shape=(batch_size, num_residues, 3),
        key=mx.random.key(0),
    )
    atom_types = mx.zeros((batch_size, num_residues), dtype=mx.int32)
    single = mx.zeros((batch_size, num_residues, 384))
    pair = mx.zeros((batch_size, num_residues, num_residues, 128))
    key = mx.random.key(42)

    # Warmup
    for _ in range(3):
        _ = diffusion_head.single_denoise_step(
            coords, atom_types, single, pair,
            sigma=1.0, sigma_next=0.5, key=key,
        )
        mx.eval(_)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = diffusion_head.single_denoise_step(
            coords, atom_types, single, pair,
            sigma=1.0, sigma_next=0.5, key=key,
        )
        mx.eval(_)
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="Diffusion benchmark")
    parser.add_argument("--num-residues", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=50)  # Reduced for faster benchmark
    parser.add_argument("--num-samples", type=int, default=1)  # Single sample by default
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="Run full 200-step benchmark")

    args = parser.parse_args()

    print("=" * 60)
    print("Diffusion Benchmark")
    print("=" * 60)

    if args.full:
        args.num_steps = 200
        args.num_samples = 5

    # Single step benchmark
    print("\nSingle Step Benchmark:")
    step_time = benchmark_single_step(args.num_residues)
    print(f"  Time per step: {step_time:.2f} ms")
    print(f"  Estimated 200 steps: {step_time * 200:.0f} ms")

    # Full diffusion benchmark
    print(f"\nFull Diffusion Benchmark ({args.num_steps} steps, {args.num_samples} samples):")
    result = benchmark_diffusion(
        num_residues=args.num_residues,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
    )

    print(f"  Total Time: {result.mean_time_ms:.0f} ± {result.std_time_ms:.0f} ms")
    print(f"  Per Step: {result.time_per_step_ms:.2f} ms")
    print(f"  Per Sample: {result.time_per_sample_ms:.0f} ms")
    print(f"  Throughput: {result.samples_per_second:.2f} samples/sec")

    # target check
    if args.full:
        target_time_minutes = 10
        actual_time_minutes = result.mean_time_ms / 1000 / 60
        print()
        print(f"Target (200 residues, 5 samples, <10 min):")
        print(f"  Actual: {actual_time_minutes:.1f} min")
        print(f"  Status: {'PASS' if actual_time_minutes < target_time_minutes else 'FAIL'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
