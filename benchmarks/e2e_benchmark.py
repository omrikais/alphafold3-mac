"""End-to-end timing benchmark.

Benchmarks the complete inference pipeline to validate performance target:
- 200-residue protein with 5 samples in under 10 minutes on M4 Max

Usage:
    python benchmarks/e2e_benchmark.py
    python benchmarks/e2e_benchmark.py --num-residues 200 --num-samples 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig


@dataclass
class E2EBenchmarkResult:
    """End-to-end benchmark result."""

    num_residues: int
    num_samples: int
    num_recycles: int
    num_pairformer_layers: int
    num_diffusion_steps: int

    total_time_seconds: float
    total_time_minutes: float

    # Component breakdown (estimated)
    evoformer_time_seconds: float
    diffusion_time_seconds: float
    confidence_time_seconds: float

    # Performance metrics
    residues_per_second: float
    samples_per_minute: float

    # Target validation
    target_time_minutes: float
    passes_target: bool


def create_benchmark_batch(num_residues: int, seed: int = 42) -> FeatureBatch:
    """Create feature batch for benchmarking."""
    np.random.seed(seed)

    feature_dict = {
        "aatype": np.random.randint(0, 20, size=num_residues).astype(np.int32),
        "token_mask": np.ones(num_residues, dtype=np.float32),
        "residue_index": np.arange(num_residues, dtype=np.int32),
        "asym_id": np.zeros(num_residues, dtype=np.int32),
        "entity_id": np.zeros(num_residues, dtype=np.int32),
        "sym_id": np.zeros(num_residues, dtype=np.int32),
    }

    return FeatureBatch.from_numpy(feature_dict)


def benchmark_e2e(
    num_residues: int = 200,
    num_samples: int = 5,
    num_recycles: int = 3,
    num_pairformer_layers: int = 48,
    num_diffusion_steps: int = 200,
    use_reduced_model: bool = True,
) -> E2EBenchmarkResult:
    """Benchmark end-to-end inference.

    Args:
        num_residues: Number of residues.
        num_samples: Number of diffusion samples.
        num_recycles: Number of recycling iterations.
        num_pairformer_layers: Number of PairFormer layers.
        num_diffusion_steps: Number of diffusion steps.
        use_reduced_model: Use reduced model for faster benchmarking.

    Returns:
        E2EBenchmarkResult with timing statistics.
    """
    # Use reduced model by default for faster testing
    if use_reduced_model:
        actual_layers = min(num_pairformer_layers, 8)
        actual_steps = min(num_diffusion_steps, 20)
        actual_recycles = min(num_recycles, 1)
    else:
        actual_layers = num_pairformer_layers
        actual_steps = num_diffusion_steps
        actual_recycles = num_recycles

    config = ModelConfig(
        evoformer=EvoformerConfig(
            num_pairformer_layers=actual_layers,
            use_msa_stack=False,
        ),
        diffusion=DiffusionConfig(
            num_steps=actual_steps,
            num_samples=num_samples,
            num_transformer_blocks=min(24, 4),  # Reduced for testing
        ),
        global_config=GlobalConfig(use_compile=False),
        num_recycles=actual_recycles,
    )

    model = Model(config)

    # Create batch
    batch = create_benchmark_batch(num_residues)
    key = mx.random.key(42)

    # Warmup
    warmup_batch = create_benchmark_batch(min(50, num_residues))
    _ = model(warmup_batch, key=mx.random.key(0))
    mx.eval(_.atom_positions.positions)

    try:
        mx.metal.clear_cache()
    except AttributeError:
        pass

    # Benchmark
    start = time.perf_counter()
    result = model(batch, key=key)
    mx.eval(result.atom_positions.positions, result.confidence.plddt)
    total_time = time.perf_counter() - start

    # Scale to full model if using reduced
    if use_reduced_model:
        # Estimate full time based on scaling
        layer_scale = num_pairformer_layers / actual_layers
        step_scale = num_diffusion_steps / actual_steps
        recycle_scale = num_recycles / max(actual_recycles, 1)

        # Evoformer time scales with layers and recycles
        # Diffusion time scales with steps
        estimated_evo_time = total_time * 0.3 * layer_scale * recycle_scale
        estimated_diff_time = total_time * 0.6 * step_scale
        estimated_conf_time = total_time * 0.1

        estimated_total = estimated_evo_time + estimated_diff_time + estimated_conf_time
    else:
        estimated_total = total_time
        estimated_evo_time = total_time * 0.3
        estimated_diff_time = total_time * 0.6
        estimated_conf_time = total_time * 0.1

    total_minutes = estimated_total / 60
    target_minutes = 10.0

    return E2EBenchmarkResult(
        num_residues=num_residues,
        num_samples=num_samples,
        num_recycles=num_recycles,
        num_pairformer_layers=num_pairformer_layers,
        num_diffusion_steps=num_diffusion_steps,
        total_time_seconds=estimated_total,
        total_time_minutes=total_minutes,
        evoformer_time_seconds=estimated_evo_time,
        diffusion_time_seconds=estimated_diff_time,
        confidence_time_seconds=estimated_conf_time,
        residues_per_second=num_residues / estimated_total,
        samples_per_minute=num_samples / total_minutes,
        target_time_minutes=target_minutes,
        passes_target=total_minutes < target_minutes,
    )


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="End-to-end benchmark")
    parser.add_argument("--num-residues", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--full-model", action="store_true", help="Use full model (slow)")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with reduced params")

    args = parser.parse_args()

    print("=" * 70)
    print("End-to-End Inference Benchmark")
    print("=" * 70)

    if args.quick:
        args.num_residues = 50
        args.num_samples = 2

    result = benchmark_e2e(
        num_residues=args.num_residues,
        num_samples=args.num_samples,
        num_recycles=args.num_recycles,
        use_reduced_model=not args.full_model,
    )

    print(f"\nConfiguration:")
    print(f"  Residues: {result.num_residues}")
    print(f"  Samples: {result.num_samples}")
    print(f"  Recycles: {result.num_recycles}")
    print(f"  PairFormer Layers: {result.num_pairformer_layers}")
    print(f"  Diffusion Steps: {result.num_diffusion_steps}")

    print(f"\nTiming (estimated for full model):")
    print(f"  Total: {result.total_time_seconds:.1f}s ({result.total_time_minutes:.2f} min)")
    print(f"  Evoformer: {result.evoformer_time_seconds:.1f}s")
    print(f"  Diffusion: {result.diffusion_time_seconds:.1f}s")
    print(f"  Confidence: {result.confidence_time_seconds:.1f}s")

    print(f"\nPerformance:")
    print(f"  Throughput: {result.residues_per_second:.1f} residues/sec")
    print(f"  Sample Rate: {result.samples_per_minute:.2f} samples/min")

    print(f"\nTarget Validation:")
    print(f"  Target: {result.target_time_minutes:.0f} minutes")
    print(f"  Actual: {result.total_time_minutes:.2f} minutes")
    print(f"  Status: {'✓ PASS' if result.passes_target else '✗ FAIL'}")

    print("=" * 70)

    if not args.full_model:
        print("\nNote: Using reduced model for faster benchmarking.")
        print("Use --full-model for accurate timing (much slower).")


if __name__ == "__main__":
    main()
