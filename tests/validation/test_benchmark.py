"""Benchmark validation tests."""

import mlx.core as mx
import numpy as np
import pytest

from alphafold3_mlx.core.benchmark import BenchmarkResult, theoretical_minimum_bytes
from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import AF3_SHAPES, MEMORY_RATIO_THRESHOLD
from alphafold3_mlx.spike.attention import MLXAttentionSpike


@pytest.mark.benchmark
class TestMemoryThreshold:
    """Test memory usage within threshold.

    Note: These tests are hardware-dependent and may fail on different systems.
    Run with: pytest -m benchmark to include these tests.
    """

    @pytest.mark.parametrize("shape", [s for s in AF3_SHAPES if s["seq"] >= 1024])
    def test_memory_within_threshold_no_bias(self, shape: dict):
        """Peak memory / theoretical minimum <= 2.0 for AF3 shapes (no bias).

        Note: Only tests seq>=1024 because smaller sequences have high measurement
        variance due to fixed overhead from MLX runtime. The benchmark script
        tests all sizes with appropriate caveats.
        """
        config = AttentionConfig(
            batch_size=shape["batch"],
            num_heads=shape["heads"],
            seq_q=shape["seq"],
            seq_k=shape["seq"],
            head_dim=shape["head_dim"],
        )

        # Generate inputs
        mx.random.seed(42)
        q = mx.random.normal(config.q_shape)
        k = mx.random.normal(config.k_shape)
        v = mx.random.normal(config.v_shape)

        spike = MLXAttentionSpike()

        # Multiple warmup runs to stabilize GPU state
        for _ in range(3):
            warmup = spike(q, k, v)
            mx.eval(warmup.output)

        # Clear and measure
        mx.clear_cache()
        mx.reset_peak_memory()
        result = spike(q, k, v)
        mx.eval(result.output)
        peak_memory = mx.get_peak_memory()

        # Calculate ratio (no bias)
        theoretical_min = theoretical_minimum_bytes(config, include_bias=False)
        ratio = peak_memory / theoretical_min if theoretical_min > 0 else float("inf")

        assert ratio <= MEMORY_RATIO_THRESHOLD, (
            f"Memory ratio {ratio:.2f}x exceeds threshold {MEMORY_RATIO_THRESHOLD}x "
            f"for seq={shape['seq']} (no bias)"
        )

    @pytest.mark.parametrize("shape", [s for s in AF3_SHAPES if s["seq"] >= 1024])
    def test_memory_within_threshold_with_bias(self, shape: dict):
        """Peak memory / theoretical minimum <= 2.0 for AF3 shapes (with bias).

        Note: Only tests seq>=1024 because smaller sequences have high measurement
        variance due to fixed overhead from MLX runtime. The benchmark script
        tests all sizes with appropriate caveats.
        """
        config = AttentionConfig(
            batch_size=shape["batch"],
            num_heads=shape["heads"],
            seq_q=shape["seq"],
            seq_k=shape["seq"],
            head_dim=shape["head_dim"],
        )

        # Generate inputs
        mx.random.seed(42)
        q = mx.random.normal(config.q_shape)
        k = mx.random.normal(config.k_shape)
        v = mx.random.normal(config.v_shape)
        bias = mx.random.normal(config.bias_shape) * 0.1

        spike = MLXAttentionSpike()

        # Multiple warmup runs to stabilize GPU state
        for _ in range(3):
            warmup = spike(q, k, v, additive_bias=bias)
            mx.eval(warmup.output)

        # Clear and measure
        mx.clear_cache()
        mx.reset_peak_memory()
        result = spike(q, k, v, additive_bias=bias)
        mx.eval(result.output)
        peak_memory = mx.get_peak_memory()

        # Calculate ratio (with bias)
        theoretical_min = theoretical_minimum_bytes(config, include_bias=True)
        ratio = peak_memory / theoretical_min if theoretical_min > 0 else float("inf")

        assert ratio <= MEMORY_RATIO_THRESHOLD, (
            f"Memory ratio {ratio:.2f}x exceeds threshold {MEMORY_RATIO_THRESHOLD}x "
            f"for seq={shape['seq']} (with bias)"
        )

    @pytest.mark.parametrize("shape", [s for s in AF3_SHAPES if s["seq"] >= 1024])
    def test_memory_within_threshold_mask_and_bias(self, shape: dict):
        """Peak memory / theoretical minimum <= 2.0 for AF3 shapes (mask + bias).

        Note: Only tests seq>=1024 because smaller sequences have high measurement
        variance due to fixed overhead. The benchmark script tests all sizes.
        """
        config = AttentionConfig(
            batch_size=shape["batch"],
            num_heads=shape["heads"],
            seq_q=shape["seq"],
            seq_k=shape["seq"],
            head_dim=shape["head_dim"],
        )

        # Generate inputs
        mx.random.seed(42)
        q = mx.random.normal(config.q_shape)
        k = mx.random.normal(config.k_shape)
        v = mx.random.normal(config.v_shape)
        mask = mx.random.uniform(shape=config.mask_shape) > 0.1
        bias = mx.random.normal(config.bias_shape) * 0.1

        spike = MLXAttentionSpike()

        # Multiple warmup runs to stabilize GPU state
        for _ in range(3):
            warmup = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
            mx.eval(warmup.output)

        # Clear and measure
        mx.clear_cache()
        mx.reset_peak_memory()
        result = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
        mx.eval(result.output)
        peak_memory = mx.get_peak_memory()

        # Calculate ratio (with bias - mask doesn't add significant memory)
        theoretical_min = theoretical_minimum_bytes(config, include_bias=True)
        ratio = peak_memory / theoretical_min if theoretical_min > 0 else float("inf")

        assert ratio <= MEMORY_RATIO_THRESHOLD, (
            f"Memory ratio {ratio:.2f}x exceeds threshold {MEMORY_RATIO_THRESHOLD}x "
            f"for seq={shape['seq']} (mask + bias)"
        )


class TestBenchmarkResultStructure:
    """Test BenchmarkResult captures all required fields."""

    def test_benchmark_result_has_required_fields(self):
        """BenchmarkResult has all fields per data-model.md."""
        config = AttentionConfig(seq_q=64, seq_k=64)

        result = BenchmarkResult.create(
            config=config,
            execution_time_s=0.001,
            peak_memory_bytes=1000000,
        )

        # Check all required fields exist
        assert hasattr(result, "config")
        assert hasattr(result, "execution_time_s")
        assert hasattr(result, "peak_memory_bytes")
        assert hasattr(result, "theoretical_minimum_bytes")
        assert hasattr(result, "memory_ratio")
        assert hasattr(result, "memory_within_threshold")

        # Check types
        assert isinstance(result.config, AttentionConfig)
        assert isinstance(result.execution_time_s, float)
        assert isinstance(result.peak_memory_bytes, int)
        assert isinstance(result.theoretical_minimum_bytes, int)
        assert isinstance(result.memory_ratio, float)
        assert isinstance(result.memory_within_threshold, bool)

    def test_benchmark_result_to_dict(self):
        """BenchmarkResult serializes to dictionary correctly."""
        config = AttentionConfig(seq_q=64, seq_k=64)

        result = BenchmarkResult.create(
            config=config,
            execution_time_s=0.001,
            peak_memory_bytes=1000000,
        )

        result_dict = result.to_dict()

        assert "config" in result_dict
        assert "execution_time_s" in result_dict
        assert "peak_memory_bytes" in result_dict
        assert "theoretical_minimum_bytes" in result_dict
        assert "memory_ratio" in result_dict
        assert "memory_within_threshold" in result_dict

        # Config should be serialized
        assert isinstance(result_dict["config"], dict)
        assert "seq_q" in result_dict["config"]

    def test_memory_ratio_calculation(self):
        """Memory ratio is calculated correctly."""
        config = AttentionConfig(
            batch_size=1,
            num_heads=4,
            seq_q=256,
            seq_k=256,
            head_dim=64,
        )

        # Calculate expected theoretical minimum (updated formula with logits)
        # Q + K + V + output = 4 tensors: 4 * 1 * 4 * 256 * 64 * 4 bytes = 1048576 bytes
        # Logits = 1 * 4 * 256 * 256 * 4 bytes = 1048576 bytes
        # Total = 2097152 bytes
        expected_theoretical = 2097152

        result = BenchmarkResult.create(
            config=config,
            execution_time_s=0.001,
            peak_memory_bytes=expected_theoretical * 2,  # 2x ratio
        )

        assert result.theoretical_minimum_bytes == expected_theoretical
        assert abs(result.memory_ratio - 2.0) < 0.01
        assert result.memory_within_threshold  # 2.0 <= 2.0

    def test_memory_within_threshold_flag(self):
        """memory_within_threshold flag is set correctly."""
        config = AttentionConfig(seq_q=64, seq_k=64)

        # Under threshold
        result_pass = BenchmarkResult.create(
            config=config,
            execution_time_s=0.001,
            peak_memory_bytes=100000,  # Small
        )
        assert result_pass.memory_within_threshold

        # Over threshold
        theoretical = theoretical_minimum_bytes(config)
        result_fail = BenchmarkResult.create(
            config=config,
            execution_time_s=0.001,
            peak_memory_bytes=int(theoretical * 10),  # 10x, way over
        )
        assert not result_fail.memory_within_threshold


class TestTheoreticalMinimum:
    """Test theoretical minimum calculation."""

    def test_theoretical_minimum_formula(self):
        """Theoretical minimum follows correct formula (with logits)."""
        config = AttentionConfig(
            batch_size=2,
            num_heads=8,
            seq_q=128,
            seq_k=128,
            head_dim=32,
            dtype="float32",
        )

        # Updated formula:
        # Q + K + V + output: 4 * batch * heads * seq * head_dim * bytes
        # Logits: batch * heads * seq_q * seq_k * bytes
        # Base tensors = 4 * 2 * 8 * 128 * 32 * 4 = 1048576 bytes
        # Logits = 2 * 8 * 128 * 128 * 4 = 1048576 bytes
        # Total = 2097152 bytes
        expected = 2097152

        actual = theoretical_minimum_bytes(config)
        assert actual == expected

    def test_theoretical_minimum_cross_attention(self):
        """Theoretical minimum handles cross-attention correctly."""
        config = AttentionConfig(
            batch_size=1,
            num_heads=4,
            seq_q=256,
            seq_k=128,  # Different from seq_q
            head_dim=64,
            dtype="float32",
        )

        # Q and output: 1 * 4 * 256 * 64 * 4 = 262144 bytes each = 524288
        # K and V: 1 * 4 * 128 * 64 * 4 = 131072 bytes each = 262144
        # Logits: 1 * 4 * 256 * 128 * 4 = 524288 bytes
        # Total = 524288 + 262144 + 524288 = 1310720 bytes
        expected = 1310720

        actual = theoretical_minimum_bytes(config)
        assert actual == expected

    def test_theoretical_minimum_float16(self):
        """Theoretical minimum uses float32 regardless of input dtype.

        The policy requires float32 internal computation for numerical stability,
        so the memory baseline must reflect actual float32 allocations even
        when inputs are float16/bfloat16.
        """
        config_f32 = AttentionConfig(seq_q=64, seq_k=64, dtype="float32")
        config_f16 = AttentionConfig(seq_q=64, seq_k=64, dtype="float16")

        min_f32 = theoretical_minimum_bytes(config_f32)
        min_f16 = theoretical_minimum_bytes(config_f16)

        # Both should be equal since upcasts to float32 internally
        assert min_f16 == min_f32

    def test_theoretical_minimum_with_bias(self):
        """Theoretical minimum includes bias when specified."""
        config = AttentionConfig(
            batch_size=1,
            num_heads=4,
            seq_q=256,
            seq_k=256,
            head_dim=64,
            dtype="float32",
        )

        min_no_bias = theoretical_minimum_bytes(config, include_bias=False)
        min_with_bias = theoretical_minimum_bytes(config, include_bias=True)

        # Bias adds seq_q * seq_k * heads * batch * bytes = 256 * 256 * 4 * 1 * 4 = 1048576
        expected_diff = 1048576
        assert min_with_bias - min_no_bias == expected_diff
