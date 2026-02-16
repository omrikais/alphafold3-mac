"""Precision validation tests for MLX attention spike.

These tests are mandatory - reduced precision must match within tolerance.
"""

import mlx.core as mx
import numpy as np
import pytest

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import TOLERANCES
from alphafold3_mlx.reference.jax_attention import JAXReferenceHarness, jax_scaled_dot_product_attention
from alphafold3_mlx.spike.attention import MLXAttentionSpike
from alphafold3_mlx.validation.validator import AttentionValidator


class TestFloat16Precision:
    """Test float16 precision handling.

    Mandatory: Both backends must use identical float16 inputs and match within tolerance.
    """

    def test_float16_basic(self):
        """Float16 inputs produce valid outputs without NaN/Inf."""
        batch, heads, seq, head_dim = 1, 4, 256, 64
        mx.random.seed(42)

        q = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.float16)
        k = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.float16)
        v = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.float16)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np)), "Float16 output contains NaN"
        assert not np.any(np.isinf(output_np)), "Float16 output contains Inf"

    def test_float16_matches_jax_with_identical_inputs(self):
        """Float16 MLX output matches JAX with identical float16 inputs.

        Both backends receive the SAME float16 inputs to ensure fair comparison.
        This is mandatory.
        """
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 256, 64

        # Generate inputs in float32, then cast to float16
        harness = JAXReferenceHarness(seed=42)
        config = AttentionConfig(seq_q=seq, seq_k=seq, seed=42)
        q_f32, k_f32, v_f32, _, _ = harness.generate_inputs(config)

        # Cast to float16 for BOTH backends (numpy float16 is cross-platform)
        q_f16_np = q_f32.astype(np.float16)
        k_f16_np = k_f32.astype(np.float16)
        v_f16_np = v_f32.astype(np.float16)

        # Run JAX with float16 inputs
        jax_output, _ = jax_scaled_dot_product_attention(
            jnp.array(q_f16_np),
            jnp.array(k_f16_np),
            jnp.array(v_f16_np),
            capture_intermediates=False,
        )
        jax_output_np = np.array(jax_output)

        # Run MLX with float16 inputs
        q_f16_mlx = mx.array(q_f16_np).astype(mx.float16)
        k_f16_mlx = mx.array(k_f16_np).astype(mx.float16)
        v_f16_mlx = mx.array(v_f16_np).astype(mx.float16)

        spike = MLXAttentionSpike()
        result = spike(q_f16_mlx, k_f16_mlx, v_f16_mlx)
        mx.eval(result.output)
        mlx_output_np = np.array(result.output)

        # Compare with float16 tolerances - MANDATORY
        tols = TOLERANCES["float16"]
        max_diff = float(np.max(np.abs(mlx_output_np - jax_output_np)))
        mean_diff = float(np.mean(np.abs(mlx_output_np - jax_output_np)))

        passed = np.allclose(mlx_output_np, jax_output_np, rtol=tols["rtol"], atol=tols["atol"])

        assert passed, (
            f"Float16 validation FAILED (mandatory): "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
            f"rtol={tols['rtol']}, atol={tols['atol']}"
        )


class TestBFloat16Precision:
    """Test bfloat16 precision handling.

    Mandatory: Both backends must use identical bfloat16 inputs and match within tolerance.
    Note: bfloat16 is cast to float32 for JAX since JAX on CPU may not support bfloat16 natively.
    """

    def test_bfloat16_basic(self):
        """BFloat16 inputs produce valid outputs without NaN/Inf."""
        batch, heads, seq, head_dim = 1, 4, 256, 64
        mx.random.seed(42)

        q = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.bfloat16)
        k = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.bfloat16)
        v = mx.random.normal((batch, heads, seq, head_dim)).astype(mx.bfloat16)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        # Convert to float32 first (NumPy doesn't support bfloat16 natively)
        output_f32 = result.output.astype(mx.float32)
        mx.eval(output_f32)
        output_np = np.array(output_f32)
        assert not np.any(np.isnan(output_np)), "BFloat16 output contains NaN"
        assert not np.any(np.isinf(output_np)), "BFloat16 output contains Inf"

    def test_bfloat16_matches_jax_with_identical_inputs(self):
        """BFloat16 MLX output matches JAX with identical inputs.

        Since NumPy/JAX-CPU may not fully support bfloat16, we:
        1. Generate inputs in float32
        2. Cast to bfloat16 in MLX
        3. Use the bfloat16→float32 roundtrip values for JAX to ensure identical inputs

        This is mandatory.
        """
        import jax.numpy as jnp

        batch, heads, seq, head_dim = 1, 4, 256, 64

        # Generate inputs in float32
        harness = JAXReferenceHarness(seed=42)
        config = AttentionConfig(seq_q=seq, seq_k=seq, seed=42)
        q_f32, k_f32, v_f32, _, _ = harness.generate_inputs(config)

        # Cast to bfloat16 in MLX, then back to float32 for JAX
        # This ensures both backends see the same quantized values
        q_bf16_mlx = mx.array(q_f32).astype(mx.bfloat16)
        k_bf16_mlx = mx.array(k_f32).astype(mx.bfloat16)
        v_bf16_mlx = mx.array(v_f32).astype(mx.bfloat16)
        mx.eval(q_bf16_mlx, k_bf16_mlx, v_bf16_mlx)

        # Convert bfloat16 back to float32 for JAX (captures quantization)
        q_for_jax = np.array(q_bf16_mlx.astype(mx.float32))
        k_for_jax = np.array(k_bf16_mlx.astype(mx.float32))
        v_for_jax = np.array(v_bf16_mlx.astype(mx.float32))

        # Run JAX with the quantized inputs
        jax_output, _ = jax_scaled_dot_product_attention(
            jnp.array(q_for_jax),
            jnp.array(k_for_jax),
            jnp.array(v_for_jax),
            capture_intermediates=False,
        )
        jax_output_np = np.array(jax_output)

        # Run MLX with bfloat16 inputs
        spike = MLXAttentionSpike()
        result = spike(q_bf16_mlx, k_bf16_mlx, v_bf16_mlx)
        mx.eval(result.output)

        # Convert output to float32 for comparison
        mlx_output_f32 = result.output.astype(mx.float32)
        mx.eval(mlx_output_f32)
        mlx_output_np = np.array(mlx_output_f32)

        # Compare with bfloat16 tolerances - MANDATORY
        tols = TOLERANCES["bfloat16"]
        max_diff = float(np.max(np.abs(mlx_output_np - jax_output_np)))
        mean_diff = float(np.mean(np.abs(mlx_output_np - jax_output_np)))

        passed = np.allclose(mlx_output_np, jax_output_np, rtol=tols["rtol"], atol=tols["atol"])

        assert passed, (
            f"BFloat16 validation FAILED (mandatory): "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
            f"rtol={tols['rtol']}, atol={tols['atol']}"
        )


class TestMixedPrecisionHandling:
    """Test handling of mixed precision inputs."""

    def test_mixed_precision_produces_output(self):
        """Mixed precision inputs still produce valid output (with warning)."""
        batch, heads, seq, head_dim = 1, 4, 64, 32

        q = mx.ones((batch, heads, seq, head_dim), dtype=mx.float32)
        k = mx.ones((batch, heads, seq, head_dim), dtype=mx.float16)
        v = mx.ones((batch, heads, seq, head_dim), dtype=mx.bfloat16)

        spike = MLXAttentionSpike()
        result = spike(q, k, v)
        mx.eval(result.output)

        output_np = np.array(result.output)
        assert not np.any(np.isnan(output_np)), "Mixed precision output contains NaN"
