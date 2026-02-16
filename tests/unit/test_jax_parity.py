"""JAX parity validation tests.

Tests that MLX implementations produce numerically equivalent results
to JAX CPU references within specified tolerances.
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx
from pathlib import Path


class TestAttentionParity:
    """Test MLX attention parity with JAX."""

    GOLDEN_FILE = Path("tests/fixtures/model_golden/jax_attention_reference.npz")

    @pytest.fixture
    def golden_data(self):
        """Load JAX reference data if available."""
        if not self.GOLDEN_FILE.exists():
            pytest.skip(
                f"JAX reference not found: {self.GOLDEN_FILE}. "
                "Run: python scripts/generate_jax_model_references.py"
            )
        return np.load(self.GOLDEN_FILE)

    def test_attention_no_mask_parity(self, golden_data):
        """Test attention without mask matches JAX.

        Validates that MLX scaled dot-product attention produces
        numerically equivalent results to JAX within tolerance.
        """
        # Load inputs
        q = mx.array(golden_data["query"])
        k = mx.array(golden_data["key"])
        v = mx.array(golden_data["value"])
        head_dim = int(golden_data["head_dim"])

        # Compute attention in MLX
        scale = 1.0 / np.sqrt(head_dim)
        scores = mx.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["output_no_mask"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="MLX attention output differs from JAX reference",
        )

    def test_attention_with_mask_bias_parity(self, golden_data):
        """Test attention with mask and bias matches JAX."""
        # Load inputs
        q = mx.array(golden_data["query"])
        k = mx.array(golden_data["key"])
        v = mx.array(golden_data["value"])
        mask = mx.array(golden_data["mask"])
        bias = mx.array(golden_data["bias"])
        head_dim = int(golden_data["head_dim"])

        # Compute attention in MLX
        scale = 1.0 / np.sqrt(head_dim)
        scores = mx.einsum("bhqd,bhkd->bhqk", q, k) * scale
        scores = scores + bias

        # Apply mask (additive masking)
        mask_value = -1e9
        scores = mx.where(mask > 0.5, scores, mask_value)

        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["output_with_mask_bias"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="MLX masked attention output differs from JAX reference",
        )


class TestPairFormerOpsParity:
    """Test MLX PairFormer operations parity with JAX."""

    GOLDEN_FILE = Path("tests/fixtures/model_golden/jax_pairformer_ops_reference.npz")

    @pytest.fixture
    def golden_data(self):
        """Load JAX reference data if available."""
        if not self.GOLDEN_FILE.exists():
            pytest.skip(
                f"JAX reference not found: {self.GOLDEN_FILE}. "
                "Run: python scripts/generate_jax_model_references.py"
            )
        return np.load(self.GOLDEN_FILE)

    def test_linear_projection_parity(self, golden_data):
        """Test linear projection matches JAX."""
        # Load inputs
        single = mx.array(golden_data["single_input"])
        weight = mx.array(golden_data["linear_weight"])
        bias = mx.array(golden_data["linear_bias"])

        # Compute in MLX
        output = mx.matmul(single, weight) + bias
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["linear_output"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="MLX linear projection differs from JAX reference",
        )

    def test_layernorm_parity(self, golden_data):
        """Test LayerNorm matches JAX."""
        # Load inputs
        single = mx.array(golden_data["single_input"])

        # Compute LayerNorm in MLX
        eps = 1e-5
        mean = mx.mean(single, axis=-1, keepdims=True)
        var = mx.var(single, axis=-1, keepdims=True)
        output = (single - mean) / mx.sqrt(var + eps)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["layernorm_output"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="MLX LayerNorm differs from JAX reference",
        )

    def test_outer_product_mean_parity(self, golden_data):
        """Test outer product mean matches JAX."""
        # Load inputs
        single = mx.array(golden_data["single_input"])
        proj_a = mx.array(golden_data["outer_proj_a"])
        proj_b = mx.array(golden_data["outer_proj_b"])
        out_proj = mx.array(golden_data["outer_out_proj"])

        batch_size = single.shape[0]
        seq_len = single.shape[1]

        # Compute outer product mean in MLX
        a = mx.matmul(single, proj_a)  # [batch, seq, 32]
        b = mx.matmul(single, proj_b)  # [batch, seq, 32]
        outer = mx.einsum("bic,bjd->bijcd", a, b)  # [batch, seq, seq, 32, 32]
        outer_flat = outer.reshape(batch_size, seq_len, seq_len, -1)
        output = mx.matmul(outer_flat, out_proj)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["outer_mean_output"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="MLX outer product mean differs from JAX reference",
        )

    def test_triangle_multiplication_parity(self, golden_data):
        """Test triangle multiplication matches JAX."""
        # Load inputs
        pair = mx.array(golden_data["pair_input"])
        proj_left = mx.array(golden_data["tri_proj_left"])
        proj_right = mx.array(golden_data["tri_proj_right"])
        gate_weight = mx.array(golden_data["tri_gate"])
        out_proj = mx.array(golden_data["tri_out"])

        # Compute triangle multiplication in MLX
        left = mx.matmul(pair, proj_left)
        right = mx.matmul(pair, proj_right)
        gate = mx.sigmoid(mx.matmul(pair, gate_weight))
        triangle = mx.einsum("bikc,bjkc->bijc", left, right)
        triangle = triangle * gate
        output = mx.matmul(triangle, out_proj)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["triangle_output"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="MLX triangle multiplication differs from JAX reference",
        )


class TestDiffusionStepParity:
    """Test MLX diffusion step parity with JAX."""

    GOLDEN_FILE = Path("tests/fixtures/model_golden/jax_diffusion_step_reference.npz")

    @pytest.fixture
    def golden_data(self):
        """Load JAX reference data if available."""
        if not self.GOLDEN_FILE.exists():
            pytest.skip(
                f"JAX reference not found: {self.GOLDEN_FILE}. "
                "Run: python scripts/generate_jax_model_references.py"
            )
        return np.load(self.GOLDEN_FILE)

    def test_noise_embedding_parity(self, golden_data):
        """Test noise level embedding matches JAX."""
        # Load inputs
        sigma = float(golden_data["sigma"])
        freq = mx.array(golden_data["freq"])

        # Compute Fourier embedding in MLX
        sigma_arr = mx.array([sigma])
        x = sigma_arr[:, None] * freq[None, :]
        output = mx.concatenate([mx.sin(x), mx.cos(x)], axis=-1)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["noise_embed"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="MLX noise embedding differs from JAX reference",
        )

    def test_coordinate_update_parity(self, golden_data):
        """Test coordinate update matches JAX."""
        # Load inputs
        coords = mx.array(golden_data["coords_input"])
        score = mx.array(golden_data["score_output"])
        sigma = float(golden_data["sigma"])
        sigma_next = float(golden_data["sigma_next"])

        # Compute coordinate update in MLX
        dt = sigma - sigma_next
        output = coords + score * dt
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["coords_output"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="MLX coordinate update differs from JAX reference",
        )


class TestConfidenceOpsParity:
    """Test MLX confidence operations parity with JAX."""

    GOLDEN_FILE = Path("tests/fixtures/model_golden/jax_confidence_ops_reference.npz")

    @pytest.fixture
    def golden_data(self):
        """Load JAX reference data if available."""
        if not self.GOLDEN_FILE.exists():
            pytest.skip(
                f"JAX reference not found: {self.GOLDEN_FILE}. "
                "Run: python scripts/generate_jax_model_references.py"
            )
        return np.load(self.GOLDEN_FILE)

    def test_plddt_computation_parity(self, golden_data):
        """Test pLDDT computation matches JAX."""
        # Load inputs
        logits = mx.array(golden_data["plddt_logits"])
        num_bins = int(golden_data["num_plddt_bins"])

        # Compute pLDDT in MLX
        probs = mx.softmax(logits, axis=-1)
        bin_centers = mx.linspace(0, 100, num_bins)
        output = mx.sum(probs * bin_centers, axis=-1)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["plddt"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="MLX pLDDT differs from JAX reference",
        )

    def test_pae_computation_parity(self, golden_data):
        """Test PAE computation matches JAX."""
        # Load inputs
        logits = mx.array(golden_data["pae_logits"])
        num_bins = int(golden_data["num_pae_bins"])

        # Compute PAE in MLX
        probs = mx.softmax(logits, axis=-1)
        bin_centers = mx.linspace(0, 32, num_bins)
        output = mx.sum(probs * bin_centers, axis=-1)
        mx.eval(output)

        # Compare to JAX reference
        output_np = np.array(output)
        expected = golden_data["pae"]

        np.testing.assert_allclose(
            output_np,
            expected,
            rtol=1e-4,
            atol=1e-5,
            err_msg="MLX PAE differs from JAX reference",
        )
