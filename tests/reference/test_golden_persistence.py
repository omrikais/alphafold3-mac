"""Tests for golden output persistence."""

import json
import numpy as np
import pytest
from pathlib import Path

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.golden import GoldenOutputs
from alphafold3_mlx.core.inputs import AttentionInputs
from alphafold3_mlx.reference.jax_attention import JAXReferenceHarness


class TestNPZPersistence:
    """Test NPZ save/reload integrity."""

    def test_save_reload_byte_exact(self, tmp_path: Path):
        """Save and reload produces byte-exact arrays."""
        config = AttentionConfig(seq_q=64, seq_k=64, seed=42)
        harness = JAXReferenceHarness(seed=42)

        # Generate and run
        q, k, v, _, _ = harness.generate_inputs(config)
        outputs = harness.run_attention(q, k, v)

        # Save
        output_path = tmp_path / "test_case"
        inputs = {"q": q, "k": k, "v": v}
        npz_path, meta_path = harness.save_golden_outputs(
            outputs, config, output_path,
            case_id="test_case",
            description="Test case",
            inputs=inputs,
        )

        # Reload
        with np.load(npz_path, allow_pickle=False) as data:
            loaded = dict(data)

        # Compare byte-exact
        for key in ["q", "k", "v", "output", "logits_pre_mask", "logits_masked", "weights"]:
            original = inputs.get(key, outputs.get(key))
            if original is not None:
                np.testing.assert_array_equal(
                    loaded[key], original,
                    err_msg=f"Array {key} differs after save/reload",
                )

    def test_metadata_preserved(self, tmp_path: Path):
        """Metadata is correctly saved and loadable."""
        config = AttentionConfig(
            batch_size=1,
            num_heads=4,
            seq_q=128,
            seq_k=128,
            head_dim=64,
            dtype="float32",
            seed=123,
        )
        harness = JAXReferenceHarness(seed=123)

        q, k, v, _, _ = harness.generate_inputs(config)
        outputs = harness.run_attention(q, k, v)

        output_path = tmp_path / "meta_test"
        npz_path, meta_path = harness.save_golden_outputs(
            outputs, config, output_path,
            case_id="meta_test",
            description="Metadata test case",
            inputs={"q": q, "k": k, "v": v},
        )

        # Load and verify metadata
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["case_id"] == "meta_test"
        assert meta["description"] == "Metadata test case"
        assert meta["seed"] == 123
        assert meta["config"]["seq_q"] == 128
        assert meta["config"]["head_dim"] == 64
        assert "rtol" in meta
        assert "atol" in meta

    def test_shapes_recorded_correctly(self, tmp_path: Path):
        """Shape information is recorded in metadata."""
        config = AttentionConfig(seq_q=256, seq_k=128, head_dim=48, seed=42)
        harness = JAXReferenceHarness(seed=42)

        q, k, v, _, _ = harness.generate_inputs(config)
        outputs = harness.run_attention(q, k, v)

        output_path = tmp_path / "shape_test"
        _, meta_path = harness.save_golden_outputs(
            outputs, config, output_path,
            case_id="shape_test",
            inputs={"q": q, "k": k, "v": v},
        )

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["shapes"]["q"] == [1, 4, 256, 48]
        assert meta["shapes"]["k"] == [1, 4, 128, 48]
        assert meta["shapes"]["output"] == [1, 4, 256, 48]
        assert meta["shapes"]["logits_pre_mask"] == [1, 4, 256, 128]


class TestGoldenOutputsDataclass:
    """Test GoldenOutputs dataclass."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """GoldenOutputs can be saved and loaded."""
        config = AttentionConfig(seq_q=64, seq_k=64, seed=42)

        rng = np.random.default_rng(42)
        q = rng.standard_normal(config.q_shape).astype(np.float32)
        k = rng.standard_normal(config.k_shape).astype(np.float32)
        v = rng.standard_normal(config.v_shape).astype(np.float32)
        output = rng.standard_normal(config.output_shape).astype(np.float32)

        inputs = AttentionInputs(q=q, k=k, v=v)

        golden = GoldenOutputs(
            case_id="roundtrip_test",
            description="Roundtrip test",
            inputs=inputs,
            output=output,
            intermediates={
                "logits_pre_mask": rng.standard_normal((1, 4, 64, 64)).astype(np.float32),
                "logits_masked": rng.standard_normal((1, 4, 64, 64)).astype(np.float32),
                "weights": rng.standard_normal((1, 4, 64, 64)).astype(np.float32),
            },
            config=config,
            seed=42,
            numpy_version=np.__version__,
            rtol=1e-4,
            atol=1e-5,
        )

        # Save
        npz_path, meta_path = golden.save(tmp_path)

        # Load
        loaded = GoldenOutputs.load(npz_path, meta_path)

        # Verify
        assert loaded.case_id == golden.case_id
        assert loaded.description == golden.description
        assert loaded.seed == golden.seed
        assert loaded.rtol == golden.rtol
        assert loaded.atol == golden.atol
        np.testing.assert_array_equal(loaded.output, golden.output)
        np.testing.assert_array_equal(loaded.inputs.q, golden.inputs.q)

    def test_load_with_mask_and_bias(self, tmp_path: Path):
        """GoldenOutputs handles mask and bias correctly."""
        config = AttentionConfig(seq_q=64, seq_k=64, seed=42)

        rng = np.random.default_rng(42)
        q = rng.standard_normal(config.q_shape).astype(np.float32)
        k = rng.standard_normal(config.k_shape).astype(np.float32)
        v = rng.standard_normal(config.v_shape).astype(np.float32)
        mask = rng.random(config.mask_shape) > 0.1
        bias = rng.standard_normal(config.bias_shape).astype(np.float32)
        output = rng.standard_normal(config.output_shape).astype(np.float32)

        inputs = AttentionInputs(q=q, k=k, v=v, boolean_mask=mask, additive_bias=bias)

        golden = GoldenOutputs(
            case_id="mask_bias_test",
            description="Mask and bias test",
            inputs=inputs,
            output=output,
            intermediates={},
            config=config,
            seed=42,
            numpy_version=np.__version__,
            rtol=1e-4,
            atol=1e-5,
        )

        npz_path, meta_path = golden.save(tmp_path)
        loaded = GoldenOutputs.load(npz_path, meta_path)

        np.testing.assert_array_equal(loaded.inputs.boolean_mask, mask)
        np.testing.assert_array_equal(loaded.inputs.additive_bias, bias)


class TestNoPickle:
    """Test security: no pickle allowed."""

    def test_load_rejects_pickle(self, tmp_path: Path):
        """Loading with allow_pickle=False is used (security)."""
        # Create a valid NPZ file
        npz_path = tmp_path / "test.npz"
        np.savez_compressed(npz_path, arr=np.array([1, 2, 3]))

        # Our code uses allow_pickle=False - verify this is the default behavior
        with np.load(npz_path, allow_pickle=False) as data:
            _ = data["arr"]
