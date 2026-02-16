"""JAX reference harness for golden output generation.

This module implements a pure JAX/NumPy attention computation for CPU validation.
The harness produces deterministic outputs with fixed seeds and captures
intermediate activations for validation.
"""

import json
import math
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import TOLERANCES, DEFAULT_MASK_VALUE


# Force JAX to use CPU backend for deterministic outputs
jax.config.update("jax_platform_name", "cpu")


class AttentionOutputs(TypedDict):
    """Type definition for attention outputs dictionary."""
    output: np.ndarray
    logits_pre_mask: np.ndarray
    logits_masked: np.ndarray
    weights: np.ndarray


def jax_scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    boolean_mask: jnp.ndarray | None = None,
    additive_bias: jnp.ndarray | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    capture_intermediates: bool = True,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray] | None]:
    """Pure JAX scaled dot-product attention for CPU validation.

    Implements: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + bias) * V

    Args:
        q: Query tensor [batch, heads, seq_q, head_dim]
        k: Key tensor [batch, heads, seq_k, head_dim]
        v: Value tensor [batch, heads, seq_k, head_dim]
        boolean_mask: Optional mask [batch, seq_k] where True=attend, False=mask
        additive_bias: Optional bias [batch, heads, seq_q, seq_k]
        mask_value: Large value for masked positions (default: 1e9)
        capture_intermediates: If True, return intermediate activations

    Returns:
        Tuple of (output, intermediates) where intermediates is None if not captured
    """
    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    # Cast to float32 for numerical stability
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    # Compute scaled dot-product logits: QK^T / sqrt(d_k)
    # Using einsum: [batch, heads, seq_q, head_dim] @ [batch, heads, seq_k, head_dim].T
    # → [batch, heads, seq_q, seq_k]
    logits_pre_mask = jnp.einsum("bhqd,bhkd->bhqk", q * scale, k)

    # Start with logits before any masking
    logits = logits_pre_mask.copy()

    # Apply boolean mask (AF3's pattern: True = attend, False = mask)
    if boolean_mask is not None:
        # Convert boolean mask to additive mask: True → 0, False → -mask_value
        # Shape: [batch, seq_k] → [batch, 1, 1, seq_k] for broadcasting
        additive_mask = mask_value * (boolean_mask.astype(jnp.float32) - 1.0)
        logits = logits + additive_mask[:, None, None, :]

    # Apply additive bias
    if additive_bias is not None:
        logits = logits + additive_bias.astype(jnp.float32)

    logits_masked = logits

    # Softmax over keys dimension
    weights = jax.nn.softmax(logits, axis=-1)

    # Handle fully masked rows: NaN → 0
    # This happens when all positions in a row are masked
    weights = jnp.where(jnp.isnan(weights), 0.0, weights)

    # Weighted sum of values
    # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
    # → [batch, heads, seq_q, head_dim]
    output = jnp.einsum("bhqk,bhkd->bhqd", weights, v_f32)

    # Explicitly zero fully-masked rows
    # When all keys are masked (all False), the output must be zeros.
    # With finite mask_value (-1e9), softmax produces uniform weights, not NaN.
    # We must detect and zero these rows explicitly.
    if boolean_mask is not None:
        # boolean_mask: [batch, seq_k] where True=attend, False=mask
        # If ALL positions are False for a batch element, zero that output
        any_attend = jnp.any(boolean_mask, axis=-1, keepdims=True)  # [batch, 1]
        any_attend = any_attend[:, None, None, :]  # [batch, 1, 1, 1]
        output = jnp.where(any_attend, output, jnp.zeros_like(output))

    if capture_intermediates:
        intermediates = {
            "logits_pre_mask": logits_pre_mask,
            "logits_masked": logits_masked,
            "weights": weights,
        }
        return output, intermediates

    return output, None


class JAXReferenceHarness:
    """JAX reference harness for generating golden outputs.

    This harness runs on JAX CPU backend and produces deterministic outputs
    with fixed seeds. It captures intermediate activations for validation.
    """

    def __init__(self, seed: int = 42):
        """Initialize harness with random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng_key = jax.random.PRNGKey(seed)

    def generate_inputs(
        self,
        config: AttentionConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Generate deterministic test inputs.

        Args:
            config: Attention configuration

        Returns:
            Tuple of (q, k, v, boolean_mask, additive_bias) as numpy arrays
        """
        # Split keys for each tensor
        keys = jax.random.split(self._rng_key, 5)

        # Map dtype string to numpy dtype
        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float32,  # Generate as float32, cast later if needed
        }
        np_dtype = dtype_map.get(config.dtype, np.float32)

        # Generate Q, K, V
        q = np.array(jax.random.normal(keys[0], config.q_shape)).astype(np_dtype)
        k = np.array(jax.random.normal(keys[1], config.k_shape)).astype(np_dtype)
        v = np.array(jax.random.normal(keys[2], config.v_shape)).astype(np_dtype)

        boolean_mask = None
        additive_bias = None

        return q, k, v, boolean_mask, additive_bias

    def generate_inputs_with_mask(
        self,
        config: AttentionConfig,
        mask_ratio: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Generate inputs with boolean mask.

        Args:
            config: Attention configuration
            mask_ratio: Fraction of positions to mask (0.0 to 1.0)

        Returns:
            Tuple of (q, k, v, boolean_mask, additive_bias)
        """
        q, k, v, _, _ = self.generate_inputs(config)

        # Generate boolean mask
        keys = jax.random.split(self._rng_key, 6)
        mask_probs = jax.random.uniform(keys[3], config.mask_shape)
        boolean_mask = np.array(mask_probs > mask_ratio)

        return q, k, v, boolean_mask, None

    def generate_inputs_with_bias(
        self,
        config: AttentionConfig,
        bias_scale: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """Generate inputs with additive bias.

        Args:
            config: Attention configuration
            bias_scale: Scale factor for bias values

        Returns:
            Tuple of (q, k, v, boolean_mask, additive_bias)
        """
        q, k, v, _, _ = self.generate_inputs(config)

        # Generate additive bias
        keys = jax.random.split(self._rng_key, 6)
        dtype_map = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}
        np_dtype = dtype_map.get(config.dtype, np.float32)
        additive_bias = np.array(
            jax.random.normal(keys[4], config.bias_shape) * bias_scale
        ).astype(np_dtype)

        return q, k, v, None, additive_bias

    def generate_inputs_with_mask_and_bias(
        self,
        config: AttentionConfig,
        mask_ratio: float = 0.1,
        bias_scale: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate inputs with both mask and bias.

        Args:
            config: Attention configuration
            mask_ratio: Fraction of positions to mask
            bias_scale: Scale factor for bias values

        Returns:
            Tuple of (q, k, v, boolean_mask, additive_bias)
        """
        q, k, v, boolean_mask, _ = self.generate_inputs_with_mask(config, mask_ratio)
        _, _, _, _, additive_bias = self.generate_inputs_with_bias(config, bias_scale)

        return q, k, v, boolean_mask, additive_bias

    def run_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        *,
        boolean_mask: np.ndarray | None = None,
        additive_bias: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Run JAX attention and capture intermediates.

        Args:
            q, k, v: Input tensors as numpy arrays
            boolean_mask: Optional boolean mask
            additive_bias: Optional additive bias

        Returns:
            Dictionary with keys: output, logits_pre_mask, logits_masked, weights
        """
        # Convert to JAX arrays
        q_jax = jnp.array(q)
        k_jax = jnp.array(k)
        v_jax = jnp.array(v)

        mask_jax = jnp.array(boolean_mask) if boolean_mask is not None else None
        bias_jax = jnp.array(additive_bias) if additive_bias is not None else None

        # Run attention
        output, intermediates = jax_scaled_dot_product_attention(
            q_jax, k_jax, v_jax,
            boolean_mask=mask_jax,
            additive_bias=bias_jax,
            capture_intermediates=True,
        )

        # Convert back to numpy
        result = {
            "output": np.array(output),
        }
        if intermediates is not None:
            for key, value in intermediates.items():
                result[key] = np.array(value)

        return result

    def save_golden_outputs(
        self,
        outputs: dict[str, np.ndarray],
        config: AttentionConfig,
        output_path: str,
        *,
        case_id: str | None = None,
        description: str = "",
        inputs: dict[str, np.ndarray] | None = None,
    ) -> tuple[Path, Path]:
        """Save golden outputs to NPZ with metadata.

        Args:
            outputs: Dictionary of numpy arrays (output, intermediates)
            config: Configuration used to generate outputs
            output_path: Path to save NPZ file (without extension)
            case_id: Optional case identifier (defaults to output_path stem)
            description: Optional description
            inputs: Optional input arrays to save alongside outputs

        Returns:
            Tuple of (npz_path, meta_path)
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if case_id is None:
            case_id = output_path.stem

        npz_path = output_path.with_suffix(".npz")
        meta_path = output_path.with_suffix(".meta.json")

        # Prepare arrays
        arrays = {}

        # Add inputs if provided
        if inputs is not None:
            for key, arr in inputs.items():
                if arr is not None:
                    arrays[key] = np.ascontiguousarray(arr)

        # Add outputs
        for key, arr in outputs.items():
            arrays[key] = np.ascontiguousarray(arr)

        # Save NPZ
        np.savez_compressed(npz_path, **arrays)

        # Get tolerances for this dtype
        tols = TOLERANCES.get(config.dtype, TOLERANCES["float32"])

        # Prepare metadata
        meta = {
            "case_id": case_id,
            "description": description,
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
            "config": config.to_dict(),
            "seed": self.seed,
            "numpy_version": np.__version__,
            "rtol": tols["rtol"],
            "atol": tols["atol"],
        }

        # Save metadata
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return npz_path, meta_path


def generate_all_golden_outputs(output_dir: Path, seed: int = 42) -> list[tuple[str, Path]]:
    """Generate golden outputs for all test cases.

    Args:
        output_dir: Directory to save golden outputs
        seed: Random seed for reproducibility

    Returns:
        List of (case_id, npz_path) tuples
    """
    from alphafold3_mlx.core.constants import AF3_SHAPES

    harness = JAXReferenceHarness(seed=seed)
    output_dir = Path(output_dir)
    results = []

    # Test configurations
    test_cases = [
        ("no_mask_no_bias", False, False),
        ("mask_only", True, False),
        ("bias_only", False, True),
        ("mask_and_bias", True, True),
    ]

    # Generate for each shape and test case
    for shape in AF3_SHAPES:
        config = AttentionConfig(
            batch_size=shape["batch"],
            num_heads=shape["heads"],
            seq_q=shape["seq"],
            seq_k=shape["seq"],
            head_dim=shape["head_dim"],
            seed=seed,
        )

        for case_name, use_mask, use_bias in test_cases:
            case_id = f"{case_name}_seq{shape['seq']}"

            # Generate inputs
            if use_mask and use_bias:
                q, k, v, mask, bias = harness.generate_inputs_with_mask_and_bias(config)
            elif use_mask:
                q, k, v, mask, bias = harness.generate_inputs_with_mask(config)
            elif use_bias:
                q, k, v, mask, bias = harness.generate_inputs_with_bias(config)
            else:
                q, k, v, mask, bias = harness.generate_inputs(config)

            # Run attention
            outputs = harness.run_attention(q, k, v, boolean_mask=mask, additive_bias=bias)

            # Save
            inputs = {"q": q, "k": k, "v": v, "boolean_mask": mask, "additive_bias": bias}
            desc = f"{case_name} with seq={shape['seq']}"
            npz_path, _ = harness.save_golden_outputs(
                outputs, config, output_dir / case_id,
                case_id=case_id, description=desc, inputs=inputs,
            )

            results.append((case_id, npz_path))

    # Generate edge cases
    edge_cases = _generate_edge_case_golden_outputs(harness, output_dir)
    results.extend(edge_cases)

    return results


def _generate_edge_case_golden_outputs(
    harness: JAXReferenceHarness,
    output_dir: Path,
) -> list[tuple[str, Path]]:
    """Generate golden outputs for edge cases.

    Args:
        harness: JAX reference harness
        output_dir: Directory to save golden outputs

    Returns:
        List of (case_id, npz_path) tuples
    """
    results = []

    # all_masked - fully masked rows
    config = AttentionConfig(seq_q=256, seq_k=256, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    boolean_mask = np.zeros((1, 256), dtype=bool)  # All False = all masked
    outputs = harness.run_attention(q, k, v, boolean_mask=boolean_mask)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "all_masked",
        case_id="all_masked",
        description="Fully masked rows must return zeros",
        inputs={"q": q, "k": k, "v": v, "boolean_mask": boolean_mask},
    )
    results.append(("all_masked", npz_path))

    # large_bias_positive
    config = AttentionConfig(seq_q=256, seq_k=256, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    large_bias = np.full(config.bias_shape, 1e9, dtype=np.float32)
    outputs = harness.run_attention(q, k, v, additive_bias=large_bias)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "large_bias_positive",
        case_id="large_bias_positive",
        description="Large positive bias without overflow",
        inputs={"q": q, "k": k, "v": v, "additive_bias": large_bias},
    )
    results.append(("large_bias_positive", npz_path))

    # large_bias_negative
    config = AttentionConfig(seq_q=256, seq_k=256, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    large_bias = np.full(config.bias_shape, -1e9, dtype=np.float32)
    outputs = harness.run_attention(q, k, v, additive_bias=large_bias)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "large_bias_negative",
        case_id="large_bias_negative",
        description="Large negative bias without overflow",
        inputs={"q": q, "k": k, "v": v, "additive_bias": large_bias},
    )
    results.append(("large_bias_negative", npz_path))

    # seq_1
    config = AttentionConfig(seq_q=1, seq_k=1, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    outputs = harness.run_attention(q, k, v)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "seq_1",
        case_id="seq_1",
        description="seq_q=seq_k=1 produces valid output",
        inputs={"q": q, "k": k, "v": v},
    )
    results.append(("seq_1", npz_path))

    # cross_attention (seq_q != seq_k)
    config = AttentionConfig(seq_q=256, seq_k=128, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    outputs = harness.run_attention(q, k, v)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "cross_attention",
        case_id="cross_attention",
        description="seq_q != seq_k with correct logits shape",
        inputs={"q": q, "k": k, "v": v},
    )
    results.append(("cross_attention", npz_path))

    # non_power2_head_dim
    config = AttentionConfig(seq_q=256, seq_k=256, head_dim=48, seed=harness.seed)
    q, k, v, _, _ = harness.generate_inputs(config)
    outputs = harness.run_attention(q, k, v)
    npz_path, _ = harness.save_golden_outputs(
        outputs, config, output_dir / "non_power2_head_dim",
        case_id="non_power2_head_dim",
        description="head_dim=48 (not power of 2) produces correct outputs",
        inputs={"q": q, "k": k, "v": v},
    )
    results.append(("non_power2_head_dim", npz_path))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate golden outputs for attention validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Output directory for golden files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating golden outputs to {args.output_dir}")
    results = generate_all_golden_outputs(args.output_dir, seed=args.seed)

    print(f"\nGenerated {len(results)} golden output files:")
    for case_id, path in results:
        print(f"  {case_id}: {path}")
