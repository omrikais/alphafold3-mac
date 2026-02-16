"""Cross-backend validation utilities.

This module provides tools for comparing MLX attention outputs against
JAX reference outputs within specified tolerances.
"""

import numpy as np

from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.constants import TOLERANCES
from alphafold3_mlx.core.outputs import AttentionOutput
from alphafold3_mlx.core.validation import ValidationResult


class AttentionValidator:
    """Cross-backend validator for attention implementations.

    Compares MLX output against JAX reference output using np.allclose
    with configurable tolerances. Reports detailed error metrics on failure.
    """

    def compare(
        self,
        mlx_output: AttentionOutput,
        jax_output: dict[str, np.ndarray],
        *,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        config: AttentionConfig | None = None,
    ) -> ValidationResult:
        """Compare MLX output against JAX reference.

        Args:
            mlx_output: Output from MLX attention spike
            jax_output: Golden output dictionary from JAX harness
            rtol: Relative tolerance
            atol: Absolute tolerance
            config: Optional configuration for metadata

        Returns:
            ValidationResult with pass/fail and error metrics
        """
        # Convert MLX output to numpy
        mlx_arrays = mlx_output.to_numpy()

        # Track per-tensor results
        tensor_results = {}
        all_passed = True
        max_abs_diff = 0.0
        total_abs_diff = 0.0
        total_elements = 0

        # Compare output tensor (required)
        output_passed, output_max, output_mean, output_count = self._compare_tensor(
            mlx_arrays["output"],
            jax_output["output"],
            rtol=rtol,
            atol=atol,
        )
        tensor_results["output"] = output_passed
        all_passed = all_passed and output_passed
        max_abs_diff = max(max_abs_diff, output_max)
        total_abs_diff += output_mean * output_count
        total_elements += output_count

        # Compare intermediates if both have them
        intermediate_keys = ["logits_pre_mask", "logits_masked", "weights"]
        for key in intermediate_keys:
            mlx_tensor = mlx_arrays.get(key)
            jax_tensor = jax_output.get(key)

            if mlx_tensor is not None and jax_tensor is not None:
                passed, t_max, t_mean, t_count = self._compare_tensor(
                    mlx_tensor,
                    jax_tensor,
                    rtol=rtol,
                    atol=atol,
                )
                tensor_results[key] = passed
                all_passed = all_passed and passed
                max_abs_diff = max(max_abs_diff, t_max)
                total_abs_diff += t_mean * t_count
                total_elements += t_count

        # Calculate mean absolute difference
        mean_abs_diff = total_abs_diff / total_elements if total_elements > 0 else 0.0

        return ValidationResult(
            passed=all_passed,
            rtol=rtol,
            atol=atol,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            tensor_results=tensor_results,
            config=config,
        )

    def _compare_tensor(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        *,
        rtol: float,
        atol: float,
    ) -> tuple[bool, float, float, int]:
        """Compare two tensors and return metrics.

        Args:
            actual: Actual tensor from MLX
            expected: Expected tensor from JAX
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            Tuple of (passed, max_abs_diff, mean_abs_diff, element_count)
        """
        # Ensure same shape
        if actual.shape != expected.shape:
            return False, float("inf"), float("inf"), 0

        # Calculate absolute differences
        abs_diff = np.abs(actual - expected)
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        element_count = actual.size

        # Check allclose
        passed = np.allclose(actual, expected, rtol=rtol, atol=atol)

        return passed, max_abs_diff, mean_abs_diff, element_count


def validate_attention(
    mlx_output: np.ndarray,
    jax_output: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationResult:
    """Convenience function for simple output comparison.

    Args:
        mlx_output: MLX attention output as numpy array
        jax_output: JAX attention output as numpy array
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ValidationResult
    """
    # Calculate metrics
    abs_diff = np.abs(mlx_output - jax_output)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    passed = np.allclose(mlx_output, jax_output, rtol=rtol, atol=atol)

    return ValidationResult(
        passed=passed,
        rtol=rtol,
        atol=atol,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        tensor_results={"output": passed},
    )


def get_tolerances_for_dtype(dtype: str) -> dict[str, float]:
    """Get appropriate tolerances for a data type.

    Args:
        dtype: Data type string ("float32", "float16", "bfloat16")

    Returns:
        Dictionary with "rtol" and "atol" keys
    """
    return TOLERANCES.get(dtype, TOLERANCES["float32"])
