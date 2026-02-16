"""Intermediate activations dataclass for attention validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np


# Type alias for array-like
Array = Any


def _to_numpy_safe(arr: Any) -> np.ndarray:
    """Convert array to numpy, handling bfloat16 which NumPy doesn't support.

    Args:
        arr: Array-like object (MLX array, NumPy array, etc.)

    Returns:
        NumPy array (bfloat16 is converted to float32)
    """
    if hasattr(arr, 'dtype'):
        dtype_str = str(arr.dtype)
        if 'bfloat16' in dtype_str:
            import mlx.core as mx
            arr = arr.astype(mx.float32)
            mx.eval(arr)
    return np.asarray(arr)


@dataclass
class AttentionIntermediates:
    """Intermediate activations captured during attention computation.

    Attributes:
        logits_pre_mask: QK^T / sqrt(d_k) before mask/bias [batch, heads, seq_q, seq_k]
        logits_masked: Logits after mask and bias applied [batch, heads, seq_q, seq_k]
        weights: Attention weights after softmax [batch, heads, seq_q, seq_k]

    State Transitions:
        1. logits_pre_mask = QK^T / sqrt(d_k)
        2. logits_masked = logits_pre_mask + mask + bias
        3. weights = softmax(logits_masked)
    """

    logits_pre_mask: Array
    logits_masked: Array
    weights: Array

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert to numpy arrays for persistence.

        Note: bfloat16 arrays are automatically converted to float32
        since NumPy doesn't support bfloat16 natively.
        """
        return {
            "logits_pre_mask": _to_numpy_safe(self.logits_pre_mask),
            "logits_masked": _to_numpy_safe(self.logits_masked),
            "weights": _to_numpy_safe(self.weights),
        }

    @classmethod
    def from_numpy(cls, data: dict[str, np.ndarray]) -> "AttentionIntermediates":
        """Create from numpy arrays."""
        return cls(
            logits_pre_mask=data["logits_pre_mask"],
            logits_masked=data["logits_masked"],
            weights=data["weights"],
        )

    def validate_shapes(self) -> None:
        """Validate that all intermediates have consistent shapes."""
        assert self.logits_pre_mask.shape == self.logits_masked.shape, (
            f"logits_pre_mask and logits_masked shapes must match: "
            f"{self.logits_pre_mask.shape} vs {self.logits_masked.shape}"
        )
        assert self.logits_masked.shape == self.weights.shape, (
            f"logits_masked and weights shapes must match: "
            f"{self.logits_masked.shape} vs {self.weights.shape}"
        )
