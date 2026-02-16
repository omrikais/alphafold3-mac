"""MLX attention spike implementation.

This module implements scaled dot-product attention using MLX's fused SDPA kernel.
It matches AlphaFold 3's attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + bias) * V
"""

import logging
import math
from typing import Any

import mlx.core as mx
import numpy as np

from alphafold3_mlx.core.constants import DEFAULT_MASK_VALUE
from alphafold3_mlx.core.intermediates import AttentionIntermediates
from alphafold3_mlx.core.outputs import AttentionOutput


logger = logging.getLogger(__name__)


def mlx_scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    boolean_mask: mx.array | None = None,
    additive_bias: mx.array | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    capture_intermediates: bool = False,
) -> AttentionOutput:
    """MLX scaled dot-product attention.

    Implements: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + bias) * V

    Uses mx.fast.scaled_dot_product_attention for fused GPU kernel execution.

    Args:
        q: Query tensor [batch, heads, seq_q, head_dim]
        k: Key tensor [batch, heads, seq_k, head_dim]
        v: Value tensor [batch, heads, seq_k, head_dim]
        boolean_mask: Optional mask [batch, seq_k] where True=attend, False=mask
        additive_bias: Optional bias [batch, heads, seq_q, seq_k]
        mask_value: Large value for masked positions (default: 1e9)
        capture_intermediates: If True, compute and return intermediate activations

    Returns:
        AttentionOutput with output tensor and optional intermediates

    Note:
        - Inputs are cast to float32 internally for numerical stability
        - Output is cast back to input dtype
        - Fully masked rows will produce zeros
    """
    # Validate dtypes match (performance warning)
    dtypes = {q.dtype, k.dtype, v.dtype}
    if len(dtypes) > 1:
        logger.warning(
            f"Mixed dtypes detected: {dtypes}. "
            "Pre-cast all inputs to same dtype for best performance."
        )

    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # Cast to float32 for numerical stability in SDPA
    # This ensures softmax accumulation is done in float32 regardless of input precision
    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)
    v_f32 = v.astype(mx.float32)

    # Build combined mask for SDPA
    combined_mask = None

    # Convert boolean mask to additive mask
    # True (attend) → 0, False (mask) → -mask_value
    if boolean_mask is not None:
        # Shape: [batch, seq_k] → [batch, 1, 1, seq_k] for broadcasting
        additive_mask = mask_value * (boolean_mask.astype(mx.float32) - 1.0)
        additive_mask = additive_mask[:, None, None, :]
        combined_mask = additive_mask

    # Add bias if provided
    if additive_bias is not None:
        if combined_mask is not None:
            combined_mask = combined_mask + additive_bias.astype(mx.float32)
        else:
            combined_mask = additive_bias.astype(mx.float32)

    # Capture intermediates if requested (manual computation for validation)
    intermediates = None
    if capture_intermediates:
        intermediates = _compute_intermediates(
            q_f32, k_f32, scale, combined_mask
        )

    # Run fused SDPA kernel with float32 inputs for numerical stability
    output = mx.fast.scaled_dot_product_attention(
        q_f32, k_f32, v_f32,
        scale=scale,
        mask=combined_mask,
    )

    # Cast output back to input dtype if not float32
    if q.dtype != mx.float32:
        output = output.astype(q.dtype)

    # Handle fully masked rows: NaN → 0
    # This happens when all positions in a query row are masked
    output = mx.where(mx.isnan(output), mx.zeros_like(output), output)

    # Explicitly zero fully-masked rows
    # When all keys are masked (all False), the output must be zeros.
    # With finite mask_value (-1e9), softmax produces uniform weights, not NaN.
    # We must detect and zero these rows explicitly.
    if boolean_mask is not None:
        # boolean_mask: [batch, seq_k] where True=attend, False=mask
        # If ALL positions are False for a batch element, zero that output
        # any_attend: [batch] - True if at least one position can be attended
        any_attend = mx.any(boolean_mask, axis=-1, keepdims=True)  # [batch, 1]
        # Broadcast to output shape: [batch, heads, seq_q, head_dim]
        any_attend = any_attend[:, None, None, :]  # [batch, 1, 1, 1]
        output = mx.where(any_attend, output, mx.zeros_like(output))

    return AttentionOutput(output=output, intermediates=intermediates)


def _compute_intermediates(
    q: mx.array,
    k: mx.array,
    scale: float,
    combined_mask: mx.array | None,
) -> AttentionIntermediates:
    """Compute intermediate activations for validation.

    This is a manual computation separate from SDPA to capture:
    - logits_pre_mask: QK^T / sqrt(d_k)
    - logits_masked: logits + mask + bias
    - weights: softmax(logits_masked)

    Args:
        q: Query tensor
        k: Key tensor
        scale: 1/sqrt(head_dim)
        combined_mask: Combined additive mask (or None)

    Returns:
        AttentionIntermediates with captured values
    """
    # Cast to float32 for numerical stability
    q_f32 = q.astype(mx.float32)
    k_f32 = k.astype(mx.float32)

    # Compute logits: QK^T / sqrt(d_k)
    # Using einsum: [batch, heads, seq_q, head_dim] @ [batch, heads, seq_k, head_dim].T
    # → [batch, heads, seq_q, seq_k]
    logits_pre_mask = mx.einsum("bhqd,bhkd->bhqk", q_f32 * scale, k_f32)

    # Apply mask
    if combined_mask is not None:
        logits_masked = logits_pre_mask + combined_mask
    else:
        logits_masked = logits_pre_mask

    # Softmax over keys dimension
    weights = mx.softmax(logits_masked, axis=-1)

    # Handle NaN weights (fully masked rows)
    weights = mx.where(mx.isnan(weights), mx.zeros_like(weights), weights)

    # Note: For fully-masked rows with finite mask values, weights will be
    # uniform (not zero). The main function zeros the output explicitly.

    return AttentionIntermediates(
        logits_pre_mask=logits_pre_mask,
        logits_masked=logits_masked,
        weights=weights,
    )


class MLXAttentionSpike:
    """MLX attention spike implementation conforming to AttentionSpike protocol.

    This class wraps the functional mlx_scaled_dot_product_attention for
    protocol-based usage with the validation framework.

    Example:
        >>> spike = MLXAttentionSpike()
        >>> result = spike(q, k, v, boolean_mask=mask, additive_bias=bias)
        >>> output = result.output
    """

    def __init__(self, mask_value: float = DEFAULT_MASK_VALUE):
        """Initialize spike with configuration.

        Args:
            mask_value: Large value for masked positions
        """
        self.mask_value = mask_value

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        boolean_mask: mx.array | None = None,
        additive_bias: mx.array | None = None,
        capture_intermediates: bool = False,
    ) -> AttentionOutput:
        """Compute scaled dot-product attention.

        Args:
            q: Query tensor [batch, heads, seq_q, head_dim]
            k: Key tensor [batch, heads, seq_k, head_dim]
            v: Value tensor [batch, heads, seq_k, head_dim]
            boolean_mask: Optional mask [batch, seq_k] where True=attend
            additive_bias: Optional bias [batch, heads, seq_q, seq_k]
            capture_intermediates: If True, capture pre-softmax logits and weights

        Returns:
            AttentionOutput with output tensor and optional intermediates

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        # Shape validation
        if q.ndim != 4:
            raise ValueError(f"Q must be 4D, got {q.ndim}D")
        if k.ndim != 4:
            raise ValueError(f"K must be 4D, got {k.ndim}D")
        if v.ndim != 4:
            raise ValueError(f"V must be 4D, got {v.ndim}D")

        batch, heads, seq_q, head_dim = q.shape
        _, _, seq_k, _ = k.shape

        # Validate K and V match
        if k.shape != v.shape:
            raise ValueError(f"K and V shapes must match: {k.shape} vs {v.shape}")

        # Validate K dimensions match Q where expected
        if k.shape[:2] != (batch, heads):
            raise ValueError(
                f"K batch/heads mismatch: expected ({batch}, {heads}), got {k.shape[:2]}"
            )
        if k.shape[3] != head_dim:
            raise ValueError(
                f"K head_dim mismatch: expected {head_dim}, got {k.shape[3]}"
            )

        # Validate mask shape
        if boolean_mask is not None:
            if boolean_mask.shape != (batch, seq_k):
                raise ValueError(
                    f"boolean_mask shape mismatch: expected ({batch}, {seq_k}), "
                    f"got {boolean_mask.shape}"
                )

        # Validate bias shape
        if additive_bias is not None:
            expected_bias_shape = (batch, heads, seq_q, seq_k)
            if additive_bias.shape != expected_bias_shape:
                raise ValueError(
                    f"additive_bias shape mismatch: expected {expected_bias_shape}, "
                    f"got {additive_bias.shape}"
                )

        return mlx_scaled_dot_product_attention(
            q, k, v,
            boolean_mask=boolean_mask,
            additive_bias=additive_bias,
            mask_value=self.mask_value,
            capture_intermediates=capture_intermediates,
        )
