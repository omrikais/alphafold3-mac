"""Attention mechanisms for AlphaFold 3 MLX.

This module implements attention variants used throughout the model:
- GatedSelfAttention: Standard gated self-attention with MLX SDPA
- GridSelfAttention: Row-wise/column-wise attention for pair representations

Key MLX patterns:
- Additive mask format (not boolean): mask_value * (boolean_mask.astype(mx.float32) - 1.0)
- Fully masked rows → zero output (explicit detection)
- Pre-cast Q/K/V to same dtype to avoid 50-100% performance regression

Chunking support is available via the global chunking configuration
for memory-efficient processing of large sequences.

JAX AF3 Compatibility:
- act_norm LayerNorm INSIDE attention (before Q/K/V projections)
- Gating with bias (initialized to 1.0 for identity start)
- GridSelfAttention computes pair_bias_projection internally
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.chunking import get_chunking_config, chunk_attention

# Default mask value for attention masking
DEFAULT_MASK_VALUE: float = 1e9


def boolean_to_additive_mask(
    boolean_mask: mx.array,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> mx.array:
    """Convert boolean mask to additive mask format.

    MLX's scaled_dot_product_attention uses additive masks where:
    - Positions to attend to have value 0
    - Positions to mask out have large negative value

    Args:
        boolean_mask: Boolean mask where True=attend, False=mask.
            Shape: [batch, seq_k] or broadcastable.
        mask_value: Large positive value for masking (default: 1e9).

    Returns:
        Additive mask with 0 for attend and -mask_value for mask.
        Shape matches input.
    """
    # Convert: True (attend) → 0, False (mask) → -mask_value
    # Formula: mask_value * (boolean_mask.astype(float) - 1.0)
    # When True (1.0): mask_value * (1.0 - 1.0) = 0
    # When False (0.0): mask_value * (0.0 - 1.0) = -mask_value
    return mask_value * (boolean_mask.astype(mx.float32) - 1.0)


def detect_fully_masked_rows(boolean_mask: mx.array) -> mx.array:
    """Detect rows where all keys are masked.

    These rows would produce NaN from softmax (all -inf inputs).
    They need special handling to produce zero output instead.

    Args:
        boolean_mask: Boolean mask where True=attend, False=mask.
            Shape: [..., seq_k]

    Returns:
        Boolean array where True indicates fully masked rows.
        Shape: [..., 1] for broadcasting.
    """
    # A row is fully masked if no position is True
    # mx.any returns True if any element is True
    has_valid_key = mx.any(boolean_mask, axis=-1, keepdims=True)
    return ~has_valid_key


def _dot_product_attention_af3(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: mx.array | None = None,
    bias: mx.array | None = None,
) -> mx.array:
    """Scaled dot-product attention using MLX SDPA.

    Args:
        query: [..., seq_q, num_heads, head_dim]
        key:   [..., seq_k, num_heads, head_dim]
        value: [..., seq_k, num_heads, head_dim]
        mask: Boolean mask broadcastable to [..., num_heads, seq_q, seq_k]
              where True=attend, False=mask
        bias: Additive bias broadcastable to [..., num_heads, seq_q, seq_k]

    Returns:
        Output: [..., seq_q, num_heads, head_dim]
    """
    head_dim = query.shape[-1]
    num_heads = query.shape[-2]
    scale = 1.0 / (head_dim ** 0.5)

    # Get leading dimensions (could be empty, [batch], or [batch, rows], etc.)
    leading_dims = query.shape[:-3]
    seq_q = query.shape[-3]
    seq_k = key.shape[-3]

    # Flatten leading dimensions into batch for SDPA
    # [..., seq, heads, dim] -> [batch_flat, seq, heads, dim]
    if len(leading_dims) > 0:
        batch_flat = int(np.prod(leading_dims))
        q_flat = query.reshape(batch_flat, seq_q, num_heads, head_dim)
        k_flat = key.reshape(batch_flat, seq_k, num_heads, head_dim)
        v_flat = value.reshape(batch_flat, seq_k, num_heads, head_dim)
    else:
        # No leading dims - add batch dim
        batch_flat = 1
        q_flat = query[None]  # [1, seq_q, heads, dim]
        k_flat = key[None]
        v_flat = value[None]

    # Transpose to SDPA format: [batch, heads, seq, dim]
    q_sdpa = q_flat.transpose(0, 2, 1, 3)
    k_sdpa = k_flat.transpose(0, 2, 1, 3)
    v_sdpa = v_flat.transpose(0, 2, 1, 3)

    # Pre-cast Q/K/V to same dtype for performance (avoid 50-100% regression)
    compute_dtype = q_sdpa.dtype
    if k_sdpa.dtype != compute_dtype:
        k_sdpa = k_sdpa.astype(compute_dtype)
    if v_sdpa.dtype != compute_dtype:
        v_sdpa = v_sdpa.astype(compute_dtype)

    # Build combined additive mask for SDPA
    combined_mask = None
    fully_masked_rows = None

    if mask is not None:
        # Convert boolean mask to additive format using helper
        # mask has shape [..., num_heads, seq_q, seq_k] with True=attend
        # Flatten leading dims to match batch_flat
        if len(leading_dims) > 0:
            mask_flat = mask.reshape(batch_flat, num_heads, seq_q, seq_k)
        else:
            mask_flat = mask[None] if mask.ndim == 3 else mask

        # Detect fully masked rows (all False in key dimension)
        # Shape: [batch, heads, seq_q]
        has_valid_key = mx.any(mask_flat, axis=-1)
        fully_masked_rows = ~has_valid_key  # [batch, heads, seq_q]

        # Convert boolean to additive: True→0, False→-mask_value
        combined_mask = boolean_to_additive_mask(mask_flat)

    if bias is not None:
        # Flatten bias leading dims to match batch_flat
        if len(leading_dims) > 0:
            bias_flat = bias.reshape(batch_flat, num_heads, seq_q, seq_k)
        else:
            bias_flat = bias[None] if bias.ndim == 3 else bias

        if combined_mask is not None:
            combined_mask = combined_mask + bias_flat
        else:
            combined_mask = bias_flat

    # Use MLX SDPA kernel
    # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
    if combined_mask is not None and combined_mask.dtype != q_sdpa.dtype:
        combined_mask = combined_mask.astype(q_sdpa.dtype)
    output = mx.fast.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        scale=scale,
        mask=combined_mask,
    )  # [batch_flat, heads, seq_q, dim]

    # Zero fully masked rows
    if fully_masked_rows is not None:
        # Expand to [batch, heads, seq_q, dim]
        fully_masked_expanded = fully_masked_rows[..., None]
        output = mx.where(fully_masked_expanded, 0.0, output)

    # Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
    output = output.transpose(0, 2, 1, 3)

    # Reshape back to original leading dimensions
    if len(leading_dims) > 0:
        output = output.reshape(*leading_dims, seq_q, num_heads, head_dim)
    else:
        output = output[0]  # Remove added batch dim

    return output


class AF3SelfAttention(nn.Module):
    """Self-attention matching AF3 diffusion_transformer.self_attention."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        key_dim: int | None = None,
        value_dim: int | None = None,
        final_init: str = "zeros",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        # AF3 uses total key_dim/value_dim, then divides by num_heads.
        total_key_dim = key_dim if key_dim is not None else input_dim
        total_value_dim = value_dim if value_dim is not None else input_dim
        assert total_key_dim % num_heads == 0
        assert total_value_dim % num_heads == 0
        self.key_dim = total_key_dim // num_heads
        self.value_dim = total_value_dim // num_heads

        # LayerNorm (no conditioning in PairFormer)
        self.act_norm = LayerNorm(input_dim)

        # Q/K/V projections
        # Q uses bias, K/V do not.
        self.q_proj = Linear(
            (num_heads, self.key_dim),
            input_dims=input_dim,
            use_bias=True,
        )
        self.k_proj = Linear(
            (num_heads, self.key_dim),
            input_dims=input_dim,
            use_bias=False,
        )
        self.v_proj = Linear(
            (num_heads, self.value_dim),
            input_dims=input_dim,
            use_bias=False,
        )

        # Gating (AF3: no bias, weights initialized to zeros)
        self.gate_proj = Linear(
            num_heads * self.value_dim,
            input_dims=input_dim,
            use_bias=False,
            initializer="zeros",
        )

        # Output projection (final_init)
        self.o_proj = Linear(
            input_dim,
            input_dims=num_heads * self.value_dim,
            use_bias=False,
            initializer=final_init,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        pair_logits: mx.array | None = None,
    ) -> mx.array:
        """Apply AF3-style self-attention using SDPA.

        Args:
            x: [batch, seq, input_dim] or [seq, input_dim]
            mask: [batch, seq] or [seq] where True/1.0=attend, False/0.0=mask
            pair_logits: [num_heads, seq, seq] or broadcastable (additive bias)
        """
        # Handle unbatched input
        is_unbatched = x.ndim == 2
        if is_unbatched:
            x = x[None]  # [1, seq, dim]
            if mask is not None:
                mask = mask[None]

        batch_size, seq_len, _ = x.shape
        x_norm = self.act_norm(x)

        # Q/K/V projections: output shape [batch, seq, num_heads, key/value_dim]
        q = self.q_proj(x_norm)  # [batch, seq, num_heads, key_dim]
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)  # [batch, seq, num_heads, value_dim]

        # Transpose to SDPA format: [batch, num_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # JAX AF3 parity: cast Q/K to float32 before computing attention
        # logits (diffusion_transformer.py lines 152-153).  This prevents
        # gradient norm blow-ups in bfloat16 across 48 PairFormer layers.
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)

        scale = 1.0 / (self.key_dim ** 0.5)

        # Build combined additive mask
        combined_mask = None
        fully_masked_rows = None

        if mask is not None:
            # Convert mask to boolean for detection
            bool_mask = mask.astype(mx.bool_)

            # Detect fully masked rows
            has_valid_key = mx.any(bool_mask, axis=-1)  # [batch]
            fully_masked_rows = ~has_valid_key  # [batch]

            # Convert to additive mask: True/1.0→0, False/0.0→-1e9
            # Shape: [batch, 1, 1, seq] for broadcasting
            additive_mask = boolean_to_additive_mask(bool_mask)
            combined_mask = additive_mask[:, None, None, :]

        if pair_logits is not None:
            # pair_logits: [num_heads, seq, seq] or [batch, num_heads, seq, seq]
            if pair_logits.ndim == 3:
                pair_logits = pair_logits[None]  # [1, heads, seq, seq]
            if combined_mask is not None:
                combined_mask = combined_mask + pair_logits
            else:
                combined_mask = pair_logits

        # Use MLX SDPA
        # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
        if combined_mask is not None and combined_mask.dtype != q.dtype:
            combined_mask = combined_mask.astype(q.dtype)
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=scale,
            mask=combined_mask,
        )  # [batch, num_heads, seq, value_dim]

        # Zero fully masked rows
        if fully_masked_rows is not None:
            # Expand to [batch, heads, seq, dim]
            fully_masked_expanded = fully_masked_rows[:, None, None, None]
            fully_masked_expanded = mx.broadcast_to(
                fully_masked_expanded,
                (batch_size, self.num_heads, seq_len, self.value_dim)
            )
            attn_output = mx.where(fully_masked_expanded, 0.0, attn_output)

        # Transpose back: [batch, seq, num_heads, value_dim]
        weighted_avg = attn_output.transpose(0, 2, 1, 3)
        weighted_avg = weighted_avg.reshape(batch_size, seq_len, -1)

        # Gating
        gate_logits = self.gate_proj(x_norm)
        weighted_avg = weighted_avg * mx.sigmoid(gate_logits)

        output = self.o_proj(weighted_avg)

        # Remove batch dim if input was unbatched
        if is_unbatched:
            output = output[0]

        return output


class GatedSelfAttention(nn.Module):
    """Gated self-attention with MLX SDPA matching JAX AF3.

    This implements the gated attention variant used in AlphaFold 3:
    - Internal LayerNorm (act_norm) before Q/K/V projections
    - Query, Key, Value projections (no bias)
    - Scaled dot-product attention via mx.fast.scaled_dot_product_attention
    - Gating mechanism with bias (initialized to 1.0)
    - Output projection

    The gating mechanism multiplies the attention output by a learnable gate
    derived from the normalized input, allowing the network to modulate attention.

    JAX AF3 Compatibility:
    - act_norm is INSIDE the attention module (before Q/K/V)
    - gating_query has NO bias (AF3 uses bias_init but use_bias=False)
    - Weight names match Haiku paths for weight loading
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        key_dim: int | None = None,
        value_dim: int | None = None,
        gated: bool = True,
        output_dim: int | None = None,
        use_internal_norm: bool = True,
    ) -> None:
        """Initialize gated self-attention.

        Args:
            input_dim: Input feature dimension.
            num_heads: Number of attention heads.
            key_dim: Key dimension per head. Defaults to input_dim // num_heads.
            value_dim: Value dimension per head. Defaults to key_dim.
            gated: Whether to use gating mechanism.
            output_dim: Output dimension. Defaults to input_dim.
            use_internal_norm: Whether to apply LayerNorm inside (JAX AF3 style).
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_dim = key_dim or (input_dim // num_heads)
        self.value_dim = value_dim or self.key_dim
        self.gated = gated
        self.output_dim = output_dim or input_dim
        self.use_internal_norm = use_internal_norm

        # Internal LayerNorm (act_norm) - JAX AF3 style
        if use_internal_norm:
            self.act_norm = LayerNorm(input_dim)

        # Q, K, V projections (no bias, matching JAX AF3)
        self.q_proj = Linear(num_heads * self.key_dim, input_dims=input_dim, use_bias=False)
        self.k_proj = Linear(num_heads * self.key_dim, input_dims=input_dim, use_bias=False)
        self.v_proj = Linear(num_heads * self.value_dim, input_dims=input_dim, use_bias=False)

        # Output projection
        self.o_proj = Linear(self.output_dim, input_dims=num_heads * self.value_dim, use_bias=False)

        # Gating projection (JAX AF3: no bias, weights initialized to zeros)
        if gated:
            self.gate_proj = Linear(
                num_heads * self.value_dim,
                input_dims=input_dim,
                use_bias=False,
                initializer="zeros",
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        bias: mx.array | None = None,
    ) -> mx.array:
        """Apply gated self-attention.

        Args:
            x: Input tensor. Shape: [batch, seq, input_dim]
            mask: Optional boolean mask where True=attend. Shape: [batch, seq]
            bias: Optional additive bias. Shape: [batch, num_heads, seq, seq]

        Returns:
            Output tensor. Shape: [batch, seq, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Apply internal LayerNorm if enabled (JAX AF3 style)
        if self.use_internal_norm:
            x_norm = self.act_norm(x)
        else:
            x_norm = x

        # Compute Q, K, V from normalized input
        q = self.q_proj(x_norm)  # [batch, seq, num_heads * key_dim]
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)  # [batch, seq, num_heads * value_dim]

        # Reshape to [batch, num_heads, seq, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
        q = q.transpose(0, 2, 1, 3)  # [batch, num_heads, seq, key_dim]

        k = k.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
        k = k.transpose(0, 2, 1, 3)

        v = v.reshape(batch_size, seq_len, self.num_heads, self.value_dim)
        v = v.transpose(0, 2, 1, 3)

        # Ensure Q, K, V have same dtype to avoid performance regression
        compute_dtype = q.dtype
        if k.dtype != compute_dtype:
            k = k.astype(compute_dtype)
        if v.dtype != compute_dtype:
            v = v.astype(compute_dtype)

        # Handle mask and bias
        attention_mask = None
        if mask is not None:
            # Detect fully masked rows for handling
            fully_masked = detect_fully_masked_rows(mask)

            # Convert boolean mask to additive format
            # Shape: [batch, 1, 1, seq] for broadcasting
            attention_mask = boolean_to_additive_mask(mask)
            attention_mask = attention_mask[:, None, None, :]

            if bias is not None:
                attention_mask = attention_mask + bias
        elif bias is not None:
            attention_mask = bias
            fully_masked = None
        else:
            fully_masked = None

        # Scaled dot-product attention via MLX fast path
        # Scale is 1/sqrt(key_dim)
        scale = 1.0 / (self.key_dim ** 0.5)

        # Use chunked attention if configured (for memory efficiency)
        chunking = get_chunking_config()
        # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
        if attention_mask is not None and attention_mask.dtype != q.dtype:
            attention_mask = attention_mask.astype(q.dtype)
        if chunking.enabled and seq_len > chunking.attention_chunk_size:
            attn_output = chunk_attention(
                q, k, v,
                chunk_size=chunking.attention_chunk_size,
                scale=scale,
                mask=attention_mask,
            )
        else:
            attn_output = mx.fast.scaled_dot_product_attention(
                q, k, v,
                scale=scale,
                mask=attention_mask,
            )  # [batch, num_heads, seq, value_dim]

        # Handle fully masked rows
        # Replace NaN outputs with zeros
        if fully_masked is not None:
            # Expand fully_masked to match attention output shape
            # [batch, 1] → [batch, num_heads, seq, value_dim]
            fully_masked_expanded = mx.broadcast_to(
                fully_masked[:, None, :, None],
                attn_output.shape,
            )
            attn_output = mx.where(fully_masked_expanded, 0.0, attn_output)

        # Reshape back to [batch, seq, num_heads * value_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Apply gating (from normalized input, JAX AF3 style)
        if self.gated:
            gate = mx.sigmoid(self.gate_proj(x_norm))
            attn_output = attn_output * gate

        # Output projection
        output = self.o_proj(attn_output)

        return output


class GridSelfAttention(nn.Module):
    """Grid self-attention for pair representations matching JAX AF3.

    This implements row-wise or column-wise attention over pair tensors,
    matching AF3 GridSelfAttention semantics.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        key_dim: int | None = None,
        value_dim: int | None = None,
        gated: bool = True,
        orientation: Literal["row", "column"] = "row",
    ) -> None:
        """Initialize grid self-attention.

        Args:
            input_dim: Input feature dimension.
            num_heads: Number of attention heads.
            key_dim: Key dimension per head. Defaults to input_dim // num_heads.
            value_dim: Value dimension per head. Defaults to key_dim.
            gated: Whether to use gating mechanism.
            orientation: "row" for row-wise, "column" for column-wise.
        """
        super().__init__()

        self.orientation = orientation
        self.num_heads = num_heads
        self.input_dim = input_dim

        # qkv_dim follows AF3: max(ch//heads, 16)
        self.qkv_dim = key_dim or max(input_dim // num_heads, 16)

        # LayerNorm
        self.act_norm = LayerNorm(input_dim)

        # Pair bias projection (no bias)
        self.pair_bias_proj = Linear(
            num_heads, input_dims=input_dim, use_bias=False
        )

        # Q/K/V projections
        self.q_proj = Linear(
            (num_heads, self.qkv_dim),
            input_dims=input_dim,
            use_bias=False,
            transpose_weights=True,
        )
        self.k_proj = Linear(
            (num_heads, self.qkv_dim),
            input_dims=input_dim,
            use_bias=False,
            transpose_weights=True,
        )
        self.v_proj = Linear(
            (num_heads, self.qkv_dim),
            input_dims=input_dim,
            use_bias=False,
        )

        # Gating projection (AF3: no bias, weights initialized to zeros)
        self.gate_proj = Linear(
            num_heads * self.qkv_dim,
            input_dims=input_dim,
            use_bias=False,
            initializer="zeros",
            transpose_weights=True,
        )

        # Output projection
        self.o_proj = Linear(
            input_dim,
            input_dims=num_heads * self.qkv_dim,
            use_bias=False,
            initializer="zeros",
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Apply grid self-attention using SDPA.

        For row orientation: attention over seq_j for each row i
        For column orientation: attention over seq_i for each column j
        """
        batch_size, seq_i, seq_j, _ = x.shape

        # Apply LayerNorm
        x_norm = self.act_norm(x)  # [batch, seq_i, seq_j, input_dim]

        # Compute pair bias from normalized activations
        # Shape: [batch, seq_i, seq_j, num_heads]
        pair_bias = self.pair_bias_proj(x_norm)
        # Transpose to [batch, num_heads, seq_i, seq_j]
        pair_bias = pair_bias.transpose(0, 3, 1, 2)

        # For column attention, transpose spatial dims before projections
        if self.orientation == "column":
            act_attn = mx.swapaxes(x_norm, 1, 2)  # [batch, seq_j, seq_i, dim]
            grid_dim = seq_j  # Number of independent attention operations
            attn_len = seq_i  # Length of each attention sequence
        else:
            act_attn = x_norm  # [batch, seq_i, seq_j, dim]
            grid_dim = seq_i
            attn_len = seq_j

        # Q/K/V projections
        # Output shape: [batch, grid_dim, attn_len, num_heads, qkv_dim]
        q = self.q_proj(act_attn)
        k = self.k_proj(act_attn)
        v = self.v_proj(act_attn)

        # Reshape for batched SDPA:
        # Treat grid_dim as additional batch dimension
        # [batch, grid_dim, attn_len, heads, dim] -> [batch*grid_dim, heads, attn_len, dim]
        q = q.reshape(batch_size * grid_dim, attn_len, self.num_heads, self.qkv_dim)
        k = k.reshape(batch_size * grid_dim, attn_len, self.num_heads, self.qkv_dim)
        v = v.reshape(batch_size * grid_dim, attn_len, self.num_heads, self.qkv_dim)

        # Transpose to SDPA format: [batch*grid, heads, attn_len, dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Pre-cast Q/K/V to same dtype for performance
        compute_dtype = q.dtype
        if k.dtype != compute_dtype:
            k = k.astype(compute_dtype)
        if v.dtype != compute_dtype:
            v = v.astype(compute_dtype)

        scale = 1.0 / (self.qkv_dim ** 0.5)

        # Build attention mask
        combined_mask = None
        fully_masked_rows = None

        if mask is not None:
            # mask: [batch, seq_i, seq_j] - AF3 swaps axes
            pair_mask = mx.swapaxes(mask, -1, -2)  # [batch, seq_j, seq_i]

            # For grid attention, expand mask to match batched structure
            # [batch, grid_dim, attn_len] -> [batch*grid_dim, 1, 1, attn_len]
            if self.orientation == "column":
                # Column attention: mask is [batch, seq_j, seq_i], attn over seq_i
                flat_mask = pair_mask.reshape(batch_size * grid_dim, attn_len)
            else:
                # Row attention: mask is [batch, seq_j, seq_i], need [batch, seq_i, seq_j]
                # But we already swapped, so pair_mask is [batch, seq_j, seq_i]
                # For row attention over seq_j, we need mask per row
                flat_mask = mx.swapaxes(pair_mask, 1, 2).reshape(batch_size * grid_dim, attn_len)

            # Detect fully masked rows
            bool_mask = flat_mask.astype(mx.bool_)
            has_valid_key = mx.any(bool_mask, axis=-1)  # [batch*grid_dim]
            fully_masked_rows = ~has_valid_key

            # Convert to additive mask
            additive_mask = boolean_to_additive_mask(bool_mask)
            combined_mask = additive_mask[:, None, None, :]  # [batch*grid, 1, 1, attn_len]

        # pair_bias: [batch, heads, N, N] — same bias for every grid slice.
        # JAX AF3 uses a NONBATCHED bias that broadcasts across all rows/columns.
        # We keep it at [batch, heads, N, N] and let SDPA broadcast lazily
        # instead of tiling to [batch*grid, heads, N, N] which is O(N^3) memory.

        # Run SDPA — chunked along grid_dim to avoid O(N^3) mask materialization
        # when combining per-grid-slice key masks with the shared pair bias.
        _GRID_CHUNK = 64
        total_grid = batch_size * grid_dim

        if total_grid <= _GRID_CHUNK and combined_mask is None:
            # Fast path: no key mask, small grid — pair_bias broadcasts in-kernel
            sdpa_mask = pair_bias  # [batch, heads, N, N] broadcasts to [batch*grid, ...]
            if sdpa_mask.dtype != q.dtype:
                sdpa_mask = sdpa_mask.astype(q.dtype)
            attn_output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=sdpa_mask,
            )
            # Zero fully masked rows
            if fully_masked_rows is not None:
                fm = fully_masked_rows[:, None, None, None]
                attn_output = mx.where(mx.broadcast_to(fm, attn_output.shape), 0.0, attn_output)
        else:
            # Chunked path: build combined mask per chunk to stay O(chunk * heads * N^2)
            chunks_out = []
            for g_start in range(0, total_grid, _GRID_CHUNK):
                g_end = min(g_start + _GRID_CHUNK, total_grid)
                q_c = q[g_start:g_end]
                k_c = k[g_start:g_end]
                v_c = v[g_start:g_end]

                # pair_bias [batch, heads, N, N] broadcasts to [chunk, heads, N, N]
                chunk_mask = pair_bias  # broadcasts on dim 0

                if combined_mask is not None:
                    # combined_mask (additive key mask): [batch*grid, 1, 1, N]
                    chunk_mask = combined_mask[g_start:g_end] + pair_bias

                if chunk_mask.dtype != q_c.dtype:
                    chunk_mask = chunk_mask.astype(q_c.dtype)

                chunk_out = mx.fast.scaled_dot_product_attention(
                    q_c, k_c, v_c, scale=scale, mask=chunk_mask,
                )

                # Zero fully masked rows for this chunk
                if fully_masked_rows is not None:
                    fm_c = fully_masked_rows[g_start:g_end, None, None, None]
                    chunk_out = mx.where(
                        mx.broadcast_to(fm_c, chunk_out.shape), 0.0, chunk_out,
                    )

                chunks_out.append(chunk_out)

                # Periodically evaluate to prevent graph explosion
                if len(chunks_out) % 4 == 0:
                    mx.eval(chunks_out[-1])

            attn_output = mx.concatenate(chunks_out, axis=0) if len(chunks_out) > 1 else chunks_out[0]

        # Transpose back: [batch*grid, attn_len, heads, dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Reshape back to grid structure: [batch, grid_dim, attn_len, heads*dim]
        attn_output = attn_output.reshape(batch_size, grid_dim, attn_len, -1)

        # Apply gating
        gate_values = self.gate_proj(act_attn)
        attn_output = attn_output * mx.sigmoid(gate_values)

        # Output projection
        out = self.o_proj(attn_output)  # [batch, grid_dim, attn_len, input_dim]

        # Transpose back for column attention
        if self.orientation == "column":
            out = mx.swapaxes(out, 1, 2)  # [batch, seq_i, seq_j, input_dim]

        return out


def chunked_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None = None,
    chunk_size: int = 512,
) -> mx.array:
    """Chunked attention for large sequences.

    Processes attention in chunks to reduce peak memory from O(N²) to O(N × chunk_size).
    Use for sequences >1000 residues to avoid OOM.

    Args:
        q: Query tensor. Shape: [batch, num_heads, seq_q, head_dim]
        k: Key tensor. Shape: [batch, num_heads, seq_k, head_dim]
        v: Value tensor. Shape: [batch, num_heads, seq_k, head_dim]
        scale: Attention scale factor (1/sqrt(head_dim)).
        mask: Optional additive mask. Shape: [batch, 1, 1, seq_k]
        chunk_size: Number of query positions per chunk.

    Returns:
        Attention output. Shape: [batch, num_heads, seq_q, head_dim]
    """
    batch_size, num_heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]
    value_dim = v.shape[-1]

    # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
    if mask is not None and mask.dtype != q.dtype:
        mask = mask.astype(q.dtype)

    # If sequence is small enough, use standard attention
    if seq_q <= chunk_size:
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

    # Process in chunks
    outputs = []
    for start in range(0, seq_q, chunk_size):
        end = min(start + chunk_size, seq_q)
        q_chunk = q[:, :, start:end, :]  # [batch, heads, chunk, head_dim]

        # Compute attention for this chunk
        # Mask applies to keys, not queries, so use full mask
        chunk_out = mx.fast.scaled_dot_product_attention(
            q_chunk, k, v, scale=scale, mask=mask
        )  # [batch, heads, chunk, value_dim]

        outputs.append(chunk_out)

        # Periodically evaluate to prevent graph explosion
        if len(outputs) % 4 == 0:
            mx.eval(outputs[-1])

    # Concatenate all chunks
    return mx.concatenate(outputs, axis=2)


class ChunkedGatedSelfAttention(nn.Module):
    """Gated self-attention with chunking for large sequences.

    Uses chunked attention when sequence length exceeds chunk_size threshold.
    This reduces peak memory from O(N²) to O(N × chunk_size).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        key_dim: int | None = None,
        value_dim: int | None = None,
        gated: bool = True,
        output_dim: int | None = None,
        chunk_size: int = 512,
    ) -> None:
        """Initialize chunked gated self-attention.

        Args:
            input_dim: Input feature dimension.
            num_heads: Number of attention heads.
            key_dim: Key dimension per head. Defaults to input_dim // num_heads.
            value_dim: Value dimension per head. Defaults to key_dim.
            gated: Whether to use gating mechanism.
            output_dim: Output dimension. Defaults to input_dim.
            chunk_size: Chunk size for chunked attention (default: 512).
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.key_dim = key_dim or (input_dim // num_heads)
        self.value_dim = value_dim or self.key_dim
        self.gated = gated
        self.output_dim = output_dim or input_dim
        self.chunk_size = chunk_size

        # Internal LayerNorm (JAX AF3 style)
        self.act_norm = LayerNorm(input_dim)

        # Q, K, V projections
        self.q_proj = Linear(num_heads * self.key_dim, input_dims=input_dim, use_bias=False)
        self.k_proj = Linear(num_heads * self.key_dim, input_dims=input_dim, use_bias=False)
        self.v_proj = Linear(num_heads * self.value_dim, input_dims=input_dim, use_bias=False)

        # Output projection
        self.o_proj = Linear(self.output_dim, input_dims=num_heads * self.value_dim, use_bias=False)

        # Gating projection (JAX AF3: no bias, weights initialized to zeros)
        if gated:
            self.gate_proj = Linear(
                num_heads * self.value_dim,
                input_dims=input_dim,
                use_bias=False,
                initializer="zeros",
            )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        bias: mx.array | None = None,
    ) -> mx.array:
        """Apply chunked gated self-attention.

        Args:
            x: Input tensor. Shape: [batch, seq, input_dim]
            mask: Optional boolean mask where True=attend. Shape: [batch, seq]
            bias: Optional additive bias. Shape: [batch, num_heads, seq, seq]

        Returns:
            Output tensor. Shape: [batch, seq, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Internal LayerNorm
        x_norm = self.act_norm(x)

        # Compute Q, K, V
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape to [batch, num_heads, seq, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
        q = q.transpose(0, 2, 1, 3)

        k = k.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
        k = k.transpose(0, 2, 1, 3)

        v = v.reshape(batch_size, seq_len, self.num_heads, self.value_dim)
        v = v.transpose(0, 2, 1, 3)

        # Ensure Q, K, V have same dtype
        compute_dtype = q.dtype
        if k.dtype != compute_dtype:
            k = k.astype(compute_dtype)
        if v.dtype != compute_dtype:
            v = v.astype(compute_dtype)

        # Handle mask and bias
        attention_mask = None
        if mask is not None:
            fully_masked = detect_fully_masked_rows(mask)
            attention_mask = boolean_to_additive_mask(mask)
            attention_mask = attention_mask[:, None, None, :]

            if bias is not None:
                attention_mask = attention_mask + bias
        elif bias is not None:
            attention_mask = bias
            fully_masked = None
        else:
            fully_masked = None

        # Scaled attention with chunking for large sequences
        scale = 1.0 / (self.key_dim ** 0.5)

        # Cast mask to Q dtype so SDPA mask promotes to output dtype (e.g. bfloat16)
        if attention_mask is not None and attention_mask.dtype != q.dtype:
            attention_mask = attention_mask.astype(q.dtype)

        if seq_len > self.chunk_size:
            attn_output = chunked_attention(
                q, k, v, scale=scale, mask=attention_mask, chunk_size=self.chunk_size
            )
        else:
            attn_output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=scale, mask=attention_mask
            )

        # Handle fully masked rows
        if fully_masked is not None:
            fully_masked_expanded = mx.broadcast_to(
                fully_masked[:, None, :, None],
                attn_output.shape,
            )
            attn_output = mx.where(fully_masked_expanded, 0.0, attn_output)

        # Reshape back to [batch, seq, num_heads * value_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Apply gating
        if self.gated:
            gate = mx.sigmoid(self.gate_proj(x_norm))
            attn_output = attn_output * gate

        # Output projection
        output = self.o_proj(attn_output)

        return output


@dataclass
class AttentionConfig:
    """Configuration for attention modules."""

    num_heads: int = 4
    key_dim: int | None = None
    value_dim: int | None = None
    gated: bool = True
    dropout_rate: float = 0.0  # Not used in inference
    chunk_size: int = 512 # Chunk size for large sequences
