"""AF3 diffusion transformer modules for MLX parity.

This implements the diffusion_transformer stack from AlphaFold 3 JAX:
- adaptive_layernorm and adaptive_zero_init
- self_attention and cross_attention (using MLX SDPA )
- Transformer and CrossAttTransformer stacks (layer_stack semantics)

Key MLX patterns:
- Additive mask format for SDPA: mask_value * (float_mask - 1.0)
- Fully masked rows → zero output (explicit detection)
- Pre-cast Q/K/V to same dtype for performance
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm

# Default mask value for attention masking
_MASK_VALUE: float = 1e9


def _float_to_additive_mask(float_mask: mx.array, mask_value: float = _MASK_VALUE) -> mx.array:
    """Convert float mask (1.0=attend, 0.0=mask) to additive mask format.

    Args:
        float_mask: Float mask where 1.0=attend, 0.0=mask.
        mask_value: Large positive value for masking (default: 1e9).

    Returns:
        Additive mask with 0 for attend and -mask_value for mask.
    """
    # Formula: mask_value * (float_mask - 1.0)
    # When 1.0: mask_value * (1.0 - 1.0) = 0
    # When 0.0: mask_value * (0.0 - 1.0) = -mask_value
    return mask_value * (float_mask - 1.0)


def _detect_fully_masked_rows(float_mask: mx.array) -> mx.array:
    """Detect rows where all keys are masked.

    Args:
        float_mask: Float mask where 1.0=attend, 0.0=mask.
            Shape: [..., seq_k]

    Returns:
        Boolean array where True indicates fully masked rows.
        Shape: [..., 1] for broadcasting.
    """
    # A row is fully masked if no position has value > 0
    has_valid_key = mx.any(float_mask > 0.5, axis=-1, keepdims=True)
    return ~has_valid_key


@dataclass
class SelfAttentionConfig:
    num_head: int = 16
    key_dim: int | None = None
    value_dim: int | None = None


@dataclass
class CrossAttentionConfig:
    num_head: int = 4
    key_dim: int = 128
    value_dim: int = 128


@dataclass
class TransformerConfig:
    attention: SelfAttentionConfig = field(default_factory=SelfAttentionConfig)
    num_blocks: int = 24
    super_block_size: int = 4
    num_intermediate_factor: int = 2


@dataclass
class CrossAttTransformerConfig:
    num_intermediate_factor: int = 2
    num_blocks: int = 3
    attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm matching AF3 JAX."""

    def __init__(self, hidden_dim: int, cond_dim: int | None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        if cond_dim is None:
            self.layer_norm = LayerNorm(hidden_dim)
            self.single_cond_layer_norm = None
            self.single_cond_scale = None
            self.single_cond_bias = None
        else:
            # No scale/offset on main LN when conditioning
            self.layer_norm = LayerNorm(hidden_dim, create_scale=False, create_offset=False)
            self.single_cond_layer_norm = LayerNorm(cond_dim, create_offset=False)
            self.single_cond_scale = Linear(
                hidden_dim,
                input_dims=cond_dim,
                use_bias=True,
                initializer="zeros",
            )
            self.single_cond_bias = Linear(
                hidden_dim,
                input_dims=cond_dim,
                use_bias=False,
                initializer="zeros",
            )

    def __call__(self, x: mx.array, single_cond: mx.array | None) -> mx.array:
        if single_cond is None:
            return self.layer_norm(x)

        x_norm = self.layer_norm(x)
        cond_norm = self.single_cond_layer_norm(single_cond)
        scale = self.single_cond_scale(cond_norm)
        bias = self.single_cond_bias(cond_norm)
        return mx.sigmoid(scale) * x_norm + bias


class AdaptiveZeroInit(nn.Module):
    """Adaptive zero init output projection matching AF3 JAX."""

    def __init__(
        self,
        num_channels: int,
        cond_dim: int | None,
        final_init: str,
        *,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.cond_dim = cond_dim
        self.final_init = final_init
        self.input_dim = input_dim or num_channels

        if cond_dim is None:
            # Use final_init when no conditioning (AF3).
            self.transition2 = Linear(
                num_channels,
                input_dims=self.input_dim,
                use_bias=False,
                initializer=final_init,
            )
            self.adaptive_zero_cond = None
        else:
            # Use default initializer when conditioning (AF3).
            self.transition2 = Linear(
                num_channels,
                input_dims=self.input_dim,
                use_bias=False,
                initializer="linear",
            )
            self.adaptive_zero_cond = Linear(
                num_channels,
                input_dims=cond_dim,
                use_bias=True,
                bias_init=-2.0,
                initializer="zeros",
            )

    def __call__(self, x: mx.array, single_cond: mx.array | None) -> mx.array:
        out = self.transition2(x)
        if single_cond is None:
            return out
        cond = self.adaptive_zero_cond(single_cond)
        return mx.sigmoid(cond) * out


class TransitionBlock(nn.Module):
    """Transition block matching AF3 diffusion_transformer.transition_block."""

    def __init__(
        self,
        num_channels: int,
        num_intermediate_factor: int,
        *,
        cond_dim: int | None,
        final_init: str,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_intermediate = num_intermediate_factor * num_channels

        self.adaptive_norm = AdaptiveLayerNorm(num_channels, cond_dim)
        # transition1 weights (relu init)
        self.transition1 = Linear(
            self.num_intermediate * 2,
            input_dims=num_channels,
            use_bias=False,
            initializer="relu",
        )
        self.adaptive_zero = AdaptiveZeroInit(
            num_channels,
            cond_dim,
            final_init,
            input_dim=self.num_intermediate,
        )

    def __call__(self, x: mx.array, single_cond: mx.array | None) -> mx.array:
        x = self.adaptive_norm(x, single_cond)
        x = self.transition1(x)
        a, b = mx.split(x, 2, axis=-1)
        x = (a * mx.sigmoid(a)) * b  # swish(a) * b
        return self.adaptive_zero(x, single_cond)


class SelfAttention(nn.Module):
    """Self-attention matching AF3 diffusion_transformer.self_attention."""

    def __init__(
        self,
        num_channels: int,
        config: SelfAttentionConfig,
        *,
        final_init: str,
        cond_dim: int | None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.config = config
        self.final_init = final_init
        self.cond_dim = cond_dim

        key_dim_total = config.key_dim if config.key_dim is not None else num_channels
        value_dim_total = config.value_dim if config.value_dim is not None else num_channels
        assert key_dim_total % config.num_head == 0
        assert value_dim_total % config.num_head == 0
        self.key_dim = key_dim_total // config.num_head
        self.value_dim = value_dim_total // config.num_head

        self.adaptive_norm = AdaptiveLayerNorm(num_channels, cond_dim)

        qk_shape = (config.num_head, self.key_dim)
        v_shape = (config.num_head, self.value_dim)
        self.q_projection = Linear(qk_shape, input_dims=num_channels, use_bias=True)
        self.k_projection = Linear(qk_shape, input_dims=num_channels, use_bias=False)
        self.v_projection = Linear(v_shape, input_dims=num_channels, use_bias=False)

        self.gating_query = Linear(
            config.num_head * self.value_dim,
            input_dims=num_channels,
            use_bias=False,
            initializer="zeros",
        )
        self.adaptive_zero = AdaptiveZeroInit(num_channels, cond_dim, final_init)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pair_logits: mx.array | None,
        single_cond: mx.array | None,
    ) -> mx.array:
        """Apply self-attention using SDPA.

        Args:
            x: Input tensor [..., seq, channels]
            mask: Float mask where 1.0=attend, 0.0=mask [..., seq]
            pair_logits: Optional additive bias [..., heads, query, key]
            single_cond: Optional conditioning tensor
        """
        x_norm = self.adaptive_norm(x, single_cond)

        # Q/K/V projections: [..., seq, num_heads, key/value_dim]
        q = self.q_projection(x_norm)
        k = self.k_projection(x_norm)
        v = self.v_projection(x_norm)

        # Get dimensions for SDPA - need to flatten leading dims into batch
        *leading_dims, seq_len, num_heads, key_dim = q.shape
        value_dim = v.shape[-1]
        has_leading = len(leading_dims) > 0

        # Flatten leading dims into batch for SDPA
        if has_leading:
            batch_flat = int(np.prod(leading_dims))
            q = q.reshape(batch_flat, seq_len, num_heads, key_dim)
            k = k.reshape(batch_flat, seq_len, num_heads, key_dim)
            v = v.reshape(batch_flat, seq_len, num_heads, value_dim)

            # Transpose to SDPA format: [batch, heads, seq, dim]
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Flatten mask to match batch
            mask_flat = mask.reshape(batch_flat, seq_len)

            # Flatten pair_logits if provided
            if pair_logits is not None:
                pair_logits = pair_logits.reshape(batch_flat, num_heads, seq_len, seq_len)
        else:
            # Unbatched: [seq, heads, dim] -> [1, heads, seq, dim]
            q = q.transpose(1, 0, 2)[None]
            k = k.transpose(1, 0, 2)[None]
            v = v.transpose(1, 0, 2)[None]
            batch_flat = 1
            mask_flat = mask[None]

        # Pre-cast Q/K/V to same dtype for performance
        compute_dtype = mx.float32
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        scale = self.key_dim ** -0.5

        # Build combined additive mask for SDPA
        # mask_flat: [batch, seq] -> [batch, 1, 1, seq]
        additive_mask = _float_to_additive_mask(mask_flat)
        combined_mask = additive_mask[:, None, None, :]

        # Detect fully masked rows for later handling
        # [batch, seq] -> [batch]
        has_valid_key = mx.any(mask_flat > 0.5, axis=-1)
        fully_masked_rows = ~has_valid_key  # [batch]

        # Add pair logits if provided
        if pair_logits is not None:
            combined_mask = combined_mask + pair_logits

        combined_mask = combined_mask.astype(compute_dtype)

        # Use MLX SDPA
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=scale,
            mask=combined_mask,
        )  # [batch, heads, seq, value_dim]

        # Zero fully masked rows
        # [batch] -> [batch, 1, 1, 1] for broadcasting
        fully_masked_expanded = fully_masked_rows[:, None, None, None]
        attn_output = mx.where(fully_masked_expanded, 0.0, attn_output)

        # Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Reshape back to original leading dimensions
        if has_leading:
            attn_output = attn_output.reshape(*leading_dims, seq_len, num_heads, value_dim)
        else:
            attn_output = attn_output[0]  # Remove added batch dim: [seq, heads, dim]

        # Cast back to input dtype
        attn_output = attn_output.astype(x.dtype)

        # Flatten heads*value_dim
        weighted_avg = attn_output.reshape(attn_output.shape[:-2] + (-1,))

        # Apply gating
        gate = mx.sigmoid(self.gating_query(x_norm))
        weighted_avg = weighted_avg * gate

        return self.adaptive_zero(weighted_avg, single_cond)


class CrossAttention(nn.Module):
    """Cross attention matching AF3 diffusion_transformer.cross_attention."""

    def __init__(
        self,
        num_channels: int,
        config: CrossAttentionConfig,
        *,
        final_init: str,
        cond_dim_q: int | None,
        cond_dim_k: int | None,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.config = config
        self.final_init = final_init
        self.cond_dim_q = cond_dim_q
        self.cond_dim_k = cond_dim_k

        assert config.key_dim % config.num_head == 0
        assert config.value_dim % config.num_head == 0
        self.key_dim = config.key_dim // config.num_head
        self.value_dim = config.value_dim // config.num_head

        self.adaptive_norm_q = AdaptiveLayerNorm(num_channels, cond_dim_q)
        self.adaptive_norm_k = AdaptiveLayerNorm(num_channels, cond_dim_k)

        q_shape = (config.num_head, self.key_dim)
        k_shape = (config.num_head, self.key_dim)
        v_shape = (config.num_head, self.value_dim)
        self.q_projection = Linear(q_shape, input_dims=num_channels, use_bias=True)
        self.k_projection = Linear(k_shape, input_dims=num_channels, use_bias=False)
        self.v_projection = Linear(v_shape, input_dims=num_channels, use_bias=False)

        self.gating_query = Linear(
            config.num_head * self.value_dim,
            input_dims=num_channels,
            use_bias=False,
            initializer="zeros",
        )
        self.adaptive_zero = AdaptiveZeroInit(num_channels, cond_dim_q, final_init)

    def __call__(
        self,
        x_q: mx.array,
        x_k: mx.array,
        mask_q: mx.array,
        mask_k: mx.array,
        pair_logits: mx.array | None,
        single_cond_q: mx.array | None,
        single_cond_k: mx.array | None,
    ) -> mx.array:
        """Apply cross-attention using SDPA.

        Args:
            x_q: Query input [..., seq_q, channels]
            x_k: Key input [..., seq_k, channels]
            mask_q: Query mask where 1.0=attend [..., seq_q]
            mask_k: Key mask where 1.0=attend [..., seq_k]
            pair_logits: Optional additive bias [..., heads, seq_q, seq_k]
            single_cond_q: Optional query conditioning
            single_cond_k: Optional key conditioning
        """
        q_in = self.adaptive_norm_q(x_q, single_cond_q)
        k_in = self.adaptive_norm_k(x_k, single_cond_k)

        # Q/K/V projections
        q = self.q_projection(q_in)  # [..., seq_q, num_heads, key_dim]
        k = self.k_projection(k_in)  # [..., seq_k, num_heads, key_dim]
        v = self.v_projection(k_in)  # [..., seq_k, num_heads, value_dim]

        # Get dimensions - need to flatten leading dims into batch
        *leading_dims_q, seq_q, num_heads, key_dim = q.shape
        *leading_dims_k, seq_k, _, _ = k.shape
        value_dim = v.shape[-1]
        has_leading = len(leading_dims_q) > 0

        # Flatten leading dims into batch for SDPA
        if has_leading:
            batch_flat = int(np.prod(leading_dims_q))
            q = q.reshape(batch_flat, seq_q, num_heads, key_dim)
            k = k.reshape(batch_flat, seq_k, num_heads, key_dim)
            v = v.reshape(batch_flat, seq_k, num_heads, value_dim)

            # Transpose to SDPA format: [batch, heads, seq, dim]
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Flatten masks to match batch
            mask_q_flat = mask_q.reshape(batch_flat, seq_q)
            mask_k_flat = mask_k.reshape(batch_flat, seq_k)

            # Flatten pair_logits if provided
            if pair_logits is not None:
                pair_logits = pair_logits.reshape(batch_flat, num_heads, seq_q, seq_k)
        else:
            # Unbatched: [seq, heads, dim] -> [1, heads, seq, dim]
            q = q.transpose(1, 0, 2)[None]
            k = k.transpose(1, 0, 2)[None]
            v = v.transpose(1, 0, 2)[None]
            batch_flat = 1
            mask_q_flat = mask_q[None]
            mask_k_flat = mask_k[None]

        # Pre-cast Q/K/V to same dtype for performance
        compute_dtype = mx.float32
        q = q.astype(compute_dtype)
        k = k.astype(compute_dtype)
        v = v.astype(compute_dtype)

        scale = self.key_dim ** -0.5

        # Build product mask matching JAX AF3:
        #   bias = 1e9 * (mask_q - 1)[..., :, None] * (mask_k - 1)[..., None, :]
        # For valid queries, ALL keys are accessible (bias=0).
        # Only (invalid_q, invalid_k) gets +1e9 positive bias.
        # mask_q_flat: [batch, seq_q] → [batch, seq_q, 1]
        # mask_k_flat: [batch, seq_k] → [batch, 1, seq_k]
        combined_mask = (
            _MASK_VALUE
            * (mask_q_flat[:, :, None] - 1.0)
            * (mask_k_flat[:, None, :] - 1.0)
        )
        combined_mask = combined_mask[:, None, :, :]  # [batch, 1, seq_q, seq_k]

        # Detect fully masked rows (where all keys are masked)
        # [batch, seq_k] -> [batch]
        has_valid_key = mx.any(mask_k_flat > 0.5, axis=-1)
        fully_masked_rows = ~has_valid_key  # [batch]

        # Add pair logits if provided
        if pair_logits is not None:
            combined_mask = combined_mask + pair_logits

        combined_mask = combined_mask.astype(compute_dtype)

        # Use MLX SDPA
        attn_output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=scale,
            mask=combined_mask,
        )  # [batch, heads, seq_q, value_dim]

        # Zero fully masked rows (all keys masked)
        # [batch] -> [batch, 1, 1, 1] for broadcasting
        fully_masked_expanded = fully_masked_rows[:, None, None, None]
        attn_output = mx.where(fully_masked_expanded, 0.0, attn_output)

        # Transpose back: [batch, heads, seq_q, dim] -> [batch, seq_q, heads, dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Reshape back to original leading dimensions
        if has_leading:
            attn_output = attn_output.reshape(*leading_dims_q, seq_q, num_heads, value_dim)
        else:
            attn_output = attn_output[0]  # Remove added batch dim: [seq_q, heads, dim]

        # Cast back to input dtype
        attn_output = attn_output.astype(x_q.dtype)

        # Flatten heads*value_dim
        weighted_avg = attn_output.reshape(attn_output.shape[:-2] + (-1,))

        # Apply gating
        gate = mx.sigmoid(self.gating_query(q_in))
        weighted_avg = weighted_avg * gate

        return self.adaptive_zero(weighted_avg, single_cond_q)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and transition."""

    def __init__(
        self,
        num_channels: int,
        config: TransformerConfig,
        *,
        final_init: str,
        cond_dim: int | None,
    ) -> None:
        super().__init__()
        self.self_attention = SelfAttention(
            num_channels,
            config.attention,
            final_init=final_init,
            cond_dim=cond_dim,
        )
        self.transition = TransitionBlock(
            num_channels,
            config.num_intermediate_factor,
            cond_dim=cond_dim,
            final_init=final_init,
        )

    def __call__(
        self,
        act: mx.array,
        mask: mx.array,
        pair_logits: mx.array | None,
        single_cond: mx.array | None,
    ) -> mx.array:
        act = act + self.self_attention(act, mask, pair_logits, single_cond)
        act = act + self.transition(act, single_cond)
        return act


class Transformer(nn.Module):
    """AF3 diffusion transformer stack."""

    def __init__(
        self,
        config: TransformerConfig,
        global_config,
        name: str = "transformer",
    ) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        self.num_blocks = config.num_blocks
        self.super_block_size = config.super_block_size
        assert self.num_blocks % self.super_block_size == 0
        self.num_super_blocks = self.num_blocks // self.super_block_size

        # Pair input layer norm (set at runtime with correct dims)
        self.pair_input_layer_norm = LayerNorm(1, create_offset=False)
        # One pair-logits projection per super-block (matches AF3 stacked weights).
        self.pair_logits_projections: list[Linear] = [
            Linear(
                (self.super_block_size, config.attention.num_head),
                input_dims=1,
                use_bias=False,
            )
            for _ in range(self.num_super_blocks)
        ]

        # Build blocks lazily once act dims are known
        self.blocks: list[TransformerBlock] = []
        self._built = False

    def _build_blocks(self, num_channels: int, cond_dim: int, pair_cond_dim: int | None) -> None:
        if self._built:
            return
        self.blocks = [
            TransformerBlock(
                num_channels,
                self.config,
                final_init=self.global_config.final_init,
                cond_dim=cond_dim,
            )
            for _ in range(self.num_blocks)
        ]
        # Pair input layer norm with correct dims
        if pair_cond_dim is not None:
            self.pair_input_layer_norm = LayerNorm(
                pair_cond_dim,
                create_offset=False,
            )
            self.pair_logits_projections = [
                Linear(
                    (self.super_block_size, self.config.attention.num_head),
                    input_dims=pair_cond_dim,
                    use_bias=False,
                )
                for _ in range(self.num_super_blocks)
            ]
        self._built = True

    def __call__(
        self,
        act: mx.array,
        mask: mx.array,
        single_cond: mx.array,
        pair_cond: mx.array | None,
    ) -> mx.array:
        num_channels = act.shape[-1]
        pair_cond_dim = None if pair_cond is None else pair_cond.shape[-1]
        cond_dim = single_cond.shape[-1]
        self._build_blocks(num_channels, cond_dim, pair_cond_dim)

        if pair_cond is None:
            pair_act = None
        else:
            pair_act = self.pair_input_layer_norm(pair_cond)

        block_idx = 0
        for super_idx in range(self.num_super_blocks):
            if pair_act is None:
                pair_logits = None
            else:
                pair_logits = self.pair_logits_projections[super_idx](pair_act)
                # [seq, seq, super_block_size, num_head] -> [super_block_size, num_head, seq, seq]
                pair_logits = mx.transpose(pair_logits, (2, 3, 0, 1))

            for inner_idx in range(self.super_block_size):
                logits_i = None if pair_logits is None else pair_logits[inner_idx]
                act = self.blocks[block_idx](act, mask, logits_i, single_cond)
                block_idx += 1

        return act


class CrossAttTransformerBlock(nn.Module):
    """Single block of CrossAttTransformer."""

    def __init__(
        self,
        num_channels: int,
        config: CrossAttTransformerConfig,
        *,
        final_init: str,
        cond_dim_q: int | None,
        cond_dim_k: int | None,
    ) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(
            num_channels,
            config.attention,
            final_init=final_init,
            cond_dim_q=cond_dim_q,
            cond_dim_k=cond_dim_k,
        )
        self.transition = TransitionBlock(
            num_channels,
            config.num_intermediate_factor,
            cond_dim=cond_dim_q,
            final_init=final_init,
        )

    def __call__(
        self,
        queries_act: mx.array,
        keys_act: mx.array,
        queries_mask: mx.array,
        keys_mask: mx.array,
        pair_logits: mx.array | None,
        queries_single_cond: mx.array | None,
        keys_single_cond: mx.array | None,
    ) -> mx.array:
        queries_act = queries_act + self.cross_attention(
            queries_act,
            keys_act,
            queries_mask,
            keys_mask,
            pair_logits,
            queries_single_cond,
            keys_single_cond,
        )
        queries_act = queries_act + self.transition(queries_act, queries_single_cond)
        return queries_act


class CrossAttTransformer(nn.Module):
    """Cross-attention transformer matching AF3."""

    def __init__(
        self,
        config: CrossAttTransformerConfig,
        global_config,
        name: str = "transformer",
    ) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        self.pair_input_layer_norm = LayerNorm(1, create_offset=False)
        self.pair_logits_projection = Linear(
            (config.num_blocks, config.attention.num_head),
            input_dims=1,
            use_bias=False,
        )

        self.blocks = []
        self._built = False

    def _build(self, num_channels: int, pair_cond_dim: int) -> None:
        if self._built:
            return
        self.blocks = [
            CrossAttTransformerBlock(
                num_channels,
                self.config,
                final_init=self.global_config.final_init,
                cond_dim_q=num_channels,
                cond_dim_k=num_channels,
            )
            for _ in range(self.config.num_blocks)
        ]
        self.pair_input_layer_norm = LayerNorm(
            pair_cond_dim,
            create_offset=False,
        )
        self.pair_logits_projection = Linear(
            (self.config.num_blocks, self.config.attention.num_head),
            input_dims=pair_cond_dim,
            use_bias=False,
        )
        self._built = True

    def __call__(
        self,
        queries_act: mx.array,
        queries_mask: mx.array,
        queries_to_keys,
        keys_mask: mx.array,
        queries_single_cond: mx.array,
        keys_single_cond: mx.array,
        pair_cond: mx.array,
    ) -> mx.array:
        num_channels = queries_act.shape[-1]
        self._build(num_channels, pair_cond.shape[-1])

        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)
        # (num_blocks, num_subsets, num_heads, num_queries, num_keys)
        pair_logits = mx.transpose(pair_logits, (3, 0, 4, 1, 2))

        # Convert queries_act to keys layout each block
        from alphafold3_mlx.atom_layout import convert as layout_convert

        for idx, block in enumerate(self.blocks):
            keys_act = layout_convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )
            queries_act = block(
                queries_act,
                keys_act,
                queries_mask,
                keys_mask,
                pair_logits[idx],
                queries_single_cond,
                keys_single_cond,
            )

        return queries_act


# Backward-compatible aliases expected by network __init__ exports.
DiffusionTransformerBlock = TransformerBlock
DiffusionTransformer = Transformer
