"""PairFormer iteration block for AlphaFold 3 MLX.

This module implements the PairFormerIteration block, which is the
core building block of the trunk processing. The iteration processes pair
representation first, then optionally updates single representation using
information from the updated pair.

PairFormer Structure (matching JAX AF3):
1. Triangle multiplication (outgoing) - pair pathway
2. Triangle multiplication (incoming) - pair pathway
3. Pair self-attention (row-wise) - pair pathway
4. Pair self-attention (column-wise) - pair pathway
5. Pair transition (feedforward) - pair pathway
6. [Optional: with_single=True]
   - Compute pair_logits from updated pair: LayerNorm → Linear → transpose
   - Single self-attention with pair_logits as bias
   - Single transition (feedforward)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import LayerNorm, Linear
from alphafold3_mlx.network.attention import AF3SelfAttention, GridSelfAttention
from alphafold3_mlx.network.transition import TransitionBlock
from alphafold3_mlx.network.triangle_ops import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)


class PairFormerIteration(nn.Module):
    """Single PairFormer iteration block matching JAX AF3.

    This block processes pair representation through:
    1. Triangle multiplication (outgoing + incoming)
    2. Pair self-attention (row and column)
    3. Pair transition

    Optionally (with_single=True), it also:
    4. Computes pair_logits from updated pair representation
    5. Applies single self-attention with pair_logits as bias
    6. Applies single transition
    """

    def __init__(
        self,
        seq_channel: int = 384,
        pair_channel: int = 128,
        num_attention_heads: int = 4,
        single_attention_heads: int = 16,
        attention_key_dim: int | None = None,
        intermediate_factor: int = 4,
        with_single: bool = True,
        dropout_rate: float = 0.0,  # Not used in inference
    ) -> None:
        """Initialize PairFormer iteration.

        Args:
            seq_channel: Single representation channel dimension.
            pair_channel: Pair representation channel dimension.
            num_attention_heads: Number of attention heads.
            single_attention_heads: Number of heads for single attention path.
            attention_key_dim: Key dimension per attention head. If None, uses
                input_dim // num_heads for each attention module.
            intermediate_factor: Expansion factor for transition blocks.
            with_single: Whether to include single representation processing.
            dropout_rate: Dropout rate (0 for inference).
        """
        super().__init__()

        self.seq_channel = seq_channel
        self.pair_channel = pair_channel
        self.num_attention_heads = num_attention_heads
        self.single_attention_heads = single_attention_heads
        self.with_single = with_single

        # Compute key_dim per module if not specified
        pair_key_dim = attention_key_dim or max(pair_channel // num_attention_heads, 16)

        # === Pair Representation Pathway ===

        # Triangle multiplication (outgoing)
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(
            pair_dim=pair_channel,
            intermediate_dim=pair_channel,
        )

        # Triangle multiplication (incoming)
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(
            pair_dim=pair_channel,
            intermediate_dim=pair_channel,
        )

        # Pair self-attention (row-wise) - transpose=False
        self.pair_attention_row = GridSelfAttention(
            input_dim=pair_channel,
            num_heads=num_attention_heads,
            key_dim=pair_key_dim,
            gated=True,
            orientation="row",
        )

        # Pair self-attention (column-wise) - transpose=True
        self.pair_attention_col = GridSelfAttention(
            input_dim=pair_channel,
            num_heads=num_attention_heads,
            key_dim=pair_key_dim,
            gated=True,
            orientation="column",
        )

        # Pair transition
        self.pair_transition = TransitionBlock(
            input_dim=pair_channel,
            intermediate_factor=intermediate_factor,
        )

        # === Single Representation Pathway (optional) ===

        if with_single:
            # Pair logits computation: LayerNorm(pair) → Linear(num_heads) → transpose
            # This provides structural prior as attention bias for single attention
            self.pair_logits_norm = LayerNorm(pair_channel)
            self.pair_logits_proj = Linear(
                single_attention_heads,
                input_dims=pair_channel,
                use_bias=False,
            )

            # Single self-attention (AF3 diffusion_transformer.self_attention)
            self.single_attention = AF3SelfAttention(
                input_dim=seq_channel,
                num_heads=single_attention_heads,
                key_dim=None,
                value_dim=None,
                final_init="zeros",
            )

            # Single transition
            self.single_transition = TransitionBlock(
                input_dim=seq_channel,
                intermediate_factor=intermediate_factor,
            )

    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        seq_mask: mx.array | None = None,
        pair_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply PairFormer iteration.

        Args:
            single: Single representation. Shape: [batch, seq, seq_channel]
            pair: Pair representation. Shape: [batch, seq, seq, pair_channel]
            seq_mask: Optional sequence mask. Shape: [batch, seq]
            pair_mask: Optional pair mask. Shape: [batch, seq, seq]

        Returns:
            Tuple of (updated_single, updated_pair).
        """
        # === Pair Pathway (runs FIRST) ===

        # Triangle multiplication (outgoing)
        pair = self.triangle_mult_outgoing(pair, mask=pair_mask)

        # Triangle multiplication (incoming)
        pair = self.triangle_mult_incoming(pair, mask=pair_mask)

        # Pair self-attention (row-wise)
        pair = pair + self.pair_attention_row(pair, mask=pair_mask)

        # Pair self-attention (column-wise)
        pair = pair + self.pair_attention_col(pair, mask=pair_mask)

        # Pair transition
        pair = self.pair_transition(pair)

        # === Single Pathway (runs AFTER pair, using updated pair info) ===

        if self.with_single:
            # Compute pair_logits from updated pair representation
            # This provides structural prior as attention bias
            pair_normed = self.pair_logits_norm(pair)  # [batch, seq, seq, pair_channel]
            pair_logits = self.pair_logits_proj(pair_normed)  # [batch, seq, seq, num_heads]

            # Transpose to [batch, num_heads, seq, seq] for attention bias
            pair_logits = pair_logits.transpose(0, 3, 1, 2)  # [batch, heads, seq, seq]

            # Single self-attention with pair_logits as bias
            # Note: GatedSelfAttention has internal act_norm, so we pass raw single
            single_attn = self.single_attention(
                single, mask=seq_mask, pair_logits=pair_logits
            )
            single = single + single_attn

            # Single transition
            single = self.single_transition(single)

        return single, pair


class PairFormerStack(nn.Module):
    """Stack of PairFormer iterations.

    This class wraps multiple PairFormerIteration blocks, optionally with
    shared weights (for layer-stacked weight loading compatibility).
    """

    def __init__(
        self,
        num_layers: int,
        seq_channel: int = 384,
        pair_channel: int = 128,
        num_attention_heads: int = 4,
        attention_key_dim: int | None = None,
        intermediate_factor: int = 4,
        with_single: bool = True,
        share_weights: bool = False,
    ) -> None:
        """Initialize PairFormer stack.

        Args:
            num_layers: Number of PairFormer iterations.
            seq_channel: Single representation channel dimension.
            pair_channel: Pair representation channel dimension.
            num_attention_heads: Number of attention heads.
            attention_key_dim: Key dimension per attention head.
            intermediate_factor: Expansion factor for transitions.
            with_single: Whether to include single representation processing.
            share_weights: If True, use single module for all layers.
        """
        super().__init__()

        self.num_layers = num_layers
        self.share_weights = share_weights

        if share_weights:
            # Single module instance used for all layers
            self.layer = PairFormerIteration(
                seq_channel=seq_channel,
                pair_channel=pair_channel,
                num_attention_heads=num_attention_heads,
                attention_key_dim=attention_key_dim,
                intermediate_factor=intermediate_factor,
                with_single=with_single,
            )
        else:
            # Separate module instances per layer
            self.layers = [
                PairFormerIteration(
                    seq_channel=seq_channel,
                    pair_channel=pair_channel,
                    num_attention_heads=num_attention_heads,
                    attention_key_dim=attention_key_dim,
                    intermediate_factor=intermediate_factor,
                    with_single=with_single,
                )
                for _ in range(num_layers)
            ]

    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        seq_mask: mx.array | None = None,
        pair_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply PairFormer stack.

        Args:
            single: Single representation. Shape: [batch, seq, seq_channel]
            pair: Pair representation. Shape: [batch, seq, seq, pair_channel]
            seq_mask: Optional sequence mask. Shape: [batch, seq]
            pair_mask: Optional pair mask. Shape: [batch, seq, seq]

        Returns:
            Tuple of (updated_single, updated_pair) after all layers.
        """
        if self.share_weights:
            for _ in range(self.num_layers):
                single, pair = self.layer(single, pair, seq_mask, pair_mask)
        else:
            for layer in self.layers:
                single, pair = layer(single, pair, seq_mask, pair_mask)

        return single, pair
