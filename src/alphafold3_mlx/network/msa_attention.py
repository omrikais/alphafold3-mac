"""MSA attention modules for AlphaFold 3 MLX.

This module implements MSA row and column attention for processing
multiple sequence alignments during Evoformer.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.attention import (
    boolean_to_additive_mask,
    detect_fully_masked_rows,
)


class MSARowAttention(nn.Module):
    """MSA row-wise attention.

    Performs self-attention across the sequence dimension for each MSA row,
    allowing each position to attend to all other positions in the same sequence.

    Args:
        msa_channel: MSA representation dimension.
        pair_channel: Pair representation dimension for bias.
        num_heads: Number of attention heads.
        key_dim: Dimension per head (default: msa_channel // num_heads).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        msa_channel: int,
        pair_channel: int,
        num_heads: int = 8,
        key_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.msa_channel = msa_channel
        self.pair_channel = pair_channel
        self.num_heads = num_heads
        self.key_dim = key_dim or (msa_channel // num_heads)
        self.dropout = dropout

        # Layer norm
        self.layer_norm_msa = LayerNorm(msa_channel)
        self.layer_norm_pair = LayerNorm(pair_channel)

        # Value projection only (matches AF3 MSAAttention).
        self.v_proj = Linear(
            (num_heads, self.key_dim),
            input_dims=msa_channel,
            use_bias=False,
        )

        # Pair bias projection
        self.pair_bias_proj = Linear(
            num_heads,
            input_dims=pair_channel,
            use_bias=False,
        )

        # Output projection (JAX hm.Linear defaults to use_bias=False)
        self.o_proj = Linear(
            msa_channel,
            input_dims=num_heads * self.key_dim,
            use_bias=False,
        )

        # Gating (JAX hm.Linear defaults to use_bias=False; bias_init is unused)
        self.gate_proj = Linear(
            msa_channel,
            input_dims=msa_channel,
            use_bias=False,
            initializer="zeros",
        )

    def __call__(
        self,
        msa: mx.array,
        pair: mx.array,
        msa_mask: mx.array,
    ) -> mx.array:
        """Apply MSA row attention.

        Args:
            msa: MSA representation [batch, num_seqs, seq_len, msa_channel].
            pair: Pair representation [batch, seq_len, seq_len, pair_channel].
            msa_mask: MSA mask [batch, num_seqs, seq_len].

        Returns:
            Updated MSA representation [batch, num_seqs, seq_len, msa_channel].
        """
        # Matches AF3 modules.MSAAttention: attention weights come from pair
        # logits, and MSA contributes values + gating.
        msa_normed = self.layer_norm_msa(msa)
        pair_normed = self.layer_norm_pair(pair)

        logits = self.pair_bias_proj(pair_normed)  # [batch, seq, seq, heads]
        logits = logits.transpose(0, 3, 1, 2)  # [batch, heads, seq_q, seq_k]

        # AF3 uses max mask across MSA rows to gate sequence-level keys.
        seq_mask = mx.max(msa_mask.astype(logits.dtype), axis=1)  # [batch, seq]
        logits = logits + 1e9 * (seq_mask[:, None, None, :] - 1.0)
        weights = mx.softmax(logits.astype(mx.float32), axis=-1).astype(msa_normed.dtype)

        v = self.v_proj(msa_normed)  # [batch, num_msa, seq_k, heads, value_dim]
        weighted = mx.einsum("bhqk,bmkhd->bmqhd", weights, v)
        weighted = weighted.reshape(weighted.shape[:-2] + (-1,))

        gate = mx.sigmoid(self.gate_proj(msa_normed))
        weighted = weighted * gate

        out = self.o_proj(weighted)
        return msa + out


class MSATransition(nn.Module):
    """MSA transition block.

    Args:
        msa_channel: MSA representation dimension.
        expansion_factor: Hidden layer expansion factor.
    """

    def __init__(
        self,
        msa_channel: int,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()

        self.layer_norm = LayerNorm(msa_channel)
        hidden_dim = msa_channel * expansion_factor
        self._hidden_dim = hidden_dim

        # Match AF3 TransitionBlock GLU parameterization.
        self.up_proj = Linear(
            hidden_dim * 2,
            input_dims=msa_channel,
            use_bias=False,
            initializer="relu",
        )
        self.down_proj = Linear(msa_channel, input_dims=hidden_dim, use_bias=False)

    def __call__(self, msa: mx.array) -> mx.array:
        """Apply transition block.

        Args:
            msa: MSA representation [batch, num_seqs, seq_len, msa_channel].

        Returns:
            Updated MSA representation.
        """
        residual = msa
        msa = self.layer_norm(msa)
        msa = self.up_proj(msa)
        a, b = mx.split(msa, 2, axis=-1)
        msa = (a * mx.sigmoid(a)) * b
        msa = self.down_proj(msa)
        return residual + msa
