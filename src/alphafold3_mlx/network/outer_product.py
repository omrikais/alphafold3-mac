"""Outer product mean for AlphaFold 3 MLX.

This module implements the OuterProductMean operation, which
computes pairwise features from single (per-residue) representations.

This is used to communicate information from the single representation
to the pair representation in the Evoformer.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm

logger = logging.getLogger(__name__)


class OuterProductMean(nn.Module):
    """Outer product mean operation.

    Computes pairwise features from single representations:
    pair[i,j] += outer_product(single[i], single[j])

    The operation:
    1. Projects single representation to left/right features
    2. Computes outer product of left[i] and right[j]
    3. Averages over any batch/MSA dimension
    4. Projects back to pair dimension
    """

    def __init__(
        self,
        seq_channel: int,
        pair_channel: int,
        num_outer_channel: int = 32,
    ) -> None:
        """Initialize outer product mean.

        Args:
            seq_channel: Single representation channel dimension.
            pair_channel: Pair representation channel dimension.
            num_outer_channel: Intermediate channel dimension for outer product.
        """
        super().__init__()

        self.seq_channel = seq_channel
        self.pair_channel = pair_channel
        self.num_outer_channel = num_outer_channel

        # Input normalization
        self.norm = LayerNorm(seq_channel)

        # Left and right projections
        self.left_proj = Linear(num_outer_channel, input_dims=seq_channel, use_bias=False)
        self.right_proj = Linear(num_outer_channel, input_dims=seq_channel, use_bias=False)

        # Output projection (from outer product to pair)
        # Outer product shape: num_outer_channel * num_outer_channel
        self.output_proj = Linear(
            pair_channel,
            input_dims=num_outer_channel * num_outer_channel,
            use_bias=False,
        )

    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Compute outer product mean and add to pair.

        Args:
            single: Single representation. Shape: [batch, seq, seq_channel]
            pair: Pair representation. Shape: [batch, seq, seq, pair_channel]
            mask: Optional sequence mask. Shape: [batch, seq]

        Returns:
            Updated pair representation.
        """
        # Normalize input
        x = self.norm(single)

        # Project to left and right features
        left = self.left_proj(x)  # [batch, seq, num_outer_channel]
        right = self.right_proj(x)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask[..., None]  # [batch, seq, 1]
            left = left * mask_expanded
            right = right * mask_expanded

        # Compute outer product: left[i] outer right[j]
        # [batch, seq_i, outer] outer [batch, seq_j, outer] -> [batch, seq_i, seq_j, outer*outer]
        # Using einsum: "bio,bjo->bijo" then reshape
        outer = mx.einsum("bio,bjo->bijo", left, right)
        # outer shape: [batch, seq_i, seq_j, outer, outer]
        # Actually einsum gives us outer product in last dim
        # Let me fix: need explicit outer product

        # Outer product via broadcasting
        # left: [batch, seq, 1, outer] * right: [batch, 1, seq, outer]
        # But we want: [batch, seq_i, seq_j, outer_left, outer_right]
        left_expanded = left[:, :, None, :, None]  # [batch, seq_i, 1, outer, 1]
        right_expanded = right[:, None, :, None, :]  # [batch, 1, seq_j, 1, outer]
        outer = left_expanded * right_expanded  # [batch, seq_i, seq_j, outer, outer]

        # Flatten outer dimensions
        batch_size, seq_i, seq_j, _, _ = outer.shape
        outer = outer.reshape(batch_size, seq_i, seq_j, -1)
        # [batch, seq_i, seq_j, outer * outer]

        # Project to pair dimension
        output = self.output_proj(outer)

        # Add to pair representation
        return pair + output


class OuterProductMeanMSA(nn.Module):
    """Outer product mean from MSA to pair.

    This variant operates on MSA representations, computing the mean
    outer product across MSA sequences.
    """

    def __init__(
        self,
        msa_channel: int,
        pair_channel: int,
        num_outer_channel: int = 32,
    ) -> None:
        """Initialize MSA outer product mean.

        Args:
            msa_channel: MSA representation channel dimension.
            pair_channel: Pair representation channel dimension.
            num_outer_channel: Intermediate channel dimension.
        """
        super().__init__()

        self.msa_channel = msa_channel
        self.pair_channel = pair_channel
        self.num_outer_channel = num_outer_channel

        self.norm = LayerNorm(msa_channel)
        self.left_proj = Linear(num_outer_channel, input_dims=msa_channel, use_bias=False)
        self.right_proj = Linear(num_outer_channel, input_dims=msa_channel, use_bias=False)
        self.output_proj = Linear(
            pair_channel,
            input_dims=num_outer_channel * num_outer_channel,
            use_bias=True,
        )

    def __call__(
        self,
        msa: mx.array,
        pair: mx.array,
        msa_mask: mx.array | None = None,
    ) -> mx.array:
        """Compute outer product mean from MSA and add to pair.

        Args:
            msa: MSA representation. Shape: [batch, num_msa, seq, msa_channel]
            pair: Pair representation. Shape: [batch, seq, seq, pair_channel]
            msa_mask: Optional MSA mask. Shape: [batch, num_msa, seq]

        Returns:
            Updated pair representation.
        """
        # Normalize
        x = self.norm(msa)

        # Project
        left = self.left_proj(x)  # [batch, num_msa, seq, outer]
        right = self.right_proj(x)

        # Use fused contractions to avoid materializing the huge intermediate:
        # [batch, msa, seq_i, seq_j, outer, outer].
        if logger.isEnabledFor(logging.DEBUG):
            b, m, s, o = left.shape
            bytes_per_elem = 2 if left.dtype in (mx.float16, mx.bfloat16) else 4
            est_bytes = b * m * s * s * o * o * bytes_per_elem
            logger.debug(
                "OuterProductMeanMSA avoiding materialization of %s (~%.2f GiB).",
                (b, m, s, s, o, o),
                est_bytes / (1024 ** 3),
            )

        # JAX AF3 computes the raw sum, projects (with bias), THEN normalizes.
        # This ensures the bias is also divided by the count, matching:
        #   output = (sum @ W + bias) / (eps + count)
        epsilon = 1e-3

        if msa_mask is not None:
            mask = msa_mask.astype(left.dtype)
            left_masked = left * mask[..., None]
            right_masked = right * mask[..., None]

            # Sum over MSA in one contraction:
            # [b,m,i,o] x [b,m,j,p] -> [b,i,j,o,p]
            outer_sum = mx.einsum("bmio,bmjp->bijop", left_masked, right_masked)

            # Count valid MSA rows where both tokens i and j are present.
            # [b,m,i] x [b,m,j] -> [b,i,j]
            norm = mx.einsum("bmi,bmj->bij", mask, mask)
        else:
            outer_sum = mx.einsum("bmio,bmjp->bijop", left, right)
            num_msa = float(left.shape[1])
            norm = mx.full(
                (left.shape[0], left.shape[2], left.shape[2]),
                num_msa,
                dtype=left.dtype,
            )
        # [batch, seq_i, seq_j, outer_left, outer_right]

        # Flatten and project (includes bias)
        batch_size, seq_i, seq_j, _, _ = outer_sum.shape
        outer_flat = outer_sum.reshape(batch_size, seq_i, seq_j, -1)
        output = self.output_proj(outer_flat)

        # Normalize AFTER projection (JAX AF3 parity)
        output = output / (epsilon + norm[..., None])

        return pair + output
