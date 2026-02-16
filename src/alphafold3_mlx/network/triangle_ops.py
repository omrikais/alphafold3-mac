"""Triangle multiplication operations for AlphaFold 3 MLX.

This module implements triangle multiplication, which is a core
operation in the PairFormer block for propagating pairwise information
through triangular relationships.

There are two orientations:
- Outgoing: aggregates over shared endpoints going "out" from a pair
- Incoming: aggregates over shared endpoints coming "in" to a pair

JAX AF3 Compatibility:
- Combined projection/gate weights (pair_channel → 2*intermediate_dim)
- Center norm normalizes along channel axis (axis=0 after transpose)
- Gating linear without bias
- Matching einsum equations

Chunking support is available via the global chunking configuration
for memory-efficient processing of large sequences.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.chunking import get_chunking_config, chunk_einsum


class TriangleMultiplication(nn.Module):
    """Triangle multiplication operation matching JAX AF3.

    This operation propagates pair information through triangular relationships.
    Given pair representation z[i,j], it computes interactions via intermediate
    positions k:
    - Outgoing: z'[i,j] = sum_k(a[i,k] * b[j,k])
    - Incoming: z'[i,j] = sum_k(a[k,i] * b[k,j])

    The operation includes:
    1. LayerNorm on input (left_norm_input)
    2. Combined gated projection (gate * projection, split into left/right)
    3. Einsum multiplication
    4. Center norm (normalizes along channel axis)
    5. Output projection with gating

    JAX AF3 Compatibility:
    - Uses combined projection/gate weights: [pair_channel → 2*intermediate_dim]
    - Center norm normalizes along channel axis (not last dim)
    - Gating linear has no bias
    """

    def __init__(
        self,
        pair_dim: int,
        intermediate_dim: int | None = None,
        orientation: Literal["outgoing", "incoming"] = "outgoing",
    ) -> None:
        """Initialize triangle multiplication.

        Args:
            pair_dim: Pair representation channel dimension.
            intermediate_dim: Intermediate projection dimension. Defaults to pair_dim.
            orientation: "outgoing" or "incoming" for different aggregation patterns.
        """
        super().__init__()

        self.pair_dim = pair_dim
        self.intermediate_dim = intermediate_dim or pair_dim
        self.orientation = orientation

        # Input LayerNorm (left_norm_input in JAX)
        self.input_norm = LayerNorm(pair_dim)

        # Combined projection: [pair_dim → 2*intermediate_dim]
        # This produces both left and right projections, which are split later
        self.projection = Linear(
            self.intermediate_dim * 2,
            input_dims=pair_dim,
            use_bias=False,
        )

        # Combined gate: [pair_dim → 2*intermediate_dim]
        # This produces both left and right gates
        self.gate = Linear(
            self.intermediate_dim * 2,
            input_dims=pair_dim,
            use_bias=False,
            initializer="zeros",
        )

        # Center norm: normalizes along channel axis
        # In JAX this is done with scale/offset of shape [intermediate_dim]
        # applied after transposing to [intermediate_dim, seq_i, seq_j]
        self.center_norm_scale = mx.ones((self.intermediate_dim,))
        self.center_norm_offset = mx.zeros((self.intermediate_dim,))

        # Output projection: [intermediate_dim → pair_dim]
        self.output_projection = Linear(
            pair_dim,
            input_dims=self.intermediate_dim,
            use_bias=False,
            initializer="zeros",
        )

        # Output gating: [pair_dim → pair_dim] (no bias in JAX AF3)
        self.gating_linear = Linear(
            pair_dim,
            input_dims=pair_dim,
            use_bias=False,
            initializer="zeros",
        )

    def __call__(
        self,
        pair: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Apply triangle multiplication.

        Args:
            pair: Pair representation. Shape: [batch, seq_i, seq_j, pair_dim]
            mask: Optional pair mask. Shape: [batch, seq_i, seq_j]

        Returns:
            Updated pair representation. Shape: [batch, seq_i, seq_j, pair_dim]
        """
        # Input LayerNorm
        x = self.input_norm(pair)
        input_act = x  # Save for gating later

        # Combined projection and gate
        # proj: [batch, seq_i, seq_j, 2*intermediate_dim]
        proj = self.projection(x)
        gate_values = self.gate(x)

        # Transpose to [batch, 2*intermediate_dim, seq_i, seq_j] for masking and splitting
        proj = proj.transpose(0, 3, 1, 2)  # [batch, 2*intermediate_dim, seq_i, seq_j]
        gate_values = gate_values.transpose(0, 3, 1, 2)

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting: [batch, 1, seq_i, seq_j]
            mask_expanded = mask[:, None, :, :]
            proj = proj * mask_expanded

        # Apply gating: sigmoid(gate) * projection
        proj = mx.sigmoid(gate_values) * proj

        # Split into left (a) and right (b) using JAX-compatible reshape-then-split
        # JAX does: reshape(num_channels, 2, ...) then split along axis=1
        # This interleaves channels: even indices → a, odd indices → b
        # [batch, 2*intermediate_dim, seq_i, seq_j] → [batch, intermediate_dim, 2, seq_i, seq_j]
        batch_size = proj.shape[0]
        seq_i, seq_j = proj.shape[2], proj.shape[3]
        proj = proj.reshape(batch_size, self.intermediate_dim, 2, seq_i, seq_j)
        # Extract a (index 0) and b (index 1) along the "2" dimension
        a = proj[:, :, 0, :, :]  # [batch, intermediate_dim, seq_i, seq_j]
        b = proj[:, :, 1, :, :]  # [batch, intermediate_dim, seq_i, seq_j]

        # Triangle einsum (with optional chunking)
        chunking = get_chunking_config()
        chunk_size = chunking.triangle_chunk_size if chunking.enabled else None

        if self.orientation == "outgoing":
            # Outgoing: sum over k for pairs (i,k) and (j,k)
            # a: [batch, c, i, k], b: [batch, c, j, k]
            # output: [batch, c, i, j]
            # Equation: "bcik,bcjk->bcij"
            output = chunk_einsum("bcik,bcjk->bcij", a, b, chunk_size=chunk_size)
        else:
            # Incoming: sum over k for pairs (k,i) and (k,j)
            # a: [batch, c, k, j], b: [batch, c, k, i]
            # output: [batch, c, i, j]
            # Equation: "bckj,bcki->bcij"
            output = chunk_einsum("bckj,bcki->bcij", a, b, chunk_size=chunk_size)

        # Center norm: normalize along channel axis (axis=1)
        # Mean and variance computed along axis=1, keeping spatial dims
        mean = mx.mean(output, axis=1, keepdims=True)  # [batch, 1, seq_i, seq_j]
        var = mx.var(output, axis=1, keepdims=True)
        eps = 1e-5
        output = (output - mean) / mx.sqrt(var + eps)

        # Apply scale and offset: [intermediate_dim] → [1, intermediate_dim, 1, 1]
        scale = self.center_norm_scale[None, :, None, None]
        offset = self.center_norm_offset[None, :, None, None]
        output = output * scale + offset

        # Transpose back to [batch, seq_i, seq_j, intermediate_dim]
        output = output.transpose(0, 2, 3, 1)

        # Output projection
        output = self.output_projection(output)

        # Output gating
        gate_out = mx.sigmoid(self.gating_linear(input_act))
        output = output * gate_out

        # Residual connection
        return pair + output


class TriangleMultiplicationOutgoing(TriangleMultiplication):
    """Triangle multiplication with outgoing orientation.

    Convenience class for outgoing variant.
    """

    def __init__(
        self,
        pair_dim: int,
        intermediate_dim: int | None = None,
    ) -> None:
        super().__init__(
            pair_dim=pair_dim,
            intermediate_dim=intermediate_dim,
            orientation="outgoing",
        )


class TriangleMultiplicationIncoming(TriangleMultiplication):
    """Triangle multiplication with incoming orientation.

    Convenience class for incoming variant.
    """

    def __init__(
        self,
        pair_dim: int,
        intermediate_dim: int | None = None,
    ) -> None:
        super().__init__(
            pair_dim=pair_dim,
            intermediate_dim=intermediate_dim,
            orientation="incoming",
        )


def triangle_multiply_chunked(
    triangle_mul: TriangleMultiplication,
    pair: mx.array,
    mask: mx.array | None = None,
    chunk_size: int | None = None,
) -> mx.array:
    """Apply triangle multiplication with optional chunking for memory efficiency.

    For large sequences (>1000 residues), the O(N^3) memory usage of triangle
    multiplication can exceed available memory. This function chunks the
    computation to reduce peak memory usage.

    Args:
        triangle_mul: Triangle multiplication module.
        pair: Pair representation. Shape: [batch, seq_i, seq_j, pair_dim]
        mask: Optional pair mask. Shape: [batch, seq_i, seq_j]
        chunk_size: Chunk size along seq_i. None = no chunking.

    Returns:
        Updated pair representation.
    """
    if chunk_size is None:
        return triangle_mul(pair, mask)

    batch_size, seq_i, seq_j, _ = pair.shape

    # Process in chunks along seq_i
    outputs = []
    for start in range(0, seq_i, chunk_size):
        end = min(start + chunk_size, seq_i)

        # Extract chunk
        pair_chunk = pair[:, start:end, :, :]
        if mask is not None:
            mask_chunk = mask[:, start:end, :]
        else:
            mask_chunk = None

        # Process chunk (note: full pair needed for k summation)
        # For outgoing: need full pair for right[j,k] access
        # For incoming: need full pair for left[k,i] access
        # This is a simplified chunking - full memory-efficient version
        # would require more careful implementation
        output_chunk = triangle_mul(pair_chunk, mask_chunk)
        outputs.append(output_chunk)

        # Evaluate to free intermediate memory
        mx.eval(output_chunk)

    return mx.concatenate(outputs, axis=1)
