"""Evoformer stack for AlphaFold 3 MLX.

This module implements the Evoformer, which is the main trunk of
the AlphaFold 3 model. It processes input features through:
1. Initial embeddings (relative position, bond features, templates)
2. 48-layer PairFormer stack
3. Optional MSA stack processing
4. Output casting to float32 for downstream stages

The Evoformer produces single and pair representations that condition
the diffusion head for coordinate generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.pairformer import PairFormerIteration
from alphafold3_mlx.network.msa_attention import (
    MSARowAttention,
    MSATransition,
)

if TYPE_CHECKING:
    from alphafold3_mlx.core.config import EvoformerConfig, GlobalConfig


# ---------------------------------------------------------------------------
# Template helper functions (JAX AF3 parity)
# ---------------------------------------------------------------------------


def _pseudo_beta_fn(
    aatype: mx.array,
    dense_atom_positions: mx.array,
    dense_atom_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compute pseudo-beta positions and mask in MLX.

    CB atom (index 3) for non-glycine, CA atom (index 1) for glycine.
    Uses RESTYPE_PSEUDOBETA_INDEX from the original AF3 codebase.

    Args:
        aatype: [num_res] amino acid types.
        dense_atom_positions: [num_res, num_atoms, 3] atom positions.
        dense_atom_mask: [num_res, num_atoms] atom mask.

    Returns:
        Tuple of (pseudo_beta_positions [num_res, 3], pseudo_beta_mask [num_res]).
    """
    from alphafold3.model import protein_data_processing

    # RESTYPE_PSEUDOBETA_INDEX: [31] -> atom index for each residue type
    pb_index_table = mx.array(protein_data_processing.RESTYPE_PSEUDOBETA_INDEX)
    pb_index = pb_index_table[aatype.astype(mx.int32)]  # [num_res]

    # Gather positions: use advanced indexing
    num_res = aatype.shape[0]
    res_idx = mx.arange(num_res)
    pseudo_beta = dense_atom_positions[res_idx, pb_index.astype(mx.int32)]  # [num_res, 3]
    pseudo_beta_mask = dense_atom_mask[res_idx, pb_index.astype(mx.int32)]  # [num_res]

    return pseudo_beta, pseudo_beta_mask.astype(mx.float32)


def _dgram_from_positions(
    positions: mx.array,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> mx.array:
    """Compute distogram from positions using squared distances.

    Args:
        positions: [num_res, 3] position coordinates.
        min_bin: Left edge of first bin.
        max_bin: Left edge of final bin.
        num_bins: Number of bins.

    Returns:
        Distogram [num_res, num_res, num_bins].
    """
    lower_breaks = mx.linspace(min_bin, max_bin, num_bins)
    lower_breaks = lower_breaks ** 2  # Squared distances
    upper_breaks = mx.concatenate([lower_breaks[1:], mx.array([1e8])])

    # Squared pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 3]
    dist2 = mx.sum(diff ** 2, axis=-1, keepdims=True)  # [N, N, 1]

    dgram = (dist2 > lower_breaks).astype(mx.float32) * (
        dist2 < upper_breaks
    ).astype(mx.float32)

    return dgram


def _make_backbone_rigid(
    positions,  # Vec3Array [num_res, num_atoms]
    mask: mx.array,  # [num_res, num_atoms]
    group_indices: mx.array,  # [num_res, 8, 3]
):
    """Make backbone Rigid3Array and mask.

    Args:
        positions: Vec3Array of atom positions [num_res, num_atoms].
        mask: [num_res, num_atoms] atom mask.
        group_indices: [num_res, 8, 3] atom indices forming rigid groups.

    Returns:
        Tuple of (Rigid namedtuple with rotation and translation, rigid_mask [num_res]).
    """
    from alphafold3_mlx.geometry.vector import Vec3Array
    from alphafold3_mlx.geometry.rotation_matrix import Rot3Array

    # Backbone frame is group 0: indices [C, CA, N]
    backbone_indices = group_indices[:, 0]  # [num_res, 3]

    c_idx = backbone_indices[:, 0].astype(mx.int32)   # C
    b_idx = backbone_indices[:, 1].astype(mx.int32)   # CA
    a_idx = backbone_indices[:, 2].astype(mx.int32)   # N

    num_res = mask.shape[0]
    res_range = mx.arange(num_res)

    # Gather masks for the 3 backbone atoms
    mask_a = mask[res_range, a_idx]
    mask_b = mask[res_range, b_idx]
    mask_c = mask[res_range, c_idx]
    rigid_mask = (mask_a * mask_b * mask_c).astype(mx.float32)

    # Gather positions for the 3 backbone atoms
    def _gather_vec3(positions_v3, indices):
        """Gather Vec3Array elements along atom axis."""
        x = positions_v3.x[res_range, indices]
        y = positions_v3.y[res_range, indices]
        z = positions_v3.z[res_range, indices]
        return Vec3Array(x=x, y=y, z=z)

    pos_a = _gather_vec3(positions, a_idx)  # N
    pos_b = _gather_vec3(positions, b_idx)  # CA
    pos_c = _gather_vec3(positions, c_idx)  # C

    # Build frame: from_two_vectors(C-CA, N-CA) with translation at CA
    rotation = Rot3Array.from_two_vectors(pos_c - pos_b, pos_a - pos_b)

    class _Rigid:
        """Minimal rigid body container."""
        def __init__(self, rotation, translation):
            self.rotation = rotation
            self.translation = translation

    rigid = _Rigid(rotation, pos_b)
    return rigid, rigid_mask


class RelativePositionEmbedding(nn.Module):
    """Relative position embedding for pair representation.

    Encodes the relative position between residues i and j, clipped to
    [-max_relative_idx, max_relative_idx].
    """

    def __init__(
        self,
        pair_channel: int,
        max_relative_idx: int = 32,
        max_relative_chain: int = 2,
    ) -> None:
        """Initialize relative position embedding.

        Args:
            pair_channel: Pair representation channel dimension.
            max_relative_idx: Maximum relative position index.
            max_relative_chain: Maximum relative chain index.
        """
        super().__init__()

        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain

        # Relative position bins: 2 * max_idx + 1 (for clipped values)
        num_pos_bins = 2 * max_relative_idx + 1
        self.pos_embedding = nn.Embedding(num_pos_bins, pair_channel)

        # Relative chain embedding (same chain vs different chain)
        num_chain_bins = 2 * max_relative_chain + 1
        self.chain_embedding = nn.Embedding(num_chain_bins, pair_channel)

    def __call__(
        self,
        residue_index: mx.array,
        asym_id: mx.array,
    ) -> mx.array:
        """Compute relative position embedding.

        Args:
            residue_index: Residue indices. Shape: [batch, seq]
            asym_id: Chain IDs. Shape: [batch, seq]

        Returns:
            Relative position embedding. Shape: [batch, seq, seq, pair_channel]
        """
        # Compute relative positions: pos[i] - pos[j]
        rel_pos = residue_index[:, :, None] - residue_index[:, None, :]
        # Clip to [-max_idx, max_idx]
        rel_pos = mx.clip(rel_pos, -self.max_relative_idx, self.max_relative_idx)
        # Shift to [0, 2*max_idx]
        rel_pos = rel_pos + self.max_relative_idx

        # Compute relative chain: 0 if same chain, 1 if different
        same_chain = (asym_id[:, :, None] == asym_id[:, None, :]).astype(mx.int32)
        # For now, simple same/different encoding
        rel_chain = 1 - same_chain + self.max_relative_chain

        # Lookup embeddings
        pos_emb = self.pos_embedding(rel_pos.astype(mx.int32))
        chain_emb = self.chain_embedding(rel_chain.astype(mx.int32))

        return pos_emb + chain_emb


class EvoformerIteration(nn.Module):
    """Single Evoformer iteration with MSA processing.

    Processes MSA and pair representations through:
    1. Outer product mean (MSA -> pair)
    2. MSA row attention (with pair bias)
    3. MSA transition
    4. Triangle multiplication (outgoing and incoming)
    5. Pair row/column attention
    6. Pair transition

    This is used when MSA features are available (conditional execution).
    """

    def __init__(
        self,
        msa_channel: int,
        seq_channel: int,
        pair_channel: int,
        msa_attention_heads: int = 8,
        pair_attention_heads: int = 4,
    ) -> None:
        """Initialize Evoformer iteration.

        Args:
            msa_channel: MSA representation dimension.
            seq_channel: Single representation dimension.
            pair_channel: Pair representation dimension.
            msa_attention_heads: Number of heads in MSA attention.
            pair_attention_heads: Number of heads in pair attention.
        """
        super().__init__()

        self.msa_channel = msa_channel
        self.seq_channel = seq_channel
        self.pair_channel = pair_channel

        # MSA processing
        self.msa_row_attention = MSARowAttention(
            msa_channel=msa_channel,
            pair_channel=pair_channel,
            num_heads=msa_attention_heads,
        )
        self.msa_transition = MSATransition(msa_channel=msa_channel)

        # Outer product mean (MSA -> pair)
        from alphafold3_mlx.network.outer_product import OuterProductMeanMSA

        self.outer_product_mean = OuterProductMeanMSA(
            msa_channel=msa_channel,
            pair_channel=pair_channel,
        )

        # Triangle operations
        from alphafold3_mlx.network.triangle_ops import (
            TriangleMultiplicationOutgoing,
            TriangleMultiplicationIncoming,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pair_dim=pair_channel,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pair_dim=pair_channel,
        )

        # Pair attention
        from alphafold3_mlx.network.attention import GridSelfAttention

        self.pair_row_attention = GridSelfAttention(
            input_dim=pair_channel,
            num_heads=pair_attention_heads,
            orientation="row",
        )
        self.pair_col_attention = GridSelfAttention(
            input_dim=pair_channel,
            num_heads=pair_attention_heads,
            orientation="column",
        )

        # Pair transition
        from alphafold3_mlx.network.transition import TransitionBlock

        self.pair_transition = TransitionBlock(
            input_dim=pair_channel,
        )

    def __call__(
        self,
        msa: mx.array,
        pair: mx.array,
        msa_mask: mx.array,
        pair_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply Evoformer iteration.

        Args:
            msa: MSA representation [batch, num_seqs, seq_len, msa_channel].
            pair: Pair representation [batch, seq_len, seq_len, pair_channel].
            msa_mask: MSA mask [batch, num_seqs, seq_len].
            pair_mask: Optional pair mask [batch, seq_len, seq_len].

        Returns:
            Tuple of (updated_msa, updated_pair).
        """
        # Outer product mean (MSA -> pair) - must run FIRST so pair is
        # updated before MSA row attention uses it as bias (JAX AF3 parity).
        pair = self.outer_product_mean(msa, pair, msa_mask)

        # MSA row attention (with pair bias)
        msa = self.msa_row_attention(msa, pair, msa_mask)

        # MSA transition
        msa = self.msa_transition(msa)

        # Triangle multiplication
        pair = self.tri_mul_out(pair, pair_mask)
        pair = self.tri_mul_in(pair, pair_mask)

        # Pair attention (GridSelfAttention has no internal residual)
        pair = pair + self.pair_row_attention(pair, mask=pair_mask)
        pair = pair + self.pair_col_attention(pair, mask=pair_mask)

        # Pair transition
        pair = self.pair_transition(pair)

        return msa, pair


class BroadcastProjection(nn.Module):
    """Linear projection with num_input_dims=0 (broadcast multiplication).

    Matches JAX AF3 hm.Linear with num_input_dims=0: the weight has shape
    (out_dim,) and the operation is x[..., None] * weight, broadcasting the
    scalar input across the output dimension.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.weight = mx.zeros((out_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x[..., None] * self.weight


class SingleTemplateEmbedding(nn.Module):
    """Embed a single template (JAX AF3 parity).

    Computes 9 input features per template:
      0: distogram (39-bin squared distance) from pseudo-beta positions
      1: pseudo_beta_mask_2d (BroadcastProjection)
      2: aatype one-hot, row-broadcast
      3: aatype one-hot, col-broadcast
      4-6: unit_vector x/y/z (BroadcastProjection)
      7: backbone_mask_2d (BroadcastProjection)
      8: query_embedding (normalized)
    Then runs a 2-layer PairFormer stack followed by output LayerNorm.
    """

    def __init__(
        self,
        num_channels: int = 64,
        pair_channel: int = 128,
        num_pairformer_layers: int = 2,
        dgram_min_bin: float = 3.25,
        dgram_max_bin: float = 50.75,
        dgram_num_bins: int = 39,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.pair_channel = pair_channel
        self.dgram_min_bin = dgram_min_bin
        self.dgram_max_bin = dgram_max_bin
        self.dgram_num_bins = dgram_num_bins

        # 9 template pair embeddings matching JAX naming
        # 0: distogram -> Linear(39, 64)
        self.template_pair_embedding_0 = Linear(
            num_channels, input_dims=dgram_num_bins, use_bias=False,
            initializer='relu',
        )
        # 1: pseudo_beta_mask_2d -> BroadcastProjection(64)
        self.template_pair_embedding_1 = BroadcastProjection(num_channels)
        # 2: aatype one-hot row -> Linear(31, 64)
        from alphafold3.constants import residue_names
        num_aatype = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP  # 31
        self.template_pair_embedding_2 = Linear(
            num_channels, input_dims=num_aatype, use_bias=False,
            initializer='relu',
        )
        # 3: aatype one-hot col -> Linear(31, 64)
        self.template_pair_embedding_3 = Linear(
            num_channels, input_dims=num_aatype, use_bias=False,
            initializer='relu',
        )
        # 4-6: unit_vector x, y, z -> BroadcastProjection(64)
        self.template_pair_embedding_4 = BroadcastProjection(num_channels)
        self.template_pair_embedding_5 = BroadcastProjection(num_channels)
        self.template_pair_embedding_6 = BroadcastProjection(num_channels)
        # 7: backbone_mask_2d -> BroadcastProjection(64)
        self.template_pair_embedding_7 = BroadcastProjection(num_channels)
        # 8: query_embedding (normalized) -> Linear(pair_channel, 64)
        self.template_pair_embedding_8 = Linear(
            num_channels, input_dims=pair_channel, use_bias=False,
            initializer='relu',
        )

        # LayerNorm on query embedding before embedding_8
        self.query_embedding_norm = LayerNorm(pair_channel)

        # 2-layer PairFormer stack (with_single=False, intermediate_factor=2)
        self.pairformer_layers = [
            PairFormerIteration(
                seq_channel=1,  # unused when with_single=False
                pair_channel=num_channels,
                num_attention_heads=4,
                intermediate_factor=2,
                with_single=False,
            )
            for _ in range(num_pairformer_layers)
        ]

        # Output layer norm
        self.output_layer_norm = LayerNorm(num_channels)

    def __call__(
        self,
        query_embedding: mx.array,
        template_aatype: mx.array,
        template_atom_positions: mx.array,
        template_atom_mask: mx.array,
        padding_mask_2d: mx.array,
        multichain_mask_2d: mx.array,
    ) -> mx.array:
        """Embed a single template.

        Args:
            query_embedding: [num_res, num_res, pair_channel]
            template_aatype: [num_res] amino acid types
            template_atom_positions: [num_res, num_atoms, 3]
            template_atom_mask: [num_res, num_atoms]
            padding_mask_2d: [num_res, num_res]
            multichain_mask_2d: [num_res, num_res]

        Returns:
            Template embedding [num_res, num_res, num_channels]
        """
        from alphafold3.constants import residue_names
        from alphafold3.model import protein_data_processing
        from alphafold3_mlx.geometry.vector import Vec3Array
        from alphafold3_mlx.geometry.rotation_matrix import Rot3Array

        dtype = query_embedding.dtype
        num_res = query_embedding.shape[0]

        # Zero out masked atom positions
        dense_atom_positions = template_atom_positions * template_atom_mask[..., None]

        # --- Pseudo-beta positions and mask ---
        pseudo_beta_positions, pseudo_beta_mask = _pseudo_beta_fn(
            template_aatype, dense_atom_positions, template_atom_mask,
        )
        pseudo_beta_mask_2d = pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d

        # --- Distogram (39-bin squared distance) ---
        dgram = _dgram_from_positions(
            pseudo_beta_positions,
            min_bin=self.dgram_min_bin,
            max_bin=self.dgram_max_bin,
            num_bins=self.dgram_num_bins,
        )
        dgram = dgram * pseudo_beta_mask_2d[..., None]
        dgram = dgram.astype(dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)

        # --- Aatype one-hot ---
        num_aatype = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        classes = mx.arange(num_aatype)
        aatype_one_hot = (
            template_aatype[..., None] == classes
        ).astype(dtype)

        # --- Backbone rigid frames and unit vectors ---
        group_indices = mx.array(
            protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX
        )[template_aatype.astype(mx.int32)]  # [num_res, 8, 3]

        rigid, backbone_mask = _make_backbone_rigid(
            Vec3Array.from_array(dense_atom_positions),
            template_atom_mask,
            group_indices,
        )

        # Compute unit vectors: rigid[:, None].inverse().apply_to_point(points)
        points = rigid.translation  # Vec3Array shape [num_res]

        # Expand rigid for broadcasting: [num_res, 1] vs points [1, num_res]
        rigid_expanded = Rot3Array(
            xx=rigid.rotation.xx[:, None], xy=rigid.rotation.xy[:, None], xz=rigid.rotation.xz[:, None],
            yx=rigid.rotation.yx[:, None], yy=rigid.rotation.yy[:, None], yz=rigid.rotation.yz[:, None],
            zx=rigid.rotation.zx[:, None], zy=rigid.rotation.zy[:, None], zz=rigid.rotation.zz[:, None],
        )
        trans_expanded = Vec3Array(
            x=rigid.translation.x[:, None],
            y=rigid.translation.y[:, None],
            z=rigid.translation.z[:, None],
        )
        points_expanded = Vec3Array(
            x=points.x[None, :],
            y=points.y[None, :],
            z=points.z[None, :],
        )

        # inverse().apply_to_point = R^T @ (p - t)
        diff = points_expanded - trans_expanded
        rigid_vec = rigid_expanded.inverse().apply_to_point(diff)
        unit_vector = rigid_vec.normalized()

        unit_vector_x = unit_vector.x.astype(dtype)
        unit_vector_y = unit_vector.y.astype(dtype)
        unit_vector_z = unit_vector.z.astype(dtype)

        backbone_mask = backbone_mask.astype(dtype)
        backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
        backbone_mask_2d = backbone_mask_2d * multichain_mask_2d

        unit_vector_x = unit_vector_x * backbone_mask_2d
        unit_vector_y = unit_vector_y * backbone_mask_2d
        unit_vector_z = unit_vector_z * backbone_mask_2d

        # --- Normalized query embedding ---
        query_emb_normed = self.query_embedding_norm(query_embedding)

        # --- Sum all 9 embeddings ---
        act = self.template_pair_embedding_0(dgram)
        act = act + self.template_pair_embedding_1(pseudo_beta_mask_2d)
        act = act + self.template_pair_embedding_2(aatype_one_hot[None, :, :])
        act = act + self.template_pair_embedding_3(aatype_one_hot[:, None, :])
        act = act + self.template_pair_embedding_4(unit_vector_x)
        act = act + self.template_pair_embedding_5(unit_vector_y)
        act = act + self.template_pair_embedding_6(unit_vector_z)
        act = act + self.template_pair_embedding_7(backbone_mask_2d)
        act = act + self.template_pair_embedding_8(query_emb_normed)

        # --- 2-layer PairFormer stack (with_single=False) ---
        # PairFormerIteration expects [batch, seq, seq, channels] but template
        # processing is unbatched. Add and remove batch dim.
        act = act[None, ...]  # [1, num_res, num_res, num_channels]
        pair_mask = padding_mask_2d[None, ...]  # [1, num_res, num_res]
        dummy_single = mx.zeros((1, num_res, 1), dtype=dtype)

        for pf_layer in self.pairformer_layers:
            dummy_single, act = pf_layer(dummy_single, act, pair_mask=pair_mask)

        act = act[0]  # Remove batch dim

        # --- Output layer norm ---
        act = self.output_layer_norm(act)

        return act


class TemplateEmbedding(nn.Module):
    """Embed a set of templates (JAX AF3 parity).

    For each template, runs SingleTemplateEmbedding and sums the results.
    Divides by num_templates, applies ReLU, then projects to pair_channel.
    """

    def __init__(
        self,
        pair_channel: int = 128,
        template_pair_channel: int = 64,
        num_template_blocks: int = 2,
        dgram_min_bin: float = 3.25,
        dgram_max_bin: float = 50.75,
        dgram_num_bins: int = 39,
    ) -> None:
        super().__init__()
        self.pair_channel = pair_channel
        self.template_pair_channel = template_pair_channel
        self.enabled = True

        self.single_template_embedding = SingleTemplateEmbedding(
            num_channels=template_pair_channel,
            pair_channel=pair_channel,
            num_pairformer_layers=num_template_blocks,
            dgram_min_bin=dgram_min_bin,
            dgram_max_bin=dgram_max_bin,
            dgram_num_bins=dgram_num_bins,
        )

        # Output projection: template_pair_channel -> pair_channel
        self.output_linear = Linear(
            pair_channel,
            input_dims=template_pair_channel,
            use_bias=False,
            initializer='relu',
        )

    def __call__(
        self,
        query_embedding: mx.array,
        template_aatype: mx.array,
        template_atom_positions: mx.array,
        template_atom_mask: mx.array,
        padding_mask_2d: mx.array,
        multichain_mask_2d: mx.array,
    ) -> mx.array:
        """Embed templates and return additive contribution to pair.

        Args:
            query_embedding: [num_res, num_res, pair_channel] current pair rep
            template_aatype: [num_templates, num_res]
            template_atom_positions: [num_templates, num_res, num_atoms, 3]
            template_atom_mask: [num_templates, num_res, num_atoms]
            padding_mask_2d: [num_res, num_res]
            multichain_mask_2d: [num_res, num_res]

        Returns:
            Template embedding [num_res, num_res, pair_channel] to add to pair.
        """
        if not self.enabled:
            return mx.zeros_like(query_embedding)

        num_templates = template_aatype.shape[0]
        num_res = query_embedding.shape[0]

        # Sum embeddings across templates
        summed = mx.zeros(
            (num_res, num_res, self.template_pair_channel),
            dtype=query_embedding.dtype,
        )

        for t in range(num_templates):
            embedding = self.single_template_embedding(
                query_embedding=query_embedding,
                template_aatype=template_aatype[t],
                template_atom_positions=template_atom_positions[t],
                template_atom_mask=template_atom_mask[t],
                padding_mask_2d=padding_mask_2d,
                multichain_mask_2d=multichain_mask_2d,
            )
            summed = summed + embedding

        # Average, ReLU, project
        embedding = summed / (1e-7 + num_templates)
        embedding = nn.relu(embedding)
        embedding = self.output_linear(embedding)

        return embedding


class Evoformer(nn.Module):
    """Evoformer stack.

    The main trunk of AlphaFold 3, consisting of:
    1. Input embedding (relative positions, bonds, templates)
    2. 48-layer PairFormer stack
    3. Optional MSA processing
    4. Output normalization and float32 casting
    """

    def __init__(
        self,
        config: "EvoformerConfig | None" = None,
        global_config: "GlobalConfig | None" = None,
    ) -> None:
        """Initialize Evoformer.

        Args:
            config: Evoformer configuration. Uses defaults if None.
            global_config: Global model configuration.
        """
        super().__init__()

        # Import here to avoid circular dependency
        from alphafold3_mlx.core.config import EvoformerConfig, GlobalConfig

        self.config = config or EvoformerConfig()
        self.global_config = global_config or GlobalConfig()

        c = self.config
        target_feat_dim = 447
        msa_feat_dim = 34

        # Relative encoding projection (JAX parity path).
        rel_encoding_dim = (
            2 * (2 * c.max_relative_idx + 2) + 1 + (2 * c.max_relative_chain + 2)
        )
        self.position_activations = Linear(
            c.pair_channel,
            input_dims=rel_encoding_dim,
            use_bias=False,
        )

        # Bond feature embedding (optional)
        # JAX AF3 uses contact_matrix[:, :, None] with input_dim=1 (binary scalar)
        self.bond_embedding = Linear(
            c.pair_channel,
            input_dims=1,
            use_bias=False,
        )

        # Template embedding (optional, JAX AF3 parity)
        self.template_embedding = TemplateEmbedding(
            pair_channel=c.pair_channel,
            template_pair_channel=c.template.template_pair_channel,
            num_template_blocks=c.template.num_template_blocks,
            dgram_min_bin=c.template.dgram_min_bin,
            dgram_max_bin=c.template.dgram_max_bin,
            dgram_num_bins=c.template.dgram_num_bins,
        )

        # JAX-parity sequence/pair embedding projections from target features.
        self.left_single_proj = Linear(
            c.pair_channel,
            input_dims=target_feat_dim,
            use_bias=False,
        )
        self.right_single_proj = Linear(
            c.pair_channel,
            input_dims=target_feat_dim,
            use_bias=False,
        )
        self.prev_embedding_layer_norm = LayerNorm(c.pair_channel)
        self.prev_pair_proj = Linear(
            c.pair_channel,
            input_dims=c.pair_channel,
            use_bias=False,
        )

        self.single_input_proj = Linear(
            c.seq_channel,
            input_dims=target_feat_dim,
            use_bias=False,
        )
        self.prev_single_embedding_layer_norm = LayerNorm(c.seq_channel)
        self.prev_single_proj = Linear(
            c.seq_channel,
            input_dims=c.seq_channel,
            use_bias=False,
        )

        # Legacy compatibility projection (used only in legacy path).
        self.pair_input_proj = Linear(
            c.pair_channel,
            input_dims=c.pair_channel,
            use_bias=False,
        )

        # 48-layer PairFormer stack
        # Using Python for-loop (no hk.experimental.layer_stack in MLX)
        self.num_layers = c.num_pairformer_layers
        self.pairformer_layers = [
            PairFormerIteration(
                seq_channel=c.seq_channel,
                pair_channel=c.pair_channel,
                num_attention_heads=c.pairformer.num_attention_heads,
                single_attention_heads=c.pairformer.single_attention_heads,
                attention_key_dim=c.pairformer.attention_key_dim,
            )
            for _ in range(self.num_layers)
        ]

        # Output LayerNorms
        self.single_output_norm = LayerNorm(c.seq_channel)
        self.pair_output_norm = LayerNorm(c.pair_channel)

        # MSA stack (optional)
        self.msa_channel = c.msa_channel
        self.num_msa_layers = c.num_msa_layers
        self.use_msa_stack = c.use_msa_stack

        if self.use_msa_stack and self.num_msa_layers > 0:
            # MSA input projection
            self.msa_input_proj = Linear(
                c.msa_channel,
                input_dims=msa_feat_dim,  # create_msa_feat() = 32 aa + 2 deletion feats
                use_bias=False,
            )
            self.extra_msa_proj = Linear(
                c.msa_channel,
                input_dims=target_feat_dim,
                use_bias=False,
            )

            # MSA Evoformer layers
            self.msa_layers = [
                EvoformerIteration(
                    msa_channel=c.msa_channel,
                    seq_channel=c.seq_channel,
                    pair_channel=c.pair_channel,
                    msa_attention_heads=c.msa_stack.num_attention_heads,
                    pair_attention_heads=c.pairformer.num_attention_heads,
                )
                for _ in range(self.num_msa_layers)
            ]

        else:
            self.msa_layers = []

    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        residue_index: mx.array,
        asym_id: mx.array,
        target_feat: mx.array | None = None,
        token_features: object | None = None,
        seq_mask: mx.array | None = None,
        pair_mask: mx.array | None = None,
        template_aatype: mx.array | None = None,
        template_atom_positions: mx.array | None = None,
        template_atom_mask: mx.array | None = None,
        bond_features: mx.array | None = None,
        msa_features: mx.array | None = None,
        msa_mask: mx.array | None = None,
        return_intermediates: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[str, mx.array]]:
        """Apply Evoformer.

        Args:
            single: Previous recycle single representation in parity mode
                [batch, seq, seq_channel], or initial single activations in legacy mode.
            pair: Previous recycle pair representation in parity mode
                [batch, seq, seq, pair_channel], or initial pair activations in legacy mode.
            target_feat: Fixed target feature embedding [batch, seq, 447] for
                JAX-parity recycling. If None, uses legacy path.
            residue_index: Residue indices. Shape: [batch, seq]
            asym_id: Chain IDs. Shape: [batch, seq]
            token_features: Optional token features for relative encoding parity.
            seq_mask: Optional sequence mask. Shape: [batch, seq]
            pair_mask: Optional pair mask. Shape: [batch, seq, seq]
            template_aatype: Optional template AA types [num_templates, seq].
            template_atom_positions: Optional template positions [num_templates, seq, atoms, 3].
            template_atom_mask: Optional template mask [num_templates, seq, atoms].
            bond_features: Optional bond features. Shape: [batch, seq, seq, 1]
            msa_features: Optional MSA features. Shape: [batch, num_seqs, seq, msa_channel]
            msa_mask: Optional MSA mask. Shape: [batch, num_seqs, seq]
            return_intermediates: If True, return intermediate layer outputs.

        Returns:
            If return_intermediates is False:
                Tuple of (single, pair) representations after Evoformer.
            If return_intermediates is True:
                Tuple of (single, pair, intermediates) where intermediates is a dict
                mapping layer indices to (single, pair) tuples at checkpoint layers.
        """
        legacy_mode = target_feat is None
        if target_feat is not None and target_feat.ndim == 2:
            target_feat = target_feat[None, ...]

        # === Input Embeddings ===

        # Add JAX-parity relative encoding to pair.
        # create_relative_encoding expects 1D token features, so build a fallback
        # from residue/asym ids if token_features were not provided.
        if token_features is None:
            from alphafold3_mlx.feat_batch import TokenFeatures as RelativeTokenFeatures

            seq_len = int(residue_index.shape[-1])
            residue_index_1d = residue_index[0] if residue_index.ndim == 2 else residue_index
            asym_id_1d = asym_id[0] if asym_id.ndim == 2 else asym_id
            token_features = RelativeTokenFeatures(
                token_index=mx.arange(seq_len, dtype=mx.int32),
                residue_index=residue_index_1d,
                asym_id=asym_id_1d,
                entity_id=mx.zeros((seq_len,), dtype=mx.int32),
                sym_id=mx.zeros((seq_len,), dtype=mx.int32),
                mask=mx.ones((seq_len,), dtype=mx.float32),
            )

        from alphafold3_mlx.network.featurization import create_relative_encoding

        rel_features = create_relative_encoding(
            seq_features=token_features,
            max_relative_idx=self.config.max_relative_idx,
            max_relative_chain=self.config.max_relative_chain,
        )
        if legacy_mode:
            # Legacy mode keeps original MLX Evoformer semantics.
            if rel_features.ndim == 3:
                rel_features = rel_features[None, ...]
            pair = pair + self.position_activations(rel_features)
            if single.shape[-1] != self.config.seq_channel:
                single = self.single_input_proj(single)
            if pair.shape[-1] != self.config.pair_channel:
                pair = self.pair_input_proj(pair)
        else:
            assert target_feat is not None
            # JAX parity: rebuild sequence/pair embeddings each recycle from target_feat
            # then inject previous recycle activations via dedicated projections.
            prev_single = single
            prev_pair = pair
            left_single = self.left_single_proj(target_feat)[:, :, None, :]
            right_single = self.right_single_proj(target_feat)[:, None, :, :]
            pair = left_single + right_single
            pair = pair + self.prev_pair_proj(
                self.prev_embedding_layer_norm(prev_pair.astype(left_single.dtype))
            )

            single = self.single_input_proj(target_feat)
            single = single + self.prev_single_proj(
                self.prev_single_embedding_layer_norm(prev_single.astype(single.dtype))
            )

            if rel_features.ndim == 3:
                rel_features = rel_features[None, ...]
            pair = pair + self.position_activations(rel_features)

        # Add bond features if provided
        if bond_features is not None:
            bond_emb = self.bond_embedding(bond_features)
            pair = pair + bond_emb

        # Add template embedding if provided (JAX AF3 parity)
        if (
            template_aatype is not None
            and template_atom_positions is not None
            and template_atom_mask is not None
            and self.template_embedding.enabled
        ):
            # Compute pair_mask for template PairFormer (unbatched)
            if pair_mask is not None:
                padding_mask_2d = pair_mask[0]  # Remove batch dim
            else:
                seq_len = pair.shape[1]
                padding_mask_2d = mx.ones((seq_len, seq_len), dtype=pair.dtype)

            # Construct multichain mask: only intra-chain template features
            asym_id_1d = asym_id[0] if asym_id.ndim == 2 else asym_id
            multichain_mask_2d = (
                asym_id_1d[:, None] == asym_id_1d[None, :]
            ).astype(pair.dtype)

            # Call template embedding on unbatched pair representation
            template_act = self.template_embedding(
                query_embedding=pair[0],  # Remove batch dim
                template_aatype=template_aatype,
                template_atom_positions=template_atom_positions,
                template_atom_mask=template_atom_mask,
                padding_mask_2d=padding_mask_2d,
                multichain_mask_2d=multichain_mask_2d,
            )
            pair = pair + template_act[None, ...]  # Add batch dim back

        # === Conditional MSA Stack ===
        if self.use_msa_stack and msa_features is not None and len(self.msa_layers) > 0:
            # JAX AF3 parity: truncate MSA to num_msa sequences (default 1024).
            # JAX does shuffle_msa + truncate_msa_batch inside _embed_process_msa.
            # We truncate here before projection to match that behaviour.
            num_msa_limit = self.config.num_msa
            num_msa_in = msa_features.shape[-3]
            if num_msa_in > num_msa_limit:
                msa_features = msa_features[:, :num_msa_limit, :, :]
                if msa_mask is not None:
                    msa_mask = msa_mask[:, :num_msa_limit, :]

            if legacy_mode:
                # Legacy path assumes caller already projected features.
                msa = msa_features
            else:
                assert target_feat is not None
                # JAX parity: learned projections for MSA features and target_feat bias.
                msa = self.msa_input_proj(msa_features)
                msa = msa + self.extra_msa_proj(target_feat)[:, None, :, :]

            # Generate MSA mask if not provided
            if msa_mask is None:
                batch_size, num_seqs, seq_len = msa.shape[:3]
                msa_mask = mx.ones((batch_size, num_seqs, seq_len))

            # Run MSA Evoformer layers
            for layer_idx, msa_layer in enumerate(self.msa_layers):
                msa, pair = msa_layer(msa, pair, msa_mask, pair_mask)

                # Periodic evaluation
                if (layer_idx + 1) % 4 == 0:
                    mx.eval(msa, pair)

        # === PairFormer Stack (48 layers) ===
        # Using Python for-loop per research.md Section 4
        # Capture intermediate checkpoints at layers 12, 24, 36 (and final)
        intermediates: dict[str, mx.array] = {} if return_intermediates else None
        checkpoint_layers = {12, 24, 36}  # Capture at 25%, 50%, 75% through stack

        for layer_idx, pairformer in enumerate(self.pairformer_layers):
            single, pair = pairformer(single, pair, seq_mask, pair_mask)

            # Periodic evaluation to prevent graph explosion
            # Every 8 layers (6 evaluations for 48 layers)
            if (layer_idx + 1) % 8 == 0:
                mx.eval(single, pair)

            # Capture intermediate checkpoints
            if return_intermediates and (layer_idx + 1) in checkpoint_layers:
                # MLX arrays don't have .copy(); use slicing to create a copy
                intermediates[f"pairformer_layer_{layer_idx + 1}_single"] = single[...]
                intermediates[f"pairformer_layer_{layer_idx + 1}_pair"] = pair[...]

        # JAX AF3 parity: Evoformer returns raw outputs without LayerNorm or
        # masking.  The float32 cast happens in the model recycling loop, not
        # here (see model.py recycle_body).  Applying LayerNorm here would
        # corrupt the representations fed to diffusion and confidence heads.

        # Capture final outputs in intermediates
        if return_intermediates:
            # MLX arrays don't have .copy(); use slicing to create a copy
            intermediates["pairformer_final_single"] = single[...]
            intermediates["pairformer_final_pair"] = pair[...]
            return single, pair, intermediates

        return single, pair

    def set_template_enabled(self, enabled: bool) -> None:
        """Enable or disable template processing.

        Args:
            enabled: Whether to use template features.
        """
        self.template_embedding.enabled = enabled
