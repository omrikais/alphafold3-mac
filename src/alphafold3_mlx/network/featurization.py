"""Featurization utilities for JAX AF3 parity in MLX."""

from __future__ import annotations

import mlx.core as mx

from alphafold3_mlx.feat_batch import TokenFeatures


def _one_hot(x: mx.array, num_classes: int) -> mx.array:
    """One-hot encode integers along a new last axis."""
    return (x[..., None] == mx.arange(num_classes)).astype(mx.float32)


def create_relative_encoding(
    seq_features: TokenFeatures,
    max_relative_idx: int,
    max_relative_chain: int,
) -> mx.array:
    """Create relative encoding features matching AF3 JAX implementation."""
    rel_feats = []

    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id[:, None]
    right_asym_id = asym_id[None, :]

    left_residue_index = residue_index[:, None]
    right_residue_index = residue_index[None, :]

    left_token_index = token_index[:, None]
    right_token_index = token_index[None, :]

    left_entity_id = entity_id[:, None]
    right_entity_id = entity_id[None, :]

    left_sym_id = sym_id[:, None]
    right_sym_id = sym_id[None, :]

    # Relative residue index
    offset = left_residue_index - right_residue_index
    clipped_offset = mx.clip(
        offset + max_relative_idx, 0, 2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = mx.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * mx.ones_like(clipped_offset),
    )
    rel_pos = _one_hot(final_offset, 2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Relative token index within residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = mx.clip(
        token_offset + max_relative_idx, 0, 2 * max_relative_idx
    )
    residue_same = (left_asym_id == right_asym_id) & (
        left_residue_index == right_residue_index
    )
    final_token_offset = mx.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * mx.ones_like(clipped_token_offset),
    )
    rel_token = _one_hot(final_token_offset, 2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Same entity id
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.astype(mx.float32)[..., None])

    # Relative chain ID inside symmetry class
    rel_sym_id = left_sym_id - right_sym_id
    clipped_rel_chain = mx.clip(
        rel_sym_id + max_relative_chain, 0, 2 * max_relative_chain
    )
    final_rel_chain = mx.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_relative_chain + 1) * mx.ones_like(clipped_rel_chain),
    )
    rel_chain = _one_hot(final_rel_chain, 2 * max_relative_chain + 2)
    rel_feats.append(rel_chain)

    return mx.concatenate(rel_feats, axis=-1)

