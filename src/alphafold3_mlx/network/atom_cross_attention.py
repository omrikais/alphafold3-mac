"""Atom cross-attention modules matching AF3 JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.atom_layout import convert as layout_convert, GatherInfo
from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.diffusion_transformer import (
    CrossAttTransformer,
    CrossAttTransformerConfig,
)

if TYPE_CHECKING:
    from alphafold3_mlx.core.config import DiffusionConfig, GlobalConfig
    from alphafold3_mlx.feat_batch import Batch


def _mask_mean(mask: mx.array, value: mx.array, axis, keepdims, eps=1e-6) -> mx.array:
    numerator = mx.sum(mask * value, axis=axis, keepdims=keepdims)
    denom = mx.sum(mask, axis=axis, keepdims=keepdims)
    return numerator / (denom + eps)


@dataclass(frozen=True)
class AtomCrossAttEncoderOutput:
    token_act: mx.array
    skip_connection: mx.array
    queries_mask: mx.array
    queries_single_cond: mx.array
    keys_mask: mx.array
    keys_single_cond: mx.array
    pair_cond: mx.array


class AtomCrossAttEncoder(nn.Module):
    """Atom cross-attention encoder matching AF3 JAX."""

    def __init__(self, config: "DiffusionConfig", global_config: "GlobalConfig", name: str):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        c = config

        # Per-atom conditioning projections
        self.embed_ref_pos = Linear(
            c.per_atom_channels, input_dims=3, use_bias=False, precision="highest"
        )
        self.embed_ref_mask = Linear(
            c.per_atom_channels, input_dims=1, use_bias=False
        )
        self.embed_ref_element = Linear(
            c.per_atom_channels, input_dims=128, use_bias=False
        )
        self.embed_ref_charge = Linear(
            c.per_atom_channels, input_dims=1, use_bias=False
        )
        self.embed_ref_atom_name = Linear(
            c.per_atom_channels, input_dims=64 * 4, use_bias=False
        )

        # Single to pair cond
        self.single_to_pair_cond_row = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_channels, use_bias=False
        )
        self.single_to_pair_cond_col = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_channels, use_bias=False
        )
        # Second instance (Haiku naming will append _1)
        self.single_to_pair_cond_row_1 = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_channels, use_bias=False
        )
        self.single_to_pair_cond_col_1 = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_channels, use_bias=False
        )

        # Trunk conditioning
        self.lnorm_trunk_single_cond = LayerNorm(
            c.conditioning_seq_channel, create_offset=False
        )
        self.embed_trunk_single_cond = Linear(
            c.per_atom_channels,
            input_dims=c.conditioning_seq_channel,
            use_bias=False,
            initializer=global_config.final_init,
            precision="highest",
        )

        self.lnorm_trunk_pair_cond = LayerNorm(
            c.conditioning_pair_channel, create_offset=False
        )
        self.embed_trunk_pair_cond = Linear(
            c.per_atom_pair_channels,
            input_dims=c.conditioning_pair_channel,
            use_bias=False,
            initializer=global_config.final_init,
            precision="highest",
        )

        # Pair offsets
        self.embed_pair_offsets = Linear(
            c.per_atom_pair_channels, input_dims=3, use_bias=False, precision="highest"
        )
        self.embed_pair_distances = Linear(
            c.per_atom_pair_channels, input_dims=1, use_bias=False
        )
        # Second instance (Haiku naming will append _1)
        self.embed_pair_offsets_1 = Linear(
            c.per_atom_pair_channels, input_dims=3, use_bias=False, precision="highest"
        )
        self.embed_pair_distances_1 = Linear(
            c.per_atom_pair_channels, input_dims=1, use_bias=False
        )
        self.embed_pair_offsets_valid = Linear(
            c.per_atom_pair_channels, input_dims=1, use_bias=False
        )

        # Pair MLP
        self.pair_mlp_1 = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_pair_channels, use_bias=False, initializer="relu"
        )
        self.pair_mlp_2 = Linear(
            c.per_atom_pair_channels, input_dims=c.per_atom_pair_channels, use_bias=False, initializer="relu"
        )
        self.pair_mlp_3 = Linear(
            c.per_atom_pair_channels,
            input_dims=c.per_atom_pair_channels,
            use_bias=False,
            initializer=global_config.final_init,
        )

        # Atom positions to features
        self.atom_positions_to_features = Linear(
            c.per_atom_channels, input_dims=3, use_bias=False, precision="highest"
        )

        # Project for aggregation
        self.project_atom_features_for_aggr = Linear(
            c.per_token_channels, input_dims=c.per_atom_channels, use_bias=False
        )

        # Cross-att transformer
        att_cfg = CrossAttTransformerConfig(
            num_intermediate_factor=c.atom_transformer_num_intermediate_factor,
            num_blocks=c.atom_transformer_num_blocks,
        )
        att_cfg.attention.num_head = c.atom_transformer_num_head
        att_cfg.attention.key_dim = c.atom_transformer_key_dim
        att_cfg.attention.value_dim = c.atom_transformer_value_dim
        self.atom_transformer_encoder = CrossAttTransformer(att_cfg, global_config, name=f"{name}_atom_transformer_encoder")

    def _per_atom_conditioning(self, batch: "Batch") -> tuple[mx.array, mx.array]:
        c = self.config
        # Single conditioning from ref structure
        act = self.embed_ref_pos(batch.ref_structure.positions)
        act = act + self.embed_ref_mask(batch.ref_structure.mask[..., None])
        act = act + self.embed_ref_element(
            _one_hot(batch.ref_structure.element, 128)
        )
        act = act + self.embed_ref_charge(mx.arcsinh(batch.ref_structure.charge)[..., None])
        atom_name_1hot = _one_hot(batch.ref_structure.atom_name_chars, 64)
        num_token, num_dense, _, _ = atom_name_1hot.shape
        act = act + self.embed_ref_atom_name(
            atom_name_1hot.reshape(num_token, num_dense, -1)
        )
        act = act * batch.ref_structure.mask[..., None]

        # Pair conditioning from single
        row_act = self.single_to_pair_cond_row(mx.maximum(act, 0))
        col_act = self.single_to_pair_cond_col(mx.maximum(act, 0))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
        # Note: pair_act from per-atom conditioning is unused downstream,
        # but its parameters must still load for parity.
        return act, pair_act

    def __call__(
        self,
        token_atoms_act: mx.array | None,
        trunk_single_cond: mx.array | None,
        trunk_pair_cond: mx.array | None,
        batch: "Batch",
    ) -> AtomCrossAttEncoderOutput:
        c = self.config

        token_atoms_single_cond, pair_act = self._per_atom_conditioning(batch)
        token_atoms_mask = batch.predicted_structure_info.atom_mask

        queries_single_cond = layout_convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )
        queries_mask = layout_convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )

        # Add trunk single conditioning
        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(
                self.lnorm_trunk_single_cond(trunk_single_cond)
            )
            queries_single_cond = queries_single_cond + layout_convert(
                batch.atom_cross_att.tokens_to_queries,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        if token_atoms_act is None:
            queries_act = queries_single_cond
        else:
            queries_act = layout_convert(
                batch.atom_cross_att.token_atoms_to_queries,
                token_atoms_act,
                layout_axes=(-3, -2),
            )
            queries_act = self.atom_positions_to_features(queries_act)
            queries_act = queries_act * queries_mask[..., None]
            queries_act = queries_act + queries_single_cond

        keys_single_cond = layout_convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        keys_mask = layout_convert(
            batch.atom_cross_att.queries_to_keys,
            queries_mask,
            layout_axes=(-2, -1),
        )

        # Embed single features into pair conditioning
        row_act = self.single_to_pair_cond_row_1(mx.maximum(queries_single_cond, 0))
        pair_cond_keys_input = layout_convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        col_act = self.single_to_pair_cond_col_1(mx.maximum(pair_cond_keys_input, 0))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        if trunk_pair_cond is not None:
            trunk_pair_cond = self.embed_trunk_pair_cond(
                self.lnorm_trunk_pair_cond(trunk_pair_cond)
            )

            num_tokens = trunk_pair_cond.shape[0]
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys
            trunk_pair_to_atom_pair = GatherInfo(
                gather_idxs=(
                    num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                    + tokens_to_keys.gather_idxs[:, None, :]
                ),
                gather_mask=(
                    tokens_to_queries.gather_mask[:, :, None]
                    & tokens_to_keys.gather_mask[:, None, :]
                ),
                input_shape=mx.array((num_tokens, num_tokens)),
            )
            pair_act = pair_act + layout_convert(
                trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
            )

        # Pairwise offsets
        queries_ref_pos = layout_convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = layout_convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = layout_convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = layout_convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_space_uid,
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
        pair_act = pair_act + self.embed_pair_offsets_1(offsets) * offsets_valid[..., None]

        sq_dists = mx.sum(offsets ** 2, axis=-1)
        pair_act = pair_act + self.embed_pair_distances_1(
            1.0 / (1 + sq_dists[..., None])
        ) * offsets_valid[..., None]

        pair_act = pair_act + self.embed_pair_offsets_valid(
            offsets_valid[..., None].astype(mx.float32)
        )

        # Pair MLP
        pair_act2 = self.pair_mlp_1(mx.maximum(pair_act, 0))
        pair_act2 = self.pair_mlp_2(mx.maximum(pair_act2, 0))
        pair_act = pair_act + self.pair_mlp_3(mx.maximum(pair_act2, 0))

        # Cross-attention transformer
        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )
        queries_act = queries_act * queries_mask[..., None]
        skip_connection = queries_act

        # Aggregate to token act
        queries_act = self.project_atom_features_for_aggr(queries_act)
        token_atoms_act = layout_convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_act,
            layout_axes=(-3, -2),
        )
        token_act = _mask_mean(
            token_atoms_mask[..., None],
            mx.maximum(token_atoms_act, 0),
            axis=-2,
            keepdims=False,
            eps=1e-6,
        )

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip_connection,
            queries_mask=queries_mask,
            queries_single_cond=queries_single_cond,
            keys_mask=keys_mask,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )


class AtomCrossAttDecoder(nn.Module):
    """Atom cross-attention decoder matching AF3 JAX."""

    def __init__(self, config: "DiffusionConfig", global_config: "GlobalConfig", name: str):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.name = name

        c = config
        self.project_token_features_for_broadcast = Linear(
            c.per_atom_channels, input_dims=c.per_token_channels, use_bias=False
        )
        self.atom_features_layer_norm = LayerNorm(
            c.per_atom_channels, create_offset=False
        )
        self.atom_features_to_position_update = Linear(
            3,
            input_dims=c.per_atom_channels,
            use_bias=False,
            initializer=global_config.final_init,
            precision="highest",
        )

        att_cfg = CrossAttTransformerConfig(
            num_intermediate_factor=c.atom_transformer_num_intermediate_factor,
            num_blocks=c.atom_transformer_num_blocks,
        )
        att_cfg.attention.num_head = c.atom_transformer_num_head
        att_cfg.attention.key_dim = c.atom_transformer_key_dim
        att_cfg.attention.value_dim = c.atom_transformer_value_dim
        self.atom_transformer_decoder = CrossAttTransformer(att_cfg, global_config, name=f"{name}_atom_transformer_decoder")

    def __call__(
        self,
        token_act: mx.array,
        enc: AtomCrossAttEncoderOutput,
        batch: "Batch",
    ) -> mx.array:
        c = self.config

        token_act = self.project_token_features_for_broadcast(token_act)
        num_token, max_atoms_per_token = batch.atom_cross_att.queries_to_token_atoms.shape
        token_atom_act = mx.broadcast_to(
            token_act[:, None, :],
            (num_token, max_atoms_per_token, c.per_atom_channels),
        )
        queries_act = layout_convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )
        queries_act = queries_act + enc.skip_connection
        queries_act = queries_act * enc.queries_mask[..., None]

        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            pair_cond=enc.pair_cond,
        )
        queries_act = queries_act * enc.queries_mask[..., None]
        queries_act = self.atom_features_layer_norm(queries_act)
        queries_position_update = self.atom_features_to_position_update(queries_act)
        position_update = layout_convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update


def _one_hot(x: mx.array, num_classes: int) -> mx.array:
    return (x[..., None] == mx.arange(num_classes)).astype(mx.float32)
