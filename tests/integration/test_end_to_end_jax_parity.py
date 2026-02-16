"""End-to-end JAX AF3 parity test.

Validates MLX end-to-end outputs against real JAX AF3 reference outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from alphafold3.constants import atom_types
from alphafold3.model import confidences as jax_confidences

from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.pairformer import PairFormerIteration
from alphafold3_mlx.network.featurization import create_relative_encoding
from alphafold3_mlx.feat_batch import Batch, TokenFeatures
from alphafold3_mlx.core.config import (
    DiffusionConfig,
    ConfidenceConfig,
    GlobalConfig,
    ModelConfig,
)
from alphafold3_mlx.model import Model


REF_PATH = Path("tests/fixtures/jax_af3_refs/end_to_end_ref.npz")
RMSD_TOL = 0.5  # Angstrom
CONF_REL_TOL = 0.01  # 1% relative error


def _expand_colon_keys(ref_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    for key in ref_data.keys():
        data[key] = ref_data[key]
        if ":" in key:
            data[key.replace(":", "/")] = ref_data[key]
    return data


def _find_module_prefix(ref_data: np.lib.npyio.NpzFile, module_name: str) -> str:
    for key in ref_data.keys():
        if key.endswith(f"{module_name}/weights") or key.endswith(f"{module_name}/w"):
            return key.rsplit("/", 1)[0]
        if key.endswith(f"{module_name}/scale") or key.endswith(f"{module_name}/offset"):
            return key.rsplit("/", 1)[0]
    raise KeyError(f"Could not find params prefix for module '{module_name}'")


def _find_module_root(ref_data: np.lib.npyio.NpzFile, module_name: str) -> str:
    token = f"/{module_name}/"
    for key in ref_data.keys():
        if key.startswith("params/") and token in key:
            base = key.split(token)[0]
            return f"{base}/{module_name}"
    raise KeyError(f"Could not find root prefix for module '{module_name}'")


def _iter_layer_stack_params(
    ref_data: np.lib.npyio.NpzFile, prefix: str, num_layers: int
) -> dict[int, dict[str, mx.array]]:
    layers: dict[int, dict[str, mx.array]] = {i: {} for i in range(num_layers)}
    for key in ref_data.keys():
        if not key.startswith(prefix):
            continue
        rel_path = key[len(prefix):].lstrip("/")
        values = ref_data[key]
        if values.shape[0] != num_layers:
            raise ValueError(
                f"Expected stacked params with leading dim {num_layers} for {key}, "
                f"got shape {values.shape}"
            )
        for layer_idx in range(num_layers):
            layers[layer_idx][rel_path] = mx.array(values[layer_idx])
    return layers


def _infer_num_layers(ref_data: np.lib.npyio.NpzFile, prefix: str) -> int:
    for key in ref_data.keys():
        if key.startswith(prefix):
            value = ref_data[key]
            if value.ndim == 0:
                continue
            return int(value.shape[0])
    raise KeyError(f"Could not infer num layers for prefix '{prefix}'")


def _select_subparams(
    params: dict[str, mx.array], prefix: str
) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in params.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def _npz_get(ref_data: np.lib.npyio.NpzFile, key: str) -> mx.array | None:
    if key in ref_data.keys():
        return mx.array(ref_data[key])
    return None


def _load_linear(linear, ref_data: np.lib.npyio.NpzFile, prefix: str) -> None:
    weight = _npz_get(ref_data, f"{prefix}/weights")
    if weight is None:
        weight = _npz_get(ref_data, f"{prefix}/w")
    if weight is not None:
        linear.weight = weight

    bias = _npz_get(ref_data, f"{prefix}/bias")
    if bias is None:
        bias = _npz_get(ref_data, f"{prefix}/b")
    if bias is not None and hasattr(linear, "bias"):
        linear.bias = bias


def _load_layer_norm(ln, ref_data: np.lib.npyio.NpzFile, prefix: str) -> None:
    scale = _npz_get(ref_data, f"{prefix}/scale")
    if scale is not None:
        ln.scale = scale
    offset = _npz_get(ref_data, f"{prefix}/offset")
    if offset is not None:
        ln.offset = offset


def load_triangle_mult_weights(module, params: dict[str, mx.array]) -> None:
    if "left_norm_input/scale" in params:
        module.input_norm.scale = params["left_norm_input/scale"]
    if "left_norm_input/offset" in params:
        module.input_norm.offset = params["left_norm_input/offset"]
    if "projection/weights" in params:
        module.projection.weight = params["projection/weights"]
    elif "projection/w" in params:
        module.projection.weight = params["projection/w"]
    if "gate/weights" in params:
        module.gate.weight = params["gate/weights"]
    elif "gate/w" in params:
        module.gate.weight = params["gate/w"]
    if "center_norm/scale" in params:
        module.center_norm_scale = params["center_norm/scale"]
    if "center_norm/offset" in params:
        module.center_norm_offset = params["center_norm/offset"]
    if "output_projection/weights" in params:
        module.output_projection.weight = params["output_projection/weights"]
    elif "output_projection/w" in params:
        module.output_projection.weight = params["output_projection/w"]
    if "gating_linear/weights" in params:
        module.gating_linear.weight = params["gating_linear/weights"]
    elif "gating_linear/w" in params:
        module.gating_linear.weight = params["gating_linear/w"]


def load_grid_attention_weights(module, params: dict[str, mx.array]) -> None:
    if "act_norm/scale" in params:
        module.act_norm.scale = params["act_norm/scale"]
    if "act_norm/offset" in params:
        module.act_norm.offset = params["act_norm/offset"]
    if "pair_bias_projection/weights" in params:
        module.pair_bias_proj.weight = params["pair_bias_projection/weights"]
    elif "pair_bias_projection/w" in params:
        module.pair_bias_proj.weight = params["pair_bias_projection/w"]
    if "q_projection/weights" in params:
        module.q_proj.weight = params["q_projection/weights"]
    elif "q_projection/w" in params:
        module.q_proj.weight = params["q_projection/w"]
    if "k_projection/weights" in params:
        module.k_proj.weight = params["k_projection/weights"]
    elif "k_projection/w" in params:
        module.k_proj.weight = params["k_projection/w"]
    if "v_projection/weights" in params:
        module.v_proj.weight = params["v_projection/weights"]
    elif "v_projection/w" in params:
        module.v_proj.weight = params["v_projection/w"]
    if "output_projection/weights" in params:
        module.o_proj.weight = params["output_projection/weights"]
    elif "output_projection/w" in params:
        module.o_proj.weight = params["output_projection/w"]
    if "gating_query/weights" in params:
        module.gate_proj.weight = params["gating_query/weights"]
    elif "gating_query/w" in params:
        module.gate_proj.weight = params["gating_query/w"]
    if "gating_query/bias" in params:
        module.gate_proj.bias = params["gating_query/bias"]
    elif "gating_query/b" in params:
        module.gate_proj.bias = params["gating_query/b"]


def load_single_attention_weights(module, params: dict[str, mx.array]) -> None:
    if "layer_norm/scale" in params:
        module.act_norm.scale = params["layer_norm/scale"]
    if "layer_norm/offset" in params:
        module.act_norm.offset = params["layer_norm/offset"]
    if "q_projection/weights" in params:
        module.q_proj.weight = params["q_projection/weights"]
    elif "q_projection/w" in params:
        module.q_proj.weight = params["q_projection/w"]
    if "q_projection/bias" in params:
        module.q_proj.bias = params["q_projection/bias"]
    elif "q_projection/b" in params:
        module.q_proj.bias = params["q_projection/b"]
    if "k_projection/weights" in params:
        module.k_proj.weight = params["k_projection/weights"]
    elif "k_projection/w" in params:
        module.k_proj.weight = params["k_projection/w"]
    if "v_projection/weights" in params:
        module.v_proj.weight = params["v_projection/weights"]
    elif "v_projection/w" in params:
        module.v_proj.weight = params["v_projection/w"]
    if "transition2/weights" in params:
        module.o_proj.weight = params["transition2/weights"]
    elif "transition2/w" in params:
        module.o_proj.weight = params["transition2/w"]
    if "gating_query/weights" in params:
        module.gate_proj.weight = params["gating_query/weights"]
    elif "gating_query/w" in params:
        module.gate_proj.weight = params["gating_query/w"]
    if "gating_query/bias" in params:
        module.gate_proj.bias = params["gating_query/bias"]
    elif "gating_query/b" in params:
        module.gate_proj.bias = params["gating_query/b"]


def load_transition_weights(module, params: dict[str, mx.array]) -> None:
    if "input_layer_norm/scale" in params:
        module.norm.scale = params["input_layer_norm/scale"]
    if "input_layer_norm/offset" in params:
        module.norm.offset = params["input_layer_norm/offset"]
    if "transition1/weights" in params:
        module.glu.linear.weight = params["transition1/weights"]
    elif "transition1/w" in params:
        module.glu.linear.weight = params["transition1/w"]
    if "transition2/weights" in params:
        module.output_proj.weight = params["transition2/weights"]
    elif "transition2/w" in params:
        module.output_proj.weight = params["transition2/w"]


# --- Diffusion/Confidence weight loading (copied from unit parity helpers) ---

def _load_adaptive_layer_norm(adaln, ref_data: np.lib.npyio.NpzFile, base: str) -> None:
    _load_layer_norm(adaln.layer_norm, ref_data, f"{base}layer_norm")
    if adaln.single_cond_layer_norm is not None:
        _load_layer_norm(adaln.single_cond_layer_norm, ref_data, f"{base}single_cond_layer_norm")
    if adaln.single_cond_scale is not None:
        _load_linear(adaln.single_cond_scale, ref_data, f"{base}single_cond_scale")
    if adaln.single_cond_bias is not None:
        _load_linear(adaln.single_cond_bias, ref_data, f"{base}single_cond_bias")


def _load_adaptive_zero_init(az, ref_data: np.lib.npyio.NpzFile, base: str) -> None:
    _load_linear(az.transition2, ref_data, f"{base}transition2")
    if az.adaptive_zero_cond is not None:
        _load_linear(az.adaptive_zero_cond, ref_data, f"{base}adaptive_zero_cond")


def _load_self_attention(att, ref_data: np.lib.npyio.NpzFile, base: str) -> None:
    _load_adaptive_layer_norm(att.adaptive_norm, ref_data, base)
    _load_linear(att.q_projection, ref_data, f"{base}q_projection")
    _load_linear(att.k_projection, ref_data, f"{base}k_projection")
    _load_linear(att.v_projection, ref_data, f"{base}v_projection")
    _load_linear(att.gating_query, ref_data, f"{base}gating_query")
    _load_adaptive_zero_init(att.adaptive_zero, ref_data, base)


def _load_cross_attention(att, ref_data: np.lib.npyio.NpzFile, base: str) -> None:
    _load_adaptive_layer_norm(att.adaptive_norm_q, ref_data, f"{base}q")
    _load_adaptive_layer_norm(att.adaptive_norm_k, ref_data, f"{base}k")
    _load_linear(att.q_projection, ref_data, f"{base}q_projection")
    _load_linear(att.k_projection, ref_data, f"{base}k_projection")
    _load_linear(att.v_projection, ref_data, f"{base}v_projection")
    _load_linear(att.gating_query, ref_data, f"{base}gating_query")
    _load_adaptive_zero_init(att.adaptive_zero, ref_data, base)


def _iter_layer_params(
    ref_data: np.lib.npyio.NpzFile, prefix: str
) -> dict[int, dict[str, mx.array]]:
    layers: dict[int, dict[str, mx.array]] = {}
    for key in ref_data.keys():
        if not key.startswith(prefix):
            continue
        parts = key.split("/")
        layer_idx = None
        layer_pos = None
        for i, part in enumerate(parts):
            if part.startswith("layer_"):
                layer_idx = int(part.split("_")[1])
                layer_pos = i
                break
            if part == "layer_stack" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                layer_pos = i + 1
                break
        if layer_idx is None or layer_pos is None:
            continue
        if layer_pos + 2 >= len(parts):
            continue
        rel_path = "/".join(parts[layer_pos + 1 :])
        layers.setdefault(layer_idx, {})[rel_path] = mx.array(ref_data[key])
    return layers


def _assign_param(obj, param_name: str, value: mx.array) -> None:
    attr_map = {
        "weights": "weight",
        "w": "weight",
        "bias": "bias",
        "b": "bias",
        "scale": "scale",
        "offset": "offset",
    }
    attr = attr_map.get(param_name)
    if attr is None:
        return
    if hasattr(obj, attr):
        setattr(obj, attr, value)


def _load_transformer(transformer, ref_data: np.lib.npyio.NpzFile, prefix: str, name_prefix: str) -> None:
    _load_layer_norm(transformer.pair_input_layer_norm, ref_data, f"{prefix}/pair_input_layer_norm")
    _load_linear(transformer.pair_logits_projection, ref_data, f"{prefix}/pair_logits_projection")

    layers = _iter_layer_params(ref_data, prefix)
    for layer_idx, params in layers.items():
        if layer_idx >= len(transformer.blocks):
            continue
        block = transformer.blocks[layer_idx]
        for rel_path, value in params.items():
            parts = rel_path.split("/")
            if len(parts) < 2:
                continue
            comp = parts[0]
            param_name = parts[1]
            comp_suffix = comp[len(name_prefix):] if comp.startswith(name_prefix) else comp

            if comp_suffix in (
                "layer_norm",
                "single_cond_layer_norm",
                "single_cond_scale",
                "single_cond_bias",
                "q_projection",
                "k_projection",
                "v_projection",
                "gating_query",
                "transition2",
                "adaptive_zero_cond",
            ):
                _assign_param(block.self_attention, param_name, value)
                continue
            if comp_suffix in (
                "ffw_layer_norm",
                "ffw_single_cond_layer_norm",
                "ffw_single_cond_scale",
                "ffw_single_cond_bias",
                "ffw_transition1",
                "ffw_transition2",
                "ffw_adaptive_zero_cond",
            ):
                _assign_param(block.transition, param_name, value)
                continue


def _load_cross_transformer(
    transformer, ref_data: np.lib.npyio.NpzFile, prefix: str, name_prefix: str
) -> None:
    _load_layer_norm(transformer.pair_input_layer_norm, ref_data, f"{prefix}/pair_input_layer_norm")
    _load_linear(transformer.pair_logits_projection, ref_data, f"{prefix}/pair_logits_projection")

    layers = _iter_layer_params(ref_data, prefix)
    for layer_idx, params in layers.items():
        if layer_idx >= len(transformer.blocks):
            continue
        block = transformer.blocks[layer_idx]
        for rel_path, value in params.items():
            parts = rel_path.split("/")
            if len(parts) < 2:
                continue
            comp = parts[0]
            param_name = parts[1]
            comp_suffix = comp[len(name_prefix):] if comp.startswith(name_prefix) else comp

            if comp_suffix in (
                "qlayer_norm",
                "qsingle_cond_layer_norm",
                "qsingle_cond_scale",
                "qsingle_cond_bias",
                "q_projection",
                "k_projection",
                "v_projection",
                "gating_query",
                "transition2",
                "adaptive_zero_cond",
            ):
                _assign_param(block.cross_attention, param_name, value)
                continue
            if comp_suffix in (
                "klayer_norm",
                "ksingle_cond_layer_norm",
                "ksingle_cond_scale",
                "ksingle_cond_bias",
            ):
                _assign_param(block.cross_attention, param_name, value)
                continue
            if comp_suffix in (
                "ffw_layer_norm",
                "ffw_single_cond_layer_norm",
                "ffw_single_cond_scale",
                "ffw_single_cond_bias",
                "ffw_transition1",
                "ffw_transition2",
                "ffw_adaptive_zero_cond",
            ):
                _assign_param(block.transition, param_name, value)
                continue


def load_diffusion_head_weights(module, ref_data: np.lib.npyio.NpzFile, prefix: str | None = None) -> None:
    if prefix is None:
        prefix = "params/diffusion_head"

    # Conditioning projections
    _load_layer_norm(module.pair_cond_initial_norm, ref_data, f"{prefix}/pair_cond_initial_norm")
    _load_linear(module.pair_cond_initial_projection, ref_data, f"{prefix}/pair_cond_initial_projection")
    _load_layer_norm(module.single_cond_initial_norm, ref_data, f"{prefix}/single_cond_initial_norm")
    _load_linear(module.single_cond_initial_projection, ref_data, f"{prefix}/single_cond_initial_projection")
    _load_layer_norm(module.noise_embedding_initial_norm, ref_data, f"{prefix}/noise_embedding_initial_norm")
    _load_linear(module.noise_embedding_initial_projection, ref_data, f"{prefix}/noise_embedding_initial_projection")

    # Pair transitions
    _load_adaptive_layer_norm(module.pair_transition_0.adaptive_norm, ref_data, f"{prefix}/pair_transition_0/")
    _load_linear(module.pair_transition_0.transition1, ref_data, f"{prefix}/pair_transition_0/transition1")
    _load_adaptive_zero_init(module.pair_transition_0.adaptive_zero, ref_data, f"{prefix}/pair_transition_0/")

    _load_adaptive_layer_norm(module.pair_transition_1.adaptive_norm, ref_data, f"{prefix}/pair_transition_1/")
    _load_linear(module.pair_transition_1.transition1, ref_data, f"{prefix}/pair_transition_1/transition1")
    _load_adaptive_zero_init(module.pair_transition_1.adaptive_zero, ref_data, f"{prefix}/pair_transition_1/")

    # Single transitions
    _load_adaptive_layer_norm(module.single_transition_0.adaptive_norm, ref_data, f"{prefix}/single_transition_0/")
    _load_linear(module.single_transition_0.transition1, ref_data, f"{prefix}/single_transition_0/transition1")
    _load_adaptive_zero_init(module.single_transition_0.adaptive_zero, ref_data, f"{prefix}/single_transition_0/")

    _load_adaptive_layer_norm(module.single_transition_1.adaptive_norm, ref_data, f"{prefix}/single_transition_1/")
    _load_linear(module.single_transition_1.transition1, ref_data, f"{prefix}/single_transition_1/transition1")
    _load_adaptive_zero_init(module.single_transition_1.adaptive_zero, ref_data, f"{prefix}/single_transition_1/")

    # Atom cross-attention encoder
    _load_linear(module.atom_cross_att_encoder.embed_ref_pos, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_ref_pos")
    _load_linear(module.atom_cross_att_encoder.embed_ref_mask, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_ref_mask")
    _load_linear(module.atom_cross_att_encoder.embed_ref_element, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_ref_element")
    _load_linear(module.atom_cross_att_encoder.embed_ref_charge, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_ref_charge")
    _load_linear(module.atom_cross_att_encoder.embed_ref_atom_name, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_ref_atom_name")
    _load_linear(module.atom_cross_att_encoder.single_to_pair_cond_row, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_single_to_pair_cond_row")
    _load_linear(module.atom_cross_att_encoder.single_to_pair_cond_col, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_single_to_pair_cond_col")
    _load_layer_norm(module.atom_cross_att_encoder.lnorm_trunk_single_cond, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_trunk_single_cond_layer_norm")
    _load_linear(module.atom_cross_att_encoder.embed_trunk_single_cond, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_trunk_single_cond")
    _load_layer_norm(module.atom_cross_att_encoder.lnorm_trunk_pair_cond, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_trunk_pair_cond_layer_norm")
    _load_linear(module.atom_cross_att_encoder.embed_trunk_pair_cond, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_trunk_pair_cond")
    _load_linear(module.atom_cross_att_encoder.embed_pair_offsets, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_pair_offsets")
    _load_linear(module.atom_cross_att_encoder.embed_pair_distances, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_pair_distances")
    _load_linear(module.atom_cross_att_encoder.embed_pair_offsets_valid, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_pair_offsets_valid")
    _load_linear(module.atom_cross_att_encoder.pair_mlp_1, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_pair_mlp_1")
    _load_linear(module.atom_cross_att_encoder.pair_mlp_2, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_pair_mlp_2")
    _load_linear(module.atom_cross_att_encoder.pair_mlp_3, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_pair_mlp_3")
    _load_linear(module.atom_cross_att_encoder.atom_positions_to_features, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_embed_pair_offsets_1")
    _load_linear(module.atom_cross_att_encoder.project_atom_features_for_aggr, ref_data, f"{prefix}/atom_cross_attention/evoformer_conditioning_project_atom_features_for_aggr")

    _load_cross_transformer(
        module.atom_cross_att_encoder.atom_transformer_encoder,
        ref_data,
        f"{prefix}/atom_cross_attention/evoformer_conditioning_atom_transformer_encoder",
        "evoformer_conditioning_atom_transformer_encoder",
    )

    # Atom cross-attention decoder
    _load_linear(module.atom_cross_att_decoder.project_token_features_for_broadcast, ref_data, f"{prefix}/atom_cross_attention/atom_cross_att_decoder_project_token_features_for_broadcast")
    _load_layer_norm(module.atom_cross_att_decoder.atom_features_layer_norm, ref_data, f"{prefix}/atom_cross_attention/atom_cross_att_decoder_atom_features_layer_norm")
    _load_linear(module.atom_cross_att_decoder.atom_features_to_position_update, ref_data, f"{prefix}/atom_cross_attention/atom_cross_att_decoder_atom_features_to_position_update")

    _load_cross_transformer(
        module.atom_cross_att_decoder.atom_transformer_decoder,
        ref_data,
        f"{prefix}/atom_cross_attention/atom_cross_att_decoder",
        "atom_cross_att_decoder",
    )

    # Transformer
    _load_transformer(module.transformer, ref_data, f"{prefix}/transformer", "")

    # Output norm
    _load_layer_norm(module.output_norm, ref_data, f"{prefix}/output_norm")

    # Single conditioning embedding
    _load_layer_norm(module.single_cond_embedding_norm, ref_data, f"{prefix}/single_cond_embedding_norm")
    _load_linear(module.single_cond_embedding_projection, ref_data, f"{prefix}/single_cond_embedding_projection")


def load_confidence_head_weights(
    module,
    ref_data: np.lib.npyio.NpzFile,
    *,
    prefix: str | None = None,
) -> None:
    """Load ConfidenceHead weights from JAX reference NPZ."""
    prefix = prefix or "params/confidence_head"

    # Feature projections
    _load_linear(module.left_target_feat_project, ref_data, f"{prefix}/left_target_feat_project")
    _load_linear(module.right_target_feat_project, ref_data, f"{prefix}/right_target_feat_project")
    _load_linear(module.distogram_feat_project, ref_data, f"{prefix}/distogram_feat_project")

    # Pairformer stack (layer_stack)
    layers = _iter_layer_params(ref_data, f"{prefix}/confidence_pairformer")
    if not layers:
        # Some Haiku scopes place layer_stack before confidence_pairformer
        layers = _iter_layer_params(ref_data, prefix)
        filtered_layers: dict[int, dict[str, mx.array]] = {}
        for layer_idx, params in layers.items():
            filtered = {
                (k.split("confidence_pairformer/")[1] if "confidence_pairformer/" in k else k): v
                for k, v in params.items()
                if "confidence_pairformer/" in k
            }
            if filtered:
                filtered_layers[layer_idx] = filtered
        layers = filtered_layers
    for layer_idx, params in layers.items():
        if layer_idx >= len(module.pairformer_layers):
            continue
        layer = module.pairformer_layers[layer_idx]

        tri_out = {k.split("triangle_multiplication_outgoing/")[1]: v
                   for k, v in params.items()
                   if k.startswith("triangle_multiplication_outgoing/")}
        load_triangle_mult_weights(layer.triangle_mult_outgoing, tri_out)

        tri_in = {k.split("triangle_multiplication_incoming/")[1]: v
                  for k, v in params.items()
                  if k.startswith("triangle_multiplication_incoming/")}
        load_triangle_mult_weights(layer.triangle_mult_incoming, tri_in)

        attn1 = {k.split("pair_attention1/")[1]: v
                 for k, v in params.items()
                 if k.startswith("pair_attention1/")}
        load_grid_attention_weights(layer.pair_attention_row, attn1)

        attn2 = {k.split("pair_attention2/")[1]: v
                 for k, v in params.items()
                 if k.startswith("pair_attention2/")}
        load_grid_attention_weights(layer.pair_attention_col, attn2)

        trans = {k.split("pair_transition/")[1]: v
                 for k, v in params.items()
                 if k.startswith("pair_transition/")}
        load_transition_weights(layer.pair_transition, trans)

        if "single_pair_logits_norm/scale" in params:
            layer.pair_logits_norm.scale = params["single_pair_logits_norm/scale"]
        if "single_pair_logits_norm/offset" in params:
            layer.pair_logits_norm.offset = params["single_pair_logits_norm/offset"]
        if "single_pair_logits_projection/weights" in params:
            layer.pair_logits_proj.weight = params["single_pair_logits_projection/weights"]
        elif "single_pair_logits_projection/w" in params:
            layer.pair_logits_proj.weight = params["single_pair_logits_projection/w"]

        single_attn = {k.split("single_attention_")[1]: v
                       for k, v in params.items()
                       if k.startswith("single_attention_")}
        load_single_attention_weights(layer.single_attention, single_attn)

        single_trans = {k.split("single_transition/")[1]: v
                        for k, v in params.items()
                        if k.startswith("single_transition/")}
        load_transition_weights(layer.single_transition, single_trans)

    # Logits and norms
    _load_layer_norm(module.logits_ln, ref_data, f"{prefix}/logits_ln")
    _load_linear(module.left_half_distance_logits, ref_data, f"{prefix}/left_half_distance_logits")
    _load_layer_norm(module.pae_logits_ln, ref_data, f"{prefix}/pae_logits_ln")
    _load_linear(module.pae_logits, ref_data, f"{prefix}/pae_logits")

    _load_layer_norm(module.plddt_logits_ln, ref_data, f"{prefix}/plddt_logits_ln")
    _load_linear(module.plddt_logits, ref_data, f"{prefix}/plddt_logits")
    _load_layer_norm(module.experimentally_resolved_ln, ref_data, f"{prefix}/experimentally_resolved_ln")
    _load_linear(module.experimentally_resolved_logits, ref_data, f"{prefix}/experimentally_resolved_logits")


def _compute_backbone_rmsd(
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
) -> float:
    diff = pred - ref
    diff2 = np.sum(diff ** 2, axis=-1)
    valid = diff2[mask]
    return float(np.sqrt(np.mean(valid)))


def _relative_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))


class TestEndToEndJAXParity:
    """End-to-end RMSD parity vs JAX AF3 outputs."""

    def test_end_to_end_rmsd_and_confidence(self):
        if not REF_PATH.exists():
            pytest.skip(f"Missing reference file: {REF_PATH}")

        ref_data = np.load(REF_PATH)

        # Build parity Batch from reference dict
        batch_keys = [
            "token_index",
            "residue_index",
            "asym_id",
            "entity_id",
            "sym_id",
            "seq_mask",
            "pred_dense_atom_mask",
            "ref_pos",
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            # Atom cross-att gather info
            "token_atoms_to_queries/gather_idxs",
            "token_atoms_to_queries/gather_mask",
            "token_atoms_to_queries/input_shape",
            "tokens_to_queries/gather_idxs",
            "tokens_to_queries/gather_mask",
            "tokens_to_queries/input_shape",
            "tokens_to_keys/gather_idxs",
            "tokens_to_keys/gather_mask",
            "tokens_to_keys/input_shape",
            "queries_to_keys/gather_idxs",
            "queries_to_keys/gather_mask",
            "queries_to_keys/input_shape",
            "queries_to_token_atoms/gather_idxs",
            "queries_to_token_atoms/gather_mask",
            "queries_to_token_atoms/input_shape",
            # Pseudo-beta gather info
            "token_atoms_to_pseudo_beta/gather_idxs",
            "token_atoms_to_pseudo_beta/gather_mask",
            "token_atoms_to_pseudo_beta/input_shape",
        ]
        ref_dict = _expand_colon_keys(ref_data)
        batch_dict = {k: ref_dict[k] for k in batch_keys if k in ref_dict}
        batch = Batch.from_data_dict(batch_dict)

        num_res = int(ref_data["num_residues"])
        num_atoms = int(ref_data["num_atoms"])
        seq_channel = int(ref_data["seq_channel"])
        pair_channel = int(ref_data["pair_channel"])
        num_heads = int(ref_data["num_heads"])
        layer_prefix = "params/af3_model/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer"
        if "num_pairformer_layers" in ref_data.keys():
            num_layers = int(ref_data["num_pairformer_layers"])
        else:
            num_layers = _infer_num_layers(ref_data, layer_prefix)

        # Target feat from JAX reference (already includes per-atom conditioning)
        target_feat = mx.array(ref_data["target_feat"])

        # Relative encoding for pair activations
        token_features = TokenFeatures(
            token_index=mx.array(ref_data["token_index"]),
            residue_index=mx.array(ref_data["residue_index"]),
            asym_id=mx.array(ref_data["asym_id"]),
            entity_id=mx.array(ref_data["entity_id"]),
            sym_id=mx.array(ref_data["sym_id"]),
            mask=mx.array(ref_data["seq_mask"]).astype(mx.float32),
        )
        max_relative_idx = int(ref_data["max_relative_idx"]) if "max_relative_idx" in ref_data.keys() else 32
        max_relative_chain = int(ref_data["max_relative_chain"]) if "max_relative_chain" in ref_data.keys() else 2
        rel_feat = create_relative_encoding(
            token_features,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
        ).astype(mx.float32)

        # Build Evoformer input embeddings (JAX-compatible)
        left_single = Linear(pair_channel, input_dims=target_feat.shape[-1], use_bias=False)
        right_single = Linear(pair_channel, input_dims=target_feat.shape[-1], use_bias=False)
        prev_pair_ln = LayerNorm(pair_channel)
        prev_pair_proj = Linear(pair_channel, input_dims=pair_channel, use_bias=False)
        position_activations = Linear(pair_channel, input_dims=rel_feat.shape[-1], use_bias=False)

        single_proj = Linear(seq_channel, input_dims=target_feat.shape[-1], use_bias=False)
        prev_single_ln = LayerNorm(seq_channel)
        prev_single_proj = Linear(seq_channel, input_dims=seq_channel, use_bias=False)

        # Load embedding weights
        _load_linear(left_single, ref_data, _find_module_prefix(ref_data, "left_single"))
        _load_linear(right_single, ref_data, _find_module_prefix(ref_data, "right_single"))
        _load_layer_norm(prev_pair_ln, ref_data, _find_module_prefix(ref_data, "prev_embedding_layer_norm"))
        _load_linear(prev_pair_proj, ref_data, _find_module_prefix(ref_data, "prev_embedding"))
        _load_linear(position_activations, ref_data, _find_module_prefix(ref_data, "position_activations"))
        _load_linear(single_proj, ref_data, _find_module_prefix(ref_data, "single_activations"))
        _load_layer_norm(prev_single_ln, ref_data, _find_module_prefix(ref_data, "prev_single_embedding_layer_norm"))
        _load_linear(prev_single_proj, ref_data, _find_module_prefix(ref_data, "prev_single_embedding"))

        # Prev embeddings are zeros for first recycle
        prev_pair = mx.zeros((num_res, num_res, pair_channel))
        prev_single = mx.zeros((num_res, seq_channel))

        pair_act = left_single(target_feat)[:, None, :] + right_single(target_feat)[None, :, :]
        pair_act = pair_act + prev_pair_proj(prev_pair_ln(prev_pair))
        pair_act = pair_act + position_activations(rel_feat)

        single_act = single_proj(target_feat)
        single_act = single_act + prev_single_proj(prev_single_ln(prev_single))

        # Add batch dims
        seq_mask = mx.array(ref_data["seq_mask"]).astype(mx.float32)
        pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(mx.float32)
        single_act = single_act[None, ...]
        pair_act = pair_act[None, ...]
        seq_mask = seq_mask[None, ...]
        pair_mask = pair_mask[None, ...]

        # PairFormer stack weights (layer_stack)
        stacked_params = _iter_layer_stack_params(ref_data, layer_prefix, num_layers)

        layers = []
        for i in range(num_layers):
            layer = PairFormerIteration(
                seq_channel=seq_channel,
                pair_channel=pair_channel,
                num_attention_heads=num_heads,
                attention_key_dim=None,
                intermediate_factor=4,
                with_single=True,
            )
            params = stacked_params[i]
            tri_out_params = _select_subparams(params, "triangle_multiplication_outgoing/")
            tri_in_params = _select_subparams(params, "triangle_multiplication_incoming/")
            pair_attn1_params = _select_subparams(params, "pair_attention1/")
            pair_attn2_params = _select_subparams(params, "pair_attention2/")
            pair_trans_params = _select_subparams(params, "pair_transition/")
            single_attn_params = _select_subparams(params, "single_attention_")
            single_trans_params = _select_subparams(params, "single_transition/")

            load_triangle_mult_weights(layer.triangle_mult_outgoing, tri_out_params)
            load_triangle_mult_weights(layer.triangle_mult_incoming, tri_in_params)
            load_grid_attention_weights(layer.pair_attention_row, pair_attn1_params)
            load_grid_attention_weights(layer.pair_attention_col, pair_attn2_params)
            load_transition_weights(layer.pair_transition, pair_trans_params)
            load_single_attention_weights(layer.single_attention, single_attn_params)
            load_transition_weights(layer.single_transition, single_trans_params)

            # Single pair logits norm/proj
            norm_scale_key = "single_pair_logits_norm/scale"
            norm_offset_key = "single_pair_logits_norm/offset"
            proj_key_w = "single_pair_logits_projection/weights"
            proj_key_fallback = "single_pair_logits_projection/w"
            if norm_scale_key in params:
                layer.pair_logits_norm.scale = params[norm_scale_key]
            if norm_offset_key in params:
                layer.pair_logits_norm.offset = params[norm_offset_key]
            if proj_key_w in params:
                layer.pair_logits_proj.weight = params[proj_key_w]
            elif proj_key_fallback in params:
                layer.pair_logits_proj.weight = params[proj_key_fallback]

            layers.append(layer)

        for layer in layers:
            single_act, pair_act = layer(single_act, pair_act, seq_mask, pair_mask)
        mx.eval(single_act, pair_act)

        single_act = single_act[0]
        pair_act = pair_act[0]

        embeddings = {
            "single": single_act,
            "pair": pair_act,
            "target_feat": target_feat,
        }

        # Build MLX Model with matching config
        global_config = GlobalConfig(bfloat16="none", final_init="zeros")
        evoformer_cfg = ModelConfig.default().evoformer
        evoformer_cfg.seq_channel = seq_channel
        evoformer_cfg.pair_channel = pair_channel
        evoformer_cfg.num_pairformer_layers = num_layers
        evoformer_cfg.pairformer.num_attention_heads = num_heads

        diff_cfg = DiffusionConfig(
            num_steps=int(ref_data["diffusion_steps"]),
            num_samples=int(ref_data["diffusion_num_samples"]),
            conditioning_seq_channel=seq_channel,
            conditioning_pair_channel=pair_channel,
        )

        conf_cfg = ConfidenceConfig()
        if "num_plddt_bins" in ref_data.keys():
            conf_cfg.num_plddt_bins = int(ref_data["num_plddt_bins"])
        if "num_pae_bins" in ref_data.keys():
            conf_cfg.num_pae_bins = int(ref_data["num_pae_bins"])
            conf_cfg.num_bins = int(ref_data["num_pae_bins"])
        if "max_error_bin" in ref_data.keys():
            conf_cfg.max_error_bin = float(ref_data["max_error_bin"])

        model_cfg = ModelConfig(
            evoformer=evoformer_cfg,
            diffusion=diff_cfg,
            confidence=conf_cfg,
            global_config=global_config,
            num_recycles=int(ref_data["num_recycles"]),
            return_embeddings=False,
        )
        model = Model(model_cfg)

        # Build conditioning and confidence heads before weight loading
        model.diffusion_head._build_conditioning(
            pair_cond_dim=pair_act.shape[-1],
            single_cond_dim=single_act.shape[-1] + target_feat.shape[-1],
        )
        model.confidence_head._build(target_feat.shape[-1], int(ref_data["num_atoms"]))

        # Load JAX weights into MLX heads
        diff_prefix = _find_module_root(ref_data, "diffusion_head")
        conf_prefix = _find_module_root(ref_data, "confidence_head")
        load_diffusion_head_weights(model.diffusion_head, ref_data, prefix=diff_prefix)
        load_confidence_head_weights(model.confidence_head, ref_data, prefix=conf_prefix)

        diffusion_override = {
            "positions_noisy_steps": ref_data["positions_noisy_steps"],
            "t_hat_steps": ref_data["t_hat_steps"],
            "noise_levels": ref_data["noise_levels"],
        }

        # Run MLX Model inference (parity path) with overrides
        result = model(
            batch,
            key=mx.random.key(int(ref_data["seed"])),
            override_embeddings=embeddings,
            diffusion_override=diffusion_override,
        )
        mx.eval(result.atom_positions.positions, result.confidence.plddt, result.confidence.pae, result.confidence.pde)

        positions_out = np.array(result.atom_positions.positions)
        confidence = result.confidence

        # RMSD on backbone atoms (N, CA, C)
        jax_coords = ref_data["atom_positions"]
        jax_mask = ref_data["atom_positions_mask"].astype(bool)

        backbone_atoms = ["N", "CA", "C"]
        backbone_indices = [atom_types.ATOM37.index(name) for name in backbone_atoms]
        backbone_indices = [i for i in backbone_indices if i < num_atoms]
        if not backbone_indices:
            pytest.skip("Backbone atoms not present in reference")

        pred_bb = positions_out[0, :, backbone_indices, :]
        ref_bb = jax_coords[0, :, backbone_indices, :]
        mlx_mask = np.array(result.atom_positions.mask).astype(bool)
        mask_bb = jax_mask[0, :, backbone_indices] & mlx_mask[0, :, backbone_indices]
        if not np.any(mask_bb):
            pytest.skip("No valid backbone atoms for RMSD comparison")

        rmsd = _compute_backbone_rmsd(pred_bb, ref_bb, mask_bb)
        assert rmsd < RMSD_TOL, f"FAILED: RMSD {rmsd:.4f}Å exceeds {RMSD_TOL:.2f}Å"

        # Confidence comparisons
        plddt_mlx = np.array(confidence.plddt)
        pae_mlx = np.array(confidence.pae)
        pde_mlx = np.array(confidence.pde)

        plddt_jax = ref_data["predicted_lddt"]
        pae_jax = ref_data["full_pae"]
        pde_jax = ref_data["full_pde"]

        assert _relative_error(plddt_mlx, plddt_jax) < CONF_REL_TOL, "pLDDT relative error too high"
        assert _relative_error(pae_mlx, pae_jax) < CONF_REL_TOL, "PAE relative error too high"
        assert _relative_error(pde_mlx, pde_jax) < CONF_REL_TOL, "PDE relative error too high"

        # pTM comparison via tm-adjusted PAE
        if "tmscore_adjusted_pae_global" in ref_data.keys() and confidence.tm_pae_global is not None:
            tm_pae_jax = ref_data["tmscore_adjusted_pae_global"]
            tm_pae_mlx = np.array(confidence.tm_pae_global)
            pair_mask = (ref_data["seq_mask"][:, None] * ref_data["seq_mask"][None, :]).astype(bool)
            asym_id = ref_data["asym_id"].astype(int)

            ptm_jax = jax_confidences.predicted_tm_score(tm_pae_jax[0], pair_mask, asym_id, interface=False)
            ptm_mlx = jax_confidences.predicted_tm_score(tm_pae_mlx[0], pair_mask, asym_id, interface=False)
            ptm_rel_err = abs(ptm_mlx - ptm_jax) / (abs(ptm_jax) + 1e-8)
            assert ptm_rel_err < CONF_REL_TOL, f"pTM relative error too high: {ptm_rel_err:.4f}"

        print("\n=== End-to-End JAX Parity PASSED ===")
        print(f"  RMSD (backbone): {rmsd:.4f} Å")
