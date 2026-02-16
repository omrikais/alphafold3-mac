"""Main AlphaFold 3 model for MLX.

This module implements the main Model class which orchestrates:
- Evoformer processing with recycling
- Diffusion-based coordinate generation
- Confidence head for quality estimation
- Weight loading from JAX format
- Model compilation for performance
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from alphafold3.constants import residue_names
import warnings

from alphafold3_mlx.core.config import ModelConfig
from alphafold3_mlx.core.entities import Embeddings, AtomPositions, ConfidenceScores
from alphafold3_mlx.core.outputs import ModelResult
from alphafold3_mlx.core.validation import (
    check_memory_requirements,
    check_nan,
    get_available_memory_gb,
)
from alphafold3_mlx.core.atom_constants import (
    get_atom37_mask,
    NUM_ATOMS,
)
from alphafold3_mlx.network.evoformer import Evoformer
from alphafold3_mlx.network.diffusion_head import DiffusionHead
from alphafold3_mlx.network.confidence_head import ConfidenceHead
from alphafold3_mlx.network.atom_cross_attention import AtomCrossAttEncoder
from alphafold3_mlx.model.recycling import run_recycling_loop

if TYPE_CHECKING:
    from alphafold3_mlx.core.inputs import FeatureBatch


def _weights_exist(model_dir: Path) -> bool:
    """Check if any supported weight format exists (C-07)."""
    patterns = ["af3.bin.zst", "af3.bin", "af3.0.bin.zst"]
    return (
        any((model_dir / p).exists() for p in patterns)
        or any(model_dir.glob("af3.*.bin.zst"))
    )


# JAX to MLX parameter name mapping
# Maps Haiku hierarchical names to MLX module paths
# Format: Haiku uses "scope/module/submodule" with "/w" or "/b" for weights/biases
# MLX uses "module.submodule.weight" or "module.submodule.bias"

# Static mappings for top-level modules
JAX_TO_MLX_MODULE_MAP: dict[str, str] = {
    # Top-level model structure
    "diffuser/evoformer": "evoformer",
    "diffuser/diffusion_head": "diffusion_head",
    "diffuser/confidence_head": "confidence_head",
    "diffuser/distogram_head": "distogram_head",

    # Evoformer internal structure
    "trunk_pairformer": "pairformer_stack",
    "msa_stack": "msa_stack",

    # Attention projections
    "q_projection": "q_proj",
    "k_projection": "k_proj",
    "v_projection": "v_proj",
    "output_projection": "o_proj",
    "pair_bias_projection": "pair_bias_proj",
    "gating_query": "gate_proj",

    # Layer norms
    "act_norm": "layer_norm",
    "pair_norm": "pair_layer_norm",
    "logits_ln": "logits_ln",
    "pae_logits_ln": "pae_logits_ln",
    "pde_logits_ln": "pde_logits_ln",

    # Attention blocks
    "pair_attention1": "pair_attention_row",
    "pair_attention2": "pair_attention_col",
    "single_attention_": "single_attention",
    "msa_attention1": "msa_row_attention",

    # Triangle operations
    "triangle_multiplication_outgoing": "triangle_mult_outgoing",
    "triangle_multiplication_incoming": "triangle_mult_incoming",

    # Embeddings
    "left_single": "left_single_proj",
    "right_single": "right_single_proj",
    "position_activations": "position_activations",
    "~_relative_encoding/position_activations": "position_activations",
    "bond_embedding": "bond_embedding",
    "prev_embedding/": "prev_pair_proj/",
    "single_activations": "single_input_proj",
    "prev_single_embedding/": "prev_single_proj/",
    "msa_activations": "msa_input_proj",
    "extra_msa_target_feat": "extra_msa_proj",
    "~_embed_features/": "",

    # Confidence head
    "left_target_feat_project": "left_target_feat_project",
    "right_target_feat_project": "right_target_feat_project",
    "distogram_feat_project": "distogram_feat_project",
    "confidence_pairformer": "pairformer_stack",
    "left_half_distance_logits": "left_half_distance_logits",
    "pae_logits": "pae_logits",
    "pde_logits": "pde_proj",

    # Diffusion head (direct matches)
    "diffusion_atom_transformer_encoder": "atom_cross_att_encoder.atom_transformer_encoder",
    "diffusion_atom_transformer_decoder": "atom_cross_att_decoder.atom_transformer_decoder",
    "diffusion_embed_ref_pos": "atom_cross_att_encoder.embed_ref_pos",
    "diffusion_embed_ref_mask": "atom_cross_att_encoder.embed_ref_mask",
    "diffusion_embed_ref_element": "atom_cross_att_encoder.embed_ref_element",
    "diffusion_embed_ref_charge": "atom_cross_att_encoder.embed_ref_charge",
    "diffusion_embed_ref_atom_name": "atom_cross_att_encoder.embed_ref_atom_name",
    "diffusion_single_to_pair_cond_row": "atom_cross_att_encoder.single_to_pair_cond_row",
    "diffusion_single_to_pair_cond_col": "atom_cross_att_encoder.single_to_pair_cond_col",
    "diffusion_single_to_pair_cond_row_1": "atom_cross_att_encoder.single_to_pair_cond_row_1",
    "diffusion_single_to_pair_cond_col_1": "atom_cross_att_encoder.single_to_pair_cond_col_1",
    "diffusion_lnorm_trunk_single_cond": "atom_cross_att_encoder.lnorm_trunk_single_cond",
    "diffusion_embed_trunk_single_cond": "atom_cross_att_encoder.embed_trunk_single_cond",
    "diffusion_lnorm_trunk_pair_cond": "atom_cross_att_encoder.lnorm_trunk_pair_cond",
    "diffusion_embed_trunk_pair_cond": "atom_cross_att_encoder.embed_trunk_pair_cond",
    "diffusion_embed_pair_offsets": "atom_cross_att_encoder.embed_pair_offsets",
    "diffusion_embed_pair_distances": "atom_cross_att_encoder.embed_pair_distances",
    "diffusion_embed_pair_offsets_1": "atom_cross_att_encoder.embed_pair_offsets_1",
    "diffusion_embed_pair_distances_1": "atom_cross_att_encoder.embed_pair_distances_1",
    "diffusion_embed_pair_offsets_valid": "atom_cross_att_encoder.embed_pair_offsets_valid",
    "diffusion_pair_mlp_1": "atom_cross_att_encoder.pair_mlp_1",
    "diffusion_pair_mlp_2": "atom_cross_att_encoder.pair_mlp_2",
    "diffusion_pair_mlp_3": "atom_cross_att_encoder.pair_mlp_3",
    "diffusion_atom_positions_to_features": "atom_cross_att_encoder.atom_positions_to_features",
    "diffusion_project_atom_features_for_aggr": "atom_cross_att_encoder.project_atom_features_for_aggr",
    "diffusion_project_token_features_for_broadcast": "atom_cross_att_decoder.project_token_features_for_broadcast",
    "diffusion_atom_features_layer_norm": "atom_cross_att_decoder.atom_features_layer_norm",
    "diffusion_atom_features_to_position_update": "atom_cross_att_decoder.atom_features_to_position_update",
    "pair_transition_0ffw_layer_norm": "pair_transition_0.adaptive_norm.layer_norm",
    "pair_transition_0ffw_transition1": "pair_transition_0.transition1",
    "pair_transition_0ffw_transition2": "pair_transition_0.adaptive_zero.transition2",
    "pair_transition_1ffw_layer_norm": "pair_transition_1.adaptive_norm.layer_norm",
    "pair_transition_1ffw_transition1": "pair_transition_1.transition1",
    "pair_transition_1ffw_transition2": "pair_transition_1.adaptive_zero.transition2",
    "single_transition_0ffw_layer_norm": "single_transition_0.adaptive_norm.layer_norm",
    "single_transition_0ffw_transition1": "single_transition_0.transition1",
    "single_transition_0ffw_transition2": "single_transition_0.adaptive_zero.transition2",
    "single_transition_1ffw_layer_norm": "single_transition_1.adaptive_norm.layer_norm",
    "single_transition_1ffw_transition1": "single_transition_1.transition1",
    "single_transition_1ffw_transition2": "single_transition_1.adaptive_zero.transition2",
    # Evoformer conditioning (per-atom target_feat embedding)
    "evoformer_conditioning_atom_transformer_encoder": "evoformer_conditioning.atom_transformer_encoder",
    "evoformer_conditioning_embed_ref_pos": "evoformer_conditioning.embed_ref_pos",
    "evoformer_conditioning_embed_ref_mask": "evoformer_conditioning.embed_ref_mask",
    "evoformer_conditioning_embed_ref_element": "evoformer_conditioning.embed_ref_element",
    "evoformer_conditioning_embed_ref_charge": "evoformer_conditioning.embed_ref_charge",
    "evoformer_conditioning_embed_ref_atom_name": "evoformer_conditioning.embed_ref_atom_name",
    "evoformer_conditioning_single_to_pair_cond_row": "evoformer_conditioning.single_to_pair_cond_row",
    "evoformer_conditioning_single_to_pair_cond_col": "evoformer_conditioning.single_to_pair_cond_col",
    "evoformer_conditioning_single_to_pair_cond_row_1": "evoformer_conditioning.single_to_pair_cond_row_1",
    "evoformer_conditioning_single_to_pair_cond_col_1": "evoformer_conditioning.single_to_pair_cond_col_1",
    "evoformer_conditioning_embed_pair_offsets": "evoformer_conditioning.embed_pair_offsets",
    "evoformer_conditioning_embed_pair_offsets_1": "evoformer_conditioning.embed_pair_offsets_1",
    "evoformer_conditioning_embed_pair_distances": "evoformer_conditioning.embed_pair_distances",
    "evoformer_conditioning_embed_pair_distances_1": "evoformer_conditioning.embed_pair_distances_1",
    "evoformer_conditioning_embed_pair_offsets_valid": "evoformer_conditioning.embed_pair_offsets_valid",
    "evoformer_conditioning_pair_mlp_1": "evoformer_conditioning.pair_mlp_1",
    "evoformer_conditioning_pair_mlp_2": "evoformer_conditioning.pair_mlp_2",
    "evoformer_conditioning_pair_mlp_3": "evoformer_conditioning.pair_mlp_3",
    "evoformer_conditioning_project_atom_features_for_aggr": "evoformer_conditioning.project_atom_features_for_aggr",
}

# Weight/bias suffix mapping
JAX_PARAM_NAME_MAP: dict[str, str] = {
    "w": "weight",
    "b": "bias",
    "weights": "weight",
    "bias": "bias",
    # LayerNorm uses scale/offset parameter names directly.
    "scale": "scale",
    "offset": "offset",
}


class Model(nn.Module):
    """Main AlphaFold 3 model.

    This class orchestrates the full inference pipeline:
    1. Input feature processing
    2. Evoformer with recycling
    3. Diffusion-based coordinate generation
    4. Confidence prediction

    Example:
        >>> from alphafold3_mlx.model import Model
        >>> from alphafold3_mlx.core import ModelConfig, FeatureBatch
        >>>
        >>> config = ModelConfig.default()
        >>> model = Model(config)
        >>> model.load_weights("path/to/weights")
        >>>
        >>> batch = FeatureBatch.from_numpy(feature_dict)
        >>> result = model(batch)
        >>>
        >>> coords = result.atom_positions.positions  # [5, N, 37, 3]
        >>> plddt = result.confidence.plddt           # [5, N, 37]
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize model.

        Args:
            config: Model configuration. Uses defaults if None.

        Raises:
            ValueError: If config validation fails.
        """
        super().__init__()

        self.config = config or ModelConfig.default()
        c = self.config

        # Set up chunking config globally
        from alphafold3_mlx.network.chunking import ChunkingConfig, set_chunking_config

        if c.global_config.chunk_size is not None:
            chunking_config = ChunkingConfig(
                enabled=True,
                chunk_size=c.global_config.chunk_size,
            )
        else:
            chunking_config = ChunkingConfig(enabled=False)
        set_chunking_config(chunking_config)

        # Input embeddings
        # Target feature embedding: one-hot aatype (22 classes) -> seq_channel
        self.target_feat_embed = nn.Embedding(22, c.evoformer.seq_channel)
        # Note: Pair initialization no longer uses relative position embedding here.
        # Relative position is added ONLY in Evoformer to avoid double-counting.

        # Evoformer conditioning (per-atom target_feat embedding)
        # Mirrors JAX create_target_feat_embedding() per-atom conditioning.
        from alphafold3_mlx.core.config import DiffusionConfig

        evoformer_cond_config = DiffusionConfig(
            per_token_channels=c.evoformer.seq_channel,
            per_atom_channels=128,
            per_atom_pair_channels=16,
            atom_transformer_num_blocks=3,
            atom_transformer_num_intermediate_factor=2,
            atom_transformer_num_head=4,
            atom_transformer_key_dim=128,
            atom_transformer_value_dim=128,
        )
        self.evoformer_conditioning = AtomCrossAttEncoder(
            evoformer_cond_config, c.global_config, name="evoformer_conditioning"
        )

        # Evoformer
        self.evoformer = Evoformer(
            config=c.evoformer,
            global_config=c.global_config,
        )

        # Diffusion head
        self.diffusion_head = DiffusionHead(
            config=c.diffusion,
            global_config=c.global_config,
        )

        # Confidence head
        self.confidence_head = ConfidenceHead(
            config=c.confidence,
            global_config=c.global_config,
            seq_channel=c.evoformer.seq_channel,
            pair_channel=c.evoformer.pair_channel,
        )

        # Track compilation state
        self._compiled = False

    def __call__(
        self,
        batch: "FeatureBatch",
        key: mx.array | None = None,
        capture_checkpoints: bool = False,
        override_embeddings: Embeddings | dict[str, mx.array] | None = None,
        diffusion_override: dict[str, mx.array] | None = None,
        check_nans: bool = True,
        diffusion_callback: "Callable[[int, int], None] | None" = None,
        recycling_callback: "Callable[[int, int], None] | None" = None,
        confidence_callback: "Callable[[str], None] | None" = None,
        guidance_fn: "Callable | None" = None,
    ) -> ModelResult:
        """Run inference on a feature batch.

        Args:
            batch: Input features from data pipeline.
            key: Random key for sampling (auto-generated if None).
            capture_checkpoints: If True, capture intermediate activations
                at key checkpoints for validation. Stored in
                result.metadata["checkpoints"].
            check_nans: If True (default), check for NaN values at validation
                checkpoints and raise NaNError if detected. Set to False to
                disable NaN checks for performance.
            diffusion_callback: Optional callback called after each diffusion
                step with (step, total_steps) for progress reporting.
            recycling_callback: Optional callback called after each recycling
                iteration with (iteration, total) for progress reporting.
            confidence_callback: Optional callback called with "start" or "end"
                to signal confidence computation phase.

        Returns:
            ModelResult containing predicted coordinates and confidence scores.

        Raises:
            MemoryError: If estimated memory exceeds hardware limits.
            NaNError: If NaN values detected at validation checkpoints
                (only when check_nans=True).
        """
        # Parity path: accept minimal Batch for tests
        from alphafold3_mlx.feat_batch import Batch as ParityBatch
        if isinstance(batch, ParityBatch):
            return self._call_parity(
                batch=batch,
                key=key,
                override_embeddings=override_embeddings,
                diffusion_override=diffusion_override,
                capture_checkpoints=capture_checkpoints,
                check_nans=check_nans,
            )

        import time
        metadata: dict = {"start_time": time.time()}
        checkpoints: dict[str, mx.array] = {} if capture_checkpoints else None

        # Memory check before processing
        num_residues = batch.num_residues
        available_memory = get_available_memory_gb()
        check_memory_requirements(
            num_residues=num_residues,
            available_gb=available_memory,
            num_samples=self.config.diffusion.num_samples,
        )

        # Generate random key if not provided
        if key is None:
            warnings.warn(
                "No seed provided; using default seed=42 for reproducibility",
                stacklevel=2,
            )
            key = mx.random.key(42)

        # Extract features
        token_features = batch.token_features
        seq_mask = token_features.mask
        residue_index = token_features.residue_index
        asym_id = token_features.asym_id
        aatype = token_features.aatype

        # Handle 1D arrays (no batch dimension) - add batch dim first
        batch_size = 1  # Currently single-sequence inference
        if seq_mask.ndim == 1:
            seq_mask = seq_mask[None, :]  # [1, seq]
        if residue_index.ndim == 1:
            residue_index = residue_index[None, :]
        if asym_id.ndim == 1:
            asym_id = asym_id[None, :]
        if aatype.ndim == 1:
            aatype = aatype[None, :]

        # Create target_feat embedding (JAX parity):
        # one-hot aatype + MSA profile + deletion mean + per-atom conditioning.
        aatype_1d = aatype[0] if aatype.ndim == 2 else aatype
        aatype_vocab = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        aatype_clamped = mx.clip(aatype_1d, 0, aatype_vocab - 1)
        target_parts = [
            (aatype_clamped[:, None] == mx.arange(aatype_vocab)).astype(mx.float32)
        ]

        if batch.msa_profile is not None:
            msa_profile = batch.msa_profile
            if msa_profile.ndim == 3:
                msa_profile = msa_profile[0]
            target_parts.append(msa_profile.astype(mx.float32))

        if batch.deletion_mean is not None:
            deletion_mean = batch.deletion_mean
            if deletion_mean.ndim == 2:
                deletion_mean = deletion_mean[0]
            target_parts.append(deletion_mean.astype(mx.float32)[..., None])

        target_feat_2d = mx.concatenate(target_parts, axis=-1)

        if batch.per_atom_features is not None:
            enc = self.evoformer_conditioning(
                token_atoms_act=None,
                trunk_single_cond=None,
                trunk_pair_cond=None,
                batch=batch.per_atom_features,
            )
            target_feat_2d = mx.concatenate([target_feat_2d, enc.token_act], axis=-1)

        # AF3 target_feat width is fixed by trained projections. Keep a stable
        # feature width by padding/truncating when optional inputs are absent.
        # Query the Evoformer's projection directly — it always knows its
        # expected input dim from __init__, regardless of loaded weights.
        expected_target_feat_dim = self.evoformer.left_single_proj._input_shape[0]

        if expected_target_feat_dim is not None:
            current_dim = int(target_feat_2d.shape[-1])
            if current_dim < expected_target_feat_dim:
                pad = mx.zeros(
                    (int(target_feat_2d.shape[0]), expected_target_feat_dim - current_dim),
                    dtype=target_feat_2d.dtype,
                )
                target_feat_2d = mx.concatenate([target_feat_2d, pad], axis=-1)
            elif current_dim > expected_target_feat_dim:
                target_feat_2d = target_feat_2d[:, :expected_target_feat_dim]

        target_feat = target_feat_2d[None, ...]  # [batch, seq, feat_dim]

        # JAX parity recycling state starts from zeros; Evoformer rebuilds
        # pair/single embeddings from target_feat at each recycle iteration.
        single = mx.zeros(
            (batch_size, num_residues, self.config.evoformer.seq_channel),
            dtype=mx.float32,
        )
        pair = mx.zeros(
            (batch_size, num_residues, num_residues, self.config.evoformer.pair_channel),
            dtype=mx.float32,
        )

        # Compute pair mask (now seq_mask is [batch, seq])
        pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # [batch, seq, seq]

        # Prepare MSA features if available
        msa_features = None
        msa_mask = None
        if batch.has_msa and batch.msa_features is not None:
            msa = batch.msa_features.msa  # [num_msa, seq]
            if msa.ndim == 2:
                msa = msa[None, :, :]  # Add batch dim: [batch, num_msa, seq]
            num_msa = int(msa.shape[1])

            # Match JAX featurization.create_msa_feat():
            # one_hot(rows, POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1) + has_deletion + deletion_value
            msa_vocab = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1
            msa_clamped = mx.clip(msa, 0, msa_vocab - 1)
            msa_one_hot = (msa_clamped[..., None] == mx.arange(msa_vocab)).astype(mx.float32)

            # Get deletion_matrix if available
            deletion_matrix = batch.msa_features.deletion_matrix
            if deletion_matrix is not None:
                if deletion_matrix.ndim == 2:
                    deletion_matrix = deletion_matrix[None, :, :]  # Add batch dim

                deletion_matrix = deletion_matrix.astype(mx.float32)
                has_deletion = mx.clip(deletion_matrix, 0.0, 1.0)
                deletion_value = mx.arctan(deletion_matrix / 3.0) * (2.0 / mx.pi)

                # [batch, num_msa, seq, 34] - JAX order: one_hot, has_deletion, deletion_value
                msa_features_raw = mx.concatenate([
                    msa_one_hot,
                    has_deletion[:, :, :, None],  # [batch, num_msa, seq, 1]
                    deletion_value[:, :, :, None],  # [batch, num_msa, seq, 1]
                ], axis=-1)
            else:
                # No deletion matrix - pad with zeros
                zeros = mx.zeros((batch_size, num_msa, num_residues, 2))
                msa_features_raw = mx.concatenate([msa_one_hot, zeros], axis=-1)

            # Keep raw MSA feature width (34); Evoformer applies learned projection.
            msa_features = msa_features_raw

            msa_mask = batch.msa_features.msa_mask
            if msa_mask.ndim == 2:
                msa_mask = msa_mask[None, :, :]

        # Prepare raw template data if available (JAX AF3 parity)
        # Raw template data is passed to Evoformer which computes features internally.
        template_aatype = None
        template_atom_positions = None
        template_atom_mask = None
        if batch.has_templates and batch.template_features is not None:
            template_aatype = batch.template_features.template_aatype
            template_atom_positions = batch.template_features.template_all_atom_positions
            template_atom_mask = batch.template_features.template_all_atom_mask

        # Prepare bond features if available
        # JAX AF3 uses a binary contact_matrix[:, :, None] with shape [seq, seq, 1]
        bond_features = None
        if batch.polymer_ligand_bond_info.num_bonds > 0 or batch.ligand_ligand_bond_info.num_bonds > 0:
            # Create binary contact matrix: [num_residues, num_residues]
            contact_matrix = mx.zeros((num_residues, num_residues), dtype=mx.float32)

            def _set_contact_positions(token_i, token_j, num_res):
                """Set symmetric contact positions in a binary matrix.

                Uses NumPy fancy indexing for O(num_res²) memory instead of
                broadcasting a [num_bonds, num_res, num_res] intermediate.
                """
                ti = np.asarray(token_i, dtype=np.intp)
                tj = np.asarray(token_j, dtype=np.intp)

                valid = (ti >= 0) & (ti < num_res) & (tj >= 0) & (tj < num_res)
                vi, vj = ti[valid], tj[valid]

                if len(vi) == 0:
                    return mx.zeros((num_res, num_res), dtype=mx.float32)

                result = np.zeros((num_res, num_res), dtype=np.float32)
                result[vi, vj] = 1.0
                result[vj, vi] = 1.0  # symmetric
                return mx.array(result)

            # Polymer-ligand bonds
            if batch.polymer_ligand_bond_info.num_bonds > 0:
                pl_contacts = _set_contact_positions(
                    batch.polymer_ligand_bond_info.token_i,
                    batch.polymer_ligand_bond_info.token_j,
                    num_residues,
                )
                contact_matrix = mx.minimum(contact_matrix + pl_contacts, 1.0)

            # Ligand-ligand bonds
            if batch.ligand_ligand_bond_info.num_bonds > 0:
                ll_contacts = _set_contact_positions(
                    batch.ligand_ligand_bond_info.token_i,
                    batch.ligand_ligand_bond_info.token_j,
                    num_residues,
                )
                contact_matrix = mx.minimum(contact_matrix + ll_contacts, 1.0)

            # Zero out padding position [0, 0] (matches JAX: contact_matrix.at[0,0].set(0.0))
            zero_mask = 1.0 - (
                (mx.arange(num_residues)[:, None] == 0).astype(mx.float32)
                * (mx.arange(num_residues)[None, :] == 0).astype(mx.float32)
            )
            contact_matrix = contact_matrix * zero_mask

            # Shape: [batch, seq, seq, 1] matching JAX contact_matrix[:, :, None]
            bond_features = contact_matrix[None, :, :, None]

        # Run Evoformer with recycling
        # Use compiled Evoformer when available
        evoformer_fn = self.evoformer
        if self._compiled and hasattr(self, '_compiled_evoformer'):
            evoformer_fn = self._compiled_evoformer

        # Track Evoformer timing
        evoformer_start = time.time()

        # Request intermediates when capturing checkpoints
        recycling_result = run_recycling_loop(
            evoformer_fn=evoformer_fn,
            initial_single=single,
            initial_pair=pair,
            target_feat=target_feat,
            residue_index=residue_index,
            asym_id=asym_id,
            num_recycles=self.config.num_recycles,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            token_features=token_features,
            msa_features=msa_features,
            msa_mask=msa_mask,
            template_aatype=template_aatype,
            template_atom_positions=template_atom_positions,
            template_atom_mask=template_atom_mask,
            bond_features=bond_features,
            return_intermediates=capture_checkpoints,
            iteration_callback=recycling_callback,
        )

        # Unpack result based on whether intermediates were requested
        if capture_checkpoints:
            single, pair, _, evoformer_intermediates = recycling_result
        else:
            single, pair, _ = recycling_result
            evoformer_intermediates = None

        # Force synchronization for accurate timing
        mx.eval(single, pair)
        metadata["evoformer_duration_seconds"] = time.time() - evoformer_start

        # NaN check after Evoformer
        if check_nans:
            check_nan(single, "evoformer.single")
            check_nan(pair, "evoformer.pair")

        # Capture Evoformer checkpoint and per-layer intermediates
        if capture_checkpoints:
            # Use slicing to create independent copy (MLX arrays don't have .copy())
            checkpoints["evoformer_single"] = single[:]
            checkpoints["evoformer_pair"] = pair[:]
            # Merge per-layer PairFormer checkpoints
            if evoformer_intermediates:
                for ck_name, value in evoformer_intermediates.items():
                    checkpoints[f"evoformer_{ck_name}"] = value

        # Get atom37 mask based on residue types
        # aatype: [batch, residues] -> atom37_mask: [batch, residues, 37]
        atom37_mask = get_atom37_mask(aatype)  # [batch, residues, 37]

        # Apply sequence mask to atom mask
        atom37_mask = atom37_mask * seq_mask[:, :, None]  # [batch, residues, 37]

        # Generate coordinates via diffusion (directly in atom37 format)
        key, diffusion_key = mx.random.split(key)

        # Build a minimal diffusion batch from FeatureBatch inputs.
        from alphafold3_mlx.atom_layout import GatherInfo
        from alphafold3_mlx.feat_batch import (
            Batch as DiffusionBatch,
            TokenFeatures as DiffusionTokenFeatures,
            PredictedStructureInfo,
            AtomCrossAtt,
            PseudoBetaInfo,
            RefStructure,
        )

        if batch.per_atom_features is not None:
            # Use real per-atom conditioning from featurisation when available.
            diffusion_batch = batch.per_atom_features
            seq_mask_1d = diffusion_batch.token_features.mask
            asym_id_1d = diffusion_batch.token_features.asym_id
            token_atoms_to_pseudo_beta = (
                diffusion_batch.pseudo_beta_info.token_atoms_to_pseudo_beta
            )
        else:
            num_atoms = int(atom37_mask.shape[-1])
            seq_mask_1d = seq_mask[0]
            residue_index_1d = residue_index[0]
            asym_id_1d = asym_id[0]
            entity_id = token_features.entity_id
            if entity_id.ndim == 2:
                entity_id = entity_id[0]
            sym_id = token_features.sym_id
            if sym_id.ndim == 2:
                sym_id = sym_id[0]

            token_index = mx.arange(num_residues, dtype=mx.int32)
            atom_mask_1d = atom37_mask[0].astype(mx.float32)

            token_atom_indices = mx.arange(
                num_residues * num_atoms, dtype=mx.int32
            ).reshape(num_residues, num_atoms)
            token_atoms_to_queries = GatherInfo(
                gather_idxs=token_atom_indices,
                gather_mask=mx.ones((num_residues, num_atoms), dtype=mx.bool_),
                input_shape=mx.array((num_residues, num_atoms)),
            )
            token_to_query_idxs = mx.broadcast_to(
                token_index[:, None], (num_residues, num_atoms)
            )
            tokens_to_queries = GatherInfo(
                gather_idxs=token_to_query_idxs,
                gather_mask=mx.ones((num_residues, num_atoms), dtype=mx.bool_),
                input_shape=mx.array((num_residues,)),
            )
            tokens_to_keys = GatherInfo(
                gather_idxs=token_to_query_idxs,
                gather_mask=mx.ones((num_residues, num_atoms), dtype=mx.bool_),
                input_shape=mx.array((num_residues,)),
            )
            queries_to_keys = GatherInfo(
                gather_idxs=token_atom_indices,
                gather_mask=mx.ones((num_residues, num_atoms), dtype=mx.bool_),
                input_shape=mx.array((num_residues, num_atoms)),
            )
            queries_to_token_atoms = GatherInfo(
                gather_idxs=token_atom_indices,
                gather_mask=mx.ones((num_residues, num_atoms), dtype=mx.bool_),
                input_shape=mx.array((num_residues, num_atoms)),
            )
            token_atoms_to_pseudo_beta = GatherInfo(
                gather_idxs=token_index * num_atoms,
                gather_mask=mx.ones((num_residues,), dtype=mx.bool_),
                input_shape=mx.array((num_residues, num_atoms)),
            )

            diffusion_batch = DiffusionBatch(
                token_features=DiffusionTokenFeatures(
                    token_index=token_index,
                    residue_index=residue_index_1d,
                    asym_id=asym_id_1d,
                    entity_id=entity_id,
                    sym_id=sym_id,
                    mask=seq_mask_1d,
                ),
                predicted_structure_info=PredictedStructureInfo(atom_mask=atom_mask_1d),
                atom_cross_att=AtomCrossAtt(
                    token_atoms_to_queries=token_atoms_to_queries,
                    tokens_to_queries=tokens_to_queries,
                    tokens_to_keys=tokens_to_keys,
                    queries_to_keys=queries_to_keys,
                    queries_to_token_atoms=queries_to_token_atoms,
                ),
                pseudo_beta_info=PseudoBetaInfo(
                    token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
                ),
                ref_structure=RefStructure(
                    positions=mx.zeros((num_residues, num_atoms, 3), dtype=mx.float32),
                    mask=atom_mask_1d,
                    element=mx.zeros((num_residues, num_atoms), dtype=mx.int32),
                    charge=mx.zeros((num_residues, num_atoms), dtype=mx.float32),
                    atom_name_chars=mx.zeros((num_residues, num_atoms, 4), dtype=mx.int32),
                    space_uid=mx.zeros((num_residues, num_atoms), dtype=mx.int32),
                ),
            )

        diffusion_embeddings = {
            "single": single[0],
            "pair": pair[0],
            "target_feat": target_feat[0],  # One-hot aatype [seq, 22]
        }

        def denoising_step(positions_noisy, noise_level):
            return self.diffusion_head(
                positions_noisy=positions_noisy,
                noise_level=noise_level,
                batch=diffusion_batch,
                embeddings=diffusion_embeddings,
                use_conditioning=True,
            )

        # Track Diffusion timing
        diffusion_start = time.time()

        diffusion_samples = self.diffusion_head.sample(
            denoising_step=denoising_step,
            batch=diffusion_batch,
            key=diffusion_key,
            num_steps=self.config.diffusion.num_steps,
            gamma_0=self.config.diffusion.gamma_0,
            gamma_min=self.config.diffusion.gamma_min,
            noise_scale=self.config.diffusion.noise_scale,
            step_scale=self.config.diffusion.step_scale,
            num_samples=self.config.diffusion.num_samples,
            capture_checkpoints=capture_checkpoints,
            check_nans=check_nans,
            step_callback=diffusion_callback,
            guidance_fn=guidance_fn,
        )
        coords = diffusion_samples["atom_positions"]
        diffusion_mask = diffusion_samples["mask"]
        # Extract diffusion checkpoints if captured
        diffusion_checkpoints = diffusion_samples.get("checkpoints") if capture_checkpoints else None

        # Force synchronization for accurate timing
        mx.eval(coords)
        metadata["diffusion_duration_seconds"] = time.time() - diffusion_start

        # NaN check after diffusion
        if check_nans:
            check_nan(coords, "diffusion.coords")

        # Capture diffusion checkpoints (per-step coords at 50, 100, 150, 200)
        if capture_checkpoints:
            # Use slicing to create independent copy (MLX arrays don't have .copy())
            checkpoints["diffusion_coords_final"] = coords[:]
            # Attach per-step diffusion checkpoints from DiffusionHead.sample()
            if diffusion_checkpoints is not None:
                checkpoints["diffusion_step_checkpoints"] = diffusion_checkpoints

        # Compute confidence scores for each sample
        num_samples = coords.shape[0]
        all_confidences = []

        # Confidence head (not compiled - takes GatherInfo which mx.compile can't handle)
        confidence_fn = self.confidence_head

        # Signal confidence computation start
        if confidence_callback is not None:
            confidence_callback("start")

        # Track Confidence timing
        confidence_start = time.time()

        for sample_idx in range(num_samples):
            sample_coords = coords[sample_idx]  # [num_residues, 37, 3]

            confidence = confidence_fn(
                dense_atom_positions=sample_coords,
                embeddings=diffusion_embeddings,
                seq_mask=seq_mask_1d,
                token_atoms_to_pseudo_beta=token_atoms_to_pseudo_beta,
                asym_id=asym_id_1d,
            )
            all_confidences.append(confidence)

        # Combine confidence scores
        # Stack across samples and squeeze out batch dimension (batch=1)
        # Each confidence has shape [batch, ...], stacking gives [num_samples, batch, ...]
        # We need [num_samples, ...] so squeeze axis=1
        combined_confidence = ConfidenceScores(
            plddt=mx.stack([c.plddt for c in all_confidences], axis=0)[:, 0],  # [samples, residues, atoms]
            pae=mx.stack([c.pae for c in all_confidences], axis=0)[:, 0],  # [samples, residues, residues]
            pde=mx.stack([c.pde for c in all_confidences], axis=0)[:, 0],  # [samples, residues, residues]
            ptm=mx.stack([c.ptm for c in all_confidences], axis=0)[:, 0],  # [samples]
            iptm=mx.stack([c.iptm for c in all_confidences], axis=0)[:, 0],  # [samples]
            # Propagate TM-adjusted PAE
            tm_pae_global=mx.stack([c.tm_pae_global for c in all_confidences], axis=0)[:, 0],  # [samples, residues, residues]
            tm_pae_interface=mx.stack([c.tm_pae_interface for c in all_confidences], axis=0)[:, 0],  # [samples, residues, residues]
        )

        # Force synchronization for accurate timing
        mx.eval(combined_confidence.plddt, combined_confidence.pae)
        metadata["confidence_duration_seconds"] = time.time() - confidence_start

        # Signal confidence computation end
        if confidence_callback is not None:
            confidence_callback("end")

        # NaN check after confidence (added per validation report)
        if check_nans:
            check_nan(combined_confidence.plddt, "confidence.plddt")
            check_nan(combined_confidence.pae, "confidence.pae")

        # Capture confidence checkpoint
        if capture_checkpoints:
            # Use slicing to create independent copy (MLX arrays don't have .copy())
            checkpoints["confidence_plddt"] = combined_confidence.plddt[:]
            checkpoints["confidence_pae"] = combined_confidence.pae[:]
            checkpoints["confidence_ptm"] = combined_confidence.ptm[:]

        # Create atom positions container
        atom_positions = AtomPositions(
            positions=coords,
            mask=diffusion_mask,
        )

        # Return embeddings if requested
        embeddings = None
        if self.config.return_embeddings:
            embeddings = Embeddings(
                single=single[0],  # Remove batch dim
                pair=pair[0],
                target_feat=target_feat[0],  # One-hot aatype [seq, 22]
            )

        metadata["end_time"] = time.time()
        metadata["duration_seconds"] = metadata["end_time"] - metadata["start_time"]
        metadata["num_residues"] = num_residues
        metadata["num_samples"] = num_samples

        # Include checkpoints in metadata for validation
        if capture_checkpoints:
            metadata["checkpoints"] = checkpoints

        return ModelResult(
            atom_positions=atom_positions,
            confidence=combined_confidence,
            embeddings=embeddings,
            metadata=metadata,
        )

    def _call_parity(
        self,
        batch: "ParityBatch",
        key: mx.array | None,
        override_embeddings: Embeddings | dict[str, mx.array] | None,
        diffusion_override: dict[str, mx.array] | None,
        capture_checkpoints: bool = False,
        check_nans: bool = True,
    ) -> ModelResult:
        """Parity-only path for using minimal Batch and overrides."""
        import time
        metadata: dict = {"start_time": time.time(), "parity_mode": True}

        if override_embeddings is None:
            raise ValueError("override_embeddings required for parity Batch")

        # Normalize embeddings input
        if isinstance(override_embeddings, Embeddings):
            embeddings_dict = {
                "single": override_embeddings.single,
                "pair": override_embeddings.pair,
                "target_feat": override_embeddings.target_feat,
            }
        else:
            embeddings_dict = override_embeddings

        # Ensure arrays are MLX arrays
        embeddings = {k: (v if isinstance(v, mx.array) else mx.array(v))
                      for k, v in embeddings_dict.items()}

        # Deterministic diffusion using precomputed intermediates if provided
        if diffusion_override is not None:
            positions_noisy_steps = mx.array(diffusion_override["positions_noisy_steps"])
            t_hat_steps = mx.array(diffusion_override["t_hat_steps"])
            noise_levels = mx.array(diffusion_override["noise_levels"])

            num_steps = positions_noisy_steps.shape[0]
            positions_out = None
            for step in range(num_steps):
                positions_noisy = positions_noisy_steps[step]
                t_hat = t_hat_steps[step]
                # End-to-end refs include a singleton sample dim; squeeze for DiffusionHead.
                if positions_noisy.ndim == 4 and positions_noisy.shape[0] == 1:
                    positions_noisy = positions_noisy[0]
                if t_hat.ndim == 1 and t_hat.shape[0] == 1:
                    t_hat = t_hat[0]
                positions_denoised = self.diffusion_head(
                    positions_noisy=positions_noisy,
                    noise_level=t_hat,
                    batch=batch,
                    embeddings=embeddings,
                    use_conditioning=True,
                )
                if t_hat.ndim == 0:
                    t_hat_b = t_hat.reshape((1, 1, 1))
                else:
                    t_hat_b = t_hat.reshape((t_hat.shape[0],) + (1, 1, 1))
                grad = (positions_noisy - positions_denoised) / t_hat_b
                d_t = noise_levels[step + 1] - t_hat
                if t_hat.ndim == 0:
                    d_t_b = d_t.reshape((1, 1, 1))
                else:
                    d_t_b = d_t.reshape((t_hat.shape[0],) + (1, 1, 1))
                positions_out = (
                    positions_noisy + self.config.diffusion.step_scale * d_t_b * grad
                )

            if positions_out is None:
                raise ValueError("No diffusion steps executed in parity override")
            coords = positions_out
        else:
            if key is None:
                warnings.warn(
                    "No seed provided; using default seed=42 for reproducibility",
                    stacklevel=2,
                )
                key = mx.random.key(42)

            def denoising_step(positions_noisy, noise_level):
                return self.diffusion_head(
                    positions_noisy=positions_noisy,
                    noise_level=noise_level,
                    batch=batch,
                    embeddings=embeddings,
                    use_conditioning=True,
                )

            diffusion_samples = self.diffusion_head.sample(
                denoising_step=denoising_step,
                batch=batch,
                key=key,
                num_steps=self.config.diffusion.num_steps,
                gamma_0=self.config.diffusion.gamma_0,
                gamma_min=self.config.diffusion.gamma_min,
                noise_scale=self.config.diffusion.noise_scale,
                step_scale=self.config.diffusion.step_scale,
                num_samples=self.config.diffusion.num_samples,
                check_nans=check_nans,
            )
            coords = diffusion_samples["atom_positions"]

        # Build mask and ensure sample dimension is present.
        atom_mask = batch.predicted_structure_info.atom_mask
        if coords.ndim == 3:
            coords = coords[None, ...]
            atom_mask = atom_mask[None, ...]
        else:
            atom_mask = mx.broadcast_to(atom_mask[None, ...], coords.shape[:-1])

        # Confidence head
        confidence = self.confidence_head(
            dense_atom_positions=coords,
            embeddings=embeddings,
            seq_mask=batch.token_features.mask,
            token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
            asym_id=batch.token_features.asym_id,
        )

        atom_positions = AtomPositions(
            positions=coords,
            mask=atom_mask,
        )

        embeddings_out = None
        if self.config.return_embeddings:
            embeddings_out = Embeddings(
                single=embeddings["single"],
                pair=embeddings["pair"],
                target_feat=embeddings["target_feat"],
            )

        metadata["end_time"] = time.time()
        metadata["duration_seconds"] = metadata["end_time"] - metadata["start_time"]

        return ModelResult(
            atom_positions=atom_positions,
            confidence=confidence,
            embeddings=embeddings_out,
            metadata=metadata,
        )

    def load_weights(self, weights_path: str | Path) -> None:
        """Load pre-trained weights from AF3 format.

        Args:
            weights_path: Path to weights file (zstd record format).

        Raises:
            FileNotFoundError: If weights file doesn't exist.
            WeightsNotFoundError: If required parameters missing.
            ShapeMismatchError: If parameter shapes don't match model.
        """
        from alphafold3_mlx.core.exceptions import WeightsNotFoundError, ShapeMismatchError

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise WeightsNotFoundError(str(weights_path))

        # Use Phase 1 weight loader
        from alphafold3_mlx.weights.loader import load_mlx_params

        # Load and convert weights using Phase 1 loader
        result = load_mlx_params(weights_path, validate=True)

        # Apply JAX-to-MLX parameter name mapping
        mlx_params = self._convert_jax_params(result.params)

        # Load parameters into model with shape validation
        loaded_keys, missing_keys = self.load_weights_dict(mlx_params, strict=True)

        # Avoid random template influence when AF3 template weights
        # are unavailable.
        if (
            "evoformer.template_embedding.output_linear.weight" not in loaded_keys
        ):
            self.evoformer.set_template_enabled(False)

        # Log loading statistics
        if missing_keys:
            # In debug mode, could log missing keys
            # For now, silently ignore extra params that don't match model
            pass

    def _convert_jax_params(self, jax_params: dict) -> dict:
        """Convert JAX parameter names to MLX format.

        Handles:
        - Haiku scope hierarchy (e.g., "diffuser/evoformer/trunk_pairformer/layer_0/...")
        - Layer indices (layer_0 -> layers[0])
        - Weight/bias naming ("/w" -> ".weight", "/b" -> ".bias")
        - Shape validation per parameter

        Args:
            jax_params: Nested dictionary of JAX parameters (Haiku format).

        Returns:
            Flat dictionary mapping MLX-compatible names to mx.array.

        Raises:
            ShapeMismatchError: If parameter shapes don't match expected model.
        """
        import re

        mlx_params: dict[str, mx.array] = {}

        def flatten_haiku_params(d: dict, prefix: str = "") -> dict:
            """Flatten Haiku's nested scope/name structure."""
            result = {}
            for key, value in d.items():
                new_key = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(flatten_haiku_params(value, new_key))
                else:
                    result[new_key] = value
            return result

        flat_jax = flatten_haiku_params(jax_params)
        # Normalize Haiku "~" scopes and stack-only pair logits projection path.
        flat_jax = {
            k.replace("diffuser/~/", "diffuser/").replace(
                "transformer/__layer_stack_with_per_layer/pair_logits_projection",
                "transformer/pair_logits_projection",
            ): v
            for k, v in flat_jax.items()
        }

        def ensure_confidence_head_built() -> None:
            """Build confidence head lazy layers before loading mapped weights."""
            if self.confidence_head._built:
                return

            target_feat_key = (
                "diffuser/confidence_head/~_embed_features/left_target_feat_project/weights"
            )
            plddt_key = "diffuser/confidence_head/plddt_logits/weights"

            target_feat_dim = None
            num_atoms = None

            target_feat_w = flat_jax.get(target_feat_key)
            if target_feat_w is not None and hasattr(target_feat_w, "shape"):
                target_feat_dim = int(target_feat_w.shape[0])

            plddt_w = flat_jax.get(plddt_key)
            if plddt_w is not None and hasattr(plddt_w, "shape"):
                # Shape: [seq_channel, num_atoms, num_bins]
                num_atoms = int(plddt_w.shape[1])

            if target_feat_dim is None:
                target_feat_dim = 447
            if num_atoms is None:
                num_atoms = 24

            self.confidence_head._build(
                target_feat_dim=target_feat_dim,
                num_atoms=num_atoms,
            )

        def to_mlx_array(param) -> mx.array:
            import numpy as np

            if isinstance(param, mx.array):
                return param
            np_arr = np.asarray(param)
            if str(np_arr.dtype) == "bfloat16":
                uint16_view = np_arr.view(np.uint16)
                return mx.array(uint16_view).view(mx.bfloat16)
            return mx.array(np_arr)

        def resolve_attr(path: str):
            module = self
            parts = path.split(".")
            for part in parts[:-1]:
                if part.isdigit():
                    if isinstance(module, (list, tuple)) and int(part) < len(module):
                        module = module[int(part)]
                    else:
                        return None
                else:
                    if not hasattr(module, part):
                        return None
                    module = getattr(module, part)
            if not hasattr(module, parts[-1]):
                return None
            return getattr(module, parts[-1])

        def add_param(path: str, param) -> None:
            """Add param if it matches an existing MLX array attribute."""
            import numpy as np

            existing = resolve_attr(path)
            if not isinstance(existing, mx.array):
                return
            arr = to_mlx_array(param)

            # Haiku OPM output weights are stored as [..., C, C, out]; MLX uses
            # flattened [..., C*C, out]. Flatten when this is an exact reshape.
            if arr.shape != existing.shape:
                if (
                    arr.ndim == 3
                    and existing.ndim == 2
                    and arr.shape[-1] == existing.shape[-1]
                    and int(np.prod(arr.shape[:-1])) == int(existing.shape[0])
                ):
                    arr = arr.reshape(existing.shape)

            if existing.shape != arr.shape:
                # Skip mismatched shapes to avoid strict loading failures.
                return
            mlx_params[path] = arr

        def ensure_diffusion_blocks_built() -> None:
            """Build diffusion transformer/cross-att blocks so weights can load."""
            c = self.diffusion_head.config
            # Diffusion transformer blocks
            self.diffusion_head.transformer._build_blocks(
                num_channels=c.per_token_channels,
                cond_dim=c.conditioning_seq_channel,
                pair_cond_dim=c.conditioning_pair_channel,
            )
            # Atom cross-attention transformer blocks
            self.diffusion_head.atom_cross_att_encoder.atom_transformer_encoder._build(
                num_channels=c.per_atom_channels,
                pair_cond_dim=c.per_atom_pair_channels,
            )
            self.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder._build(
                num_channels=c.per_atom_channels,
                pair_cond_dim=c.per_atom_pair_channels,
            )
            # Evoformer conditioning atom transformer blocks
            evo_c = self.evoformer_conditioning.config
            self.evoformer_conditioning.atom_transformer_encoder._build(
                num_channels=evo_c.per_atom_channels,
                pair_cond_dim=evo_c.per_atom_pair_channels,
            )

        def ensure_diffusion_conditioning_built() -> None:
            """Build diffusion conditioning projections with AF3 weight dimensions."""
            c = self.diffusion_head.config
            rel_dim = (2 * 32 + 2) * 2 + 1 + (2 * 2 + 2)

            pair_cond_dim = c.conditioning_pair_channel
            single_cond_dim = c.conditioning_seq_channel

            pair_norm = flat_jax.get("diffuser/diffusion_head/pair_cond_initial_norm/scale")
            if pair_norm is not None and hasattr(pair_norm, "shape") and len(pair_norm.shape) > 0:
                inferred = int(pair_norm.shape[0]) - rel_dim
                if inferred > 0:
                    pair_cond_dim = inferred

            single_norm = flat_jax.get("diffuser/diffusion_head/single_cond_initial_norm/scale")
            if single_norm is not None and hasattr(single_norm, "shape") and len(single_norm.shape) > 0:
                inferred = int(single_norm.shape[0])
                if inferred > 0:
                    single_cond_dim = inferred

            self.diffusion_head._build_conditioning(
                pair_cond_dim=pair_cond_dim,
                single_cond_dim=single_cond_dim,
            )

        def map_layer_norm(module_path: str, param_name: str) -> str | None:
            if param_name == "scale":
                return f"{module_path}.scale"
            if param_name == "offset":
                return f"{module_path}.offset"
            return None

        def map_linear(module_path: str, param_name: str) -> str | None:
            if param_name in ("weights", "w"):
                return f"{module_path}.weight"
            if param_name in ("bias", "b"):
                return f"{module_path}.bias"
            return None

        def map_transition(module_path: str, sub: str, param_name: str) -> str | None:
            if sub == "input_layer_norm":
                return map_layer_norm(f"{module_path}.norm", param_name)
            if sub == "transition1":
                return map_linear(f"{module_path}.glu.linear", param_name)
            if sub == "transition2":
                return map_linear(f"{module_path}.output_proj", param_name)
            return None

        def map_grid_attention(module_path: str, sub: str, param_name: str) -> str | None:
            if sub == "act_norm":
                return map_layer_norm(f"{module_path}.act_norm", param_name)
            if sub == "pair_bias_projection":
                return map_linear(f"{module_path}.pair_bias_proj", param_name)
            if sub == "q_projection":
                return map_linear(f"{module_path}.q_proj", param_name)
            if sub == "k_projection":
                return map_linear(f"{module_path}.k_proj", param_name)
            if sub == "v_projection":
                return map_linear(f"{module_path}.v_proj", param_name)
            if sub == "output_projection":
                return map_linear(f"{module_path}.o_proj", param_name)
            if sub == "gating_query":
                return map_linear(f"{module_path}.gate_proj", param_name)
            return None

        def map_triangle(module_path: str, sub: str, param_name: str) -> str | None:
            if sub == "left_norm_input":
                return map_layer_norm(f"{module_path}.input_norm", param_name)
            if sub == "center_norm":
                if param_name == "scale":
                    return f"{module_path}.center_norm_scale"
                if param_name == "offset":
                    return f"{module_path}.center_norm_offset"
                return None
            if sub == "projection":
                return map_linear(f"{module_path}.projection", param_name)
            if sub == "gate":
                return map_linear(f"{module_path}.gate", param_name)
            if sub == "output_projection":
                return map_linear(f"{module_path}.output_projection", param_name)
            if sub == "gating_linear":
                return map_linear(f"{module_path}.gating_linear", param_name)
            return None

        def map_outer_product(module_path: str, sub: str, param_name: str) -> str | None:
            if sub == "layer_norm_input":
                return map_layer_norm(f"{module_path}.norm", param_name)
            if sub == "left_projection":
                return map_linear(f"{module_path}.left_proj", param_name)
            if sub == "right_projection":
                return map_linear(f"{module_path}.right_proj", param_name)
            if sub == "output_w":
                return map_linear(f"{module_path}.output_proj", "weights")
            if sub == "output_b":
                return map_linear(f"{module_path}.output_proj", "bias")
            return None

        def map_single_attention(module_path: str, sub: str, param_name: str) -> str | None:
            if sub in ("layer_norm", "act_norm"):
                return map_layer_norm(f"{module_path}.act_norm", param_name)
            if sub == "q_projection":
                return map_linear(f"{module_path}.q_proj", param_name)
            if sub == "k_projection":
                return map_linear(f"{module_path}.k_proj", param_name)
            if sub == "v_projection":
                return map_linear(f"{module_path}.v_proj", param_name)
            if sub == "gating_query":
                return map_linear(f"{module_path}.gate_proj", param_name)
            if sub == "transition2":
                return map_linear(f"{module_path}.o_proj", param_name)
            return None

        def map_diffusion_transformer_rel_path(rel_path: str) -> str | None:
            parts = rel_path.split("/")
            head = parts[0]
            param_name = parts[-1] if len(parts) > 1 else None

            if head.startswith("transformerffw_"):
                sub = head[len("transformerffw_"):]
                if sub == "transition1":
                    return map_linear("transition.transition1", param_name)
                if sub == "transition2":
                    return map_linear("transition.adaptive_zero.transition2", param_name)
                if sub == "adaptive_zero_cond":
                    return map_linear("transition.adaptive_zero.adaptive_zero_cond", param_name)
                if sub == "single_cond_layer_norm":
                    return map_layer_norm("transition.adaptive_norm.single_cond_layer_norm", param_name)
                if sub == "single_cond_scale":
                    return map_linear("transition.adaptive_norm.single_cond_scale", param_name)
                if sub == "single_cond_bias":
                    return map_linear("transition.adaptive_norm.single_cond_bias", param_name)
                return None

            if head.startswith("transformer"):
                sub = head[len("transformer"):]
                if sub == "q_projection":
                    return map_linear("self_attention.q_projection", param_name)
                if sub == "k_projection":
                    return map_linear("self_attention.k_projection", param_name)
                if sub == "v_projection":
                    return map_linear("self_attention.v_projection", param_name)
                if sub == "gating_query":
                    return map_linear("self_attention.gating_query", param_name)
                if sub == "transition2":
                    return map_linear("self_attention.adaptive_zero.transition2", param_name)
                if sub == "adaptive_zero_cond":
                    return map_linear("self_attention.adaptive_zero.adaptive_zero_cond", param_name)
                if sub == "single_cond_layer_norm":
                    return map_layer_norm("self_attention.adaptive_norm.single_cond_layer_norm", param_name)
                if sub == "single_cond_scale":
                    return map_linear("self_attention.adaptive_norm.single_cond_scale", param_name)
                if sub == "single_cond_bias":
                    return map_linear("self_attention.adaptive_norm.single_cond_bias", param_name)
            return None

        def map_atom_transformer_rel_path(rel_path: str, prefix: str) -> str | None:
            parts = rel_path.split("/")
            head = parts[0]
            param_name = parts[-1] if len(parts) > 1 else None
            if not head.startswith(prefix):
                return None
            sub = head[len(prefix):]

            if sub.startswith("ffw_"):
                sub = sub[len("ffw_"):]
                if sub == "transition1":
                    return map_linear("transition.transition1", param_name)
                if sub == "transition2":
                    return map_linear("transition.adaptive_zero.transition2", param_name)
                if sub == "adaptive_zero_cond":
                    return map_linear("transition.adaptive_zero.adaptive_zero_cond", param_name)
                if sub == "single_cond_layer_norm":
                    return map_layer_norm("transition.adaptive_norm.single_cond_layer_norm", param_name)
                if sub == "single_cond_scale":
                    return map_linear("transition.adaptive_norm.single_cond_scale", param_name)
                if sub == "single_cond_bias":
                    return map_linear("transition.adaptive_norm.single_cond_bias", param_name)
                return None

            if sub.startswith("qsingle_cond_"):
                sub = sub[len("qsingle_cond_"):]
                if sub == "layer_norm":
                    return map_layer_norm("cross_attention.adaptive_norm_q.single_cond_layer_norm", param_name)
                if sub == "scale":
                    return map_linear("cross_attention.adaptive_norm_q.single_cond_scale", param_name)
                if sub == "bias":
                    return map_linear("cross_attention.adaptive_norm_q.single_cond_bias", param_name)
                return None

            if sub.startswith("ksingle_cond_"):
                sub = sub[len("ksingle_cond_"):]
                if sub == "layer_norm":
                    return map_layer_norm("cross_attention.adaptive_norm_k.single_cond_layer_norm", param_name)
                if sub == "scale":
                    return map_linear("cross_attention.adaptive_norm_k.single_cond_scale", param_name)
                if sub == "bias":
                    return map_linear("cross_attention.adaptive_norm_k.single_cond_bias", param_name)
                return None

            if sub == "q_projection":
                return map_linear("cross_attention.q_projection", param_name)
            if sub == "k_projection":
                return map_linear("cross_attention.k_projection", param_name)
            if sub == "v_projection":
                return map_linear("cross_attention.v_projection", param_name)
            if sub == "gating_query":
                return map_linear("cross_attention.gating_query", param_name)
            if sub == "transition2":
                return map_linear("cross_attention.adaptive_zero.transition2", param_name)
            if sub == "adaptive_zero_cond":
                return map_linear("cross_attention.adaptive_zero.adaptive_zero_cond", param_name)
            return None

        def map_pairformer_rel_path(rel_path: str) -> str | None:
            parts = rel_path.split("/")
            head = parts[0]
            param_name = parts[-1] if len(parts) > 1 else None

            if head in ("pair_attention1", "pair_attention2"):
                module = "pair_attention_row" if head == "pair_attention1" else "pair_attention_col"
                return map_grid_attention(module, parts[1], param_name)

            if head in ("triangle_multiplication_outgoing", "triangle_multiplication_incoming"):
                module = "triangle_mult_outgoing" if head.endswith("outgoing") else "triangle_mult_incoming"
                return map_triangle(module, parts[1], param_name)

            if head == "pair_transition":
                return map_transition("pair_transition", parts[1], param_name)

            if head == "single_pair_logits_norm":
                return map_layer_norm("pair_logits_norm", param_name)

            if head == "single_pair_logits_projection":
                return map_linear("pair_logits_proj", param_name)

            if head.startswith("single_attention_"):
                sub = head[len("single_attention_"):]
                return map_single_attention("single_attention", sub, param_name)

            if head == "single_transition":
                return map_transition("single_transition", parts[1], param_name)

            return None

        def map_msa_rel_path(rel_path: str) -> str | None:
            parts = rel_path.split("/")
            head = parts[0]
            param_name = parts[-1] if len(parts) > 1 else None

            if head == "msa_attention1":
                sub = parts[1]
                if sub == "act_norm":
                    return map_layer_norm("msa_row_attention.layer_norm_msa", param_name)
                if sub == "pair_norm":
                    return map_layer_norm("msa_row_attention.layer_norm_pair", param_name)
                if sub == "pair_logits":
                    return map_linear("msa_row_attention.pair_bias_proj", param_name)
                if sub == "v_projection":
                    return map_linear("msa_row_attention.v_proj", param_name)
                if sub == "output_projection":
                    return map_linear("msa_row_attention.o_proj", param_name)
                if sub == "gating_query":
                    return map_linear("msa_row_attention.gate_proj", param_name)
                return None

            if head == "msa_transition":
                sub = parts[1]
                if sub == "input_layer_norm":
                    return map_layer_norm("msa_transition.layer_norm", param_name)
                if sub == "transition1":
                    return map_linear("msa_transition.up_proj", param_name)
                if sub == "transition2":
                    return map_linear("msa_transition.down_proj", param_name)
                return None

            if head == "outer_product_mean":
                return map_outer_product("outer_product_mean", parts[1], param_name)

            if head in ("triangle_multiplication_outgoing", "triangle_multiplication_incoming"):
                module = "tri_mul_out" if head.endswith("outgoing") else "tri_mul_in"
                return map_triangle(module, parts[1], param_name)

            if head in ("pair_attention1", "pair_attention2"):
                module = "pair_row_attention" if head == "pair_attention1" else "pair_col_attention"
                return map_grid_attention(module, parts[1], param_name)

            if head == "pair_transition":
                return map_transition("pair_transition", parts[1], param_name)

            return None

        # --- Handle stacked weights for Evoformer/Confidence stacks ---
        used_keys: set[str] = set()

        def expand_stack(prefix: str, num_layers: int, base_path: str, mapper) -> None:
            import numpy as np

            for jax_name, param in flat_jax.items():
                if not jax_name.startswith(prefix + "/"):
                    continue
                used_keys.add(jax_name)

                if isinstance(param, mx.array):
                    if param.shape[0] == num_layers:
                        per_layer = [param[i] for i in range(num_layers)]
                    elif len(param.shape) >= 2 and int(param.shape[0]) * int(param.shape[1]) == num_layers:
                        per_layer = [
                            param[i, j]
                            for i in range(int(param.shape[0]))
                            for j in range(int(param.shape[1]))
                        ]
                    else:
                        continue
                else:
                    np_arr = np.asarray(param)
                    if np_arr.shape[0] == num_layers:
                        per_layer = [np_arr[i] for i in range(num_layers)]
                    elif np_arr.ndim >= 2 and int(np_arr.shape[0]) * int(np_arr.shape[1]) == num_layers:
                        per_layer = [
                            np_arr[i, j]
                            for i in range(int(np_arr.shape[0]))
                            for j in range(int(np_arr.shape[1]))
                        ]
                    else:
                        continue
                rel_path = jax_name[len(prefix) + 1 :]
                mapped_rel = mapper(rel_path)
                if mapped_rel is None:
                    continue
                for layer_idx in range(num_layers):
                    full_path = f"{base_path}.{layer_idx}.{mapped_rel}"
                    add_param(full_path, per_layer[layer_idx])

        # --- Template embedding weight mapping (JAX AF3 parity) ---
        _te_prefix = "diffuser/evoformer/template_embedding"
        _ste_prefix = f"{_te_prefix}/single_template_embedding"

        # Direct (non-stacked) template weights
        _template_direct_map = {
            # TemplateEmbedding.output_linear
            f"{_te_prefix}/output_linear/weights": "evoformer.template_embedding.output_linear.weight",
            # SingleTemplateEmbedding.query_embedding_norm
            f"{_ste_prefix}/query_embedding_norm/scale": "evoformer.template_embedding.single_template_embedding.query_embedding_norm.scale",
            f"{_ste_prefix}/query_embedding_norm/offset": "evoformer.template_embedding.single_template_embedding.query_embedding_norm.offset",
            # SingleTemplateEmbedding.output_layer_norm
            f"{_ste_prefix}/output_layer_norm/scale": "evoformer.template_embedding.single_template_embedding.output_layer_norm.scale",
            f"{_ste_prefix}/output_layer_norm/offset": "evoformer.template_embedding.single_template_embedding.output_layer_norm.offset",
            # template_pair_embedding_0 (distogram, Linear)
            f"{_ste_prefix}/template_pair_embedding_0/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_0.weight",
            # template_pair_embedding_1 (pseudo_beta_mask_2d, BroadcastProjection)
            f"{_ste_prefix}/template_pair_embedding_1/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_1.weight",
            # template_pair_embedding_2 (aatype row, Linear)
            f"{_ste_prefix}/template_pair_embedding_2/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_2.weight",
            # template_pair_embedding_3 (aatype col, Linear)
            f"{_ste_prefix}/template_pair_embedding_3/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_3.weight",
            # template_pair_embedding_4 (unit_vector.x, BroadcastProjection)
            f"{_ste_prefix}/template_pair_embedding_4/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_4.weight",
            # template_pair_embedding_5 (unit_vector.y, BroadcastProjection)
            f"{_ste_prefix}/template_pair_embedding_5/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_5.weight",
            # template_pair_embedding_6 (unit_vector.z, BroadcastProjection)
            f"{_ste_prefix}/template_pair_embedding_6/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_6.weight",
            # template_pair_embedding_7 (backbone_mask_2d, BroadcastProjection)
            f"{_ste_prefix}/template_pair_embedding_7/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_7.weight",
            # template_pair_embedding_8 (query_embedding, Linear)
            f"{_ste_prefix}/template_pair_embedding_8/weights": "evoformer.template_embedding.single_template_embedding.template_pair_embedding_8.weight",
        }

        for jax_name, mlx_name in _template_direct_map.items():
            if jax_name in flat_jax:
                used_keys.add(jax_name)
                add_param(mlx_name, flat_jax[jax_name])

        def map_template_pairformer_rel_path(rel_path: str) -> str | None:
            """Map template PairFormer relative path (with_single=False)."""
            parts = rel_path.split("/")
            head = parts[0]
            param_name = parts[-1] if len(parts) > 1 else None

            if head in ("pair_attention1", "pair_attention2"):
                module = "pair_attention_row" if head == "pair_attention1" else "pair_attention_col"
                return map_grid_attention(module, parts[1], param_name)

            if head in ("triangle_multiplication_outgoing", "triangle_multiplication_incoming"):
                module = "triangle_mult_outgoing" if head.endswith("outgoing") else "triangle_mult_incoming"
                return map_triangle(module, parts[1], param_name)

            if head == "pair_transition":
                return map_transition("pair_transition", parts[1], param_name)

            return None

        # Template 2-layer PairFormer stack
        expand_stack(
            prefix=f"{_ste_prefix}/__layer_stack_no_per_layer/template_embedding_iteration",
            num_layers=len(self.evoformer.template_embedding.single_template_embedding.pairformer_layers),
            base_path="evoformer.template_embedding.single_template_embedding.pairformer_layers",
            mapper=map_template_pairformer_rel_path,
        )

        if self.evoformer.msa_layers:
            expand_stack(
                prefix="diffuser/evoformer/__layer_stack_no_per_layer/msa_stack",
                num_layers=len(self.evoformer.msa_layers),
                base_path="evoformer.msa_layers",
                mapper=map_msa_rel_path,
            )

        expand_stack(
            prefix="diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer",
            num_layers=len(self.evoformer.pairformer_layers),
            base_path="evoformer.pairformer_layers",
            mapper=map_pairformer_rel_path,
        )

        expand_stack(
            prefix="diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer",
            num_layers=len(self.confidence_head.pairformer_layers),
            base_path="confidence_head.pairformer_layers",
            mapper=map_pairformer_rel_path,
        )

        # Diffusion transformer / atom cross-attention stacks
        ensure_diffusion_blocks_built()
        ensure_diffusion_conditioning_built()
        ensure_confidence_head_built()

        expand_stack(
            prefix="diffuser/diffusion_head/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer",
            num_layers=len(self.diffusion_head.transformer.blocks),
            base_path="diffusion_head.transformer.blocks",
            mapper=map_diffusion_transformer_rel_path,
        )

        expand_stack(
            prefix="diffuser/diffusion_head/diffusion_atom_transformer_encoder/__layer_stack_with_per_layer",
            num_layers=len(self.diffusion_head.atom_cross_att_encoder.atom_transformer_encoder.blocks),
            base_path="diffusion_head.atom_cross_att_encoder.atom_transformer_encoder.blocks",
            mapper=lambda rel: map_atom_transformer_rel_path(
                rel, "diffusion_atom_transformer_encoder"
            ),
        )

        expand_stack(
            prefix="diffuser/diffusion_head/diffusion_atom_transformer_decoder/__layer_stack_with_per_layer",
            num_layers=len(self.diffusion_head.atom_cross_att_decoder.atom_transformer_decoder.blocks),
            base_path="diffusion_head.atom_cross_att_decoder.atom_transformer_decoder.blocks",
            mapper=lambda rel: map_atom_transformer_rel_path(
                rel, "diffusion_atom_transformer_decoder"
            ),
        )

        expand_stack(
            prefix="diffuser/evoformer_conditioning_atom_transformer_encoder/__layer_stack_with_per_layer",
            num_layers=len(self.evoformer_conditioning.atom_transformer_encoder.blocks),
            base_path="evoformer_conditioning.atom_transformer_encoder.blocks",
            mapper=lambda rel: map_atom_transformer_rel_path(
                rel, "evoformer_conditioning_atom_transformer_encoder"
            ),
        )

        # Layer index pattern: layer_0, layer_1, etc.
        layer_pattern = re.compile(r"layer_(\d+)")

        for jax_name, param in flat_jax.items():
            if jax_name in used_keys:
                continue

            # Diffusion transformer has one pair-logits projection per super-block.
            if jax_name == "diffuser/diffusion_head/transformer/pair_logits_projection/weights":
                arr = to_mlx_array(param)
                projections = self.diffusion_head.transformer.pair_logits_projections
                if arr.ndim == 4 and arr.shape[0] == len(projections):
                    for super_idx in range(len(projections)):
                        add_param(
                            f"diffusion_head.transformer.pair_logits_projections.{super_idx}.weight",
                            arr[super_idx],
                        )
                continue

            # Start with the JAX path
            mlx_name = jax_name

            # Step 1: Apply module name mappings
            for jax_pattern, mlx_pattern in JAX_TO_MLX_MODULE_MAP.items():
                mlx_name = mlx_name.replace(jax_pattern, mlx_pattern)

            # Step 2: Convert layer indices (layer_0 -> layers.0)
            def replace_layer_index(match: re.Match) -> str:
                idx = match.group(1)
                return f"layers.{idx}"

            mlx_name = layer_pattern.sub(replace_layer_index, mlx_name)

            # Step 3: Convert path separators and weight/bias suffixes
            # Split into path parts
            parts = mlx_name.split("/")

            # Last part is the parameter name (w, b, scale, offset)
            if len(parts) >= 2:
                param_name = parts[-1]
                module_path = parts[:-1]

                # Map parameter name
                mlx_param_name = JAX_PARAM_NAME_MAP.get(param_name, param_name)

                # Construct MLX path
                mlx_name = ".".join(module_path) + "." + mlx_param_name
            else:
                # Single-part name, just convert
                mlx_name = mlx_name.replace("/", ".")

            # Step 4: Convert to MLX array
            if hasattr(param, "__array__") or isinstance(param, mx.array):
                add_param(mlx_name, param)
            else:
                mlx_params[mlx_name] = param

        return mlx_params

    def load_weights_dict(self, params: dict, strict: bool = True) -> tuple[list[str], list[str]]:
        """Load weights from a flat parameter dictionary with shape validation.

        Args:
            params: Dictionary mapping parameter names to arrays.
            strict: If True, raise error on mismatched or missing params.

        Returns:
            Tuple of (loaded_keys, missing_keys).

        Raises:
            ShapeMismatchError: If shapes don't match and strict=True.
        """
        from alphafold3_mlx.core.exceptions import ShapeMismatchError

        loaded_keys: list[str] = []
        missing_keys: list[str] = []
        shape_mismatches: list[tuple[str, tuple, tuple]] = []

        for name, param in params.items():
            parts = name.split(".")
            module = self
            found = True

            # Navigate to the target module
            for i, part in enumerate(parts[:-1]):
                # Handle list indexing (e.g., "layers.0" -> layers[0])
                if part.isdigit():
                    # Previous part should be a list attribute
                    idx = int(part)
                    if isinstance(module, (list, tuple)) and idx < len(module):
                        module = module[idx]
                    else:
                        found = False
                        break
                elif hasattr(module, part):
                    attr = getattr(module, part)
                    # Check if next part is an index
                    if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                        module = attr  # Keep as list for indexing
                    else:
                        module = attr
                else:
                    found = False
                    break

            if not found:
                missing_keys.append(name)
                continue

            # Set the final attribute
            param_name = parts[-1]

            # Handle the case where module is actually a list
            if isinstance(module, (list, tuple)):
                missing_keys.append(name)
                continue

            if hasattr(module, param_name):
                existing = getattr(module, param_name)
                if isinstance(existing, mx.array):
                    # Validate shape
                    if existing.shape != param.shape:
                        shape_mismatches.append(
                            (name, tuple(existing.shape), tuple(param.shape))
                        )
                        if strict:
                            continue  # Don't load mismatched params
                setattr(module, param_name, param)
                loaded_keys.append(name)
            else:
                missing_keys.append(name)

        # Report shape mismatches
        if shape_mismatches and strict:
            mismatch_details = "\n".join(
                f"  {path}: model expects {expected}, got {actual}"
                for path, expected, actual in shape_mismatches[:10]
            )
            total = len(shape_mismatches)
            msg = f"Weight loading found {total} shape mismatch(es):\n{mismatch_details}"
            if total > 10:
                msg += f"\n  ... and {total - 10} more"
            raise ShapeMismatchError(msg, mismatches=shape_mismatches)

        return loaded_keys, missing_keys

    def compile(self) -> None:
        """Compile model for improved performance.

        Applies mx.compile to performance-critical functions.
        Call after load_weights and before inference.

        MLX compilation captures model state and compiles forward passes.
        This provides significant speedups for repeated inference.

        Compilation Coverage:

        1. Evoformer: FULLY COMPILED
           - Takes pure array inputs (single, pair, masks)
           - Called num_recycles times per inference
           - Compilation provides significant speedup

        2. DiffusionHead.denoise_step_arrays: COMPILED
           - Array-only denoising step (embedding + transformer + norm)
           - Called num_steps × num_samples times per inference
           - This is the denoising step's computational core
           - DiffusionHead.__call__ performs Batch-dependent operations
             (conditioning, encoder, decoder) then calls compiled step

        3. ConfidenceHead.confidence_forward_arrays: COMPILED
           - Array-only confidence computation (feature embedding,
             PairFormer stack, all metric computations)
           - ConfidenceHead.__call__ performs layout_convert (GatherInfo-
             dependent) then calls compiled forward
           - Only called num_samples times, but compilation still beneficial
             for larger sequences

        Note on denoising step compilation:
        The "denoising step" in DiffusionHead.sample() calls DiffusionHead.__call__
        which uses Batch objects for:
        - _conditioning(): creates relative features from batch.token_features
        - atom_cross_att_encoder(): uses batch for atom layout
        - atom_cross_att_decoder(): uses batch for atom layout

        The array-only denoise_step_arrays is the performance-critical core
        and IS compiled. The surrounding Batch-dependent logic cannot be compiled.
        """
        if self._compiled:
            return

        if not self.config.global_config.use_compile:
            # Still call compile() methods to set _compiled flags
            # (they will be no-ops internally since use_compile=False)
            self.diffusion_head.compile()
            self.confidence_head.compile()
            self._compiled = True
            return

        from functools import partial

        # Compile Evoformer forward pass
        # Capture model state for compilation
        evoformer_state = [self.evoformer.state]

        # Create compiled Evoformer function
        @partial(mx.compile, inputs=evoformer_state, outputs=evoformer_state)
        def compiled_evoformer(single, pair, residue_index, asym_id, seq_mask, pair_mask, **kwargs):
            return self.evoformer(
                single=single,
                pair=pair,
                residue_index=residue_index,
                asym_id=asym_id,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                **kwargs,
            )

        self._compiled_evoformer = compiled_evoformer

        # Compile DiffusionHead's internal Transformer
        # The full DiffusionHead.__call__ takes Batch objects that mx.compile
        # cannot handle, but the internal Transformer takes pure arrays.
        # Compiling it provides speedup since it's called many times during
        # diffusion (num_steps × num_samples iterations).
        self.diffusion_head.compile()

        # Compile ConfidenceHead's array-only forward path
        # The full ConfidenceHead.__call__ takes GatherInfo for layout_convert,
        # but confidence_forward_arrays takes pure arrays and is compiled.
        self.confidence_head.compile()

        self._compiled = True

    def set_precision(self, precision: str) -> None:
        """Set model precision mode.

        Converts model parameters to the specified precision.
        Supports 'float32', 'float16', and 'bfloat16'.

        Note: bfloat16 requires M3+ chip for native support.

        Args:
            precision: Target precision - 'float32', 'float16', or 'bfloat16'.

        Raises:
            ValueError: If precision is not valid.
        """
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"precision must be 'float32', 'float16', or 'bfloat16', "
                f"got '{precision}'"
            )

        # Map precision string to MLX dtype
        dtype_map = {
            "float32": mx.float32,
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
        }
        target_dtype = dtype_map[precision]

        # Convert all parameters
        def convert_params(module: nn.Module) -> None:
            """Recursively convert module parameters."""
            children = module.children()
            for name, child in children.items():
                if isinstance(child, list):
                    # Handle lists of modules (e.g., pairformer_layers)
                    for item in child:
                        if isinstance(item, nn.Module):
                            convert_params(item)
                elif isinstance(child, nn.Module):
                    convert_params(child)

            # Convert leaf parameters
            for name, param in vars(module).items():
                if isinstance(param, mx.array) and param.dtype in (mx.float32, mx.float16, mx.bfloat16):
                    setattr(module, name, param.astype(target_dtype))

        convert_params(self)

        # Update config
        # Note: GlobalConfig is frozen, so we recreate it
        from dataclasses import replace
        new_global = replace(self.config.global_config, precision=precision)
        object.__setattr__(self.config, "global_config", new_global)

    @property
    def precision(self) -> str:
        """Return current precision mode."""
        return self.config.global_config.precision

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path | str,
        config: ModelConfig | None = None,
    ) -> "Model":
        """Load a pre-trained model from weights directory.

        Combines model instantiation and weight loading into a single
        classmethod for convenience. Uses the Phase 1 weight loader.

        Args:
            model_dir: Path to directory containing weight files (af3.bin.zst).
            config: Optional model configuration. Uses defaults if None.

        Returns:
            Model instance with loaded weights, ready for inference.

        Raises:
            WeightsNotFoundError: If weights not found at model_dir.
            CorruptedWeightsError: If weight files are corrupted.
            ShapeMismatchError: If parameter shapes don't match model.

        Example:
            >>> from pathlib import Path
            >>> from alphafold3_mlx.model import Model
            >>>
            >>> model = Model.from_pretrained(Path("weights/model"))
            >>> result = model(batch)
        """
        from alphafold3_mlx.weights import WeightsNotFoundError

        model_dir = Path(model_dir)

        # Validate weights directory exists before creating model (C-07)
        if not _weights_exist(model_dir):
            raise WeightsNotFoundError(
                f"Model weights not found at: {model_dir}. "
                "Expected directory containing af3.bin.zst (or af3.bin, sharded af3.*.bin.zst).\n"
                "  Download weights from: https://github.com/google-deepmind/alphafold3\n"
                "  Place in: ~/.alphafold3/weights/model/ or set AF3_WEIGHTS_DIR"
            )

        # Create model instance
        model = cls(config)

        # Load weights using Phase 1 loader via load_weights
        model.load_weights(model_dir)

        return model
