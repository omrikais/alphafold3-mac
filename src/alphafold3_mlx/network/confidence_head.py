"""Confidence head matching AF3 JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from alphafold3_mlx.atom_layout import convert as layout_convert, GatherInfo
from alphafold3_mlx.modules import Linear, LayerNorm
from alphafold3_mlx.network.template_modules import DistogramFeaturesConfig, dgram_from_positions
from alphafold3_mlx.network.pairformer import PairFormerIteration

if TYPE_CHECKING:
    from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig
    from alphafold3_mlx.core.entities import ConfidenceScores


def _safe_norm(x: mx.array, keepdims: bool, axis, eps=1e-8) -> mx.array:
    return mx.sqrt(eps + mx.sum(x * x, axis=axis, keepdims=keepdims))


class ConfidenceHead(nn.Module):
    """AF3 confidence head.

    Compilation Note:
        ConfidenceHead CAN be compiled via mx.compile by separating the
        GatherInfo-dependent layout_convert() from the core computation.

        The forward pass is structured as:
        1. __call__: Performs layout_convert (GatherInfo-dependent) to get
           pseudo-beta positions, then delegates to confidence_forward_arrays.
        2. confidence_forward_arrays: Array-only computation suitable for
           mx.compile, including feature embedding, PairFormer stack, and
           all confidence metric computations.

        Call compile() to enable the compiled path. When compiled:
        - __call__() performs layout_convert() then calls compiled function
        - Performance benefit is modest since ConfidenceHead is called once
          per sample (not iteratively like diffusion)
    """

    # Precision string to MLX dtype mapping
    _PRECISION_TO_DTYPE = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }

    def __init__(
        self,
        config: "ConfidenceConfig | None" = None,
        global_config: "GlobalConfig | None" = None,
        seq_channel: int = 384,
        pair_channel: int = 128,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        from alphafold3_mlx.core.config import ConfidenceConfig, GlobalConfig

        self.config = config or ConfidenceConfig()
        self.global_config = global_config or GlobalConfig()

        c = self.config
        self.seq_channel = seq_channel
        self.pair_channel = pair_channel

        # Distogram features config
        self.dgram_features = DistogramFeaturesConfig(
            min_bin=c.dgram_min_bin, max_bin=c.dgram_max_bin, num_bins=c.dgram_num_bins
        )

        # PairFormer refinement stack
        self.pairformer_layers = [
            PairFormerIteration(
                seq_channel=seq_channel,
                pair_channel=pair_channel,
                num_attention_heads=num_attention_heads,
                attention_key_dim=None,
                intermediate_factor=4,
                with_single=True,
            )
            for _ in range(c.num_pairformer_layers)
        ]

        # Feature embeddings
        self.left_target_feat_project = None
        self.right_target_feat_project = None
        self.distogram_feat_project = None

        # Logits layers
        self.logits_ln = LayerNorm(pair_channel)
        self.left_half_distance_logits = Linear(
            c.num_bins,
            input_dims=pair_channel,
            use_bias=False,
            initializer=self.global_config.final_init,
        )
        self.pae_logits_ln = LayerNorm(pair_channel)
        self.pae_logits = Linear(
            c.num_pae_bins,
            input_dims=pair_channel,
            use_bias=False,
            initializer=self.global_config.final_init,
        )

        # pLDDT logits
        self.plddt_logits_ln = LayerNorm(seq_channel)
        self.plddt_logits = None

        # Experimentally resolved logits
        self.experimentally_resolved_ln = LayerNorm(seq_channel)
        self.experimentally_resolved_logits = None

        self._built = False
        self._compiled = False
        self._compiled_confidence_forward = None

    def compile(self) -> None:
        """Compile the array-only confidence forward path.

        This method compiles confidence_forward_arrays(), which encapsulates
        the core confidence computation after GatherInfo/layout_convert has
        been performed externally:
        - Feature embedding from precomputed pseudo-beta positions
        - PairFormer refinement stack
        - Distance error, PAE, pLDDT, and TM-score computations

        The __call__ method performs layout_convert() outside the compiled
        function to convert dense atom positions to pseudo-beta positions,
        then delegates to confidence_forward_arrays() for the core computation.

        Note: ConfidenceHead is only called once per sample (not iteratively
        like diffusion), so compilation benefit is modest but still worthwhile
        for larger sequences.
        """
        if self._compiled:
            return

        if not self.global_config.use_compile:
            self._compiled = True
            return

        from functools import partial

        # Collect state from all submodules used in confidence_forward_arrays
        # Note: pairformer_layers is a list, need to collect each layer's state
        pairformer_states = [layer.state for layer in self.pairformer_layers]
        confidence_state = [
            self.left_target_feat_project.state if self.left_target_feat_project else {},
            self.right_target_feat_project.state if self.right_target_feat_project else {},
            self.distogram_feat_project.state if self.distogram_feat_project else {},
            self.logits_ln.state,
            self.left_half_distance_logits.state,
            self.pae_logits_ln.state,
            self.pae_logits.state,
            self.plddt_logits_ln.state,
            self.plddt_logits.state if self.plddt_logits else {},
            *pairformer_states,
        ]

        @partial(mx.compile, inputs=confidence_state, outputs=confidence_state)
        def compiled_confidence_forward(
            pseudo_beta_positions, pair_act, single_act, target_feat, seq_mask, pair_mask, asym_id
        ):
            return self.confidence_forward_arrays(
                pseudo_beta_positions=pseudo_beta_positions,
                pair_act=pair_act,
                single_act=single_act,
                target_feat=target_feat,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                asym_id=asym_id,
            )

        self._compiled_confidence_forward = compiled_confidence_forward
        self._compiled = True

    def _build(self, target_feat_dim: int, num_atoms: int) -> None:
        if self._built:
            return
        # rebuild projections with correct input dims
        self.left_target_feat_project = Linear(
            self.pair_channel,
            input_dims=target_feat_dim,
            use_bias=False,
        )
        self.right_target_feat_project = Linear(
            self.pair_channel,
            input_dims=target_feat_dim,
            use_bias=False,
        )
        self.distogram_feat_project = Linear(
            self.pair_channel,
            input_dims=self.config.dgram_num_bins,
            use_bias=False,
        )
        self.plddt_logits = Linear(
            (num_atoms, self.config.num_plddt_bins),
            input_dims=self.seq_channel,
            use_bias=False,
            initializer=self.global_config.final_init,
        )
        self.experimentally_resolved_logits = Linear(
            (num_atoms, 2),
            input_dims=self.seq_channel,
            use_bias=False,
            initializer=self.global_config.final_init,
        )
        self._built = True

    def _embed_features_arrays(
        self,
        pseudo_beta_positions: mx.array,
        pair_mask: mx.array,
        pair_act: mx.array,
        target_feat: mx.array,
    ) -> mx.array:
        """Array-only feature embedding (no GatherInfo/layout_convert).

        Args:
            pseudo_beta_positions: Precomputed pseudo-beta positions [seq, 3].
            pair_mask: [seq, seq] pair mask.
            pair_act: [seq, seq, pair_channel] pair activations.
            target_feat: [seq, feat_dim] target features.

        Returns:
            Embedded pair features [seq, seq, pair_channel].
        """
        out = self.left_target_feat_project(target_feat).astype(pair_act.dtype)
        out = out + self.right_target_feat_project(target_feat).astype(pair_act.dtype)[:, None]

        dgram = dgram_from_positions(pseudo_beta_positions, self.dgram_features)
        dgram = dgram * pair_mask[..., None]

        out = out + self.distogram_feat_project(dgram.astype(pair_act.dtype))
        return out

    def _embed_features(
        self,
        dense_atom_positions: mx.array,
        token_atoms_to_pseudo_beta: GatherInfo,
        pair_mask: mx.array,
        pair_act: mx.array,
        target_feat: mx.array,
    ) -> mx.array:
        """Feature embedding with layout conversion (uses GatherInfo)."""
        positions = layout_convert(
            token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2)
        )
        return self._embed_features_arrays(positions, pair_mask, pair_act, target_feat)

    def confidence_forward_arrays(
        self,
        pseudo_beta_positions: mx.array,
        pair_act: mx.array,
        single_act: mx.array,
        target_feat: mx.array,
        seq_mask: mx.array,
        pair_mask: mx.array,
        asym_id: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Array-only confidence head forward pass.

        This function contains no GatherInfo or layout_convert usage and is suitable
        for mx.compile. It encapsulates the core confidence computation:
        1. Feature embedding from precomputed pseudo-beta positions
        2. PairFormer refinement stack
        3. Distance error, PAE, pLDDT, and TM-score computations

        Args:
            pseudo_beta_positions: Precomputed pseudo-beta positions [seq, 3].
            pair_act: Pair activations [seq, seq, pair_channel].
            single_act: Single activations [seq, seq_channel].
            target_feat: Target features [seq, feat_dim].
            seq_mask: Sequence mask [seq].
            pair_mask: Pair mask [seq, seq].
            asym_id: Chain identifiers [seq].

        Returns:
            Tuple of (plddt, pae, pde, ptm, iptm, tm_global, tm_interface).
        """
        # Embed features using precomputed pseudo-beta positions
        pair_act = pair_act + self._embed_features_arrays(
            pseudo_beta_positions,
            pair_mask,
            pair_act,
            target_feat,
        )

        # PairFormer stack
        for layer in self.pairformer_layers:
            single_act, pair_act = layer(
                single_act[None, ...],
                pair_act[None, ...],
                seq_mask[None, ...],
                pair_mask[None, ...],
            )
            single_act = single_act[0]
            pair_act = pair_act[0]

        pair_act = pair_act.astype(mx.float32)

        # Distance error logits
        left_distance_logits = self.left_half_distance_logits(self.logits_ln(pair_act))
        right_distance_logits = left_distance_logits
        distance_logits = left_distance_logits + mx.swapaxes(right_distance_logits, -2, -3)

        distance_breaks = mx.linspace(0.0, self.config.max_error_bin, self.config.num_bins - 1)
        step = distance_breaks[1] - distance_breaks[0]
        # Add half-step to get the center of each bin
        bin_centers = distance_breaks + step / 2
        # Add a catch-all bin at the end (JAX: bin_centers[-1:] + step)
        bin_centers = mx.concatenate([bin_centers, bin_centers[-1:] + step], axis=0)
        distance_probs = mx.softmax(distance_logits, axis=-1)
        pred_distance_error = mx.sum(distance_probs * bin_centers, axis=-1) * pair_mask

        # PAE logits
        pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
        pae_breaks = mx.linspace(0.0, self.config.max_error_bin, self.config.num_pae_bins - 1)
        step = pae_breaks[1] - pae_breaks[0]
        # Add half-step to get the center of each bin
        pae_bin_centers = pae_breaks + step / 2
        # Add a catch-all bin at the end (JAX: bin_centers[-1:] + step)
        pae_bin_centers = mx.concatenate([pae_bin_centers, pae_bin_centers[-1:] + step], axis=0)
        pae_probs = mx.softmax(pae_logits, axis=-1)

        seq_mask_bool = seq_mask.astype(mx.bool_)
        pair_mask_bool = (seq_mask_bool[:, None] * seq_mask_bool[None, :]).astype(mx.float32)
        pae = mx.sum(pae_probs * pae_bin_centers, axis=-1) * pair_mask_bool

        tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = self._get_tmscore_adjusted_pae(
            asym_id=asym_id,
            seq_mask=seq_mask_bool.astype(mx.int32),
            pair_mask=pair_mask_bool.astype(mx.int32),
            bin_centers=pae_bin_centers,
            pae_probs=pae_probs,
        )

        single_act = single_act.astype(mx.float32)

        # pLDDT
        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))
        bin_width = 1.0 / self.config.num_plddt_bins
        plddt_bin_centers = mx.arange(0.5 * bin_width, 1.0, bin_width)
        predicted_lddt = mx.sum(mx.softmax(plddt_logits, axis=-1) * plddt_bin_centers, axis=-1)
        predicted_lddt = predicted_lddt * 100.0

        # Compute pTM and ipTM from TM-score adjusted PAE
        ptm_scalar = self._compute_ptm(
            tm_adjusted_pae=tmscore_adjusted_pae_global,
            pair_mask=pair_mask_bool,
        )
        iptm_scalar = self._compute_iptm(
            tm_adjusted_pae=tmscore_adjusted_pae_global,
            pair_mask=pair_mask_bool,
            asym_id=asym_id,
        )

        return (
            predicted_lddt,
            pae,
            pred_distance_error,
            ptm_scalar,
            iptm_scalar,
            tmscore_adjusted_pae_global,
            tmscore_adjusted_pae_interface,
        )

    def _get_tmscore_adjusted_pae(
        self,
        asym_id: mx.array,
        seq_mask: mx.array,
        pair_mask: mx.array,
        bin_centers: mx.array,
        pae_probs: mx.array,
    ) -> tuple[mx.array, mx.array]:
        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            clipped_num_res = mx.maximum(num_interface_tokens, 19)
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
            d0 = d0[:, :, None]
            bin_centers_b = bin_centers[None, None, :]
            tm_per_bin = 1.0 / (1 + (bin_centers_b ** 2) / (d0 ** 2))
            predicted_tm_term = mx.sum(pae_probs * tm_per_bin, axis=-1)
            return predicted_tm_term

        x = asym_id[None, :] == asym_id[:, None]
        num_chain_tokens = mx.sum(x * pair_mask, axis=-1)
        num_interface_tokens = num_chain_tokens[None, :] + num_chain_tokens[:, None]
        num_interface_tokens = num_interface_tokens - x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = mx.full(pair_mask.shape, seq_mask.sum())
        global_apae = get_tmscore_adjusted_pae(num_global_tokens, bin_centers, pae_probs)
        interface_apae = get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs)
        return global_apae, interface_apae

    def _compute_ptm(
        self,
        tm_adjusted_pae: mx.array,
        pair_mask: mx.array,
    ) -> mx.array:
        """Compute pTM score from TM-adjusted PAE matrix.

        pTM is computed as max over aligned positions of mean TM-score contribution.

        Args:
            tm_adjusted_pae: [seq, seq] TM-score adjusted PAE values
            pair_mask: [seq, seq] mask for valid residue pairs

        Returns:
            Scalar pTM score in [0, 1]
        """
        # For each aligned position i, compute mean TM contribution from all positions j
        # ptm = max_i (sum_j tm_adjusted_pae[i,j] * pair_mask[i,j]) / (sum_j pair_mask[i,j])
        masked_tm = tm_adjusted_pae * pair_mask
        row_sums = mx.sum(masked_tm, axis=-1)  # [seq]
        row_counts = mx.sum(pair_mask, axis=-1)  # [seq]
        row_means = row_sums / mx.maximum(row_counts, 1.0)  # [seq]

        # Take max over alignment positions, weighted by valid mask
        seq_mask = row_counts > 0
        # Set invalid positions to -inf for max
        row_means_masked = mx.where(seq_mask, row_means, -mx.inf)
        ptm = mx.max(row_means_masked)

        # Handle edge case: no valid positions
        ptm = mx.where(mx.any(seq_mask), ptm, mx.array(0.0))
        return ptm

    def _compute_iptm(
        self,
        tm_adjusted_pae: mx.array,
        pair_mask: mx.array,
        asym_id: mx.array,
    ) -> mx.array:
        """Compute ipTM score from TM-adjusted PAE matrix.

        ipTM focuses on inter-chain (interface) residue pairs only.

        Args:
            tm_adjusted_pae: [seq, seq] TM-score adjusted PAE values
            pair_mask: [seq, seq] mask for valid residue pairs
            asym_id: [seq] chain identifiers

        Returns:
            Scalar ipTM score in [0, 1]
        """
        # Create interface mask: different chains only
        interface_mask = (asym_id[None, :] != asym_id[:, None]).astype(pair_mask.dtype)
        interface_pair_mask = pair_mask * interface_mask

        # Check if there are any interface pairs (multi-chain structure)
        has_interface = mx.sum(interface_pair_mask) > 0

        # Compute ipTM for interface pairs
        masked_tm = tm_adjusted_pae * interface_pair_mask
        row_sums = mx.sum(masked_tm, axis=-1)
        row_counts = mx.sum(interface_pair_mask, axis=-1)
        row_means = row_sums / mx.maximum(row_counts, 1.0)

        # Take max over alignment positions
        seq_mask = row_counts > 0
        row_means_masked = mx.where(seq_mask, row_means, -mx.inf)
        iptm = mx.max(row_means_masked)

        # If no interface (single chain), return 0
        iptm = mx.where(has_interface, iptm, mx.array(0.0))
        # Handle edge case: no valid interface positions
        iptm = mx.where(mx.any(seq_mask), iptm, mx.array(0.0))
        return iptm

    def __call__(
        self,
        dense_atom_positions: mx.array,
        embeddings: dict[str, mx.array],
        seq_mask: mx.array,
        token_atoms_to_pseudo_beta: GatherInfo,
        asym_id: mx.array,
    ) -> "ConfidenceScores":
        from alphafold3_mlx.core.entities import ConfidenceScores

        # Support optional sample dimension
        if dense_atom_positions.ndim == 4:
            # [num_samples, num_res, num_atom, 3]
            outputs = []
            for i in range(dense_atom_positions.shape[0]):
                outputs.append(
                    self.__call__(
                        dense_atom_positions[i],
                        embeddings,
                        seq_mask,
                        token_atoms_to_pseudo_beta,
                        asym_id,
                    )
                )
            # Stack outputs
            plddt = mx.stack([o.plddt[0] for o in outputs], axis=0)
            pae = mx.stack([o.pae[0] for o in outputs], axis=0)
            pde = mx.stack([o.pde[0] for o in outputs], axis=0)
            ptm = mx.stack([o.ptm[0] for o in outputs], axis=0)
            iptm = mx.stack([o.iptm[0] for o in outputs], axis=0)
            tm_pae_global = mx.stack([o.tm_pae_global[0] for o in outputs], axis=0) if outputs[0].tm_pae_global is not None else None
            tm_pae_interface = mx.stack([o.tm_pae_interface[0] for o in outputs], axis=0) if outputs[0].tm_pae_interface is not None else None
            return ConfidenceScores(
                plddt=plddt,
                pae=pae,
                pde=pde,
                ptm=ptm,
                iptm=iptm,
                tm_pae_global=tm_pae_global,
                tm_pae_interface=tm_pae_interface,
            )

        # Single sample path
        # Use global_config.precision for dtype selection (consistency with Model.set_precision)
        # Note: LayerNorm will upcast 16-bit inputs to float32 internally for numerical stability
        dtype = self._PRECISION_TO_DTYPE.get(self.global_config.precision, mx.float32)
        seq_mask_cast = seq_mask.astype(dtype)
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
        pair_mask = pair_mask.astype(dtype)

        pair_act = embeddings["pair"].astype(dtype)
        single_act = embeddings["single"].astype(dtype)
        target_feat = embeddings["target_feat"].astype(dtype)

        num_atoms = dense_atom_positions.shape[-2]
        self._build(target_feat.shape[-1], num_atoms)

        # Weight-loading parity path may pre-build projections with AF3 dims (e.g. 447)
        # while runtime target_feat can be narrower. Pad/truncate to match loaded weights.
        if self.left_target_feat_project is not None:
            expected_target_feat_dim = int(self.left_target_feat_project.weight.shape[0])
            current_target_feat_dim = int(target_feat.shape[-1])
            if current_target_feat_dim < expected_target_feat_dim:
                pad = mx.zeros(
                    target_feat.shape[:-1] + (expected_target_feat_dim - current_target_feat_dim,),
                    dtype=target_feat.dtype,
                )
                target_feat = mx.concatenate([target_feat, pad], axis=-1)
            elif current_target_feat_dim > expected_target_feat_dim:
                target_feat = target_feat[..., :expected_target_feat_dim]

        # === GatherInfo-dependent: Convert to pseudo-beta positions ===
        pseudo_beta_positions = layout_convert(
            token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2)
        )

        # === Array-only: Core confidence computation ===
        if self._compiled and self._compiled_confidence_forward is not None:
            (
                predicted_lddt,
                pae,
                pred_distance_error,
                ptm_scalar,
                iptm_scalar,
                tmscore_adjusted_pae_global,
                tmscore_adjusted_pae_interface,
            ) = self._compiled_confidence_forward(
                pseudo_beta_positions, pair_act, single_act, target_feat, seq_mask, pair_mask, asym_id
            )
        else:
            (
                predicted_lddt,
                pae,
                pred_distance_error,
                ptm_scalar,
                iptm_scalar,
                tmscore_adjusted_pae_global,
                tmscore_adjusted_pae_interface,
            ) = self.confidence_forward_arrays(
                pseudo_beta_positions=pseudo_beta_positions,
                pair_act=pair_act,
                single_act=single_act,
                target_feat=target_feat,
                seq_mask=seq_mask,
                pair_mask=pair_mask,
                asym_id=asym_id,
            )

        # Wrap outputs into ConfidenceScores (single sample)
        plddt = predicted_lddt[None, ...]
        pae_out = pae[None, ...]
        pde_out = pred_distance_error[None, ...]
        ptm = ptm_scalar[None]
        iptm = iptm_scalar[None]
        tm_global = tmscore_adjusted_pae_global[None, ...]
        tm_interface = tmscore_adjusted_pae_interface[None, ...]

        return ConfidenceScores(
            plddt=plddt,
            pae=pae_out,
            pde=pde_out,
            ptm=ptm,
            iptm=iptm,
            tm_pae_global=tm_global,
            tm_pae_interface=tm_interface,
        )
