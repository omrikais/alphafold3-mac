"""Input dataclasses for AlphaFold 3 MLX.

This module provides input containers including:
- AttentionInputs: Low-level attention inputs (Phase 0)
- TokenFeatures: Token-level input features
- MSAFeatures: MSA input features
- TemplateFeatures: Template input features
- FrameFeatures: Frame alignment features
- BondInfo: Bond information
- FeatureBatch: Complete input batch from data pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


# Type alias for array-like (works with both numpy and mlx arrays)
Array = Any


@dataclass
class AttentionInputs:
    """Input tensors for scaled dot-product attention.

    Attributes:
        q: Query tensor [batch, heads, seq_q, head_dim]
        k: Key tensor [batch, heads, seq_k, head_dim]
        v: Value tensor [batch, heads, seq_k, head_dim]
        boolean_mask: Optional mask [batch, seq_k] where True=attend, False=mask
        additive_bias: Optional bias [batch, heads, seq_q, seq_k]
    """

    q: Array
    k: Array
    v: Array
    boolean_mask: Array | None = None
    additive_bias: Array | None = None

    def validate(self) -> None:
        """Validate tensor shapes and dtypes.

        Raises:
            AssertionError: If tensor shapes are invalid
        """
        # Check Q, K, V are 4D
        assert self.q.ndim == 4, f"Q must be 4D, got {self.q.ndim}D"
        assert self.k.ndim == 4, f"K must be 4D, got {self.k.ndim}D"
        assert self.v.ndim == 4, f"V must be 4D, got {self.v.ndim}D"

        # Extract dimensions from Q
        batch, heads, seq_q, head_dim = self.q.shape
        seq_k = self.k.shape[2]

        # Check K shape: [batch, heads, seq_k, head_dim]
        assert self.k.shape == (batch, heads, seq_k, head_dim), (
            f"K shape mismatch: expected ({batch}, {heads}, {seq_k}, {head_dim}), "
            f"got {self.k.shape}"
        )

        # Check V shape matches K
        assert self.v.shape == self.k.shape, (
            f"V shape must match K: expected {self.k.shape}, got {self.v.shape}"
        )

        # Check boolean mask if provided
        if self.boolean_mask is not None:
            assert self.boolean_mask.ndim == 2, (
                f"boolean_mask must be 2D, got {self.boolean_mask.ndim}D"
            )
            assert self.boolean_mask.shape == (batch, seq_k), (
                f"boolean_mask shape mismatch: expected ({batch}, {seq_k}), "
                f"got {self.boolean_mask.shape}"
            )

        # Check additive bias if provided
        if self.additive_bias is not None:
            assert self.additive_bias.ndim == 4, (
                f"additive_bias must be 4D, got {self.additive_bias.ndim}D"
            )
            assert self.additive_bias.shape == (batch, heads, seq_q, seq_k), (
                f"additive_bias shape mismatch: expected ({batch}, {heads}, {seq_q}, {seq_k}), "
                f"got {self.additive_bias.shape}"
            )

    def to_numpy(self) -> dict[str, np.ndarray | None]:
        """Convert all tensors to numpy arrays."""
        result = {
            "q": np.asarray(self.q),
            "k": np.asarray(self.k),
            "v": np.asarray(self.v),
        }
        if self.boolean_mask is not None:
            result["boolean_mask"] = np.asarray(self.boolean_mask)
        else:
            result["boolean_mask"] = None
        if self.additive_bias is not None:
            result["additive_bias"] = np.asarray(self.additive_bias)
        else:
            result["additive_bias"] = None
        return result


# =============================================================================
# Model Input Features (Phase 3)
# =============================================================================


@dataclass
class TokenFeatures:
    """Token-level input features.

    These features describe each token (residue/nucleotide/ligand atom)
    in the input sequence.
    """

    aatype: mx.array
    """Amino acid type index. Shape: [num_tokens]"""

    token_index: mx.array
    """Absolute token index. Shape: [num_tokens]"""

    mask: mx.array
    """Token validity mask (1=valid, 0=padding). Shape: [num_tokens]"""

    residue_index: mx.array
    """Residue index within chain. Shape: [num_tokens]"""

    asym_id: mx.array
    """Asymmetric unit (chain) identifier. Shape: [num_tokens]"""

    entity_id: mx.array
    """Entity identifier. Shape: [num_tokens]"""

    sym_id: mx.array
    """Symmetry identifier. Shape: [num_tokens]"""

    @property
    def num_tokens(self) -> int:
        """Return number of tokens."""
        return self.mask.shape[0]


@dataclass
class MSAFeatures:
    """MSA (Multiple Sequence Alignment) input features."""

    msa: mx.array
    """MSA sequences. Shape: [num_msa, num_tokens]"""

    msa_mask: mx.array
    """MSA validity mask. Shape: [num_msa, num_tokens]"""

    deletion_matrix: mx.array
    """Deletion counts per position. Shape: [num_msa, num_tokens]"""

    @property
    def num_msa(self) -> int:
        """Return number of MSA sequences."""
        return self.msa.shape[0]


@dataclass
class TemplateFeatures:
    """Template input features."""

    template_aatype: mx.array
    """Template amino acid types. Shape: [num_templates, num_tokens]"""

    template_all_atom_positions: mx.array
    """Template atom positions. Shape: [num_templates, num_tokens, max_atoms, 3]"""

    template_all_atom_mask: mx.array
    """Template atom validity mask. Shape: [num_templates, num_tokens, max_atoms]"""

    @property
    def num_templates(self) -> int:
        """Return number of templates."""
        return self.template_aatype.shape[0]


@dataclass
class FrameFeatures:
    """Frame alignment features for coordinate generation."""

    mask: mx.array
    """Frame validity mask. Shape: [num_tokens]"""

    rotation: mx.array
    """Frame rotation matrices. Shape: [num_tokens, 3, 3]"""

    translation: mx.array
    """Frame translation vectors. Shape: [num_tokens, 3]"""


@dataclass
class BondInfo:
    """Bond information for molecular connectivity."""

    token_i: mx.array
    """First token indices. Shape: [num_bonds]"""

    token_j: mx.array
    """Second token indices. Shape: [num_bonds]"""

    bond_type: mx.array
    """Bond type indices. Shape: [num_bonds]"""

    @property
    def num_bonds(self) -> int:
        """Return number of bonds."""
        return self.token_i.shape[0]


def _extract_bond_info(
    batch_dict: dict,
    gather_key_prefix: str,
    to_mlx,
) -> "BondInfo":
    """Extract BondInfo from JAX AF3 GatherInfo-format bond data.

    JAX AF3 stores bonds as GatherInfo with gather_idxs shape (N, 2) where
    each row is (token_i, token_j).  Valid bonds have both mask entries True.
    Padded rows have gather_idxs == 0 and gather_mask == False.
    """
    empty = np.array([], dtype=np.int32)

    # Try colon-separated keys (JAX default) then slash-separated
    idxs_key = f"{gather_key_prefix}:gather_idxs"
    mask_key = f"{gather_key_prefix}:gather_mask"
    if idxs_key not in batch_dict:
        idxs_key = f"{gather_key_prefix}/gather_idxs"
        mask_key = f"{gather_key_prefix}/gather_mask"

    gather_idxs = batch_dict.get(idxs_key)
    gather_mask = batch_dict.get(mask_key)

    if gather_idxs is not None and gather_mask is not None and gather_idxs.size > 0:
        # Valid bonds: both endpoints must be masked in
        valid = gather_mask.prod(axis=1).astype(bool)
        bond_pairs = gather_idxs[valid]
        if bond_pairs.size > 0:
            token_i = bond_pairs[:, 0].astype(np.int32)
            token_j = bond_pairs[:, 1].astype(np.int32)
            return BondInfo(
                token_i=to_mlx(token_i),
                token_j=to_mlx(token_j),
                bond_type=to_mlx(np.zeros(len(token_i), dtype=np.int32)),
            )

    return BondInfo(
        token_i=to_mlx(empty),
        token_j=to_mlx(empty),
        bond_type=to_mlx(empty),
    )


@dataclass
class FeatureBatch:
    """Input features for model inference.

    This class wraps the feature dictionary from the AlphaFold data pipeline
    and converts NumPy arrays to MLX arrays at the model boundary.

    Relationship to Existing Code:
        - Wraps `alphafold3.model.feat_batch.Batch`
        - Converts NumPy arrays to MLX arrays at model boundary
        - Preserves all feature names and shapes
    """

    token_features: TokenFeatures
    """Token-level features (aatype, mask, residue_index, etc.)"""

    msa_features: MSAFeatures | None
    """MSA features (optional - None if no MSA available)"""

    template_features: TemplateFeatures | None
    """Template features (optional - None if no templates available)"""

    frames: FrameFeatures
    """Frame alignment features"""

    polymer_ligand_bond_info: BondInfo
    """Bond information between polymer and ligand atoms"""

    ligand_ligand_bond_info: BondInfo
    """Bond information between ligand atoms"""

    msa_profile: mx.array | None = None
    """Per-token MSA profile features (optional). Shape: [num_tokens, 22]"""

    deletion_mean: mx.array | None = None
    """Per-token mean deletion values (optional). Shape: [num_tokens]"""

    per_atom_features: Any | None = None
    """Optional per-atom conditioning batch (alphafold3_mlx.feat_batch.Batch)."""

    @property
    def num_residues(self) -> int:
        """Return number of residues/tokens."""
        return self.token_features.num_tokens

    @property
    def has_msa(self) -> bool:
        """Return whether MSA features are available."""
        return self.msa_features is not None

    @property
    def has_templates(self) -> bool:
        """Return whether template features are available."""
        return self.template_features is not None

    @classmethod
    def from_numpy(cls, batch_dict: dict[str, np.ndarray]) -> "FeatureBatch":
        """Convert NumPy batch dict to MLX FeatureBatch.

        This is the bridge function from the AlphaFold data pipeline to MLX.

        Args:
            batch_dict: Dictionary of NumPy arrays from data pipeline.

        Returns:
            FeatureBatch with MLX arrays.
        """

        def to_mlx(arr: np.ndarray | None) -> mx.array | None:
            """Convert NumPy array to MLX array."""
            if arr is None:
                return None
            return mx.array(arr)

        # Token features
        # AF3 pipeline uses "seq_mask" while MLX expects "token_mask" (bridge)
        token_mask = batch_dict.get("token_mask", batch_dict.get("seq_mask"))
        if token_mask is None:
            raise KeyError("batch_dict is missing required token mask (token_mask or seq_mask)")
        token_index = batch_dict.get("token_index")
        if token_index is None:
            token_index = np.arange(token_mask.shape[0], dtype=np.int32)
        token_features = TokenFeatures(
            aatype=to_mlx(batch_dict["aatype"]),
            token_index=to_mlx(token_index),
            mask=to_mlx(token_mask),
            residue_index=to_mlx(batch_dict["residue_index"]),
            asym_id=to_mlx(batch_dict["asym_id"]),
            entity_id=to_mlx(batch_dict["entity_id"]),
            sym_id=to_mlx(batch_dict["sym_id"]),
        )

        # MSA features (optional)
        msa_features = None
        if "msa" in batch_dict and batch_dict["msa"] is not None:
            msa_features = MSAFeatures(
                msa=to_mlx(batch_dict["msa"]),
                msa_mask=to_mlx(batch_dict["msa_mask"]),
                deletion_matrix=to_mlx(batch_dict["deletion_matrix"]),
            )

        # Template features (optional)
        template_features = None
        if "template_aatype" in batch_dict and batch_dict["template_aatype"] is not None:
            # Support AF3 pipeline naming (template_atom_positions/mask)
            template_positions = batch_dict.get(
                "template_all_atom_positions",
                batch_dict.get("template_atom_positions"),
            )
            template_masks = batch_dict.get(
                "template_all_atom_mask",
                batch_dict.get("template_atom_mask"),
            )
            template_features = TemplateFeatures(
                template_aatype=to_mlx(batch_dict["template_aatype"]),
                template_all_atom_positions=to_mlx(template_positions),
                template_all_atom_mask=to_mlx(template_masks),
            )

        # Frame features
        frames = FrameFeatures(
            mask=to_mlx(batch_dict.get("frame_mask", token_mask)),
            rotation=to_mlx(
                batch_dict.get(
                    "frame_rotation",
                    np.tile(np.eye(3), (token_mask.shape[0], 1, 1)),
                )
            ),
            translation=to_mlx(
                batch_dict.get(
                    "frame_translation",
                    np.zeros((token_mask.shape[0], 3)),
                )
            ),
        )

        # Bond info -- JAX AF3 stores bonds as GatherInfo with gather_idxs
        # shape (N, 2) where each row is (token_i, token_j).  Valid bonds
        # have gather_mask.prod(axis=1) == True.  Extract flat token_i/token_j
        # arrays from this format.
        polymer_ligand_bond_info = _extract_bond_info(
            batch_dict, "tokens_to_polymer_ligand_bonds", to_mlx,
        )
        ligand_ligand_bond_info = _extract_bond_info(
            batch_dict, "tokens_to_ligand_ligand_bonds", to_mlx,
        )

        # Optional target features for parity (MSA profile + deletion mean)
        msa_profile = to_mlx(batch_dict.get("profile"))
        deletion_mean = to_mlx(batch_dict.get("deletion_mean"))

        # Optional per-atom conditioning batch (for evoformer conditioning)
        per_atom_features = None
        try:
            from alphafold3_mlx.feat_batch import Batch as PerAtomBatch

            has_token_atoms_to_queries = (
                "token_atoms_to_queries/gather_idxs" in batch_dict
                or "token_atoms_to_queries:gather_idxs" in batch_dict
            )
            if "ref_pos" in batch_dict and has_token_atoms_to_queries:
                per_atom_features = PerAtomBatch.from_data_dict(batch_dict)
        except Exception:
            per_atom_features = None

        return cls(
            token_features=token_features,
            msa_features=msa_features,
            template_features=template_features,
            frames=frames,
            polymer_ligand_bond_info=polymer_ligand_bond_info,
            ligand_ligand_bond_info=ligand_ligand_bond_info,
            msa_profile=msa_profile,
            deletion_mean=deletion_mean,
            per_atom_features=per_atom_features,
        )
