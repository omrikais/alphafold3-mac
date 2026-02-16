"""Minimal Batch dataclasses for JAX AF3 parity paths in MLX.

This mirrors a subset of alphafold3.model.feat_batch used by the
DiffusionHead and ConfidenceHead for parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from alphafold3_mlx.atom_layout import GatherInfo


Array = Any


@dataclass(frozen=True)
class TokenFeatures:
    """Token-level features required for relative encoding."""

    token_index: mx.array
    residue_index: mx.array
    asym_id: mx.array
    entity_id: mx.array
    sym_id: mx.array
    mask: mx.array


@dataclass(frozen=True)
class PredictedStructureInfo:
    """Predicted structure metadata."""

    atom_mask: mx.array


@dataclass(frozen=True)
class RefStructure:
    """Reference structure metadata for per-atom conditioning."""

    positions: mx.array
    mask: mx.array
    element: mx.array
    charge: mx.array
    atom_name_chars: mx.array
    space_uid: mx.array


@dataclass(frozen=True)
class AtomCrossAtt:
    """Atom cross-attention gather infos."""

    token_atoms_to_queries: GatherInfo
    tokens_to_queries: GatherInfo
    tokens_to_keys: GatherInfo
    queries_to_keys: GatherInfo
    queries_to_token_atoms: GatherInfo

    @classmethod
    def from_data_dict(cls, data: dict[str, np.ndarray]) -> "AtomCrossAtt":
        return cls(
            token_atoms_to_queries=GatherInfo.from_dict(
                data, "token_atoms_to_queries"
            ),
            tokens_to_queries=GatherInfo.from_dict(data, "tokens_to_queries"),
            tokens_to_keys=GatherInfo.from_dict(data, "tokens_to_keys"),
            queries_to_keys=GatherInfo.from_dict(data, "queries_to_keys"),
            queries_to_token_atoms=GatherInfo.from_dict(
                data, "queries_to_token_atoms"
            ),
        )


@dataclass(frozen=True)
class PseudoBetaInfo:
    """Pseudo-beta gather info."""

    token_atoms_to_pseudo_beta: GatherInfo

    @classmethod
    def from_data_dict(cls, data: dict[str, np.ndarray]) -> "PseudoBetaInfo":
        return cls(
            token_atoms_to_pseudo_beta=GatherInfo.from_dict(
                data, "token_atoms_to_pseudo_beta"
            )
        )


@dataclass(frozen=True)
class Batch:
    """Minimal Batch for diffusion/confidence heads."""

    token_features: TokenFeatures
    predicted_structure_info: PredictedStructureInfo
    atom_cross_att: AtomCrossAtt
    pseudo_beta_info: PseudoBetaInfo
    ref_structure: RefStructure

    @property
    def num_res(self) -> int:
        return int(self.token_features.mask.shape[0])

    @classmethod
    def from_data_dict(cls, data: dict[str, np.ndarray]) -> "Batch":
        """Build Batch from a flat feature dict (NumPy arrays)."""

        def to_mx(arr: np.ndarray) -> mx.array:
            return mx.array(arr)

        token_features = TokenFeatures(
            token_index=to_mx(data["token_index"]),
            residue_index=to_mx(data["residue_index"]),
            asym_id=to_mx(data["asym_id"]),
            entity_id=to_mx(data["entity_id"]),
            sym_id=to_mx(data["sym_id"]),
            mask=to_mx(data["seq_mask"]),
        )

        predicted_structure_info = PredictedStructureInfo(
            atom_mask=to_mx(data["pred_dense_atom_mask"])
        )

        atom_cross_att = AtomCrossAtt.from_data_dict(data)
        pseudo_beta_info = PseudoBetaInfo.from_data_dict(data)

        ref_structure = RefStructure(
            positions=to_mx(data["ref_pos"]),
            mask=to_mx(data["ref_mask"]),
            element=to_mx(data["ref_element"]),
            charge=to_mx(data["ref_charge"]),
            atom_name_chars=to_mx(data["ref_atom_name_chars"]),
            space_uid=to_mx(data["ref_space_uid"]),
        )

        return cls(
            token_features=token_features,
            predicted_structure_info=predicted_structure_info,
            atom_cross_att=atom_cross_att,
            pseudo_beta_info=pseudo_beta_info,
            ref_structure=ref_structure,
        )

