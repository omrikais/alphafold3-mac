"""Index resolution for restraint references.

Maps user-facing (chain_id, residue_num, atom_name) references to internal
(token_index, dense_atom_index) pairs using the token layout from the data
pipeline.  The dense-24 format is what the diffusion head uses for positions
(num_tokens, 24, 3), NOT the atom37 superset (37 slots).
"""

from __future__ import annotations

import numpy as np

from alphafold3_mlx.core.atom_constants import ATOM37_ORDER, RESTYPE_NAMES


def _build_atom37_to_dense() -> np.ndarray:
    """Build per-residue-type atom37 → dense-24 index mapping.

    Returns:
        Array of shape (31, 37) where entry [aatype, atom37_idx] gives the
        dense-24 index for that atom, or -1 if the atom is not present in
        the dense layout for that residue type.
    """
    from alphafold3.model.protein_data_processing import (
        PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
    )

    num_restypes, dense_dim = PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37.shape
    # Initialize with -1 (invalid)
    table = np.full((num_restypes, 37), -1, dtype=np.int32)
    for rt in range(num_restypes):
        for d_idx in range(dense_dim):
            a37 = int(PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37[rt, d_idx])
            # Only map if the dense slot is actually used (skip padding zeros
            # after the first occurrence, since index 0 = N is always at d_idx=0)
            if d_idx == 0 or a37 != 0:
                table[rt, a37] = d_idx
    return table


# Lazy singleton so the table is built at most once.
_ATOM37_TO_DENSE: np.ndarray | None = None


def _get_atom37_to_dense() -> np.ndarray:
    global _ATOM37_TO_DENSE
    if _ATOM37_TO_DENSE is None:
        _ATOM37_TO_DENSE = _build_atom37_to_dense()
    return _ATOM37_TO_DENSE
from alphafold3_mlx.restraints.types import (
    ContactRestraint,
    DistanceRestraint,
    RepulsiveRestraint,
    ResolvedContactRestraint,
    ResolvedDistanceRestraint,
    ResolvedRepulsiveRestraint,
    RestraintConfig,
)


class RestraintResolutionError(ValueError):
    """Raised when a restraint reference cannot be resolved to atom indices."""


def resolve_restraints(
    config: RestraintConfig,
    chain_ids: list[str],
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    aatype: np.ndarray,
    mask: np.ndarray,
) -> tuple[
    list[ResolvedDistanceRestraint],
    list[ResolvedContactRestraint],
    list[ResolvedRepulsiveRestraint],
]:
    """Resolve restraint references to flat atom indices.

    Args:
        config: Validated restraint configuration.
        chain_ids: Ordered list of string chain IDs. chain_ids[i] maps to asym_id=i+1.
        asym_id: Integer chain identifiers per token. Shape: [num_tokens]. 1-based.
        residue_index: PDB residue numbers per token. Shape: [num_tokens]. 1-based.
        aatype: Amino acid type indices per token. Shape: [num_tokens].
        mask: Token validity mask. Shape: [num_tokens]. 1=valid, 0=padding.

    Returns:
        Tuple of (resolved_distance, resolved_contact, resolved_repulsive) lists.

    Raises:
        RestraintResolutionError: If any restraint reference cannot be resolved.
    """
    # Build chain_id -> asym_id mapping
    chain_to_asym = {cid: i + 1 for i, cid in enumerate(chain_ids)}

    # Build lookup index: (asym_id, residue_num) -> token_index
    # Only consider valid (unpadded) tokens
    valid_mask = mask.astype(bool)
    token_lookup: dict[tuple[int, int], int] = {}
    for tok_idx in range(len(asym_id)):
        if valid_mask[tok_idx]:
            key = (int(asym_id[tok_idx]), int(residue_index[tok_idx]))
            # First occurrence wins (for proteins, there's exactly one token per residue)
            if key not in token_lookup:
                token_lookup[key] = tok_idx

    # Resolve distance restraints
    resolved_distance = []
    for i, r in enumerate(config.distance):
        atom_i_idx = _resolve_atom_ref(
            f"distance[{i}].i", r.chain_i, r.residue_i, r.atom_i,
            chain_to_asym, token_lookup, aatype,
        )
        atom_j_idx = _resolve_atom_ref(
            f"distance[{i}].j", r.chain_j, r.residue_j, r.atom_j,
            chain_to_asym, token_lookup, aatype,
        )
        resolved_distance.append(ResolvedDistanceRestraint(
            atom_i_idx=atom_i_idx,
            atom_j_idx=atom_j_idx,
            target_distance=r.target_distance,
            sigma=r.sigma,
            weight=r.weight,
        ))

    # Resolve contact restraints (source and candidates use CA)
    # Use per-residue-type lookup instead of hardcoded index — UNK/X residues
    # have no CA in the dense-24 layout, so _resolve_atom_ref will raise a
    # clear RestraintResolutionError rather than silently using wrong coords.
    resolved_contact = []
    for i, r in enumerate(config.contact):
        source_atom_idx = _resolve_atom_ref(
            f"contact[{i}].source", r.chain_i, r.residue_i, "CA",
            chain_to_asym, token_lookup, aatype,
        )
        candidate_idxs = []
        for j, cand in enumerate(r.candidates):
            cand_atom_idx = _resolve_atom_ref(
                f"contact[{i}].candidate[{j}]", cand.chain_j, cand.residue_j, "CA",
                chain_to_asym, token_lookup, aatype,
            )
            candidate_idxs.append(cand_atom_idx)
        resolved_contact.append(ResolvedContactRestraint(
            source_atom_idx=source_atom_idx,
            candidate_atom_idxs=candidate_idxs,
            threshold=r.threshold,
            weight=r.weight,
        ))

    # Resolve repulsive restraints (uses CA atoms)
    resolved_repulsive = []
    for i, r in enumerate(config.repulsive):
        atom_i_idx = _resolve_atom_ref(
            f"repulsive[{i}].i", r.chain_i, r.residue_i, "CA",
            chain_to_asym, token_lookup, aatype,
        )
        atom_j_idx = _resolve_atom_ref(
            f"repulsive[{i}].j", r.chain_j, r.residue_j, "CA",
            chain_to_asym, token_lookup, aatype,
        )
        resolved_repulsive.append(ResolvedRepulsiveRestraint(
            atom_i_idx=atom_i_idx,
            atom_j_idx=atom_j_idx,
            min_distance=r.min_distance,
            weight=r.weight,
        ))

    return resolved_distance, resolved_contact, resolved_repulsive


def _resolve_atom_ref(
    label: str,
    chain_id: str,
    residue_num: int,
    atom_name: str,
    chain_to_asym: dict[str, int],
    token_lookup: dict[tuple[int, int], int],
    aatype: np.ndarray,
) -> tuple[int, int]:
    """Resolve a (chain, residue, atom) reference to (token_idx, dense_atom_idx).

    The returned atom index is in the dense-24 layout used by the diffusion
    head, NOT the atom37 superset.
    """
    if atom_name not in ATOM37_ORDER:
        raise RestraintResolutionError(
            f"Restraint {label}: atom '{atom_name}' is not a valid atom37 name"
        )
    atom37_idx = ATOM37_ORDER[atom_name]

    token_idx = _resolve_residue_ref(label, chain_id, residue_num, chain_to_asym, token_lookup)

    # Verify atom is valid for this residue type
    aa = int(aatype[token_idx])
    if 0 <= aa < len(RESTYPE_NAMES):
        res_name = RESTYPE_NAMES[aa]
    else:
        res_name = "UNK"

    from alphafold3_mlx.core.atom_constants import RESTYPE_ATOMS
    valid_atoms = RESTYPE_ATOMS.get(res_name, RESTYPE_ATOMS["UNK"])
    if atom_name not in valid_atoms:
        raise RestraintResolutionError(
            f"Restraint {label}: atom '{atom_name}' is not valid for {res_name} "
            f"at chain {chain_id} residue {residue_num} "
            f"(valid: {', '.join(valid_atoms)})"
        )

    # Convert atom37 index → dense-24 index (the format used by diffusion head)
    a37_to_dense = _get_atom37_to_dense()
    dense_idx = int(a37_to_dense[aa, atom37_idx])
    if dense_idx < 0:
        raise RestraintResolutionError(
            f"Restraint {label}: atom '{atom_name}' (atom37={atom37_idx}) "
            f"has no dense-24 mapping for {res_name} (aatype={aa})"
        )

    return (token_idx, dense_idx)


def _resolve_residue_ref(
    label: str,
    chain_id: str,
    residue_num: int,
    chain_to_asym: dict[str, int],
    token_lookup: dict[tuple[int, int], int],
) -> int:
    """Resolve a (chain, residue) reference to a token_index."""
    if chain_id not in chain_to_asym:
        available = sorted(chain_to_asym.keys())
        raise RestraintResolutionError(
            f"Restraint {label}: chain '{chain_id}' not found "
            f"(available: {', '.join(available)})"
        )

    asym = chain_to_asym[chain_id]
    key = (asym, residue_num)
    if key not in token_lookup:
        raise RestraintResolutionError(
            f"Restraint {label}: residue {residue_num} in chain '{chain_id}' "
            f"(asym_id={asym}) not found in token layout"
        )

    return token_lookup[key]
