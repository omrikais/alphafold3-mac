"""Restraint reference validation.

Validates that restraint chain/residue/atom references are consistent with the
input sequences. Reports actionable error messages for invalid references.

## Validation Stages

This module supports **early fail-fast validation** to catch errors BEFORE
expensive operations like MSA search and feature preparation.

### Two-Stage Validation Architecture

1. **Early Validation** (before expensive operations):
   - Runs BEFORE: MSA search (`_run_data_pipeline`), feature preparation (`_prepare_features`)
   - Validates using ONLY: Input sequences (no featurized batch required)
   - Checks:
     * Chain existence
     * Residue range checks (1-indexed)
     * Atom name validity for residue types
     * Guidance config parameter ranges (start_step, end_step, scale)
     * Restraint count warnings (>100 restraints)
     * Conflict detection (distance vs repulsive on same atom pair)
   - Implementation: `validate_restraints()` called from `InferenceRunner._early_validate_restraints()`

2. **Late Validation** (after feature preparation):
   - Runs AFTER: Feature preparation (requires featurized batch)
   - Validates: Token index resolution (chain, residue, atom) → (token_index, atom37_index)
   - Implementation: `resolve.resolve_restraints()` called from `InferenceRunner._build_guidance_fn()`

### Why Two Stages?

**Early validation** catches most common errors (typos, out-of-range references) immediately,
providing fast feedback without wasting time on expensive MSA searches. This is especially
important for interactive use cases where users may iteratively refine restraints.

**Late validation** handles index resolution which requires the featurized batch's token layout
(asym_id, residue_index, aatype). This cannot be done earlier because the token layout is only
available after featurization.

### Example: Early vs Late Error Detection

**Early-caught errors** (fast feedback, before MSA search):
```python
# Invalid chain Z (chains A, B exist)
errors = validate_restraints(config, chains)
# Error: "chain 'Z' does not exist in input sequences (available: A, B)"
```

**Late-caught errors** (only possible after featurization):
```python
# Index resolution failure (valid chain/residue but token not found in batch)
resolved = resolve_restraints(config, chain_ids, asym_id, residue_index, aatype, mask)
# Raises: "Failed to resolve chain A residue 10 to token index"
```

### Usage

For fail-fast validation in the pipeline:
```python
from alphafold3_mlx.restraints.validate import (
    build_chain_info_from_input,
    validate_restraints,
)

# Build chain info from input sequences (no featurization needed)
chains = build_chain_info_from_input(fold_input.input)

# Early validation before expensive MSA search
errors = validate_restraints(restraint_config, chains, guidance_config, num_steps=200)
if errors:
    raise InputError("Invalid restraints (caught before MSA search): " + "; ".join(errors))

# ... expensive MSA search happens here ...
# ... feature preparation happens here ...

# Late validation with featurized batch
resolved = resolve_restraints(
    config=restraint_config,
    chain_ids=chain_ids,
    asym_id=batch.token_features.asym_id,
    residue_index=batch.token_features.residue_index,
    aatype=batch.token_features.aatype,
    mask=batch.token_features.mask,
)
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from alphafold3_mlx.core.atom_constants import ATOM37_ORDER, RESTYPE_ATOMS
from alphafold3_mlx.restraints.types import (
    ContactRestraint,
    DistanceRestraint,
    GuidanceConfig,
    RepulsiveRestraint,
    RestraintConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ChainInfo:
    """Information about a chain in the input sequences."""

    chain_id: str
    length: int
    residue_types: list[str]  # 3-letter codes per residue position (1-indexed)


# Standard single-letter → 3-letter amino acid mapping
_AA_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def build_chain_info_from_input(af3_input) -> dict[str, ChainInfo]:
    """Build ChainInfo dict from alphafold3.common.folding_input.Input.

    This is the canonical source of chain metadata for restraint validation.
    Only protein chains are supported for restraints currently.

    Args:
        af3_input: The alphafold3.common.folding_input.Input instance.

    Returns:
        Dictionary mapping chain_id to ChainInfo for protein chains only.
        Non-protein chains (RNA, DNA, ligands) are excluded.
    """
    from alphafold3.common import folding_input

    chains: dict[str, ChainInfo] = {}
    for chain in af3_input.chains:
        if isinstance(chain, folding_input.ProteinChain):
            residue_types = [
                _AA_1TO3.get(aa, "UNK") for aa in chain.sequence
            ]
            chains[chain.id] = ChainInfo(
                chain_id=chain.id,
                length=len(chain.sequence),
                residue_types=residue_types,
            )
        # Note: RNA, DNA, and Ligand chains are intentionally excluded.
        # Restraints on non-protein chains are not currently supported.

    return chains


def validate_restraints(
    config: RestraintConfig,
    chains: dict[str, ChainInfo],
    guidance: GuidanceConfig | None = None,
    num_diffusion_steps: int = 200,
    all_chain_ids: set[str] | None = None,
) -> list[str]:
    """Validate restraint references against input sequences.

    This function performs all restraint validation that can be done using
    only the input sequences, WITHOUT requiring featurized batches or
    index resolution. It should be called BEFORE expensive MSA search and
    feature preparation stages to provide fail-fast error detection.

    **Validations performed:**
    - Chain existence (with non-protein chain detection when all_chain_ids provided)
    - Residue range checks (1-indexed)
    - Atom name validity for residue types
    - Guidance config parameter ranges (start_step, end_step, scale)
    - Restraint count warnings (>100 restraints)
    - Conflict detection (distance vs repulsive on same atom pair)

    **Validations NOT performed** (require featurized batch):
    - Token index resolution (requires batch.token_features.asym_id, residue_index)
    - Atom37 index mapping (requires token layout)

    These late-stage validations happen during index resolution in
    `resolve.resolve_restraints()`.

    Args:
        config: The restraint configuration to validate.
        chains: Map from chain_id to ChainInfo for each chain in the input.
            Can be built from folding_input.Input using `build_chain_info_from_input()`.
        guidance: Optional guidance config to validate step ranges.
        num_diffusion_steps: Total number of diffusion steps (for step range validation).
        all_chain_ids: Optional set of ALL chain IDs (protein + non-protein).
            When provided, enables distinguishing "chain doesn't exist" from
            "chain exists but is non-protein (restraints not supported)".

    Returns:
        List of error messages. Empty list means all restraints are valid.

    Example:
        >>> # Build chain info from input sequences (no featurization needed)
        >>> chains = build_chain_info_from_input(fold_input.input)
        >>> all_ids = {c.id for c in fold_input.input.chains}
        >>> errors = validate_restraints(restraint_config, chains, guidance_config, 200, all_ids)
        >>> if errors:
        ...     raise InputError("Invalid restraints: " + "; ".join(errors))
    """
    errors: list[str] = []
    available_chains = sorted(chains.keys())

    # When no protein chains exist but restraints are present, reject early
    if not chains and config.total_count > 0:
        errors.append(
            "Restraints are only supported on protein chains. "
            "This input contains no protein chains."
        )
        return errors

    # Validate distance restraints
    for i, r in enumerate(config.distance):
        prefix = f"Restraint distance[{i}]"
        _validate_atom_ref(
            prefix, "i", r.chain_i, r.residue_i, r.atom_i,
            chains, available_chains, errors, all_chain_ids,
        )
        _validate_atom_ref(
            prefix, "j", r.chain_j, r.residue_j, r.atom_j,
            chains, available_chains, errors, all_chain_ids,
        )

    # Validate contact restraints
    for i, r in enumerate(config.contact):
        prefix = f"Restraint contact[{i}]"
        # Source uses CA implicitly
        _validate_residue_ref(
            prefix, "i", r.chain_i, r.residue_i,
            chains, available_chains, errors, all_chain_ids,
        )
        if len(r.candidates) < 1:
            errors.append(f"{prefix}: must have at least 1 candidate")
        for j, cand in enumerate(r.candidates):
            cand_prefix = f"{prefix} candidate[{j}]"
            _validate_residue_ref(
                cand_prefix, "j", cand.chain_j, cand.residue_j,
                chains, available_chains, errors, all_chain_ids,
            )

    # Validate repulsive restraints
    for i, r in enumerate(config.repulsive):
        prefix = f"Restraint repulsive[{i}]"
        # Repulsive uses CA implicitly
        _validate_residue_ref(
            prefix, "i", r.chain_i, r.residue_i,
            chains, available_chains, errors, all_chain_ids,
        )
        _validate_residue_ref(
            prefix, "j", r.chain_j, r.residue_j,
            chains, available_chains, errors, all_chain_ids,
        )

    # Validate guidance config step range
    if guidance is not None:
        end = guidance.end_step if guidance.end_step is not None else num_diffusion_steps
        if guidance.start_step >= end:
            errors.append(
                f"Guidance: start_step ({guidance.start_step}) must be < "
                f"end_step ({end})"
            )
        if end > num_diffusion_steps:
            errors.append(
                f"Guidance: end_step ({end}) must be <= "
                f"num_diffusion_steps ({num_diffusion_steps})"
            )
        if guidance.scale > 10.0:
            logger.warning(
                "Guidance scale %.1f is very high (> 10.0). "
                "This may distort the predicted structure.",
                guidance.scale,
            )

    # Warn if total restraint count > 100
    if config.total_count > 100:
        logger.warning(
            "Total restraint count (%d) exceeds 100. "
            "Large numbers of restraints may slow inference.",
            config.total_count,
        )

    # Conflict detection: distance and repulsive targeting same atom pair
    _detect_conflicts(config)

    return errors


def _validate_atom_ref(
    prefix: str,
    suffix: str,
    chain_id: str,
    residue_num: int,
    atom_name: str,
    chains: dict[str, ChainInfo],
    available_chains: list[str],
    errors: list[str],
    all_chain_ids: set[str] | None = None,
) -> None:
    """Validate a (chain, residue, atom) reference."""
    # Check chain exists
    if chain_id not in chains:
        _append_chain_error(prefix, chain_id, chains, available_chains, errors, all_chain_ids)
        return

    chain = chains[chain_id]

    # Check residue in range (1-indexed)
    if residue_num < 1 or residue_num > chain.length:
        errors.append(
            f"{prefix}: residue_{suffix} {residue_num} is out of range for "
            f"chain '{chain_id}' (valid: 1-{chain.length})"
        )
        return

    # Check atom is valid for residue type
    res_type = chain.residue_types[residue_num - 1]  # 1-indexed → 0-indexed

    # Reject UNK/X residues early — they have no dense-24 atom mapping and
    # will fail during resolve_restraints() after expensive MSA search.
    if res_type in ("UNK", "X"):
        errors.append(
            f"{prefix}: residue_{suffix} at chain '{chain_id}' position {residue_num} "
            f"has unknown residue type ({res_type}). Restraints cannot target "
            f"unknown residues (no dense-24 atom mapping available)."
        )
        return

    valid_atoms = RESTYPE_ATOMS.get(res_type, RESTYPE_ATOMS["UNK"])
    if atom_name not in valid_atoms:
        errors.append(
            f"{prefix}: atom '{atom_name}' is not valid for {res_type} at "
            f"chain {chain_id} residue {residue_num} "
            f"(valid atoms: {', '.join(valid_atoms)})"
        )

    # Also check atom exists in atom37 ordering
    if atom_name not in ATOM37_ORDER:
        errors.append(
            f"{prefix}: atom '{atom_name}' is not a recognized atom37 atom name"
        )


def _validate_residue_ref(
    prefix: str,
    suffix: str,
    chain_id: str,
    residue_num: int,
    chains: dict[str, ChainInfo],
    available_chains: list[str],
    errors: list[str],
    all_chain_ids: set[str] | None = None,
) -> None:
    """Validate a (chain, residue) reference (no atom validation needed)."""
    if chain_id not in chains:
        _append_chain_error(prefix, chain_id, chains, available_chains, errors, all_chain_ids)
        return

    chain = chains[chain_id]
    if residue_num < 1 or residue_num > chain.length:
        errors.append(
            f"{prefix}: residue_{suffix} {residue_num} is out of range for "
            f"chain '{chain_id}' (valid: 1-{chain.length})"
        )
        return

    # Reject UNK/X residues early — contact and repulsive restraints use CA
    # implicitly, but UNK/X have no dense-24 atom mapping.
    res_type = chain.residue_types[residue_num - 1]
    if res_type in ("UNK", "X"):
        errors.append(
            f"{prefix}: residue_{suffix} at chain '{chain_id}' position {residue_num} "
            f"has unknown residue type ({res_type}). Restraints cannot target "
            f"unknown residues."
        )


def _append_chain_error(
    prefix: str,
    chain_id: str,
    chains: dict[str, ChainInfo],
    available_chains: list[str],
    errors: list[str],
    all_chain_ids: set[str] | None = None,
) -> None:
    """Append the correct error for an invalid chain reference.

    Distinguishes between:
    - Chain exists but is non-protein (restraints not supported)
    - Chain does not exist at all
    """
    if all_chain_ids is not None and chain_id in all_chain_ids:
        # Chain exists but is not a protein chain
        protein_list = ", ".join(available_chains) if available_chains else "(none)"
        errors.append(
            f"{prefix}: restraints on non-protein chain '{chain_id}' are not supported. "
            f"Restraints can only reference protein chains: {protein_list}"
        )
    else:
        errors.append(
            f"{prefix}: chain '{chain_id}' does not exist in input sequences "
            f"(available: {', '.join(available_chains)})"
        )


def _detect_conflicts(config: RestraintConfig) -> None:
    """Log warnings for conflicting restraint pairs.

    Warns when distance and repulsive restraints target the same atom pair.
    Conflict detection operates at the atom-pair level (not residue-pair),
    so two restraints on the same residue pair but different atoms are NOT
    considered conflicting.

    For repulsive restraints (which implicitly use CA-CA), the comparison
    uses CA explicitly.
    """
    # Build set of atom pairs from distance restraints
    distance_atom_pairs: set[tuple[str, int, str, str, int, str]] = set()
    for r in config.distance:
        pair = _normalize_atom_pair(
            r.chain_i, r.residue_i, r.atom_i,
            r.chain_j, r.residue_j, r.atom_j,
        )
        distance_atom_pairs.add(pair)

    # Check repulsive restraints against distance pairs.
    # Repulsive restraints implicitly operate on CA-CA.
    for r in config.repulsive:
        pair = _normalize_atom_pair(
            r.chain_i, r.residue_i, "CA",
            r.chain_j, r.residue_j, "CA",
        )
        if pair in distance_atom_pairs:
            logger.warning(
                "Conflict: distance and repulsive restraints both target "
                "atom pair (%s:%d:%s, %s:%d:%s). These may produce opposing forces.",
                pair[0], pair[1], pair[2], pair[3], pair[4], pair[5],
            )


def _normalize_atom_pair(
    chain_i: str, res_i: int, atom_i: str,
    chain_j: str, res_j: int, atom_j: str,
) -> tuple[str, int, str, str, int, str]:
    """Normalize an atom pair so (A:10:NZ, B:20:CA) == (B:20:CA, A:10:NZ)."""
    if (chain_i, res_i, atom_i) <= (chain_j, res_j, atom_j):
        return (chain_i, res_i, atom_i, chain_j, res_j, atom_j)
    return (chain_j, res_j, atom_j, chain_i, res_i, atom_i)
