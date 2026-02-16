"""Parse PDB/mmCIF files and RCSB fetches into AlphaFold Server JSON format."""

from __future__ import annotations

import logging
import socket
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PdbNotFoundError(Exception):
    """Raised when a PDB ID is not found in RCSB."""

    def __init__(self, pdb_id: str):
        self.pdb_id = pdb_id
        super().__init__(f"PDB ID '{pdb_id}' not found in RCSB")


class RcsbTimeoutError(Exception):
    """Raised when the RCSB request times out."""

    def __init__(self, pdb_id: str):
        self.pdb_id = pdb_id
        super().__init__(f"RCSB request timed out after {RCSB_TIMEOUT_SECONDS}s")


class RcsbUnavailableError(Exception):
    """Raised when RCSB returns a non-404 HTTP error."""

    def __init__(self, pdb_id: str, detail: str):
        self.pdb_id = pdb_id
        self.detail = detail
        super().__init__(f"RCSB unavailable: {detail}")


# ---------------------------------------------------------------------------
# RCSB fetch
# ---------------------------------------------------------------------------


def fetch_pdb(pdb_id: str) -> str:
    """Fetch an mmCIF file from RCSB for the given PDB ID.

    Returns the mmCIF string content.
    Raises PdbNotFoundError, RcsbTimeoutError, or RcsbUnavailableError.
    """
    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id.upper())
    try:
        with urllib.request.urlopen(url, timeout=RCSB_TIMEOUT_SECONDS) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise PdbNotFoundError(pdb_id) from e
        raise RcsbUnavailableError(pdb_id, f"HTTP {e.code}") from e
    except (socket.timeout, urllib.error.URLError) as e:
        # URLError wraps socket.timeout on some platforms
        if isinstance(e, urllib.error.URLError) and not isinstance(
            e.reason, (socket.timeout, TimeoutError)
        ):
            raise RcsbUnavailableError(pdb_id, str(e)) from e
        raise RcsbTimeoutError(pdb_id) from e


# ---------------------------------------------------------------------------
# Gemmi-based structure parsing
# ---------------------------------------------------------------------------


def _parse_with_gemmi(content: str, fmt: str) -> dict[str, Any]:
    """Parse PDB/mmCIF using gemmi and extract entities directly.

    This bypasses Input.from_mmcif() which requires scheme tables that
    gemmi's PDB→mmCIF converter does not generate.
    """
    import gemmi

    try:
        from alphafold3.constants.chemical_component_sets import IONS
    except (ImportError, KeyError):
        IONS = frozenset()  # type: ignore[assignment]

    # Standard residue names for detecting modifications
    _STANDARD_AA = frozenset({
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
    })
    _STANDARD_RNA = frozenset({"A", "C", "G", "U"})
    _STANDARD_DNA = frozenset({"DA", "DC", "DG", "DT"})

    if fmt == "pdb":
        structure = gemmi.read_pdb_string(content)
    else:
        structure = gemmi.read_pdb_string(content) if not content.lstrip().startswith("data_") else None
        if structure is None or fmt in ("cif", "mmcif"):
            doc = gemmi.cif.read_string(content)
            structure = gemmi.make_structure_from_block(doc.sole_block())

    structure.setup_entities()

    name = structure.name or "structure"
    warnings: list[str] = []
    total_residues = 0

    # Group chains by (type, content_key) for deduplication
    groups: dict[tuple, tuple[str, dict[str, Any], int]] = {}
    group_order: list[tuple] = []
    model = structure[0]  # First model only

    for chain in model:
        polymer = chain.get_polymer()
        if polymer and len(polymer) > 0:
            # Determine polymer type from entity
            entity = structure.get_entity_of(polymer)
            if entity is None:
                continue

            poly_type = entity.polymer_type
            # make_one_letter_sequence() uses '-' for gaps in numbering;
            # strip them to get the actual sequence letters
            seq = polymer.make_one_letter_sequence().replace("-", "")
            total_residues += len(seq)

            if poly_type == gemmi.PolymerType.PeptideL:
                # Extract modification fingerprint for dedup
                mods = tuple(
                    (i + 1, res.name)
                    for i, res in enumerate(polymer)
                    if res.name not in _STANDARD_AA
                )
                content_key = (seq, mods)
                group_key: tuple = ("proteinChain", content_key)
                if group_key in groups:
                    _, payload, count = groups[group_key]
                    groups[group_key] = ("proteinChain", payload, count + 1)
                else:
                    payload: dict[str, Any] = {"sequence": seq, "count": 1}
                    if mods:
                        payload["modifications"] = [
                            {"ptmType": f"CCD_{name}", "ptmPosition": pos}
                            for pos, name in mods
                        ]
                    groups[group_key] = ("proteinChain", payload, 1)
                    group_order.append(group_key)

            elif poly_type in (gemmi.PolymerType.Rna, gemmi.PolymerType.DnaRnaHybrid):
                mods = tuple(
                    (i + 1, res.name)
                    for i, res in enumerate(polymer)
                    if res.name not in _STANDARD_RNA
                )
                content_key = (seq, mods)
                group_key = ("rnaSequence", content_key)
                if group_key in groups:
                    _, payload, count = groups[group_key]
                    groups[group_key] = ("rnaSequence", payload, count + 1)
                else:
                    payload = {"sequence": seq, "count": 1}
                    if mods:
                        payload["modifications"] = [
                            {"modificationType": name, "basePosition": pos}
                            for pos, name in mods
                        ]
                    groups[group_key] = ("rnaSequence", payload, 1)
                    group_order.append(group_key)

            elif poly_type == gemmi.PolymerType.Dna:
                mods = tuple(
                    (i + 1, res.name)
                    for i, res in enumerate(polymer)
                    if res.name not in _STANDARD_DNA
                )
                content_key = (seq, mods)
                group_key = ("dnaSequence", content_key)
                if group_key in groups:
                    _, payload, count = groups[group_key]
                    groups[group_key] = ("dnaSequence", payload, count + 1)
                else:
                    payload = {"sequence": seq, "count": 1}
                    if mods:
                        payload["modifications"] = [
                            {"modificationType": name, "basePosition": pos}
                            for pos, name in mods
                        ]
                    groups[group_key] = ("dnaSequence", payload, 1)
                    group_order.append(group_key)

        # Process ligands/ions (non-polymer, non-water residues)
        # Group by entity (subchain) for multi-residue detection.
        # Independent ligands (e.g. MG + ATP in the same chain) get separate
        # entity/subchain IDs from setup_entities() and are processed
        # individually.  A multi-residue entity (len(ccd_ids) > 1 in plan
        # terms) is skipped with a warning — whether the residues have the
        # same CCD (e.g. ATP×2) or different CCDs (e.g. ALA+GLY).
        ligand_by_entity: dict[str, list] = {}
        for residue in chain.get_ligands():
            if residue.name == "HOH":
                continue
            # Use subchain (label_asym_id) as entity key; fall back to
            # seq-id so independent residues are never merged.
            key = residue.subchain or f"__{residue.seqid}"
            ligand_by_entity.setdefault(key, []).append(residue)

        for entity_key, entity_residues in ligand_by_entity.items():
            # Plan rule: len(ccd_ids) > 1 → skip.  A multi-residue entity
            # can have distinct CCDs (true multi-component, e.g. ALA+GLY)
            # or repeated same CCD (e.g. ATP×2 in one entity).  Either
            # case cannot be represented as a single CCD ligand entry.
            if len(entity_residues) > 1:
                distinct_ccds = {r.name for r in entity_residues}
                if len(distinct_ccds) > 1:
                    ccd_list = ", ".join(sorted(distinct_ccds))
                    detail = f"multi-component ligand ({ccd_list})"
                else:
                    ccd_name = next(iter(distinct_ccds))
                    detail = (
                        f"multi-residue ligand "
                        f"({ccd_name} x{len(entity_residues)})"
                    )
                warnings.append(
                    f"Skipped chain {chain.name}: {detail} is not supported "
                    f"in structure import. Add it manually if needed."
                )
                continue

            for residue in entity_residues:
                ccd_id = residue.name
                total_residues += 1

                if ccd_id in IONS:
                    content_key = (ccd_id,)
                    group_key = ("ion", content_key)
                    if group_key in groups:
                        _, payload, count = groups[group_key]
                        groups[group_key] = ("ion", payload, count + 1)
                    else:
                        payload = {"ion": ccd_id, "count": 1}
                        groups[group_key] = ("ion", payload, 1)
                        group_order.append(group_key)
                else:
                    content_key = (ccd_id,)
                    group_key = ("ligand_ccd", content_key)
                    if group_key in groups:
                        _, payload, count = groups[group_key]
                        groups[group_key] = ("ligand", payload, count + 1)
                    else:
                        payload = {"ligand": f"CCD_{ccd_id}", "count": 1}
                        groups[group_key] = ("ligand", payload, 1)
                        group_order.append(group_key)

    # Build sequences list in order
    sequences: list[dict[str, Any]] = []
    for key in group_order:
        server_type, payload, count = groups[key]
        payload_copy = dict(payload)
        payload_copy["count"] = count
        sequences.append({server_type: payload_copy})

    if not sequences:
        raise ValueError(
            "No supported entities could be imported from this structure."
        )

    return {
        "name": name,
        "sequences": sequences,
        "dialect": "alphafoldserver",
        "version": 1,
        "source": "upload",
        "pdb_id": None,
        "num_chains": sum(
            next(iter(entry.values()))["count"]
            for entry in sequences
        ),
        "num_residues": total_residues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_structure(content: str, fmt: str) -> dict[str, Any]:
    """Parse a PDB or mmCIF string into AlphaFold Server JSON format.

    Uses gemmi directly to parse the structure and extract entities,
    bypassing Input.from_mmcif() which requires PDBx scheme tables
    that gemmi's PDB→mmCIF converter does not generate.

    Args:
        content: The file content as a string.
        fmt: File format - "pdb", "cif", or "mmcif".

    Returns:
        Dict with name, sequences, dialect, version, warnings, source, etc.
    """
    return _parse_with_gemmi(content, fmt)


def fetch_and_parse(pdb_id: str) -> dict[str, Any]:
    """Fetch a structure from RCSB and parse it.

    Args:
        pdb_id: 4-character PDB identifier.

    Returns:
        Dict with name, sequences, dialect, version, warnings, source, pdb_id.
    """
    mmcif_str = fetch_pdb(pdb_id)
    result = parse_structure(mmcif_str, "cif")
    result["source"] = "rcsb"
    result["pdb_id"] = pdb_id.upper()
    result["name"] = pdb_id.upper()
    return result
