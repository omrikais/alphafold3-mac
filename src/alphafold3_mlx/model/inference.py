"""Inference orchestration for AlphaFold 3 MLX.

This module provides high-level inference utilities including:
- End-to-end inference with mmCIF output
- Memory checkpointing at each component
- NaN detection after each major component
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx

from alphafold3_mlx.core.validation import check_nan, NaNCheckpoint, get_available_memory_gb

if TYPE_CHECKING:
    from typing import Callable
    from alphafold3_mlx.core.inputs import FeatureBatch
    from alphafold3_mlx.core.outputs import ModelResult
    from alphafold3_mlx.model.model import Model


@dataclass
class InferenceStats:
    """Statistics from inference run."""

    total_duration_seconds: float = 0.0
    evoformer_duration_seconds: float = 0.0
    diffusion_duration_seconds: float = 0.0
    confidence_duration_seconds: float = 0.0

    peak_memory_gb: float = 0.0
    memory_checkpoints: dict[str, float] = field(default_factory=dict)

    num_residues: int = 0
    num_samples: int = 0
    num_recycles: int = 0


def run_inference(
    model: "Model",
    batch: "FeatureBatch",
    key: mx.array | None = None,
    output_path: str | Path | None = None,
    track_memory: bool = True,
    check_nans: bool = True,
    diffusion_callback: "Callable[[int, int], None] | None" = None,
    recycling_callback: "Callable[[int, int], None] | None" = None,
    confidence_callback: "Callable[[str], None] | None" = None,
    guidance_fn: "Callable | None" = None,
) -> tuple["ModelResult", InferenceStats]:
    """Run inference with full monitoring.

    This function orchestrates inference with:
    - Per-component timing
    - Memory checkpoint logging
    - NaN detection at each stage
    - Optional mmCIF output
    - Progress callbacks for diffusion, recycling, and confidence
    - Restraint guidance via guidance_fn

    Args:
        model: AlphaFold 3 model.
        batch: Input features.
        key: Random key (auto-generated if None).
        output_path: Optional path for mmCIF output.
        track_memory: Whether to track memory usage.
        check_nans: Whether to check for NaN values.
        diffusion_callback: Optional callback called after each diffusion step
            with (step, total_steps) for progress reporting.
        recycling_callback: Optional callback called after each recycling
            iteration with (iteration, total) for progress reporting.
        confidence_callback: Optional callback called with "start" or "end"
            to signal confidence computation phase.
        guidance_fn: Optional restraint guidance function built by
            build_guidance_fn. Passed through to Model.__call__.

    Returns:
        Tuple of (ModelResult, InferenceStats).

    Raises:
        NaNError: If NaN values detected and check_nans=True.
    """
    stats = InferenceStats()
    stats.num_residues = batch.num_residues
    stats.num_samples = model.config.diffusion.num_samples
    stats.num_recycles = model.config.num_recycles

    start_time = time.time()

    # Memory check before processing
    if track_memory:
        initial_memory = _get_current_memory_gb()
        stats.memory_checkpoints["initial"] = initial_memory

    # Generate key if not provided
    if key is None:
        key = mx.random.key(int(time.time() * 1000) % 2**32)

    # Run model with check_nans flag, callbacks, and optional guidance
    result = model(
        batch,
        key=key,
        check_nans=check_nans,
        diffusion_callback=diffusion_callback,
        recycling_callback=recycling_callback,
        confidence_callback=confidence_callback,
        guidance_fn=guidance_fn,
    )

    # Record timing
    stats.total_duration_seconds = time.time() - start_time

    # Populate per-component timing from model metadata
    metadata = result.metadata
    if "evoformer_duration_seconds" in metadata:
        stats.evoformer_duration_seconds = metadata["evoformer_duration_seconds"]
    if "diffusion_duration_seconds" in metadata:
        stats.diffusion_duration_seconds = metadata["diffusion_duration_seconds"]
    if "confidence_duration_seconds" in metadata:
        stats.confidence_duration_seconds = metadata["confidence_duration_seconds"]

    # Memory checkpoint logging
    if track_memory:
        final_memory = _get_current_memory_gb()
        stats.memory_checkpoints["final"] = final_memory
        stats.peak_memory_gb = max(stats.memory_checkpoints.values())

    # Write mmCIF output
    if output_path is not None:
        write_mmcif_output(result, batch, output_path)

    return result, stats


def run_inference_with_checkpoints(
    model: "Model",
    batch: "FeatureBatch",
    key: mx.array | None = None,
) -> tuple["ModelResult", dict[str, mx.array]]:
    """Run inference with intermediate checkpoint capture.

    Captures activations at key validation checkpoints:
    - evoformer_single: Single representation after Evoformer [batch, seq, seq_channel]
    - evoformer_pair: Pair representation after Evoformer [batch, seq, seq, pair_channel]
    - diffusion_coords_final: Final diffusion coordinates [samples, batch, seq, 37, 3]
    - confidence_plddt: Per-atom pLDDT scores [samples, seq, 37]
    - confidence_pae: PAE matrix [samples, seq, seq]
    - confidence_ptm: pTM scores [samples]

    Useful for validation against JAX reference and debugging.

    Args:
        model: AlphaFold 3 model.
        batch: Input features.
        key: Random key.

    Returns:
        Tuple of (ModelResult, checkpoints_dict).
    """
    if key is None:
        warnings.warn(
            "No seed provided; using default seed=42 for reproducibility",
            stacklevel=2,
        )
        key = mx.random.key(42)

    # Run model with checkpoint capture enabled
    result = model(batch, key=key, capture_checkpoints=True)

    # Extract checkpoints from metadata
    checkpoints: dict[str, mx.array] = {}

    if "checkpoints" in result.metadata:
        checkpoints = result.metadata["checkpoints"]

    # Also include final embeddings if available (for backwards compatibility)
    if result.embeddings is not None:
        checkpoints["single_final"] = result.embeddings.single
        checkpoints["pair_final"] = result.embeddings.pair

    return result, checkpoints


def write_mmcif_output(
    result: "ModelResult",
    batch: "FeatureBatch",
    output_path: str | Path,
    sample_index: int | None = None,
) -> Path:
    """Write prediction to mmCIF format.

    Generates mmCIF file with:
    - Predicted coordinates
    - Per-atom B-factors from pLDDT
    - PAE data in _ma_qa_metric_global table

    Args:
        result: Model prediction result.
        batch: Input features (for sequence information).
        output_path: Output file path.
        sample_index: Sample to write (default: best by pLDDT).

    Returns:
        Path to written file.
    """
    import numpy as np

    output_path = Path(output_path)

    # Select sample
    if sample_index is None:
        sample_index = result.best_sample_index

    # Convert to numpy
    np_result = result.to_numpy()
    coords = np_result["atom_positions"][sample_index]  # [N, 37, 3]
    atom_mask = np_result["atom_mask"][sample_index]  # [N, 37]
    plddt = np_result["plddt"][sample_index]  # [N, 37]
    pae = np_result["pae"][sample_index]  # [N, N]
    ptm = float(np_result["ptm"][sample_index])
    iptm = float(np_result["iptm"][sample_index])

    # Get sequence info
    aatype = np.array(batch.token_features.aatype)
    residue_index = np.array(batch.token_features.residue_index)
    asym_id = np.array(batch.token_features.asym_id)

    # Generate mmCIF content with full atom37 and PAE data
    mmcif_content = _generate_mmcif(
        coords=coords,
        plddt=plddt,
        pae=pae,
        ptm=ptm,
        iptm=iptm,
        aatype=aatype,
        residue_index=residue_index,
        asym_id=asym_id,
        atom_mask=atom_mask,
    )

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(mmcif_content)

    return output_path


def _generate_mmcif(
    coords: "np.ndarray",
    plddt: "np.ndarray",
    pae: "np.ndarray",
    ptm: float,
    iptm: float,
    aatype: "np.ndarray",
    residue_index: "np.ndarray",
    asym_id: "np.ndarray",
    atom_mask: "np.ndarray | None" = None,
) -> str:
    """Generate mmCIF file content with full atom37 and PAE data.

    Args:
        coords: Atom coordinates [N, 37, 3].
        plddt: Per-atom confidence [N, 37].
        pae: Predicted aligned error [N, N].
        ptm: Global TM-score.
        iptm: Interface TM-score.
        aatype: Amino acid types [N].
        residue_index: Residue indices [N].
        asym_id: Chain IDs [N].
        atom_mask: Optional atom validity mask [N, 37].

    Returns:
        mmCIF format string.
    """
    import numpy as np
    from alphafold3_mlx.core.atom_constants import (
        ATOM37_NAMES,
        RESTYPE_ATOM37_MASK,
        NUM_ATOMS,
    )
    from alphafold3.model.protein_data_processing import (
        PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37,
    )

    # Three-letter amino acid codes
    AA_CODES = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK",
    ]

    # Element symbols for atom names
    ELEMENT_MAP = {
        "N": "N", "C": "C", "O": "O", "S": "S", "H": "H",
    }

    def get_element(atom_name: str) -> str:
        """Extract element symbol from atom name.

        For protein backbone/sidechain atoms, the element is determined by
        the first character (C, N, O, S, H). Multi-character elements like
        CL, BR, FE, ZN, MG, SE are only for ligand/metal atoms which are
        explicitly named (e.g., "CL1" for chlorine, not "CA" for C-alpha).

        The "CA" atom in proteins is Carbon Alpha (element "C"), NOT Calcium.
        """
        # Strip whitespace from atom name
        atom_name = atom_name.strip()

        # Standard protein atom names: first character is the element
        # CA = Carbon Alpha (element C), CB = Carbon Beta (element C), etc.
        # N = Nitrogen, O = Oxygen, S = Sulfur (in CYS, MET), H = Hydrogen

        # Check for multi-character elements that are EXPLICITLY named as such
        # These are typically ligand/metal atoms, NOT backbone atoms
        # E.g., "CL" = Chlorine (explicit), "FE" = Iron, "ZN" = Zinc
        # But "CA" in protein context = Carbon Alpha, NOT Calcium
        if len(atom_name) >= 2:
            two_char = atom_name[:2].upper()
            # Only match elements that aren't ambiguous with protein atoms
            # CL, BR are clear (no protein atom starts with CL or BR)
            # FE, ZN, MG, SE are clear (no protein atoms)
            # CA is AMBIGUOUS - in proteins it's Carbon Alpha (C), not Calcium
            # NA is AMBIGUOUS - could be backbone N, not Sodium
            if two_char in {"CL", "BR", "FE", "ZN", "MG", "SE"}:
                return two_char

        # Standard single-character elements for protein atoms
        if atom_name and atom_name[0] in "CNOSHP":
            return atom_name[0]

        return atom_name[0].upper() if atom_name else "X"

    num_residues = len(aatype)

    # Convert non-atom37 layouts (e.g. dense-24) to atom37 for mmCIF output.
    if (
        coords.shape[1] != NUM_ATOMS
        or plddt.shape[1] != NUM_ATOMS
        or (atom_mask is not None and atom_mask.shape[1] != NUM_ATOMS)
    ):
        dense_to_atom37_table = np.asarray(PROTEIN_AATYPE_DENSE_ATOM_TO_ATOM37)
        aatype_safe = np.clip(aatype.astype(np.int32), 0, dense_to_atom37_table.shape[0] - 1)

        num_dense = next(
            d for d in (
                coords.shape[1],
                plddt.shape[1],
                atom_mask.shape[1] if atom_mask is not None else None,
            )
            if d is not None and d != NUM_ATOMS
        )

        coords_atom37 = (
            np.array(coords, copy=True)
            if coords.shape[1] == NUM_ATOMS
            else np.zeros((num_residues, NUM_ATOMS, 3), dtype=coords.dtype)
        )
        plddt_atom37 = (
            np.array(plddt, copy=True)
            if plddt.shape[1] == NUM_ATOMS
            else np.zeros((num_residues, NUM_ATOMS), dtype=plddt.dtype)
        )
        mask_atom37 = (
            np.array(atom_mask, copy=True)
            if atom_mask is not None and atom_mask.shape[1] == NUM_ATOMS
            else np.zeros((num_residues, NUM_ATOMS), dtype=np.float32)
        )

        dense_mask_available = atom_mask is not None and atom_mask.shape[1] == num_dense
        written = np.zeros((num_residues, NUM_ATOMS), dtype=bool)

        for i in range(num_residues):
            dense_to_atom37 = dense_to_atom37_table[aatype_safe[i]]
            if dense_to_atom37.shape[0] > num_dense:
                dense_to_atom37 = dense_to_atom37[:num_dense]

            if dense_mask_available:
                valid_dense = atom_mask[i] > 0.5
            else:
                valid_dense = np.ones((num_dense,), dtype=bool)
            valid_dense = valid_dense & (dense_to_atom37 >= 0) & (dense_to_atom37 < NUM_ATOMS)
            atom37_idx = dense_to_atom37[valid_dense]
            dense_idx = np.nonzero(valid_dense)[0]

            for d_idx, a_idx in zip(dense_idx.tolist(), atom37_idx.tolist()):
                if not dense_mask_available and written[i, a_idx]:
                    continue

                if coords.shape[1] == num_dense:
                    coords_atom37[i, a_idx] = coords[i, d_idx]
                if plddt.shape[1] == num_dense:
                    plddt_atom37[i, a_idx] = plddt[i, d_idx]
                if dense_mask_available:
                    mask_atom37[i, a_idx] = atom_mask[i, d_idx]
                else:
                    mask_atom37[i, a_idx] = 1.0
                written[i, a_idx] = True

        coords = coords_atom37
        plddt = plddt_atom37
        atom_mask = mask_atom37

    # Compute atom mask from residue types if not provided
    if atom_mask is None:
        atom_mask = np.zeros((num_residues, NUM_ATOMS), dtype=np.float32)
        for i, aa in enumerate(aatype):
            aa_idx = min(int(aa), len(AA_CODES) - 1)
            atom_mask[i] = RESTYPE_ATOM37_MASK[aa_idx]

    lines = []
    lines.append("data_alphafold3_mlx_prediction")
    lines.append("#")

    # Entry info
    lines.append("_entry.id   alphafold3_mlx")
    lines.append("#")

    # Global quality metrics
    lines.append("loop_")
    lines.append("_ma_qa_metric_global.id")
    lines.append("_ma_qa_metric_global.model_id")
    lines.append("_ma_qa_metric_global.metric_value")
    lines.append("_ma_qa_metric_global.metric_type")
    lines.append(f"1 1 {ptm:.4f} pTM")
    lines.append(f"2 1 {iptm:.4f} ipTM")
    lines.append("#")

    # Atom site table with all atom37 atoms
    lines.append("loop_")
    lines.append("_atom_site.group_PDB")
    lines.append("_atom_site.id")
    lines.append("_atom_site.type_symbol")
    lines.append("_atom_site.label_atom_id")
    lines.append("_atom_site.label_comp_id")
    lines.append("_atom_site.label_asym_id")
    lines.append("_atom_site.label_seq_id")
    lines.append("_atom_site.Cartn_x")
    lines.append("_atom_site.Cartn_y")
    lines.append("_atom_site.Cartn_z")
    lines.append("_atom_site.B_iso_or_equiv")
    lines.append("_atom_site.occupancy")

    atom_id = 1
    for res_idx in range(num_residues):
        aa_type = int(aatype[res_idx])
        if aa_type >= len(AA_CODES):
            aa_type = len(AA_CODES) - 1  # UNK
        comp_id = AA_CODES[aa_type]

        chain_id = chr(ord("A") + (int(asym_id[res_idx]) - 1) % 26)
        seq_id = int(residue_index[res_idx])

        # Write all valid atoms for this residue
        for atom_idx in range(NUM_ATOMS):
            # Skip atoms that don't exist for this residue type
            if atom_mask[res_idx, atom_idx] < 0.5:
                continue

            x, y, z = coords[res_idx, atom_idx]
            b_factor = plddt[res_idx, atom_idx]
            atom_name = ATOM37_NAMES[atom_idx]
            element = get_element(atom_name)

            lines.append(
                f"ATOM   {atom_id:5d} {element:2s} {atom_name:4s} {comp_id:3s} "
                f"{chain_id:1s} {seq_id:4d}    "
                f"{x:8.3f} {y:8.3f} {z:8.3f} {b_factor:6.2f} 1.00"
            )
            atom_id += 1

    lines.append("#")

    # PAE data as _ma_qa_metric_local_pairwise table
    # This table stores the predicted aligned error for each residue pair
    lines.append("loop_")
    lines.append("_ma_qa_metric_local_pairwise.id")
    lines.append("_ma_qa_metric_local_pairwise.model_id")
    lines.append("_ma_qa_metric_local_pairwise.label_asym_id_1")
    lines.append("_ma_qa_metric_local_pairwise.label_seq_id_1")
    lines.append("_ma_qa_metric_local_pairwise.label_asym_id_2")
    lines.append("_ma_qa_metric_local_pairwise.label_seq_id_2")
    lines.append("_ma_qa_metric_local_pairwise.metric_value")
    lines.append("_ma_qa_metric_local_pairwise.metric_type")

    pae_id = 1
    for i in range(num_residues):
        chain_i = chr(ord("A") + (int(asym_id[i]) - 1) % 26)
        seq_i = int(residue_index[i])
        for j in range(num_residues):
            chain_j = chr(ord("A") + (int(asym_id[j]) - 1) % 26)
            seq_j = int(residue_index[j])
            pae_value = pae[i, j]
            lines.append(
                f"{pae_id} 1 {chain_i} {seq_i} {chain_j} {seq_j} {pae_value:.2f} PAE"
            )
            pae_id += 1

    lines.append("#")

    return "\n".join(lines)


def _get_current_memory_gb() -> float:
    """Get current MLX memory usage in GB."""
    try:
        # MLX memory tracking
        # Note: mx.metal.get_active_memory() is deprecated
        # Use mx.get_peak_memory() instead
        peak_bytes = mx.get_peak_memory()
        return peak_bytes / (1024 ** 3)
    except AttributeError:
        # Fallback if function not available
        return 0.0
