"""Output handling for AlphaFold 3 MLX pipeline.

This module provides output file management .

Example:
    output_bundle = create_output_bundle(output_dir, num_samples=5)
    write_ranked_outputs(result, output_bundle, ranking)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from alphafold3_mlx.pipeline.errors import InputError, ResourceError


@dataclass
class OutputBundle:
    """Collection of output files.

    Tracks all output file paths for a single inference run.

    Attributes:
        output_dir: Directory containing all output files.
        structure_files: Ranked structure files [structure_rank_1.cif, ...].
        confidence_scores_file: Path to confidence_scores.json.
        timing_file: Path to timing.json.
        ranking_debug_file: Path to ranking_debug.json.
        failure_log_file: Path to failure_log.json (only on error).
    """

    output_dir: Path
    structure_files: list[Path] = field(default_factory=list)
    confidence_scores_file: Path = field(init=False)
    timing_file: Path = field(init=False)
    ranking_debug_file: Path = field(init=False)
    failure_log_file: Path | None = None

    def __post_init__(self) -> None:
        """Initialize computed file paths."""
        self.confidence_scores_file = self.output_dir / "confidence_scores.json"
        self.timing_file = self.output_dir / "timing.json"
        self.ranking_debug_file = self.output_dir / "ranking_debug.json"

    def structure_path(self, rank: int) -> Path:
        """Get path to structure file by rank (1-indexed).

        Args:
            rank: Structure rank (1 = best).

        Returns:
            Path to the structure file.
        """
        return self.output_dir / f"structure_rank_{rank}.cif"

    def all_files(self) -> list[Path]:
        """Return list of all output file paths."""
        files = list(self.structure_files)
        files.extend([
            self.confidence_scores_file,
            self.timing_file,
            self.ranking_debug_file,
        ])
        if self.failure_log_file:
            files.append(self.failure_log_file)
        return files

    def initialize_structure_files(self, num_samples: int) -> None:
        """Initialize structure file paths for given number of samples.

        Args:
            num_samples: Number of structure samples.
        """
        self.structure_files = [
            self.structure_path(rank) for rank in range(1, num_samples + 1)
        ]


def create_output_directory(
    output_dir: Path,
    required_space_gb: float | None = None,
    num_samples: int | None = None,
    num_residues: int | None = None,
) -> None:
    """Create output directory with disk space check.

    Args:
        output_dir: Directory path to create.
        required_space_gb: Minimum required disk space in GB. If None, estimated
            from num_samples and num_residues.
        num_samples: Number of samples (used for estimation if required_space_gb not set).
        num_residues: Number of residues (used for estimation if required_space_gb not set).

    Raises:
        ResourceError: If insufficient disk space or directory creation fails.
    """
    # Determine required space: explicit value or estimate
    if required_space_gb is None:
        required_space_gb = estimate_output_size_gb(
            num_samples=num_samples if num_samples is not None else 5,
            num_residues=num_residues,
        )

    # Check disk space before creating directory
    parent = output_dir.parent if output_dir.parent.exists() else Path.home()
    available_gb = get_available_disk_space_gb(parent)

    if available_gb < required_space_gb:
        raise ResourceError(
            f"Insufficient disk space. Required: {required_space_gb:.1f}GB, "
            f"Available: {available_gb:.1f}GB"
        )

    # Create directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ResourceError(f"Failed to create output directory: {e}")


def get_available_disk_space_gb(path: Path) -> float:
    """Get available disk space at path in GB.

    Args:
        path: Path to check.

    Returns:
        Available disk space in GB.
    """
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def estimate_output_size_gb(
    num_samples: int = 5,
    num_residues: int | None = None,
) -> float:
    """Estimate output size based on job parameters.

    Calculates expected output size considering:
    - mmCIF structure files (~500 bytes/residue per sample)
    - PAE matrix in confidence_scores.json (O(num_residues²) per sample)
    - Other JSON files (timing, ranking_debug)

    Args:
        num_samples: Number of structure samples to generate.
        num_residues: Total residues (if known). If None, uses conservative default.

    Returns:
        Estimated output size in GB with 2x safety margin.
    """
    # Conservative defaults if num_residues not provided
    if num_residues is None:
        num_residues = 500  # Medium-sized protein assumption

    # Base overhead for JSON files and directory structure
    base_bytes = 1 * 1024 * 1024  # 1 MB

    # mmCIF structure files: ~500 bytes per residue per sample
    # (includes coordinates, pLDDT per atom, formatting)
    structure_bytes = 500 * num_residues * num_samples

    # PAE matrix: 8 bytes per float * num_residues² per sample
    # JSON overhead roughly doubles this
    pae_bytes = 16 * (num_residues ** 2) * num_samples

    # pLDDT per residue: 8 bytes * num_residues * num_samples
    plddt_bytes = 8 * num_residues * num_samples

    # Total with 2x safety margin
    total_bytes = (base_bytes + structure_bytes + pae_bytes + plddt_bytes) * 2

    # Convert to GB with minimum floor of 0.1 GB (100 MB)
    return max(total_bytes / (1024 ** 3), 0.1)


def handle_existing_outputs(
    output_dir: Path,
    num_samples: int,
    no_overwrite: bool = False,
) -> None:
    """Handle existing output files.

    Args:
        output_dir: Output directory to check.
        num_samples: Number of structure samples expected.
        no_overwrite: If True, raise error if files exist.

    Raises:
        InputError: If no_overwrite is True and files exist.
    """
    if not output_dir.exists():
        return

    # Check for existing files
    existing_files = []

    # Check structure files
    for rank in range(1, num_samples + 1):
        structure_path = output_dir / f"structure_rank_{rank}.cif"
        if structure_path.exists():
            existing_files.append(structure_path.name)

    # Check JSON files
    for filename in ["confidence_scores.json", "timing.json", "ranking_debug.json"]:
        if (output_dir / filename).exists():
            existing_files.append(filename)

    if existing_files:
        if no_overwrite:
            raise InputError(
                f"Output directory contains existing files and --no-overwrite is set. "
                f"Remove existing files or omit --no-overwrite to allow overwriting."
            )
        else:
            # Warn that files will be overwritten
            print(
                f"Warning: Existing files will be overwritten: {', '.join(existing_files)}",
                file=sys.stderr,
            )


@contextmanager
def atomic_write(final_path: Path) -> Generator[Path, None, None]:
    """Write to temp file, then atomic rename on success.

    Ensures that output files are either complete or not present,
    preventing corrupted partial files on interrupt.

    Args:
        final_path: Final destination path.

    Yields:
        Temporary file path to write to.

    Raises:
        Any exception from the writing context is re-raised after cleanup.
    """
    # Create temp file in same directory for atomic rename
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    try:
        yield temp_path
        # Success: move (works cross-filesystem, L-01)
        shutil.move(str(temp_path), str(final_path))
    except Exception:
        # Failure: clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_mmcif_file(
    structure_data: dict[str, Any],
    output_path: Path,
    rank: int,
) -> None:
    """Write structure to mmCIF file with atomic write.

    Args:
        structure_data: Structure data from model output.
            Expected keys: coords, plddt, pae, ptm, iptm, and optionally
            atom_mask, aatype, residue_index, asym_id.
        output_path: Path to write mmCIF file.
        rank: Rank of this structure (1 = best).
    """
    with atomic_write(output_path) as temp_path:
        # Try Phase 3's mmCIF generation with full atom data
        try:
            from alphafold3_mlx.model.inference import _generate_mmcif
            import numpy as np

            # Extract required fields
            coords = structure_data.get("coords")
            plddt = structure_data.get("plddt")
            pae = structure_data.get("pae")
            ptm = structure_data.get("ptm", 0.0)
            iptm = structure_data.get("iptm", 0.0)
            atom_mask = structure_data.get("atom_mask")
            aatype = structure_data.get("aatype")
            residue_index = structure_data.get("residue_index")
            asym_id = structure_data.get("asym_id")

            # Infer defaults if not provided
            if coords is not None:
                num_residues = len(coords) if hasattr(coords, "__len__") else 1
                if aatype is None:
                    aatype = np.zeros(num_residues, dtype=np.int32)
                if residue_index is None:
                    residue_index = np.arange(1, num_residues + 1, dtype=np.int32)
                if asym_id is None:
                    asym_id = np.zeros(num_residues, dtype=np.int32)
                if pae is None:
                    pae = np.zeros((num_residues, num_residues), dtype=np.float32)

                mmcif_content = _generate_mmcif(
                    coords=np.asarray(coords),
                    plddt=np.asarray(plddt),
                    pae=np.asarray(pae),
                    ptm=float(ptm),
                    iptm=float(iptm),
                    aatype=np.asarray(aatype),
                    residue_index=np.asarray(residue_index),
                    asym_id=np.asarray(asym_id),
                    atom_mask=np.asarray(atom_mask) if atom_mask is not None else None,
                )
            else:
                # Fallback if coords not provided
                mmcif_content = _generate_minimal_mmcif(structure_data, rank)
        except (ImportError, TypeError, KeyError):
            # Fallback: generate minimal mmCIF
            mmcif_content = _generate_minimal_mmcif(structure_data, rank)

        with open(temp_path, "w") as f:
            f.write(mmcif_content)


def _generate_minimal_mmcif(structure_data: dict[str, Any], rank: int) -> str:
    """Generate minimal mmCIF content (fallback).

    Args:
        structure_data: Structure data dictionary.
        rank: Rank of this structure.

    Returns:
        mmCIF format string.
    """
    lines = [
        "data_alphafold3_mlx",
        "#",
        f"_entry.id alphafold3_mlx_rank_{rank}",
        "#",
    ]
    return "\n".join(lines)


def write_confidence_scores(
    confidence_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Write confidence scores to JSON.

    Args:
        confidence_data: Confidence scores dictionary.
        output_path: Path to write JSON file.
    """
    with atomic_write(output_path) as temp_path:
        with open(temp_path, "w") as f:
            json.dump(confidence_data, f, indent=2)


def write_timing(
    timing_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Write timing data to JSON.

    Args:
        timing_data: Timing data dictionary.
        output_path: Path to write JSON file.
    """
    with atomic_write(output_path) as temp_path:
        with open(temp_path, "w") as f:
            json.dump(timing_data, f, indent=2)


def write_ranking_debug(
    ranking_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Write ranking debug info to JSON.

    Args:
        ranking_data: Ranking debug data dictionary.
        output_path: Path to write JSON file.
    """
    with atomic_write(output_path) as temp_path:
        with open(temp_path, "w") as f:
            json.dump(ranking_data, f, indent=2)


def write_failure_log(
    failure_data: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Write failure log to JSON.

    Args:
        failure_data: Failure log data dictionary.
        output_dir: Output directory.

    Returns:
        Path to written failure log file.
    """
    output_path = output_dir / "failure_log.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    with atomic_write(output_path) as temp_path:
        with open(temp_path, "w") as f:
            json.dump(failure_data, f, indent=2)

    return output_path


def check_disk_space(
    output_dir: Path,
    required_gb: float | None = None,
    num_samples: int | None = None,
    num_residues: int | None = None,
) -> bool:
    """Check if sufficient disk space is available.

    Args:
        output_dir: Directory to check.
        required_gb: Required disk space in GB. If None, estimated from
            num_samples and num_residues.
        num_samples: Number of samples (used for estimation if required_gb not set).
        num_residues: Number of residues (used for estimation if required_gb not set).

    Returns:
        True if sufficient space available, False otherwise.
    """
    # Determine required space: explicit value or estimate
    if required_gb is None:
        required_gb = estimate_output_size_gb(
            num_samples=num_samples if num_samples is not None else 5,
            num_residues=num_residues,
        )

    parent = output_dir if output_dir.exists() else output_dir.parent
    if not parent.exists():
        parent = Path.home()

    available_gb = get_available_disk_space_gb(parent)
    return available_gb >= required_gb


def write_ranked_outputs(
    result: Any,
    output_bundle: OutputBundle,
    ranking: Any,
    timing_data: dict[str, Any] | None = None,
    token_metadata: dict[str, Any] | None = None,
    token_mask: Any | None = None,
    restraint_data: dict[str, Any] | None = None,
) -> None:
    """Write all output files with structures in ranked order.

    This is the main entry point for writing inference outputs. It:
    1. Writes structure files in ranked order (rank 1 = best)
    2. Writes confidence_scores.json with per-sample metrics
    3. Writes ranking_debug.json explaining ranking criteria
    4. Writes timing.json with stage timings (if provided)

    Args:
        result: ModelResult containing predictions and confidence scores.
            Expected to have a to_numpy() method returning a dict with:
            - atom_positions: [num_samples, num_residues, max_atoms, 3]
            - atom_mask: [num_samples, num_residues, max_atoms]
            - plddt: [num_samples, num_residues, max_atoms]
            - pae: [num_samples, num_residues, num_residues]
            - ptm: [num_samples]
            - iptm: [num_samples]
            And a num_samples property.
        output_bundle: OutputBundle with file paths.
        ranking: SampleRanking with ranked_indices and scores.
        timing_data: Optional timing data dict for timing.json.
        token_metadata: Optional dict with aatype, residue_index, asym_id arrays
            from the feature batch. Required for correct multi-chain mmCIF output.
            If not provided, falls back to np_result fields or inferred defaults.
        token_mask: Optional token validity mask to strip bucket padding before
            writing mmCIF outputs.
        restraint_data: Optional dict with restraint satisfaction data.
            Keys: "resolved_distance", "resolved_contact", "resolved_repulsive",
            "restraint_config", "input_json" for computing satisfaction metrics.

    Example:
        >>> write_ranked_outputs(result, output_bundle, ranking, token_metadata=token_metadata)
        # Writes structure_rank_1.cif through structure_rank_N.cif
        # where rank_1 has highest confidence score with correct chain labels
    """
    import numpy as np

    np_result = result.to_numpy()
    num_samples = result.num_samples
    masked_token_metadata = token_metadata
    masked_token_mask = None

    if token_mask is not None:
        mask = np.array(token_mask)
        if mask.ndim > 1:
            mask = mask.reshape(-1)
        mask = mask > 0
        masked_token_mask = mask

        if token_metadata is not None:
            expected_len = mask.shape[0]
            metadata_len = len(token_metadata.get("aatype", []))
            if metadata_len == expected_len:
                masked_token_metadata = {
                    "aatype": np.array(token_metadata["aatype"])[mask],
                    "residue_index": np.array(token_metadata["residue_index"])[mask],
                    "asym_id": np.array(token_metadata["asym_id"])[mask],
                }
        if masked_token_metadata is None:
            expected_len = mask.shape[0]
            if "aatype" in np_result and len(np_result["aatype"]) == expected_len:
                masked_token_metadata = {
                    "aatype": np.array(np_result["aatype"])[mask],
                    "residue_index": np.array(np_result["residue_index"])[mask],
                    "asym_id": np.array(np_result["asym_id"])[mask],
                }

    # Write structure files in ranked order
    for rank, sample_idx in enumerate(ranking.ranked_indices, start=1):
        structure_data = {
            "coords": np_result["atom_positions"][sample_idx],
            "atom_mask": np_result["atom_mask"][sample_idx],
            "plddt": np_result["plddt"][sample_idx],
            "pae": np_result["pae"][sample_idx],
            "ptm": float(np_result["ptm"][sample_idx]),
            "iptm": float(np_result["iptm"][sample_idx]),
        }

        if masked_token_mask is not None:
            token_len = masked_token_mask.shape[0]
            coords = structure_data.get("coords")
            if coords is not None and coords.shape[0] == token_len:
                structure_data["coords"] = coords[masked_token_mask]
            atom_mask = structure_data.get("atom_mask")
            if atom_mask is not None and atom_mask.shape[0] == token_len:
                structure_data["atom_mask"] = atom_mask[masked_token_mask]
            plddt = structure_data.get("plddt")
            if plddt is not None and plddt.shape[0] == token_len:
                structure_data["plddt"] = plddt[masked_token_mask]
            pae = structure_data.get("pae")
            if pae is not None and pae.shape[0] == token_len:
                structure_data["pae"] = pae[np.ix_(masked_token_mask, masked_token_mask)]

        # Use token_metadata from batch (preferred - preserves chain order deterministically)
        # Fall back to np_result fields if token_metadata not provided
        if masked_token_metadata is not None:
            structure_data["aatype"] = masked_token_metadata["aatype"]
            structure_data["residue_index"] = masked_token_metadata["residue_index"]
            structure_data["asym_id"] = masked_token_metadata["asym_id"]
        else:
            # Fallback: check np_result (legacy path)
            if "aatype" in np_result:
                structure_data["aatype"] = np_result["aatype"]
            if "residue_index" in np_result:
                structure_data["residue_index"] = np_result["residue_index"]
            if "asym_id" in np_result:
                structure_data["asym_id"] = np_result["asym_id"]

        write_mmcif_file(
            structure_data,
            output_bundle.structure_path(rank),
            rank=rank,
        )

    # Write confidence scores
    confidence_data = _build_confidence_scores_dict(
        np_result, ranking, num_samples, restraint_data=restraint_data,
    )
    write_confidence_scores(confidence_data, output_bundle.confidence_scores_file)

    # Write ranking debug
    ranking_debug_data = ranking.to_ranking_debug_dict()
    write_ranking_debug(ranking_debug_data, output_bundle.ranking_debug_file)

    # Write timing
    if timing_data is not None:
        write_timing(timing_data, output_bundle.timing_file)


def _build_confidence_scores_dict(
    np_result: dict[str, Any],
    ranking: Any,
    num_samples: int,
    restraint_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build confidence_scores.json content.

    Per spec, plddt is per-residue (mean over atoms using atom_mask),
    NOT the raw per-atom values.

    Args:
        np_result: NumPy result dictionary from model.
            - plddt: [num_samples, num_residues, max_atoms] per-atom confidence
            - atom_mask: [num_samples, num_residues, max_atoms] validity mask
            - pae: [num_samples, num_residues, num_residues]
            - ptm, iptm: [num_samples]
        ranking: SampleRanking with ranking information.
        num_samples: Number of samples.
        restraint_data: Optional restraint satisfaction data dict.

    Returns:
        Dictionary formatted for confidence_scores.json.
    """
    import numpy as np

    # Build sample_index → rank lookup (rank 1 = best)
    index_to_rank = {idx: rank for rank, idx in enumerate(ranking.ranked_indices, start=1)}

    confidence_data: dict[str, Any] = {
        "num_samples": num_samples,
        "ranking_metric": ranking.ranking_metric,
        "is_complex": ranking.is_complex,
        "samples": {},
        "best_sample_index": ranking.best_index,
    }

    # Get atom_mask for computing per-residue pLDDT
    atom_mask = np_result.get("atom_mask")

    for i in range(num_samples):
        plddt_per_atom = np_result["plddt"][i]  # [num_residues, max_atoms]
        pae = np_result["pae"][i]  # [num_residues, num_residues]

        # Compute per-residue pLDDT: mean over atoms using atom_mask
        if atom_mask is not None:
            sample_mask = atom_mask[i]  # [num_residues, max_atoms]
            if plddt_per_atom.shape[-1] != sample_mask.shape[-1]:
                atom_dim = min(plddt_per_atom.shape[-1], sample_mask.shape[-1])
                plddt_per_atom = plddt_per_atom[:, :atom_dim]
                sample_mask = sample_mask[:, :atom_dim]
            # Masked mean: sum(plddt * mask) / sum(mask) for each residue
            masked_plddt = plddt_per_atom * sample_mask
            residue_plddt = masked_plddt.sum(axis=-1) / np.maximum(sample_mask.sum(axis=-1), 1.0)
            # Residue is valid if it has at least one valid atom (excludes bucketing padding)
            residue_mask = sample_mask.sum(axis=-1) > 0  # [num_residues]
        else:
            # Fallback: simple mean over atoms (treats all atoms as valid)
            residue_plddt = plddt_per_atom.mean(axis=-1)
            residue_mask = np.ones(len(residue_plddt), dtype=bool)

        # Trim to valid residues only (removes bucketing padding from output)
        valid_plddt = residue_plddt[residue_mask]
        valid_pae = pae[np.ix_(residue_mask, residue_mask)]  # Trim both dimensions

        # Build flat per-atom pLDDT array (matching golden atom_plddts format)
        # For each valid residue, include pLDDT for each valid atom
        atom_plddts_flat = []
        plddt_src = np_result["plddt"][i]  # [num_residues, max_atoms]
        if atom_mask is not None:
            src_mask = atom_mask[i]
        else:
            src_mask = np.ones_like(plddt_src)
        for res_i in range(len(residue_mask)):
            if not residue_mask[res_i]:
                continue
            for atom_j in range(plddt_src.shape[-1]):
                if atom_j < src_mask.shape[-1] and src_mask[res_i, atom_j] > 0.5:
                    atom_plddts_flat.append(float(plddt_src[res_i, atom_j]))

        # Convert to lists for JSON serialization
        plddt_list = valid_plddt.tolist() if hasattr(valid_plddt, "tolist") else list(valid_plddt)
        pae_list = valid_pae.tolist() if hasattr(valid_pae, "tolist") else list(valid_pae)

        # Compute mean_plddt from valid residues
        mean_plddt = float(valid_plddt.mean()) if valid_plddt.size > 0 else 0.0

        sample_dict: dict[str, Any] = {
            "plddt": plddt_list,
            "atom_plddts": atom_plddts_flat,
            "pae": pae_list,
            "ptm": float(np_result["ptm"][i]),
            "iptm": float(np_result["iptm"][i]),
            "mean_plddt": mean_plddt,
            "rank": index_to_rank.get(i, i + 1),
        }

        # Add restraint satisfaction per-sample
        if restraint_data is not None:
            satisfaction = _compute_restraint_satisfaction(
                np_result, i, restraint_data,
            )
            if satisfaction is not None:
                sample_dict["restraint_satisfaction"] = satisfaction

        confidence_data["samples"][str(i)] = sample_dict

    return confidence_data


def _compute_restraint_satisfaction(
    np_result: dict[str, Any],
    sample_idx: int,
    restraint_data: dict[str, Any],
) -> dict[str, list[dict[str, Any]]] | None:
    """Compute per-restraint satisfaction for a single sample.

    Computes actual distances from final atom positions and compares
    against restraint targets.

    Args:
        np_result: NumPy result dict with atom_positions.
        sample_idx: Index of the current sample.
        restraint_data: Dict with resolved restraints and original config.

    Returns:
        Satisfaction dict with "distance", "contact", "repulsive" lists,
        or None if no restraints.
    """
    import numpy as np

    positions = np_result["atom_positions"][sample_idx]  # [num_tokens, num_atoms, 3]

    resolved_distance = restraint_data.get("resolved_distance", [])
    restraint_config = restraint_data.get("restraint_config")

    if not resolved_distance and restraint_config is None:
        return None

    satisfaction: dict[str, list[dict[str, Any]]] = {}

    # Distance satisfaction
    if resolved_distance and restraint_config is not None:
        distance_sat = []
        for r, orig in zip(resolved_distance, restraint_config.distance):
            pos_i = positions[r.atom_i_idx[0], r.atom_i_idx[1]]
            pos_j = positions[r.atom_j_idx[0], r.atom_j_idx[1]]
            diff = pos_i - pos_j
            actual_dist = float(np.sqrt(np.sum(diff * diff) + 1e-8))
            # Satisfied if within 3 * sigma
            is_satisfied = abs(actual_dist - r.target_distance) <= 3 * r.sigma
            distance_sat.append({
                "chain_i": orig.chain_i,
                "residue_i": orig.residue_i,
                "atom_i": orig.atom_i,
                "chain_j": orig.chain_j,
                "residue_j": orig.residue_j,
                "atom_j": orig.atom_j,
                "target_distance": r.target_distance,
                "actual_distance": round(actual_dist, 3),
                "satisfied": is_satisfied,
            })
        satisfaction["distance"] = distance_sat

    # Contact satisfaction (data-model.md §7b)
    resolved_contact = restraint_data.get("resolved_contact", [])
    if resolved_contact and restraint_config is not None:
        contact_sat = []
        for r, orig in zip(resolved_contact, restraint_config.contact):
            pos_source = positions[r.source_atom_idx[0], r.source_atom_idx[1]]
            # Find closest candidate
            best_dist = float("inf")
            best_cand_idx = 0
            for ci, cand_idx in enumerate(r.candidate_atom_idxs):
                pos_cand = positions[cand_idx[0], cand_idx[1]]
                diff = pos_source - pos_cand
                d = float(np.sqrt(np.sum(diff * diff) + 1e-8))
                if d < best_dist:
                    best_dist = d
                    best_cand_idx = ci
            closest_cand = orig.candidates[best_cand_idx]
            contact_sat.append({
                "chain_i": orig.chain_i,
                "residue_i": orig.residue_i,
                "closest_candidate_chain": closest_cand.chain_j,
                "closest_candidate_residue": closest_cand.residue_j,
                "threshold": r.threshold,
                "actual_distance": round(best_dist, 3),
                "satisfied": best_dist <= r.threshold,
            })
        satisfaction["contact"] = contact_sat

    # Repulsive satisfaction (data-model.md §7c)
    resolved_repulsive = restraint_data.get("resolved_repulsive", [])
    if resolved_repulsive and restraint_config is not None:
        repulsive_sat = []
        for r, orig in zip(resolved_repulsive, restraint_config.repulsive):
            pos_i = positions[r.atom_i_idx[0], r.atom_i_idx[1]]
            pos_j = positions[r.atom_j_idx[0], r.atom_j_idx[1]]
            diff = pos_i - pos_j
            actual_dist = float(np.sqrt(np.sum(diff * diff) + 1e-8))
            repulsive_sat.append({
                "chain_i": orig.chain_i,
                "residue_i": orig.residue_i,
                "chain_j": orig.chain_j,
                "residue_j": orig.residue_j,
                "min_distance": r.min_distance,
                "actual_distance": round(actual_dist, 3),
                "satisfied": actual_dist >= r.min_distance,
            })
        satisfaction["repulsive"] = repulsive_sat

    return satisfaction if satisfaction else None
