"""Pre-flight validation for AlphaFold 3 data pipeline on macOS.

This module provides validation utilities to check that all required
databases and dependencies are available before running the data pipeline.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path


class DatabaseNotFoundError(Exception):
    """Raised when required database paths are inaccessible.

    Attributes:
        missing_paths: List of paths that were not found or not readable.
    """

    def __init__(self, message: str, missing_paths: list[str]):
        super().__init__(message)
        self.missing_paths = missing_paths


@dataclass
class DatabaseConfig:
    """Configuration for required sequence databases.

    Attributes:
        uniref90_path: Path to UniRef90 database.
        bfd_path: Path to BFD database (optional for some workflows).
        mgnify_path: Path to MGnify database (optional for some workflows).
        pdb70_path: Path to PDB70 template database (optional).
        pdb_mmcif_path: Path to PDB mmCIF structure files (optional).
    """

    uniref90_path: Path | None = None
    bfd_path: Path | None = None
    mgnify_path: Path | None = None
    pdb70_path: Path | None = None
    pdb_mmcif_path: Path | None = None

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables.

        Recognized environment variables (M-15: AF3_* names preferred):
            AF3_UNIREF90_DB / UNIREF90_PATH: Path to UniRef90 database
            AF3_BFD_DB / BFD_PATH: Path to BFD database
            AF3_MGNIFY_DB / MGNIFY_PATH: Path to MGnify database
            AF3_PDB70_DB / PDB70_PATH: Path to PDB70 database
            AF3_PDB_MMCIF_DIR / AF3_PDB_MMCIF_DB / PDB_MMCIF_PATH: Path to PDB mmCIF files

        Returns:
            DatabaseConfig populated from environment.
        """

        def _get_path(new_var: str, *legacy_vars: str) -> Path | None:
            value = os.environ.get(new_var)
            if value:
                return Path(value)
            for legacy_var in legacy_vars:
                value = os.environ.get(legacy_var)
                if value:
                    warnings.warn(
                        f"Env var {legacy_var} is deprecated, use {new_var}",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    return Path(value)
            return None

        return cls(
            uniref90_path=_get_path("AF3_UNIREF90_DB", "UNIREF90_PATH"),
            bfd_path=_get_path("AF3_BFD_DB", "BFD_PATH"),
            mgnify_path=_get_path("AF3_MGNIFY_DB", "MGNIFY_PATH"),
            pdb70_path=_get_path("AF3_PDB70_DB", "PDB70_PATH"),
            pdb_mmcif_path=_get_path("AF3_PDB_MMCIF_DIR", "AF3_PDB_MMCIF_DB", "PDB_MMCIF_PATH"),
        )


def validate_database_paths(
    config: DatabaseConfig | None = None,
    *,
    require_all: bool = False,
) -> dict[str, bool]:
    """Verify that all required database paths exist and are readable.

    Args:
        config: Database configuration. If None, reads from environment.
        require_all: If True, raise error if any database is missing.
                    If False, only check paths that are configured.

    Returns:
        Dict mapping database name to availability status.

    Raises:
        DatabaseNotFoundError: If require_all=True and any configured
            database path is inaccessible.

    Example:
        >>> from alphafold3_mlx.data import validate_database_paths
        >>> status = validate_database_paths()
        >>> if not status.get("uniref90", False):
        ...     print("UniRef90 not available - MSA search will fail")

    Note:
        For pre-flight validation before running the pipeline, use
        `validate_pipeline_requirements()` which requires all configured
        databases to be accessible.
    """
    if config is None:
        config = DatabaseConfig.from_env()

    # Database paths to check
    databases = {
        "uniref90": config.uniref90_path,
        "bfd": config.bfd_path,
        "mgnify": config.mgnify_path,
        "pdb70": config.pdb70_path,
        "pdb_mmcif": config.pdb_mmcif_path,
    }

    status: dict[str, bool] = {}
    missing: list[str] = []

    for name, path in databases.items():
        if path is None:
            status[name] = False
            if require_all:
                missing.append(f"{name}: not configured")
        elif not path.exists():
            status[name] = False
            if require_all:
                missing.append(f"{name}: {path} does not exist")
        elif not os.access(path, os.R_OK):
            status[name] = False
            if require_all:
                missing.append(f"{name}: {path} is not readable")
        else:
            status[name] = True

    if missing:
        raise DatabaseNotFoundError(
            f"Required databases are inaccessible:\n" + "\n".join(f"  - {m}" for m in missing),
            missing_paths=missing,
        )

    return status


def validate_pipeline_requirements(
    config: DatabaseConfig | None = None,
    *,
    check_hmmer: bool = True,
    require_all_databases: bool = True,
    require_uniref90: bool | None = None,
    require_bfd: bool | None = None,
    require_mgnify: bool | None = None,
    require_pdb70: bool | None = None,
    require_pdb_mmcif: bool | None = None,
) -> None:
    """Fail-fast validation for all pipeline requirements.

    This function should be called before running the data pipeline to ensure
    all required databases and tools are available. It raises an exception
    listing ALL missing requirements (not just the first one found).

    By default, all databases are required (require_all_databases=True) to
    match the spec requirement of listing all missing databases before search.
    For partial pipelines, set require_all_databases=False and specify which
    databases are needed.

    Args:
        config: Database configuration. If None, reads from environment.
        check_hmmer: Whether to also validate HMMER installation.
        require_all_databases: If True (default), require all databases to be
            configured. Individual require_* flags override this for specific DBs.
        require_uniref90: Override for UniRef90 requirement.
        require_bfd: Override for BFD requirement.
        require_mgnify: Override for MGnify requirement.
        require_pdb70: Override for PDB70 requirement.
        require_pdb_mmcif: Override for PDB mmCIF requirement.

    Raises:
        DatabaseNotFoundError: If any required database is not configured
            or inaccessible. The exception's missing_paths attribute lists
            all issues found.

    Example:
        >>> from alphafold3_mlx.data import validate_pipeline_requirements
        >>> # Full pipeline validation (default - all databases required)
        >>> validate_pipeline_requirements()
        >>>
        >>> # MSA-only validation (just UniRef90)
        >>> validate_pipeline_requirements(
        ...     require_all_databases=False,
        ...     require_uniref90=True
        ... )
    """
    errors: list[str] = []

    # Check databases - collect all errors, don't stop at first
    if config is None:
        config = DatabaseConfig.from_env()

    # Resolve individual requirements (override takes precedence over require_all_databases)
    def _is_required(override: bool | None) -> bool:
        if override is not None:
            return override
        return require_all_databases

    # Define which databases are required vs optional
    databases = {
        "uniref90": (config.uniref90_path, _is_required(require_uniref90)),
        "bfd": (config.bfd_path, _is_required(require_bfd)),
        "mgnify": (config.mgnify_path, _is_required(require_mgnify)),
        "pdb70": (config.pdb70_path, _is_required(require_pdb70)),
        "pdb_mmcif": (config.pdb_mmcif_path, _is_required(require_pdb_mmcif)),
    }

    for name, (path, required) in databases.items():
        if path is None:
            if required:
                errors.append(f"{name}: not configured (set {name.upper()}_PATH environment variable)")
        elif not path.exists():
            errors.append(f"{name}: {path} does not exist")
        elif not os.access(path, os.R_OK):
            errors.append(f"{name}: {path} is not readable")

    # Check HMMER installation
    if check_hmmer:
        hmmer_valid, hmmer_msg = validate_hmmer_installation()
        if not hmmer_valid:
            errors.append(f"hmmer: {hmmer_msg}")

    if errors:
        raise DatabaseNotFoundError(
            f"Pipeline requirements not met ({len(errors)} issue(s)):\n"
            + "\n".join(f"  - {e}" for e in errors),
            missing_paths=errors,
        )


def validate_hmmer_installation() -> tuple[bool, str]:
    """Check that HMMER is installed with the seq_limit patch.

    Returns:
        Tuple of (is_valid, message).
        is_valid is True if HMMER is properly installed with the patch.
        message contains version info or error details.

    Example:
        >>> valid, msg = validate_hmmer_installation()
        >>> if not valid:
        ...     print(f"HMMER issue: {msg}")
    """
    try:
        result = subprocess.run(
            ["jackhmmer", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return False, f"jackhmmer returned error: {result.stderr}"

        # Check both stdout and stderr - HMMER may output help to either stream
        combined_output = result.stdout + result.stderr
        if "--seq_limit" not in combined_output:
            return False, (
                "jackhmmer found but missing --seq_limit option. "
                "Rebuild with: ./scripts/build_hmmer_macos.sh"
            )

        # Extract version from output
        for line in result.stdout.split("\n"):
            if "HMMER" in line and "(" in line:
                return True, line.strip()

        return True, "HMMER with seq_limit patch available"

    except FileNotFoundError:
        return False, (
            "jackhmmer not found in PATH. "
            "Install with: ./scripts/build_hmmer_macos.sh"
        )
    except subprocess.TimeoutExpired:
        return False, "jackhmmer timed out"


class DataPipelineNotConfiguredError(Exception):
    """Raised when AF3 data pipeline tools or databases are missing."""

    def __init__(self, message: str, missing: dict[str, list[str]]):
        super().__init__(message)
        self.missing = missing


def _resolve_binary(binary_name: str, env_var: str | None) -> tuple[Path | None, list[str]]:
    """Resolve an external binary path with explicit and heuristic fallbacks.

    Resolution order:
    1) Explicit env var override (if provided).
    2) `PATH` via shutil.which().
    3) Common macOS install prefixes (Homebrew Intel/ARM, local HMMER build).
    """
    tried: list[str] = []

    if env_var:
        override = os.environ.get(env_var)
        tried.append(f"env:{env_var}")
        if override:
            candidate = Path(os.path.expanduser(override))
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate, tried

    tried.append("PATH")
    found = shutil.which(binary_name)
    if found:
        return Path(found), tried

    extra_dirs = [
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path.home() / "hmmer" / "bin",  # scripts/build_hmmer_macos.sh default prefix
        Path.home() / ".alphafold3_mlx" / "hmmer" / "bin",  # install.sh default
    ]
    for d in extra_dirs:
        tried.append(str(d / binary_name))
        candidate = d / binary_name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate, tried

    return None, tried


def _resolve_db_path(
    *,
    key: str,
    env_var: str,
    legacy_env_vars: list[str],
    db_dir: Path | None,
    default_name: str | None,
    is_dir: bool = False,
) -> tuple[Path | None, list[str]]:
    tried: list[str] = [f"env:{env_var}"]
    value = os.environ.get(env_var)
    if value:
        candidate = Path(os.path.expanduser(value))
        if candidate.exists() and (candidate.is_dir() if is_dir else candidate.is_file()):
            return candidate, tried

    for legacy in legacy_env_vars:
        tried.append(f"env:{legacy}")
        legacy_value = os.environ.get(legacy)
        if legacy_value:
            candidate = Path(os.path.expanduser(legacy_value))
            if candidate.exists() and (candidate.is_dir() if is_dir else candidate.is_file()):
                return candidate, tried

    if db_dir is not None and default_name is not None:
        candidate = db_dir / default_name
        tried.append(str(candidate))
        if candidate.exists() and (candidate.is_dir() if is_dir else candidate.is_file()):
            return candidate, tried

    return None, tried


def _parse_date(value: str) -> datetime.date:
    try:
        parts = value.strip().split("-")
        if len(parts) != 3:
            raise ValueError
        y, m, d = (int(p) for p in parts)
        return datetime.date(y, m, d)
    except Exception as e:
        raise ValueError(f"Invalid date {value!r}. Expected YYYY-MM-DD.") from e


def build_af3_data_pipeline_config(
    *,
    db_dir: Path | None = None,
    require_rna: bool = False,
    max_template_date: datetime.date | None = None,
) -> tuple["object", dict[str, Path], dict[str, list[str]]]:
    """Build a full AF3 DataPipelineConfig from env vars / db_dir.

    This is used by the MLX runner when `--run_data_pipeline` is enabled.

    Configuration inputs:
    - `db_dir` argument, or `AF3_DB_DIR` env var to locate standard filenames.
    - Explicit env vars for each database path override db_dir-derived defaults.
    - HMMER binaries resolved via env overrides, PATH, and common macOS prefixes.

    Returns:
        Tuple of (DataPipelineConfig, resolved_paths, tried_by_key).

    Raises:
        DataPipelineNotConfiguredError: if required tools or databases are missing.
    """
    # Import lazily to avoid import-time overhead for users not using the data pipeline.
    from alphafold3.data import pipeline as af3_pipeline

    if db_dir is None:
        env_db_dir = os.environ.get("AF3_DB_DIR")
        if env_db_dir:
            db_dir = Path(os.path.expanduser(env_db_dir))

    # fetch_databases.sh downloads these standard filenames into <DB_DIR>/.
    expected_relpaths: dict[str, list[str]] = {
        "uniref90": ["uniref90_2022_05.fa"],
        "mgnify": ["mgy_clusters_2022_05.fa"],
        "small_bfd": ["bfd-first_non_consensus_sequences.fasta"],
        "uniprot": ["uniprot_all_2021_04.fa"],
        "pdb_seqres": ["pdb_seqres_2022_09_28.fasta"],
        # The PDB mmCIF tar sometimes extracts into a wrapper folder.
        "pdb_mmcif_dir": [
            "mmcif_files/",
            "pdb_2022_09_28_mmcif_files/mmcif_files/",
            "pdb_2022_09_28_mmcif_files/",
        ],
        # RNA DBs (optional).
        "ntrna": ["nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"],
        "rfam": ["rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"],
        "rna_central": ["rnacentral_active_seq_id_90_cov_80_linclust.fasta"],
    }

    max_template_date = max_template_date or datetime.date(2021, 9, 30)
    env_max_date = os.environ.get("AF3_MAX_TEMPLATE_DATE")
    if env_max_date:
        max_template_date = _parse_date(env_max_date)

    resolved: dict[str, Path] = {}
    tried: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}

    def require_path(key: str, path: Path | None, attempts: list[str]) -> None:
        tried[key] = attempts
        if path is None:
            missing[key] = attempts
        else:
            resolved[key] = path

    # --- Binaries (protein) ---
    jackhmmer_path, attempts = _resolve_binary("jackhmmer", "AF3_JACKHMMER")
    require_path("jackhmmer", jackhmmer_path, attempts)
    hmmsearch_path, attempts = _resolve_binary("hmmsearch", "AF3_HMMSEARCH")
    require_path("hmmsearch", hmmsearch_path, attempts)
    hmmbuild_path, attempts = _resolve_binary("hmmbuild", "AF3_HMMBUILD")
    require_path("hmmbuild", hmmbuild_path, attempts)

    # --- Binaries (RNA; optional) ---
    nhmmer_path, attempts = _resolve_binary("nhmmer", "AF3_NHMMER")
    hmmalign_path, attempts2 = _resolve_binary("hmmalign", "AF3_HMMALIGN")
    tried["nhmmer"] = attempts
    tried["hmmalign"] = attempts2
    nhmmer_binary_path = str(nhmmer_path) if nhmmer_path is not None else ""
    hmmalign_binary_path = str(hmmalign_path) if hmmalign_path is not None else ""
    if require_rna:
        if nhmmer_path is None:
            missing["nhmmer"] = attempts
        else:
            resolved["nhmmer"] = nhmmer_path
        if hmmalign_path is None:
            missing["hmmalign"] = attempts2
        else:
            resolved["hmmalign"] = hmmalign_path

    # --- Databases (protein) ---
    uniref90, attempts = _resolve_db_path(
        key="uniref90",
        env_var="AF3_UNIREF90_DB",
        legacy_env_vars=["UNIREF90_PATH"],
        db_dir=db_dir,
        default_name="uniref90_2022_05.fa",
    )
    require_path("uniref90", uniref90, attempts)

    mgnify, attempts = _resolve_db_path(
        key="mgnify",
        env_var="AF3_MGNIFY_DB",
        legacy_env_vars=["MGNIFY_PATH"],
        db_dir=db_dir,
        default_name="mgy_clusters_2022_05.fa",
    )
    require_path("mgnify", mgnify, attempts)

    small_bfd, attempts = _resolve_db_path(
        key="small_bfd",
        env_var="AF3_SMALL_BFD_DB",
        legacy_env_vars=["BFD_PATH"],
        db_dir=db_dir,
        default_name="bfd-first_non_consensus_sequences.fasta",
    )
    require_path("small_bfd", small_bfd, attempts)

    # Paired MSA database. The upstream fetch_databases.sh downloads uniprot_all_2021_04.fa.
    uniprot, attempts = _resolve_db_path(
        key="uniprot",
        env_var="AF3_UNIPROT_DB",
        legacy_env_vars=[],
        db_dir=db_dir,
        default_name="uniprot_all_2021_04.fa",
    )
    require_path("uniprot", uniprot, attempts)

    seqres, attempts = _resolve_db_path(
        key="pdb_seqres",
        env_var="AF3_PDB_SEQRES_DB",
        legacy_env_vars=[],
        db_dir=db_dir,
        default_name="pdb_seqres_2022_09_28.fasta",
    )
    require_path("pdb_seqres", seqres, attempts)

    pdb_mmcif_dir, attempts = _resolve_db_path(
        key="pdb_mmcif_dir",
        env_var="AF3_PDB_MMCIF_DIR",
        legacy_env_vars=["PDB_MMCIF_PATH"],
        db_dir=db_dir,
        default_name="mmcif_files",
        is_dir=True,
    )
    if pdb_mmcif_dir is None and db_dir is not None:
        # fetch_databases.sh extracts `pdb_2022_09_28_mmcif_files.tar.zst`.
        # Depending on the tar layout, mmCIFs may be under:
        # - <DB_DIR>/mmcif_files/
        # - <DB_DIR>/pdb_2022_09_28_mmcif_files/mmcif_files/
        # - <DB_DIR>/pdb_2022_09_28_mmcif_files/
        for rel in expected_relpaths["pdb_mmcif_dir"][1:]:
            candidate = db_dir / rel.rstrip("/")
            attempts.append(str(candidate))
            if candidate.exists() and candidate.is_dir():
                pdb_mmcif_dir = candidate
                break
    require_path("pdb_mmcif_dir", pdb_mmcif_dir, attempts)

    # --- Databases (RNA; optional) ---
    ntrna, attempts = _resolve_db_path(
        key="ntrna",
        env_var="AF3_NTRNA_DB",
        legacy_env_vars=[],
        db_dir=db_dir,
        default_name="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
    )
    rfam, attempts2 = _resolve_db_path(
        key="rfam",
        env_var="AF3_RFAM_DB",
        legacy_env_vars=[],
        db_dir=db_dir,
        default_name="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
    )
    rna_central, attempts3 = _resolve_db_path(
        key="rna_central",
        env_var="AF3_RNACENTRAL_DB",
        legacy_env_vars=[],
        db_dir=db_dir,
        default_name="rnacentral_active_seq_id_90_cov_80_linclust.fasta",
    )
    tried["ntrna"] = attempts
    tried["rfam"] = attempts2
    tried["rna_central"] = attempts3
    ntrna_db_path = str(ntrna) if ntrna is not None else ""
    rfam_db_path = str(rfam) if rfam is not None else ""
    rna_central_db_path = str(rna_central) if rna_central is not None else ""
    if require_rna:
        if ntrna is None:
            missing["ntrna"] = attempts
        else:
            resolved["ntrna"] = ntrna
        if rfam is None:
            missing["rfam"] = attempts2
        else:
            resolved["rfam"] = rfam
        if rna_central is None:
            missing["rna_central"] = attempts3
        else:
            resolved["rna_central"] = rna_central

    # Z-values (optional; required for sharded specs)
    def _env_int(var: str) -> int | None:
        val = os.environ.get(var)
        if val is None or not val.strip():
            return None
        try:
            return int(val)
        except ValueError as e:
            raise ValueError(f"{var} must be an int, got {val!r}") from e

    def _env_float(var: str) -> float | None:
        val = os.environ.get(var)
        if val is None or not val.strip():
            return None
        try:
            return float(val)
        except ValueError as e:
            raise ValueError(f"{var} must be a float, got {val!r}") from e

    uniref90_z = _env_int("AF3_UNIREF90_Z")
    mgnify_z = _env_int("AF3_MGNIFY_Z")
    small_bfd_z = _env_int("AF3_SMALL_BFD_Z")
    uniprot_z = _env_int("AF3_UNIPROT_Z")

    ntrna_z = _env_float("AF3_NTRNA_Z")
    rfam_z = _env_float("AF3_RFAM_Z")
    rna_central_z = _env_float("AF3_RNACENTRAL_Z")

    if missing:
        lines = ["AF3 data pipeline is not configured (missing requirements):"]
        for k in sorted(missing.keys()):
            lines.append(f"- {k}: tried {', '.join(missing[k])}")
        lines.append("")
        if db_dir is None:
            lines.append(
                "AF3_DB_DIR is not set (and --db_dir was not provided)."
            )
            lines.append(
                "Set AF3_DB_DIR (or pass --db_dir) to a directory created by fetch_databases.sh,"
            )
            lines.append(
                "or set explicit env vars like AF3_UNIREF90_DB / AF3_PDB_MMCIF_DIR."
            )
        else:
            lines.append(f"Using db_dir: {db_dir}")

        lines.append("")
        lines.append("Expected database layout under db_dir (fetch_databases.sh):")
        for key in ("uniref90", "mgnify", "small_bfd", "uniprot", "pdb_seqres", "pdb_mmcif_dir"):
            rels = expected_relpaths.get(key)
            if not rels:
                continue
            rel_str = " or ".join(rels)
            lines.append(f"- {key}: {rel_str}")
        if require_rna:
            for key in ("ntrna", "rfam", "rna_central"):
                rels = expected_relpaths.get(key)
                if not rels:
                    continue
                lines.append(f"- {key}: {rels[0]}")
        raise DataPipelineNotConfiguredError("\n".join(lines), missing=missing)

    # Build config. Note: for protein-only, RNA paths may be empty strings.
    cfg = af3_pipeline.DataPipelineConfig(
        jackhmmer_binary_path=str(resolved["jackhmmer"]),
        nhmmer_binary_path=nhmmer_binary_path,
        hmmalign_binary_path=hmmalign_binary_path,
        hmmsearch_binary_path=str(resolved["hmmsearch"]),
        hmmbuild_binary_path=str(resolved["hmmbuild"]),
        small_bfd_database_path=str(resolved["small_bfd"]),
        small_bfd_z_value=small_bfd_z,
        mgnify_database_path=str(resolved["mgnify"]),
        mgnify_z_value=mgnify_z,
        uniprot_cluster_annot_database_path=str(resolved["uniprot"]),
        uniprot_cluster_annot_z_value=uniprot_z,
        uniref90_database_path=str(resolved["uniref90"]),
        uniref90_z_value=uniref90_z,
        ntrna_database_path=ntrna_db_path,
        ntrna_z_value=ntrna_z,
        rfam_database_path=rfam_db_path,
        rfam_z_value=rfam_z,
        rna_central_database_path=rna_central_db_path,
        rna_central_z_value=rna_central_z,
        seqres_database_path=str(resolved["pdb_seqres"]),
        pdb_database_path=str(resolved["pdb_mmcif_dir"]),
        max_template_date=max_template_date,
    )

    return cfg, resolved, tried
