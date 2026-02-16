"""Input handling for AlphaFold 3 MLX pipeline.

This module provides JSON input parsing and validation .

Uses alphafold3.common.folding_input.Input.from_json for parsing to ensure
compatibility with the official AF3 input format.

Example:
    fold_input = parse_input_json(Path("input.json"))
    errors = validate_input(fold_input)
    memory_gb = estimate_memory_gb(fold_input, num_samples=5)
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from alphafold3.common import folding_input
from alphafold3_mlx.pipeline.errors import InputError, ResourceError


@dataclass
class Modification:
    """Chemical modification on a residue.

    Attributes:
        position: 1-indexed position in the sequence.
        modification_type: Type of modification (e.g., "phosphorylation").
    """

    position: int
    modification_type: str


@dataclass
class Sequence:
    """Single chain sequence.

    Attributes:
        sequence: Amino acid or nucleotide sequence.
        chain_type: Type of molecular entity.
        chain_id: Unique identifier for this chain.
        modifications: Optional post-translational modifications.
    """

    sequence: str
    chain_type: Literal["protein", "rna", "dna", "ligand"]
    chain_id: str
    modifications: list[Modification] | None = None

    def __post_init__(self) -> None:
        """Validate sequence after initialization."""
        if not self.sequence:
            raise InputError(f"Empty sequence for chain {self.chain_id}")

        if not self.chain_id:
            raise InputError("Chain ID cannot be empty")

        # Skip character validation for ligands (use SMILES notation with different rules)
        if self.chain_type == "ligand":
            return

        # Validate sequence characters based on type
        valid_chars = self._get_valid_chars
        invalid_chars = set(self.sequence.upper) - valid_chars
        if invalid_chars:
            raise InputError(
                f"Invalid characters in {self.chain_type} sequence for chain {self.chain_id}: "
                f"{', '.join(sorted(invalid_chars))}"
            )

    def _get_valid_chars(self) -> set[str]:
        """Get valid characters for this chain type."""
        if self.chain_type == "protein":
            return set("ARNDCQEGHILKMFPSTWYVXUBZJO") # Standard + ambiguous
        elif self.chain_type in ("rna", "dna"):
            return set("ACGTURYSWKMBDHVN") # Standard + IUPAC ambiguous
        elif self.chain_type == "ligand":
            return set # Ligands use SMILES, different validation
        return set


@dataclass
class InputJSON:
    """Parsed AF3 input file.

    Attributes:
        name: Job name / identifier.
        sequences: List of chains in the input.
        model_seeds: Random seeds for inference.
        msa_paths: Pre-computed MSA file paths by chain ID.
        template_paths: Pre-computed template file paths.
    """

    name: str
    sequences: list[Sequence]
    model_seeds: list[int]
    msa_paths: dict[str, Path] | None = None
    template_paths: dict[str, Path] | None = None

    def __post_init__(self) -> None:
        """Validate input after initialization."""
        if not self.sequences:
            raise InputError("At least one sequence is required")

        # Validate unique chain IDs
        chain_ids = [s.chain_id for s in self.sequences]
        if len(chain_ids) != len(set(chain_ids)):
            raise InputError("Duplicate chain IDs found")

        # Validate seeds
        for seed in self.model_seeds:
            if seed < 0:
                raise InputError(f"Model seed must be non-negative, got {seed}")

    @property
    def total_residues(self) -> int:
        """Total number of residues across all chains."""
        return sum(len(s.sequence) for s in self.sequences)

    @property
    def is_complex(self) -> bool:
        """Whether this is a multi-chain complex."""
        return len(self.sequences) > 1

    @property
    def chain_ids(self) -> list[str]:
        """List of all chain identifiers."""
        return [s.chain_id for s in self.sequences]


class FoldInput:
    """Wrapper around alphafold3.common.folding_input.Input.

    This class provides a compatible interface for the pipeline while
    using the official AF3 Input class for parsing and validation.

    Attributes:
        _input: The underlying alphafold3.common.folding_input.Input instance.
        _restraints: Optional RestraintConfig parsed from input JSON.
        _guidance: Optional GuidanceConfig parsed from input JSON.
    """

    def __init__(
        self,
        af3_input: folding_input.Input,
        restraints: Any = None,
        guidance: Any = None,
    ) -> None:
        """Initialize with an AF3 Input instance.

        Args:
            af3_input: The parsed folding_input.Input instance.
            restraints: Optional RestraintConfig from restraint-guided docking.
            guidance: Optional GuidanceConfig from restraint-guided docking.
        """
        self._input = af3_input
        self._restraints = restraints
        self._guidance = guidance

    @classmethod
    def from_simple(
        cls,
        name: str,
        sequences: list[Sequence],
        model_seeds: list[int] | None = None,
    ) -> "FoldInput":
        """Create a FoldInput from simple parameters (for testing).

        Args:
            name: Job name.
            sequences: List of Sequence objects.
            model_seeds: Random seeds (defaults to [42]).

        Returns:
            FoldInput instance.

        Raises:
            InputError: If input is invalid.
        """
        if model_seeds is None:
            model_seeds = [42]

        chains = []
        for seq in sequences:
            if seq.chain_type == "protein":
                ptms = []
                if seq.modifications:
                    ptms = [(m.modification_type, m.position) for m in seq.modifications]
                chains.append(folding_input.ProteinChain(
                    id=seq.chain_id,
                    sequence=seq.sequence,
                    ptms=ptms,
                ))
            elif seq.chain_type == "rna":
                modifications = []
                if seq.modifications:
                    modifications = [(m.modification_type, m.position) for m in seq.modifications]
                chains.append(folding_input.RnaChain(
                    id=seq.chain_id,
                    sequence=seq.sequence,
                    modifications=modifications,
                ))
            elif seq.chain_type == "dna":
                modifications = []
                if seq.modifications:
                    modifications = [(m.modification_type, m.position) for m in seq.modifications]
                chains.append(folding_input.DnaChain(
                    id=seq.chain_id,
                    sequence=seq.sequence,
                    modifications=modifications,
                ))
            elif seq.chain_type == "ligand":
                # For ligands, sequence is treated as SMILES
                chains.append(folding_input.Ligand(
                    id=seq.chain_id,
                    smiles=seq.sequence,
                ))
            else:
                raise InputError(f"Unknown chain type: {seq.chain_type}")

        try:
            af3_input = folding_input.Input(
                name=name,
                chains=chains,
                rng_seeds=model_seeds,
            )
        except ValueError as e:
            raise InputError(f"Invalid input: {e}")

        return cls(af3_input)

    @property
    def input(self) -> folding_input.Input:
        """Return the underlying AF3 Input for use with data pipeline."""
        return self._input

    @property
    def name(self) -> str:
        """Job name / identifier."""
        return self._input.name

    @property
    def model_seeds(self) -> list[int]:
        """Random seeds for inference."""
        return list(self._input.rng_seeds)

    @property
    def sequences(self) -> list[Sequence]:
        """List of chains as Sequence objects."""
        result = []
        for chain in self._input.chains:
            if isinstance(chain, folding_input.ProteinChain):
                modifications = [
                    Modification(position=ptm[1], modification_type=ptm[0])
                    for ptm in chain.ptms
                ] if chain.ptms else None
                result.append(Sequence(
                    sequence=chain.sequence,
                    chain_type="protein",
                    chain_id=chain.id,
                    modifications=modifications,
                ))
            elif isinstance(chain, folding_input.RnaChain):
                modifications = [
                    Modification(position=mod[1], modification_type=mod[0])
                    for mod in chain.modifications
                ] if chain.modifications else None
                result.append(Sequence(
                    sequence=chain.sequence,
                    chain_type="rna",
                    chain_id=chain.id,
                    modifications=modifications,
                ))
            elif isinstance(chain, folding_input.DnaChain):
                modifications = [
                    Modification(position=mod[1], modification_type=mod[0])
                    for mod in chain.modifications
                ] if chain.modifications else None
                result.append(Sequence(
                    sequence=chain.sequence,
                    chain_type="dna",
                    chain_id=chain.id,
                    modifications=modifications,
                ))
            elif isinstance(chain, folding_input.Ligand):
                # Ligands use SMILES or CCD IDs, sequence is the SMILES/CCD
                smiles_or_ccd = chain.smiles if chain.smiles else ",".join(chain.ccd_ids or [])
                result.append(Sequence(
                    sequence=smiles_or_ccd,
                    chain_type="ligand",
                    chain_id=chain.id,
                ))
        return result

    @property
    def total_residues(self) -> int:
        """Total number of residues across all chains."""
        total = 0
        for chain in self._input.chains:
            total += len(chain)
        return total

    @property
    def is_complex(self) -> bool:
        """Whether this is a multi-chain complex."""
        return len(self._input.chains) > 1

    @property
    def chain_ids(self) -> list[str]:
        """List of all chain identifiers."""
        return [chain.id for chain in self._input.chains]

    @property
    def restraints(self) -> Any:
        """Optional restraint configuration."""
        return self._restraints

    @property
    def guidance(self) -> Any:
        """Optional guidance configuration."""
        return self._guidance


# Type alias for backward compatibility
InputJSON = FoldInput


def parse_input_json(input_path: Path) -> FoldInput:
    """Parse AF3-format JSON input file.

    Uses alphafold3.common.folding_input.Input.from_json to ensure full
    compatibility with the official AlphaFold 3 input format. Also supports
    AlphaFold Server format and simple MLX test format.

    Supported formats:
    1. AlphaFold 3 format: Has dialect="alphafold3" and version fields
    2. AlphaFold Server format: Has proteinChain/rnaSequence/dnaSequence keys
    3. Simple format: Has protein/rna/dna/ligand keys without dialect/version

    Args:
        input_path: Path to input JSON file.

    Returns:
        FoldInput wrapper around alphafold3.common.folding_input.Input.

    Raises:
        InputError: If file cannot be parsed or is invalid.
    """
    if not input_path.exists:
        raise InputError(f"Input file not found: {input_path}")

    try:
        with open(input_path, "r") as f:
            json_str = f.read
    except IOError as e:
        raise InputError(f"Failed to read input file: {e}")

    try:
        raw_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise InputError(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")

    # Handle AlphaFold Server list format (top-level list of fold jobs)
    if isinstance(raw_json, list):
        if not raw_json:
            raise InputError("Empty AlphaFold Server fold job list")
        if len(raw_json) > 1:
            import warnings
            warnings.warn(
                f"AlphaFold Server file contains {len(raw_json)} fold jobs, "
                "processing only the first one. Use load_fold_inputs for batch processing.",
                stacklevel=2,
            )
        raw_json = raw_json[0]

    if not isinstance(raw_json, dict):
        raise InputError(
            f"Expected a JSON object at top level, got {type(raw_json).__name__}"
        )

    # Extract restraints and guidance BEFORE passing to AF3 parser
    # (upstream folding_input.Input rejects unknown keys)
    restraints_dict = raw_json.pop("restraints", None)
    guidance_dict = raw_json.pop("guidance", None)

    # Re-serialize JSON string after popping restraint keys
    if restraints_dict is not None or guidance_dict is not None:
        json_str = json.dumps(raw_json)

    # Parse restraint/guidance configs if present
    restraint_config = None
    guidance_config = None
    if restraints_dict is not None:
        from alphafold3_mlx.restraints.types import restraint_config_from_dict
        try:
            restraint_config = restraint_config_from_dict(restraints_dict)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise InputError(f"Invalid restraints in input JSON: {e}")
    if guidance_dict is not None:
        from alphafold3_mlx.restraints.types import guidance_config_from_dict
        try:
            guidance_config = guidance_config_from_dict(guidance_dict)
        except (ValueError, TypeError) as e:
            raise InputError(f"Invalid guidance in input JSON: {e}")

    # Try to detect and parse based on format
    af3_input = _parse_any_format(raw_json, json_str, input_path)
    return FoldInput(af3_input, restraints=restraint_config, guidance=guidance_config)


def _strip_unsupported_server_fields(raw_json: dict[str, Any]) -> dict[str, Any]:
    """Strip fields that from_alphafoldserver_fold_job doesn't support.

    The AlphaFold Server web UI includes fields like maxTemplateDate that
    the local parser rejects. We strip them so users can paste JSON directly
    from the server without editing.
    """
    import copy
    cleaned = copy.deepcopy(raw_json)
    for seq in cleaned.get("sequences", []):
        for entity_key in ("proteinChain", "rnaSequence", "dnaSequence", "ligand", "ion"):
            if entity_key in seq:
                seq[entity_key].pop("maxTemplateDate", None)
    return cleaned


def _parse_any_format(
    raw_json: dict[str, Any],
    json_str: str,
    input_path: Path,
) -> folding_input.Input:
    """Parse input JSON detecting the format automatically.

    Args:
        raw_json: Parsed JSON dict.
        json_str: Original JSON string.
        input_path: Path to the JSON file.

    Returns:
        folding_input.Input instance.

    Raises:
        InputError: If parsing fails for all formats.
    """
    # Check if it has dialect/version fields
    if "dialect" in raw_json and "version" in raw_json:
        dialect = raw_json.get("dialect", "")
        if dialect == "alphafoldserver":
            # AlphaFold Server format with explicit dialect
            cleaned = _strip_unsupported_server_fields(raw_json)
            try:
                return folding_input.Input.from_alphafoldserver_fold_job(cleaned)
            except ValueError as e:
                raise InputError(f"Invalid AlphaFold Server input: {e}")
        else:
            # Official AlphaFold 3 format (dialect="alphafold3")
            try:
                return folding_input.Input.from_json(
                    json_str, json_path=pathlib.Path(input_path)
                )
            except ValueError as e:
                raise InputError(f"Invalid AlphaFold 3 input: {e}")

    # Check for AlphaFold Server format without dialect (sequences contain proteinChain, etc.)
    sequences = raw_json.get("sequences", [])
    if sequences and any(
        "proteinChain" in s or "rnaSequence" in s or "dnaSequence" in s
        for s in sequences
    ):
        cleaned = _strip_unsupported_server_fields(raw_json)
        try:
            return folding_input.Input.from_alphafoldserver_fold_job(cleaned)
        except ValueError as e:
            raise InputError(f"Invalid AlphaFold Server input: {e}")

    # Simple format: convert to AlphaFold 3 format
    return _convert_simple_format(raw_json)


def _convert_simple_format(data: dict[str, Any]) -> folding_input.Input:
    """Convert simple test format to folding_input.Input.

    Simple format uses:
    - protein/rna/dna/ligand keys in sequences (without dialect/version)
    - id field for chain IDs
    - sequence field for sequences

    Args:
        data: Parsed JSON dict.

    Returns:
        folding_input.Input instance.

    Raises:
        InputError: If format is invalid.
    """
    # Check for required fields and aggregate missing ones
    missing_fields = []
    if "sequences" not in data or not data.get("sequences"):
        missing_fields.append("sequences")

    if missing_fields:
        raise InputError(f"Missing required fields in input: {', '.join(missing_fields)}")

    name = data.get("name", "unnamed")
    model_seeds = data.get("modelSeeds", data.get("model_seeds", [42]))
    if not isinstance(model_seeds, list):
        model_seeds = [model_seeds]

    sequences_data = data["sequences"]

    chains = []
    for i, seq_entry in enumerate(sequences_data):
        default_id = chr(ord("A") + i)

        if "protein" in seq_entry:
            chain_data = seq_entry["protein"]
            chain_id = chain_data.get("id", default_id)
            sequence = chain_data.get("sequence", "")
            ptms = []
            for mod in chain_data.get("modifications", []):
                ptm_type = mod.get("ptmType", mod.get("type", ""))
                ptm_pos = mod.get("ptmPosition", mod.get("position", 0))
                ptms.append((ptm_type, ptm_pos))
            chains.append(folding_input.ProteinChain(
                id=chain_id,
                sequence=sequence,
                ptms=ptms,
            ))
        elif "rna" in seq_entry:
            chain_data = seq_entry["rna"]
            chain_id = chain_data.get("id", default_id)
            sequence = chain_data.get("sequence", "")
            modifications = []
            for mod in chain_data.get("modifications", []):
                mod_type = mod.get("modificationType", mod.get("type", ""))
                base_pos = mod.get("basePosition", mod.get("position", 0))
                modifications.append((mod_type, base_pos))
            chains.append(folding_input.RnaChain(
                id=chain_id,
                sequence=sequence,
                modifications=modifications,
            ))
        elif "dna" in seq_entry:
            chain_data = seq_entry["dna"]
            chain_id = chain_data.get("id", default_id)
            sequence = chain_data.get("sequence", "")
            modifications = []
            for mod in chain_data.get("modifications", []):
                mod_type = mod.get("modificationType", mod.get("type", ""))
                base_pos = mod.get("basePosition", mod.get("position", 0))
                modifications.append((mod_type, base_pos))
            chains.append(folding_input.DnaChain(
                id=chain_id,
                sequence=sequence,
                modifications=modifications,
            ))
        elif "ligand" in seq_entry:
            chain_data = seq_entry["ligand"]
            chain_id = chain_data.get("id", default_id)
            ccd_ids = chain_data.get("ccdCodes")
            smiles = chain_data.get("smiles")
            chains.append(folding_input.Ligand(
                id=chain_id,
                ccd_ids=ccd_ids,
                smiles=smiles,
            ))
        else:
            raise InputError(f"Unknown sequence type in entry: {seq_entry}")

    if not chains:
        raise InputError("No valid chains found in input")

    try:
        return folding_input.Input(
            name=name,
            chains=chains,
            rng_seeds=model_seeds,
        )
    except ValueError as e:
        raise InputError(f"Invalid input: {e}")


def _parse_input_dict(data: dict[str, Any]) -> InputJSON:
    """Parse input from dictionary.

    Args:
        data: Input data dictionary.

    Returns:
        Parsed InputJSON instance.

    Raises:
        InputError: If required fields are missing or invalid.
    """
    # Check for required fields and aggregate missing ones
    missing_fields = []
    if "sequences" not in data or not data.get("sequences"):
        missing_fields.append("sequences")

    if missing_fields:
        raise InputError(f"Missing required fields in input: {', '.join(missing_fields)}")

    # Extract name
    name = data.get("name", "unnamed")

    # Extract seeds (AF3 format uses "modelSeeds")
    model_seeds = data.get("modelSeeds", data.get("model_seeds", [42]))
    if not isinstance(model_seeds, list):
        model_seeds = [model_seeds]

    # Extract sequences
    sequences_data = data["sequences"]

    sequences = []
    for i, seq_entry in enumerate(sequences_data):
        seq = _parse_sequence_entry(seq_entry, default_chain_id=chr(ord("A") + i))
        sequences.append(seq)

    # Extract optional MSA/template paths
    msa_paths = None
    if "msa_paths" in data:
        msa_paths = {k: Path(v) for k, v in data["msa_paths"].items}

    template_paths = None
    if "template_paths" in data:
        template_paths = {k: Path(v) for k, v in data["template_paths"].items}

    return InputJSON(
        name=name,
        sequences=sequences,
        model_seeds=model_seeds,
        msa_paths=msa_paths,
        template_paths=template_paths,
    )


def _parse_sequence_entry(entry: dict[str, Any], default_chain_id: str) -> Sequence:
    """Parse a single sequence entry from AF3 JSON format.

    AF3 format uses nested structure:
    {
        "protein": {"id": "A", "sequence": "MKTAY..."},
        ...
    }

    Args:
        entry: Sequence entry from JSON.
        default_chain_id: Default chain ID if not specified.

    Returns:
        Parsed Sequence instance.

    Raises:
        InputError: If entry is invalid.
    """
    # AF3 format: {"protein": {...}} or {"rna": {...}} etc.
    chain_types = ["protein", "rna", "dna", "ligand"]

    for chain_type in chain_types:
        if chain_type in entry:
            chain_data = entry[chain_type]
            sequence = chain_data.get("sequence", "")
            chain_id = chain_data.get("id", default_chain_id)

            # Parse modifications if present
            modifications = None
            if "modifications" in chain_data:
                modifications = [
                    Modification(position=m["position"], modification_type=m["type"])
                    for m in chain_data["modifications"]
                ]

            return Sequence(
                sequence=sequence,
                chain_type=chain_type, # type: ignore
                chain_id=chain_id,
                modifications=modifications,
            )

    # Fallback: simple format with just sequence string
    if "sequence" in entry:
        return Sequence(
            sequence=entry["sequence"],
            chain_type="protein",
            chain_id=entry.get("id", entry.get("chain_id", default_chain_id)),
        )

    raise InputError(f"Invalid sequence entry format: {entry}")


def parse_multichain_sequences(fold_input: FoldInput) -> list[Sequence]:
    """Parse multi-chain complex inputs.

    Args:
        fold_input: Parsed fold input.

    Returns:
        List of sequences with proper chain IDs.
    """
    return fold_input.sequences


def load_fold_inputs(input_path: Path) -> Iterator[FoldInput]:
    """Load multiple fold inputs from an AlphaFold Server JSON file.

    This function handles the AlphaFold Server format where the JSON file
    contains a top-level list of fold jobs. Each job is yielded as a
    separate FoldInput.

    For single-job files (dict format), yields the single job.

    Supports restraint-guided docking: extracts optional "restraints" and
    "guidance" keys from each job and attaches them to the FoldInput.
    These keys are removed before passing to the upstream AF3 parser
    (which rejects unknown keys).

    Args:
        input_path: Path to input JSON file.

    Yields:
        FoldInput for each fold job in the file, with optional restraints
        and guidance attached.

    Raises:
        InputError: If file cannot be parsed or is invalid.

    Example:
        for fold_input in load_fold_inputs(Path("batch_jobs.json")):
            result = run_inference(fold_input)
            # fold_input.restraints and fold_input.guidance are available
    """
    if not input_path.exists:
        raise InputError(f"Input file not found: {input_path}")

    try:
        with open(input_path, "r") as f:
            json_str = f.read
    except IOError as e:
        raise InputError(f"Failed to read input file: {e}")

    try:
        raw_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise InputError(f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")

    # Handle AlphaFold Server list format (top-level list of fold jobs)
    if isinstance(raw_json, list):
        if not raw_json:
            raise InputError("Empty AlphaFold Server fold job list")
        for fold_job_idx, fold_job in enumerate(raw_json):
            if not isinstance(fold_job, dict):
                raise InputError(
                    f"Fold job {fold_job_idx} in {input_path} is not a JSON object"
                )
            try:
                # Extract restraints and guidance BEFORE parsing
                # (upstream folding_input.Input rejects unknown keys)
                restraints_dict = fold_job.pop("restraints", None)
                guidance_dict = fold_job.pop("guidance", None)

                # Parse restraint/guidance configs if present
                restraint_config = None
                guidance_config = None
                if restraints_dict is not None:
                    from alphafold3_mlx.restraints.types import restraint_config_from_dict
                    try:
                        restraint_config = restraint_config_from_dict(restraints_dict)
                    except (ValueError, TypeError, KeyError, AttributeError) as e:
                        raise InputError(
                            f"Invalid restraints in fold job {fold_job_idx}: {e}"
                        )
                if guidance_dict is not None:
                    from alphafold3_mlx.restraints.types import guidance_config_from_dict
                    try:
                        guidance_config = guidance_config_from_dict(guidance_dict)
                    except (ValueError, TypeError) as e:
                        raise InputError(
                            f"Invalid guidance in fold job {fold_job_idx}: {e}"
                        )

                af3_input = folding_input.Input.from_alphafoldserver_fold_job(fold_job)
                yield FoldInput(af3_input, restraints=restraint_config, guidance=guidance_config)
            except ValueError as e:
                raise InputError(
                    f"Failed to load fold job {fold_job_idx} from {input_path}: {e}"
                )
    else:
        # Single job (dict format)
        if not isinstance(raw_json, dict):
            raise InputError(
                f"Expected a JSON object or array in {input_path}, "
                f"got {type(raw_json).__name__}"
            )
        # Extract restraints and guidance BEFORE parsing
        restraints_dict = raw_json.pop("restraints", None)
        guidance_dict = raw_json.pop("guidance", None)

        # Re-serialize JSON string after popping restraint keys
        if restraints_dict is not None or guidance_dict is not None:
            json_str = json.dumps(raw_json)

        # Parse restraint/guidance configs if present
        restraint_config = None
        guidance_config = None
        if restraints_dict is not None:
            from alphafold3_mlx.restraints.types import restraint_config_from_dict
            try:
                restraint_config = restraint_config_from_dict(restraints_dict)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                raise InputError(f"Invalid restraints in input JSON: {e}")
        if guidance_dict is not None:
            from alphafold3_mlx.restraints.types import guidance_config_from_dict
            try:
                guidance_config = guidance_config_from_dict(guidance_dict)
            except (ValueError, TypeError) as e:
                raise InputError(f"Invalid guidance in input JSON: {e}")

        af3_input = _parse_any_format(raw_json, json_str, input_path)
        yield FoldInput(af3_input, restraints=restraint_config, guidance=guidance_config)


def validate_input(fold_input: FoldInput) -> list[str]:
    """Validate input JSON schema and content.

    Since we use folding_input.Input.from_json, most validation is already
    done during parsing. This function performs additional checks.

    Args:
        fold_input: Parsed fold input to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    # Check sequences
    if not fold_input.sequences:
        errors.append("At least one sequence is required")

    # Check for empty sequences
    for seq in fold_input.sequences:
        if not seq.sequence:
            errors.append(f"Empty sequence for chain {seq.chain_id}")

    # Check chain ID uniqueness
    chain_ids = fold_input.chain_ids
    if len(chain_ids) != len(set(chain_ids)):
        errors.append("Duplicate chain IDs found")

    # Check seeds
    if not fold_input.model_seeds:
        errors.append("At least one model seed is required")

    for seed in fold_input.model_seeds:
        if seed < 0:
            errors.append(f"Invalid negative seed: {seed}")

    # Note: MSA/template paths are now handled by folding_input.Input,
    # which loads them during parsing. No additional path validation needed.

    return errors


def estimate_memory_gb(fold_input: FoldInput, num_samples: int = 5) -> float:
    """Estimate memory requirements using Phase 3 API.

    Uses check_memory_requirements from alphafold3_mlx.core.validation.

    Args:
        fold_input: Parsed fold input.
        num_samples: Number of structure samples.

    Returns:
        Estimated peak memory in GB.
    """
    try:
        from alphafold3_mlx.core.validation import estimate_peak_memory_gb
        return estimate_peak_memory_gb(fold_input.total_residues, num_samples)
    except ImportError:
        # Fallback estimation if Phase 3 module not available
        # Based on empirical data from research.md
        num_residues = fold_input.total_residues
        # Rough formula: 2.5GB base + O(n^2) scaling
        return 2.5 + (num_residues / 100) ** 2 * 2.0 * num_samples / 5


def check_memory_available(
    fold_input: FoldInput,
    num_samples: int = 5,
    safety_factor: float = 0.8,
) -> None:
    """Check if sufficient memory is available for inference.

    Args:
        fold_input: Parsed fold input.
        num_samples: Number of structure samples.
        safety_factor: Fraction of available memory to use (default 0.8 = 80%).

    Raises:
        ResourceError: If estimated memory exceeds safe limit.
    """
    try:
        from alphafold3_mlx.core.exceptions import MemoryError as MLXMemoryError
        from alphafold3_mlx.core.validation import (
            check_memory_requirements,
            get_available_memory_gb,
        )
    except ImportError:
        # Fallback if Phase 3 module not available
        estimated = estimate_memory_gb(fold_input, num_samples)
        # M-02: Detect actual memory instead of assuming 128GB
        try:
            import psutil
            available = psutil.virtual_memory.total / (1024**3)
        except ImportError:
            try:
                pages = os.sysconf('SC_PHYS_PAGES')
                page_size = os.sysconf('SC_PAGE_SIZE')
                available = (pages * page_size) / (1024**3)
            except (ValueError, OSError, AttributeError):
                available = 32.0 # Conservative fallback
        threshold = available * safety_factor

        if estimated > threshold:
            raise ResourceError(
                f"Estimated memory: {estimated:.1f} GB, Available: {available:.1f} GB "
                f"({int(safety_factor * 100)}% safety threshold: {threshold:.1f} GB). "
                "Reduce num_samples, diffusion_steps, or use a smaller protein."
            )
        return

    # Phase 3 module available, use proper validation
    num_residues = fold_input.total_residues
    available_gb = get_available_memory_gb

    try:
        check_memory_requirements(
            num_residues=num_residues,
            available_gb=available_gb,
            num_samples=num_samples,
            safety_factor=safety_factor,
        )
    except MLXMemoryError as e:
        raise ResourceError(str(e))


def load_restraints_file(restraints_path: Path) -> tuple[Any, Any]:
    """Load a standalone restraints JSON file.

    Parses the file and returns (RestraintConfig, GuidanceConfig | None).
    The file must conform to the StandaloneRestraintsFile schema.

    Args:
        restraints_path: Path to the restraints JSON file.

    Returns:
        Tuple of (RestraintConfig, GuidanceConfig | None).

    Raises:
        InputError: If the file cannot be read, parsed, or is invalid.
    """
    if not restraints_path.exists:
        raise InputError(f"Restraints file not found: {restraints_path}")

    try:
        with open(restraints_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InputError(
            f"Invalid JSON in restraints file {restraints_path}: "
            f"line {e.lineno}, column {e.colno}: {e.msg}"
        )
    except IOError as e:
        raise InputError(f"Failed to read restraints file: {e}")

    if not isinstance(data, dict):
        raise InputError("Restraints file must contain a JSON object")

    if "restraints" not in data and "guidance" not in data:
        raise InputError(
            "Restraints file must contain at least one of 'restraints' or 'guidance'"
        )

    allowed_keys = {"restraints", "guidance"}
    unexpected = set(data.keys) - allowed_keys
    if unexpected:
        raise InputError(
            f"Restraints file contains unexpected top-level keys: "
            f"{sorted(unexpected)}. Only 'restraints' and 'guidance' are allowed."
        )

    from alphafold3_mlx.restraints.types import (
        guidance_config_from_dict,
        restraint_config_from_dict,
    )

    restraint_config = None
    guidance_config = None

    if "restraints" in data:
        try:
            restraint_config = restraint_config_from_dict(data["restraints"])
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise InputError(
                f"Invalid restraints in {restraints_path}: {e}"
            )
    if "guidance" in data:
        try:
            guidance_config = guidance_config_from_dict(data["guidance"])
        except (ValueError, TypeError) as e:
            raise InputError(
                f"Invalid guidance in {restraints_path}: {e}"
            )

    return restraint_config, guidance_config


def apply_restraints_file(
    fold_input: FoldInput,
    restraints_path: Path,
) -> FoldInput:
    """Load restraints from file and apply to FoldInput.

    Enforces mutual exclusion: if the FoldInput already has inline
    restraints (from the input JSON), raises an error.

    Args:
        fold_input: Parsed fold input (may already contain inline restraints).
        restraints_path: Path to the standalone restraints JSON file.

    Returns:
        FoldInput with restraints applied from the file.

    Raises:
        InputError: If restraints are present in both inline JSON and --restraints file.
    """
    if fold_input._restraints is not None:
        raise InputError(
            "Restraints are specified in both the input JSON and the "
            "--restraints file. Use only one source of restraints."
        )

    restraint_config, guidance_config = load_restraints_file(restraints_path)

    # Also check if guidance is duplicated
    if fold_input._guidance is not None and guidance_config is not None:
        raise InputError(
            "Guidance configuration is specified in both the input JSON "
            "and the --restraints file. Use only one source."
        )

    fold_input._restraints = restraint_config
    if guidance_config is not None:
        fold_input._guidance = guidance_config

    return fold_input
