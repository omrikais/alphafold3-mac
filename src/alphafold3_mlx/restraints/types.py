"""Restraint type dataclasses and resolved/satisfaction entities.

Defines all restraint types (distance, contact, repulsive), guidance configuration,
and their resolved (internal) and satisfaction (output) counterparts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── User-facing restraint types ──────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateResidue:
    """A candidate residue for a contact restraint."""

    chain_j: str
    residue_j: int


@dataclass(frozen=True)
class DistanceRestraint:
    """Attractive harmonic distance restraint between two atoms.

    Loss: weight * ((distance - target_distance) / sigma)^2
    """

    chain_i: str
    residue_i: int
    chain_j: str
    residue_j: int
    target_distance: float
    atom_i: str = "CA"
    atom_j: str = "CA"
    sigma: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.target_distance <= 0.0:
            raise ValueError(
                f"target_distance must be > 0, got {self.target_distance}"
            )
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class ContactRestraint:
    """One-to-many contact restraint: source must be near at least one candidate.

    Loss: weight * smooth_min_j(max(0, distance(i, j) - threshold)^2)
    Uses CA atoms for all distance computations.
    """

    chain_i: str
    residue_i: int
    candidates: list[CandidateResidue]
    threshold: float = 8.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.candidates) < 1:
            raise ValueError("Contact restraint must have at least 1 candidate")
        if self.threshold <= 0.0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class RepulsiveRestraint:
    """Minimum distance constraint between two residues (CA atoms).

    Loss: weight * max(0, min_distance - distance(CA_i, CA_j))^2
    """

    chain_i: str
    residue_i: int
    chain_j: str
    residue_j: int
    min_distance: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.min_distance <= 0.0:
            raise ValueError(f"min_distance must be > 0, got {self.min_distance}")
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class GuidanceConfig:
    """Parameters controlling how restraint gradients are applied during diffusion."""

    scale: float = 1.0
    annealing: str = "linear"
    start_step: int = 0
    end_step: int | None = None  # None means num_steps

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValueError(f"scale must be >= 0, got {self.scale}")
        if self.annealing not in ("linear", "cosine", "constant"):
            raise ValueError(
                f"annealing must be 'linear', 'cosine', or 'constant', "
                f"got '{self.annealing}'"
            )
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.end_step is not None and self.end_step < self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be >= start_step ({self.start_step})"
            )


@dataclass(frozen=True)
class RestraintConfig:
    """Top-level container aggregating all restraint types."""

    distance: list[DistanceRestraint] = field(default_factory=list)
    contact: list[ContactRestraint] = field(default_factory=list)
    repulsive: list[RepulsiveRestraint] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of restraints across all types."""
        return len(self.distance) + len(self.contact) + len(self.repulsive)

    @property
    def is_empty(self) -> bool:
        """True if no restraints are defined."""
        return self.total_count == 0


# ── Resolved restraints (internal, after index resolution) ───────────────────


@dataclass(frozen=True)
class ResolvedDistanceRestraint:
    """Distance restraint resolved to flat atom indices.

    atom_i_idx and atom_j_idx are (token_index, atom37_index) tuples.
    """

    atom_i_idx: tuple[int, int]
    atom_j_idx: tuple[int, int]
    target_distance: float
    sigma: float
    weight: float


@dataclass(frozen=True)
class ResolvedContactRestraint:
    """Contact restraint resolved to flat atom indices.

    source_atom_idx is (token_index, atom37_index) for source CA.
    candidate_atom_idxs is a list of (token_index, atom37_index) for candidate CAs.
    """

    source_atom_idx: tuple[int, int]
    candidate_atom_idxs: list[tuple[int, int]]
    threshold: float
    weight: float


@dataclass(frozen=True)
class ResolvedRepulsiveRestraint:
    """Repulsive restraint resolved to flat atom indices.

    atom_i_idx and atom_j_idx are (token_index, atom37_index) for CAs.
    """

    atom_i_idx: tuple[int, int]
    atom_j_idx: tuple[int, int]
    min_distance: float
    weight: float


# ── Satisfaction entities (output) ───────────────────────────────────────────


@dataclass(frozen=True)
class DistanceSatisfaction:
    """Satisfaction metric for a single distance restraint."""

    chain_i: str
    residue_i: int
    atom_i: str
    chain_j: str
    residue_j: int
    atom_j: str
    target_distance: float
    actual_distance: float
    satisfied: bool


@dataclass(frozen=True)
class ContactSatisfaction:
    """Satisfaction metric for a single contact restraint."""

    chain_i: str
    residue_i: int
    closest_candidate_chain: str
    closest_candidate_residue: int
    threshold: float
    actual_distance: float
    satisfied: bool


@dataclass(frozen=True)
class RepulsiveSatisfaction:
    """Satisfaction metric for a single repulsive restraint."""

    chain_i: str
    residue_i: int
    chain_j: str
    residue_j: int
    min_distance: float
    actual_distance: float
    satisfied: bool


# ── Parsing helpers ──────────────────────────────────────────────────────────


_KNOWN_RESTRAINT_KEYS = {"distance", "contact", "repulsive"}
_KNOWN_GUIDANCE_KEYS = {"scale", "annealing", "start_step", "end_step"}
_KNOWN_CONTACT_ENTRY_KEYS = {"chain_i", "residue_i", "candidates", "threshold", "weight"}


def restraint_config_from_dict(data: dict) -> RestraintConfig:
    """Parse a RestraintConfig from a JSON-like dict.

    Raises:
        ValueError: If input is not a dict, unknown keys are present,
            or entry values are invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"'restraints' must be a JSON object, got {type(data).__name__}"
        )
    unknown = set(data.keys()) - _KNOWN_RESTRAINT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in restraints: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_KNOWN_RESTRAINT_KEYS))}"
        )
    if not data:
        raise ValueError(
            "'restraints' must define at least one of: distance, contact, repulsive"
        )

    distance = []
    for i, d in enumerate(data.get("distance", [])):
        try:
            distance.append(DistanceRestraint(**d))
        except TypeError as e:
            raise ValueError(f"Invalid distance restraint [{i}]: {e}") from e

    contact = []
    for i, c in enumerate(data.get("contact", [])):
        if not isinstance(c, dict):
            raise ValueError(
                f"Invalid contact restraint [{i}]: must be a JSON object, "
                f"got {type(c).__name__}"
            )
        c_unknown = set(c.keys()) - _KNOWN_CONTACT_ENTRY_KEYS
        if c_unknown:
            raise ValueError(
                f"Unknown keys in contact restraint [{i}]: "
                f"{', '.join(sorted(c_unknown))}. "
                f"Valid keys: {', '.join(sorted(_KNOWN_CONTACT_ENTRY_KEYS))}"
            )
        try:
            candidates = [CandidateResidue(**cand) for cand in c.get("candidates", [])]
        except TypeError as e:
            raise ValueError(
                f"Invalid candidate in contact restraint [{i}]: {e}"
            ) from e
        contact.append(
            ContactRestraint(
                chain_i=c["chain_i"],
                residue_i=c["residue_i"],
                candidates=candidates,
                threshold=c.get("threshold", 8.0),
                weight=c.get("weight", 1.0),
            )
        )

    repulsive = []
    for i, r in enumerate(data.get("repulsive", [])):
        try:
            repulsive.append(RepulsiveRestraint(**r))
        except TypeError as e:
            raise ValueError(f"Invalid repulsive restraint [{i}]: {e}") from e

    return RestraintConfig(distance=distance, contact=contact, repulsive=repulsive)


def guidance_config_from_dict(data: dict) -> GuidanceConfig:
    """Parse a GuidanceConfig from a JSON-like dict.

    Raises:
        ValueError: If input is not a dict, unknown keys are present,
            or values are invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"'guidance' must be a JSON object, got {type(data).__name__}"
        )
    unknown = set(data.keys()) - _KNOWN_GUIDANCE_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in guidance: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_KNOWN_GUIDANCE_KEYS))}"
        )
    return GuidanceConfig(
        scale=data.get("scale", 1.0),
        annealing=data.get("annealing", "linear"),
        start_step=data.get("start_step", 0),
        end_step=data.get("end_step", None),
    )
