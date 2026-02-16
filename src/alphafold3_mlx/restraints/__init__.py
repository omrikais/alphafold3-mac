"""Restraint-guided diffusion docking for AlphaFold 3 MLX.

This package provides spatial restraints (distance, contact, repulsive) that guide
the diffusion sampling loop to produce docked structures satisfying user-specified
geometric constraints.
"""

# Pure-Python submodules — no MLX dependency
from alphafold3_mlx.restraints.types import (
    CandidateResidue,
    ContactRestraint,
    ContactSatisfaction,
    DistanceRestraint,
    DistanceSatisfaction,
    GuidanceConfig,
    RepulsiveRestraint,
    RepulsiveSatisfaction,
    ResolvedContactRestraint,
    ResolvedDistanceRestraint,
    ResolvedRepulsiveRestraint,
    RestraintConfig,
    guidance_config_from_dict,
    restraint_config_from_dict,
)
from alphafold3_mlx.restraints.validate import validate_restraints
from alphafold3_mlx.restraints.resolve import resolve_restraints

# MLX-dependent submodules — deferred so that pure-Python restraint
# parsing/validation works in environments without MLX (e.g. Linux CI).
_mlx_available = False
try:
    import mlx.core  # noqa: F401
    _mlx_available = True
except ImportError:
    pass

if _mlx_available:
    from alphafold3_mlx.restraints.loss import combined_restraint_loss
    from alphafold3_mlx.restraints.guidance import build_guidance_fn

__all__ = [
    # Types
    "CandidateResidue",
    "ContactRestraint",
    "ContactSatisfaction",
    "DistanceRestraint",
    "DistanceSatisfaction",
    "GuidanceConfig",
    "RepulsiveRestraint",
    "RepulsiveSatisfaction",
    "ResolvedContactRestraint",
    "ResolvedDistanceRestraint",
    "ResolvedRepulsiveRestraint",
    "RestraintConfig",
    # Parsing
    "guidance_config_from_dict",
    "restraint_config_from_dict",
    # Functions (pure-Python)
    "validate_restraints",
    "resolve_restraints",
]

if _mlx_available:
    __all__ += [
        "combined_restraint_loss",
        "build_guidance_fn",
    ]
