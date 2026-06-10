"""Core data structures and utilities for alphafold3_mlx.

This module exports:
- Configuration dataclasses (AttentionConfig, EvoformerConfig, etc.)
- Input/output containers (AttentionInputs, FeatureBatch, ModelResult, etc.)
- Core entities (Embeddings, AtomPositions, ConfidenceScores, etc.)
- Constants and tolerances
- Custom exceptions
- Validation utilities
"""

# Pure-Python modules — no MLX dependency
from alphafold3_mlx.core.constants import (
    TOLERANCES,
    AF3_SHAPES,
    MEMORY_RATIO_THRESHOLD,
    DEFAULT_MASK_VALUE,
    SIGMA_DATA,
    SIGMA_MAX,
    SIGMA_MIN,
    RHO,
    MAX_ATOMS,
    SEQ_CHANNEL,
    PAIR_CHANNEL,
    MSA_CHANNEL,
    NUM_PAIRFORMER_LAYERS,
    NUM_MSA_LAYERS,
    NUM_DIFFUSION_STEPS,
    NUM_DIFFUSION_TRANSFORMER_BLOCKS,
    NUM_SAMPLES,
    NUM_PLDDT_BINS,
    NUM_PAE_BINS,
    MAX_ERROR_BIN,
    DEFAULT_NUM_RECYCLES,
    DIFFUSION_EVAL_INTERVAL,
    BOND_LENGTH_TOLERANCE,
    BOND_ANGLE_TOLERANCE,
    MIN_VALID_FRACTION,
)

from alphafold3_mlx.core.exceptions import (
    NaNError,
    MemoryError,
    ShapeMismatchError,
    WeightsNotFoundError,
    ValidationError,
)

# MLX-dependent modules — deferred so pure-Python code (restraint
# validation, constants) can be imported on Linux CI without MLX.
_mlx_available = False
try:
    import mlx.core  # noqa: F401
    _mlx_available = True
except ImportError:
    pass

if _mlx_available:
    from alphafold3_mlx.core.config import AttentionConfig
    from alphafold3_mlx.core.inputs import AttentionInputs
    from alphafold3_mlx.core.outputs import AttentionOutput
    from alphafold3_mlx.core.intermediates import AttentionIntermediates
    from alphafold3_mlx.core.validation import ValidationResult
    from alphafold3_mlx.core.golden import GoldenOutputs
    from alphafold3_mlx.core.benchmark import BenchmarkResult

    from alphafold3_mlx.core.config import (
        PairFormerConfig,
        TemplateConfig,
        MSAStackConfig,
        SampleConfig,
        EvoformerConfig,
        DiffusionConfig,
        ConfidenceConfig,
        GlobalConfig,
        ModelConfig,
    )

    from alphafold3_mlx.core.entities import (
        Embeddings,
        AtomPositions,
        ConfidenceScores,
        NoiseSchedule,
        GatherInfo,
    )

    from alphafold3_mlx.core.inputs import (
        TokenFeatures,
        MSAFeatures,
        TemplateFeatures,
        FrameFeatures,
        BondInfo,
        FeatureBatch,
    )

    from alphafold3_mlx.core.outputs import ModelResult

__all__ = [
    # Constants (always available)
    "TOLERANCES",
    "AF3_SHAPES",
    "MEMORY_RATIO_THRESHOLD",
    "DEFAULT_MASK_VALUE",
    "SIGMA_DATA",
    "SIGMA_MAX",
    "SIGMA_MIN",
    "RHO",
    "MAX_ATOMS",
    "SEQ_CHANNEL",
    "PAIR_CHANNEL",
    "MSA_CHANNEL",
    "NUM_PAIRFORMER_LAYERS",
    "NUM_MSA_LAYERS",
    "NUM_DIFFUSION_STEPS",
    "NUM_DIFFUSION_TRANSFORMER_BLOCKS",
    "NUM_SAMPLES",
    "NUM_PLDDT_BINS",
    "NUM_PAE_BINS",
    "MAX_ERROR_BIN",
    "DEFAULT_NUM_RECYCLES",
    "DIFFUSION_EVAL_INTERVAL",
    "BOND_LENGTH_TOLERANCE",
    "BOND_ANGLE_TOLERANCE",
    "MIN_VALID_FRACTION",
    # Exceptions (always available)
    "NaNError",
    "MemoryError",
    "ShapeMismatchError",
    "WeightsNotFoundError",
    "ValidationError",
]

if _mlx_available:
    __all__ += [
        # Attention
        "AttentionConfig",
        "AttentionInputs",
        "AttentionOutput",
        "AttentionIntermediates",
        "ValidationResult",
        "GoldenOutputs",
        "BenchmarkResult",
        # Configuration
        "PairFormerConfig",
        "TemplateConfig",
        "MSAStackConfig",
        "SampleConfig",
        "EvoformerConfig",
        "DiffusionConfig",
        "ConfidenceConfig",
        "GlobalConfig",
        "ModelConfig",
        # Entities
        "Embeddings",
        "AtomPositions",
        "ConfidenceScores",
        "NoiseSchedule",
        "GatherInfo",
        # Inputs
        "TokenFeatures",
        "MSAFeatures",
        "TemplateFeatures",
        "FrameFeatures",
        "BondInfo",
        "FeatureBatch",
        # Outputs
        "ModelResult",
    ]
