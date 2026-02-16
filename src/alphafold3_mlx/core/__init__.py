"""Core data structures and utilities for alphafold3_mlx.

This module exports:
- Configuration dataclasses (AttentionConfig, EvoformerConfig, etc.)
- Input/output containers (AttentionInputs, FeatureBatch, ModelResult, etc.)
- Core entities (Embeddings, AtomPositions, ConfidenceScores, etc.)
- Constants and tolerances
- Custom exceptions
- Validation utilities
"""

# Phase 0: Attention-specific types
from alphafold3_mlx.core.config import AttentionConfig
from alphafold3_mlx.core.inputs import AttentionInputs
from alphafold3_mlx.core.outputs import AttentionOutput
from alphafold3_mlx.core.intermediates import AttentionIntermediates
from alphafold3_mlx.core.validation import ValidationResult
from alphafold3_mlx.core.golden import GoldenOutputs
from alphafold3_mlx.core.benchmark import BenchmarkResult

# Phase 0: Constants
from alphafold3_mlx.core.constants import (
    TOLERANCES,
    AF3_SHAPES,
    MEMORY_RATIO_THRESHOLD,
    DEFAULT_MASK_VALUE,
)

# Phase 3: Model configuration dataclasses
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

# Phase 3: Model constants
from alphafold3_mlx.core.constants import (
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

# Phase 3: Core entities
from alphafold3_mlx.core.entities import (
    Embeddings,
    AtomPositions,
    ConfidenceScores,
    NoiseSchedule,
    GatherInfo,
)

# Phase 3: Input types
from alphafold3_mlx.core.inputs import (
    TokenFeatures,
    MSAFeatures,
    TemplateFeatures,
    FrameFeatures,
    BondInfo,
    FeatureBatch,
)

# Phase 3: Output types
from alphafold3_mlx.core.outputs import ModelResult

# Phase 3: Exceptions
from alphafold3_mlx.core.exceptions import (
    NaNError,
    MemoryError,
    ShapeMismatchError,
    WeightsNotFoundError,
    ValidationError,
)

__all__ = [
    # Phase 0: Attention
    "AttentionConfig",
    "AttentionInputs",
    "AttentionOutput",
    "AttentionIntermediates",
    "ValidationResult",
    "GoldenOutputs",
    "BenchmarkResult",
    # Phase 0: Constants
    "TOLERANCES",
    "AF3_SHAPES",
    "MEMORY_RATIO_THRESHOLD",
    "DEFAULT_MASK_VALUE",
    # Phase 3: Configuration
    "PairFormerConfig",
    "TemplateConfig",
    "MSAStackConfig",
    "SampleConfig",
    "EvoformerConfig",
    "DiffusionConfig",
    "ConfidenceConfig",
    "GlobalConfig",
    "ModelConfig",
    # Phase 3: Constants
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
    # Phase 3: Entities
    "Embeddings",
    "AtomPositions",
    "ConfidenceScores",
    "NoiseSchedule",
    "GatherInfo",
    # Phase 3: Inputs
    "TokenFeatures",
    "MSAFeatures",
    "TemplateFeatures",
    "FrameFeatures",
    "BondInfo",
    "FeatureBatch",
    # Phase 3: Outputs
    "ModelResult",
    # Phase 3: Exceptions
    "NaNError",
    "MemoryError",
    "ShapeMismatchError",
    "WeightsNotFoundError",
    "ValidationError",
]
