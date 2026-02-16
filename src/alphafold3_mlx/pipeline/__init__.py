"""AlphaFold 3 MLX - Pipeline integration module.

This module provides the user-facing CLI integration layer connecting
the MLX model implementation (Phase 3) to a complete inference pipeline.

Components:
- cli: Command-line argument parsing and validation
- input_handler: AF3 JSON input parsing and validation
- output_handler: Output file management (mmCIF, JSON)
- progress: Progress reporting during inference
- ranking: Sample ranking by confidence metrics
- errors: User-facing error handling and logging
- runner: High-level inference orchestration

Example usage:
    python run_alphafold_mlx.py --input input.json --output_dir output/
"""

from alphafold3_mlx.pipeline.errors import (
    InputError,
    ResourceError,
    InferenceError,
    InterruptError,
    FailureLog,
    format_error_message,
)
from alphafold3_mlx.pipeline.cli import (
    CLIArguments,
    parse_args,
    auto_select_precision,
)
from alphafold3_mlx.pipeline.input_handler import (
    FoldInput,
    InputJSON,  # Backward-compatible alias for FoldInput
    Sequence,
    Modification,
    parse_input_json,
    validate_input,
    estimate_memory_gb,
    check_memory_available,
)
from alphafold3_mlx.pipeline.output_handler import (
    OutputBundle,
    create_output_directory,
    handle_existing_outputs,
    atomic_write,
    write_mmcif_file,
    write_confidence_scores,
    write_timing,
    write_ranking_debug,
    write_failure_log,
    write_ranked_outputs,
    check_disk_space,
)
from alphafold3_mlx.pipeline.progress import (
    ProgressReporter,
    TimingData,
    StageInfo,
)
from alphafold3_mlx.pipeline.ranking import (
    RankingScores,
    SampleRanking,
    rank_samples,
    auto_detect_complex,
    compute_aggregate_metrics,
)
from alphafold3_mlx.pipeline.runner import (
    InferenceRunner,
    InterruptHandler,
    create_failure_log,
)
from alphafold3_mlx.pipeline.logging_config import (
    get_logger,
    setup_logging,
    set_log_level,
    LoggingContext,
)

__all__ = [
    # Exceptions
    "InputError",
    "ResourceError",
    "InferenceError",
    "InterruptError",
    # Error utilities
    "FailureLog",
    "format_error_message",
    # CLI
    "CLIArguments",
    "parse_args",
    "auto_select_precision",
    # Input handling
    "FoldInput",
    "InputJSON",  # Backward-compatible alias for FoldInput
    "Sequence",
    "Modification",
    "parse_input_json",
    "validate_input",
    "estimate_memory_gb",
    "check_memory_available",
    # Output handling
    "OutputBundle",
    "create_output_directory",
    "handle_existing_outputs",
    "atomic_write",
    "write_mmcif_file",
    "write_confidence_scores",
    "write_timing",
    "write_ranking_debug",
    "write_failure_log",
    "write_ranked_outputs",
    "check_disk_space",
    # Progress
    "ProgressReporter",
    "TimingData",
    "StageInfo",
    # Ranking
    "RankingScores",
    "SampleRanking",
    "rank_samples",
    "auto_detect_complex",
    "compute_aggregate_metrics",
    # Runner
    "InferenceRunner",
    "InterruptHandler",
    "create_failure_log",
    # Logging
    "get_logger",
    "setup_logging",
    "set_log_level",
    "LoggingContext",
]
