"""Main model components for AlphaFold 3 MLX implementation.

This subpackage contains the high-level model orchestration including:
- Model: Main AlphaFold 3 model class
- Recycling loop implementation
- Inference orchestration with mmCIF output
"""

from alphafold3_mlx.model.model import Model
from alphafold3_mlx.model.recycling import (
    RecyclingState,
    run_recycling_loop,
    check_convergence,
    compute_embedding_difference,
)
from alphafold3_mlx.model.inference import (
    run_inference,
    run_inference_with_checkpoints,
    write_mmcif_output,
    InferenceStats,
)

__all__ = [
    # Main model
    "Model",
    # Recycling
    "RecyclingState",
    "run_recycling_loop",
    "check_convergence",
    "compute_embedding_difference",
    # Inference
    "run_inference",
    "run_inference_with_checkpoints",
    "write_mmcif_output",
    "InferenceStats",
]
