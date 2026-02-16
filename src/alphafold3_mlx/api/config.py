"""API configuration for AlphaFold 3 MLX web server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _default_model_dir() -> Path:
    """Resolve default model weights directory (C-01)."""
    env = os.environ.get("AF3_WEIGHTS_DIR")
    if env:
        return Path(env)
    home_default = Path(os.path.expanduser("~/.alphafold3/weights/model"))
    if home_default.is_dir():
        return home_default
    return Path("weights/model")


def _default_data_dir() -> Path:
    """Resolve default persistent data directory (H-03)."""
    return Path(os.environ.get("AF3_DATA_DIR", os.path.expanduser("~/.alphafold3_mlx/data")))


@dataclass
class APIConfig:
    """Configuration for the API server.

    Attributes:
        host: Bind address.
        port: Bind port.
        model_dir: Path to model weights directory.
        data_dir: Path to persistent data directory (jobs DB, outputs).
        db_dir: Path to genetic databases (optional, for full pipeline).
        num_samples: Default number of structure samples per job.
        diffusion_steps: Default number of diffusion steps.
        precision: Model precision mode (float32, float16, bfloat16, or None for auto).
        max_queue_size: Maximum number of pending jobs.
        run_data_pipeline: Whether to run MSA/template search by default.
        cors_origins: Allowed CORS origins (empty = same-origin only).
        api_key: Optional Bearer token for simple auth (M-10).
    """

    host: str = "127.0.0.1"
    port: int = 8642
    model_dir: Path = field(default_factory=_default_model_dir)
    data_dir: Path = field(default_factory=_default_data_dir)
    db_dir: Path | None = None
    num_samples: int = 5
    diffusion_steps: int = 200
    precision: Literal["float32", "float16", "bfloat16"] | None = None
    max_queue_size: int = 20
    run_data_pipeline: bool = False
    cors_origins: list[str] = field(default_factory=list)
    api_key: str | None = None
