"""CLI argument handling for AlphaFold 3 MLX pipeline.

This module provides command-line argument parsing and validation
according to the CLI interface contract.

Example:
    args = parse_args()
    cli_args = CLIArguments.from_namespace(args)
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alphafold3_mlx.pipeline.errors import InputError


def _resolve_model_dir() -> str:
    """Resolve default model weights directory (C-01).

    Checks in order:
    1. ``AF3_WEIGHTS_DIR`` env var
    2. ``~/.alphafold3/weights/model``
    3. ``weights/model`` (relative fallback for dev)

    Returns:
        Best available default path as a string.
    """
    env = os.environ.get("AF3_WEIGHTS_DIR")
    if env:
        return env
    home_default = os.path.expanduser("~/.alphafold3/weights/model")
    if os.path.isdir(home_default):
        return home_default
    return "weights/model"


@dataclass(frozen=True)
class CLIArguments:
    """Parsed command-line arguments.

    Immutable dataclass containing validated CLI arguments.

    Attributes:
        input_path: Path to input JSON file.
        output_dir: Directory for output files.
        model_dir: Path to model weights directory.
        num_samples: Number of structure samples to generate.
        diffusion_steps: Number of diffusion steps.
        seed: Random seed for reproducibility. None = time-based.
        precision: Compute precision. None = auto-detect.
        verbose: Enable detailed progress output.
        no_overwrite: Prevent overwriting existing outputs.
        run_data_pipeline: Run AF3-style data pipeline (MSA/template search)
            before featurisation. Requires HMMER binaries and databases.
        db_dir: Optional database directory used to resolve standard AF3
            database filenames. Overrides AF3_DB_DIR env var when set.
        msa_cache_dir: Optional directory for caching MSA/template pipeline
            results. When set, repeated runs with the same sequences skip
            HMMER search.
        max_template_date: Maximum template release date in YYYY-MM-DD format.
            Templates newer than this date are excluded.
        max_tokens: Maximum token bucket size to cap memory usage. When set,
            filters available bucket sizes to those at or below this limit.
    """

    # Required arguments
    input_path: Path
    output_dir: Path

    # Optional arguments with defaults
    model_dir: Path = None  # type: ignore[assignment]  # resolved in __post_init__
    num_samples: int = 5
    diffusion_steps: int = 200
    seed: int | None = None
    precision: Literal["float32", "float16", "bfloat16"] | None = None
    verbose: bool = False
    no_overwrite: bool = False
    run_data_pipeline: bool = False
    db_dir: Path | None = None
    msa_cache_dir: Path | None = None
    max_template_date: str = "2021-09-30"
    max_tokens: int | None = None
    restraints_file: Path | None = None

    def __post_init__(self) -> None:
        """Validate arguments after initialization."""
        # Resolve model_dir default if not explicitly set
        if self.model_dir is None:
            object.__setattr__(self, "model_dir", Path(_resolve_model_dir()))

        # Validate num_samples
        if self.num_samples < 1:
            raise InputError(f"num_samples must be >= 1, got {self.num_samples}")

        # Validate diffusion_steps
        if self.diffusion_steps < 1:
            raise InputError(f"diffusion_steps must be >= 1, got {self.diffusion_steps}")

        # L-03: Warn on unusually high diffusion_steps
        if self.diffusion_steps > 2000:
            warnings.warn(
                f"diffusion_steps={self.diffusion_steps} is unusually high (default: 200)",
                stacklevel=2,
            )

        # Validate seed
        if self.seed is not None and self.seed < 0:
            raise InputError(f"seed must be non-negative, got {self.seed}")

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CLIArguments":
        """Create CLIArguments from argparse Namespace.

        Args:
            args: Parsed argparse namespace.

        Returns:
            Validated CLIArguments instance.

        Raises:
            InputError: If arguments are invalid.
        """
        return cls(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            num_samples=args.num_samples,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
            precision=args.precision,
            verbose=args.verbose,
            no_overwrite=args.no_overwrite,
            run_data_pipeline=getattr(args, "run_data_pipeline", False),
            db_dir=Path(args.db_dir) if getattr(args, "db_dir", None) else None,
            msa_cache_dir=Path(args.msa_cache_dir) if getattr(args, "msa_cache_dir", None) else None,
            max_template_date=getattr(args, "max_template_date", "2021-09-30"),
            max_tokens=getattr(args, "max_tokens", None),
            restraints_file=Path(args.restraints) if getattr(args, "restraints", None) else None,
        )


def auto_select_precision() -> Literal["float32", "float16", "bfloat16"]:
    """Auto-select precision based on chip family.

    Returns:
        Recommended precision for the current hardware:
        - bfloat16 for M3/M4 (native bfloat16 support)
        - float16 for M2
        - float32 as fallback (on errors or unknown platforms)

    Note:
        This function never raises exceptions - it always returns a valid
        precision value, falling back to float32 on any error.
    """
    try:
        from alphafold3_mlx.weights.platform import get_platform_info

        info = get_platform_info()

        # Check if bfloat16 is supported (M3/M4)
        if info.supports_bfloat16:
            return "bfloat16"

        # M2 supports float16 well
        if info.chip_family in ("M2", "M3", "M4"):
            return "float16"

        # Conservative default for unknown chips
        return "float32"
    except Exception:
        # Fallback on any error (ImportError, PlatformError, etc.)
        return "float32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list. If None, uses sys.argv.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        prog="run_alphafold_mlx.py",
        description="AlphaFold 3 structure prediction on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic inference:
    python run_alphafold_mlx.py --input input.json --output_dir output/

  Quick test with fewer samples:
    python run_alphafold_mlx.py --input input.json --output_dir output/ \\
        --num_samples 1 --diffusion_steps 50

  Reproducible inference:
    python run_alphafold_mlx.py --input input.json --output_dir output/ \\
        --seed 12345

  High precision mode:
    python run_alphafold_mlx.py --input input.json --output_dir output/ \\
        --precision float32 --verbose
""",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 3.0.1",
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to input JSON file with sequence and features",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory for output files (mmCIF, confidence scores)",
    )

    # Optional arguments
    _default_model_dir = _resolve_model_dir()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=_default_model_dir,
        metavar="DIR",
        help=f"Path to model weights directory (default: {_default_model_dir})",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        metavar="N",
        help="Number of structure samples to generate (default: 5)",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=200,
        metavar="N",
        help="Number of diffusion iteration steps (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Compute precision. Default: bfloat16 on M3/M4, float16 on M2",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed progress output with per-stage timing",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        dest="no_overwrite",
        help="Prevent overwriting existing output files",
    )
    parser.add_argument(
        "--run_data_pipeline",
        action="store_true",
        help=(
            "Run AF3-style MSA/template search before featurisation. "
            "Requires HMMER binaries (jackhmmer/hmmsearch/hmmbuild) and "
            "database paths configured via AF3_DB_DIR or explicit env vars."
        ),
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing AF3 genetic databases (as downloaded by "
            "fetch_databases.sh). Overrides AF3_DB_DIR env var."
        ),
    )
    parser.add_argument(
        "--msa_cache_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for caching MSA/template results (skip HMMER on repeated sequences)",
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default="2021-09-30",
        metavar="DATE",
        help="Maximum template release date (YYYY-MM-DD, default: 2021-09-30)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Maximum token bucket size to cap memory usage (default: no limit)",
    )
    parser.add_argument(
        "--restraints",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to standalone restraints JSON file (mutually exclusive with inline restraints in input JSON)",
    )

    return parser.parse_args(argv)
