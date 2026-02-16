"""AlphaFold 3 MLX Web API.

Start the server::

    python -m alphafold3_mlx.api --model-dir weights/model --port 8642

Or from Python::

    from alphafold3_mlx.api import create_app
    app = create_app()
"""

from alphafold3_mlx.api.app import create_app
from alphafold3_mlx.api.config import APIConfig

__all__ = ["create_app", "APIConfig"]


def main() -> None:
    """CLI entry point for the API server."""
    import argparse
    import logging
    import os
    from pathlib import Path

    from alphafold3_mlx.api.config import _default_model_dir, _default_data_dir

    parser = argparse.ArgumentParser(
        description="AlphaFold 3 MLX — Local protein structure prediction server"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=_default_model_dir(),
        help="Path to model weights directory (default: resolved via AF3_WEIGHTS_DIR / ~/.alphafold3/weights/model)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=_default_data_dir(),
        help="Path to persistent data directory for jobs DB (default: ~/.alphafold3_mlx/data)",
    )
    parser.add_argument(
        "--db-dir", type=Path,
        default=Path(os.environ["AF3_DB_DIR"]) if os.environ.get("AF3_DB_DIR") else None,
        help="Path to genetic databases for full pipeline mode (default: $AF3_DB_DIR)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8642,
        help="Bind port (default: 8642)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Default number of structure samples (default: 5)",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=200,
        help="Default number of diffusion steps (default: 200)",
    )
    parser.add_argument(
        "--precision", type=str, default=None, choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: auto-detect)",
    )
    parser.add_argument(
        "--run-data-pipeline", action=argparse.BooleanOptionalAction, default=None,
        help="Enable MSA/template search pipeline (default: on when --db-dir or $AF3_DB_DIR is set)",
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("AF3_API_KEY"),
        help="Optional Bearer token for API authentication (or set AF3_API_KEY env var)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # If --run-data-pipeline wasn't explicitly passed, enable it when db_dir is available
    run_data_pipeline = args.run_data_pipeline
    if run_data_pipeline is None:
        run_data_pipeline = args.db_dir is not None

    config = APIConfig(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        db_dir=args.db_dir,
        num_samples=args.num_samples,
        diffusion_steps=args.diffusion_steps,
        precision=args.precision,
        run_data_pipeline=run_data_pipeline,
        api_key=args.api_key,
    )

    app = create_app(config)

    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port, log_level=args.log_level.lower())
