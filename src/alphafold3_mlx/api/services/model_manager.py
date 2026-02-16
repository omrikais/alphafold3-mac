"""Model singleton manager for the API server.

Loads the model once at startup and provides it to the job queue worker.
Thread-safe because inference is single-worker (one job at a time).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from alphafold3_mlx.api.config import APIConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the model singleton lifecycle.

    Loads model weights once during server startup and caches the instance.
    The model is shared across jobs via the single-worker job queue.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._config: Any | None = None
        self._loaded = False
        self._loading = False
        self._startup_precision: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def model(self) -> Any:
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def model_config(self) -> Any:
        return self._config

    @property
    def startup_precision(self) -> str:
        """The resolved precision from server startup (immutable)."""
        if self._startup_precision is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._startup_precision

    async def load(self, api_config: APIConfig) -> None:
        """Load model weights from disk.

        This is called during FastAPI lifespan startup.
        Runs in a thread to avoid blocking the event loop.
        """
        import asyncio

        self._loading = True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_sync, api_config)
            self._loaded = True
            logger.info("Model loaded successfully")
        except Exception:
            logger.exception("Failed to load model")
            raise
        finally:
            self._loading = False

    def _load_sync(self, api_config: APIConfig) -> None:
        """Synchronous model loading (runs in thread pool)."""
        from alphafold3_mlx import Model, ModelConfig
        from alphafold3_mlx.core import DiffusionConfig, GlobalConfig

        model_dir = api_config.model_dir

        precision = api_config.precision
        if precision is None:
            from alphafold3_mlx.pipeline.cli import auto_select_precision
            precision = auto_select_precision()

        diffusion_config = DiffusionConfig(
            num_samples=api_config.num_samples,
            num_steps=api_config.diffusion_steps,
        )
        global_config = GlobalConfig(precision=precision)
        config = ModelConfig(diffusion=diffusion_config, global_config=global_config)

        logger.info("Loading model from %s (precision=%s)", model_dir, precision)
        self._model = Model.from_pretrained(model_dir, config)
        self._config = config
        self._startup_precision = precision

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._config = None
        self._loaded = False
        logger.info("Model unloaded")
