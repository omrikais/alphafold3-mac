"""Single-worker async job queue.

Only one inference job runs at a time (MLX unified memory constraint).
Jobs are FIFO. The worker runs in a background asyncio task and delegates
actual inference to a thread pool executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any

from alphafold3_mlx.api.config import APIConfig
from alphafold3_mlx.api.models import JobStatus
from alphafold3_mlx.api.services.job_store import JobStore
from alphafold3_mlx.api.services.model_manager import ModelManager
from alphafold3_mlx.api.services.progress_bridge import (
    APIProgressReporter,
    ProgressHub,
)

logger = logging.getLogger(__name__)


class JobQueue:
    """Single-worker async job queue.

    Attributes:
        store: Job persistence.
        model_manager: Model singleton.
        hub: Progress hub for WebSocket fan-out.
        config: API configuration.
    """

    def __init__(
        self,
        store: JobStore,
        model_manager: ModelManager,
        hub: ProgressHub,
        config: APIConfig,
    ) -> None:
        self.store = store
        self.model_manager = model_manager
        self.hub = hub
        self.config = config
        self._queue: asyncio.Queue[str] = asyncio.Queue()  # unbounded; capacity managed via _pending_count
        self._worker_task: asyncio.Task[None] | None = None
        self._active_job_id: str | None = None
        self._cancel_flags: dict[str, bool] = {}
        self._pending_count: int = 0  # tracks non-cancelled jobs waiting in queue

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    @property
    def queue_size(self) -> int:
        return self._pending_count

    def start(self) -> None:
        """Start the background worker."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Job queue worker started")

    async def stop(self) -> None:
        """Stop the background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # Transition any active job out of RUNNING so it doesn't remain
        # permanently stuck after restart.
        if self._active_job_id:
            logger.warning(
                "Shutting down with active job %s — marking as FAILED",
                self._active_job_id,
            )
            self.store.update_status(
                self._active_job_id,
                JobStatus.FAILED,
                error_message="Server shut down while job was running",
            )
            self._active_job_id = None
        logger.info("Job queue worker stopped")

    async def enqueue(self, job_id: str) -> None:
        """Add a job to the queue.

        Raises:
            asyncio.QueueFull: If queue is at capacity.
        """
        if self._pending_count >= self.config.max_queue_size:
            raise asyncio.QueueFull()
        self._cancel_flags[job_id] = False
        self._queue.put_nowait(job_id)
        self._pending_count += 1

    def request_cancel(self, job_id: str) -> bool:
        """Request cancellation of a job.

        Returns True if the job was found and cancellation was requested.
        """
        # If the job is currently running, set the cancel flag
        if self._active_job_id == job_id:
            self._cancel_flags[job_id] = True
            return True

        # If it's still in the queue, we can't remove it from asyncio.Queue,
        # so set the flag (worker will skip it) and free the capacity slot now.
        if job_id in self._cancel_flags:
            self._cancel_flags[job_id] = True
            self._pending_count = max(0, self._pending_count - 1)
            self.store.update_status(job_id, JobStatus.CANCELLED)
            self.hub.publish_cancelled(job_id)
            return True

        return False

    async def _worker_loop(self) -> None:
        """Main worker loop: process one job at a time."""
        while True:
            try:
                job_id = await self._queue.get()

                # Check if already cancelled while queued
                if self._cancel_flags.get(job_id, False):
                    logger.info("Skipping cancelled job %s", job_id)
                    self._cancel_flags.pop(job_id, None)
                    # _pending_count already decremented in request_cancel
                    continue

                self._pending_count = max(0, self._pending_count - 1)
                self._active_job_id = job_id
                self.store.update_status(job_id, JobStatus.RUNNING)

                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._run_job_sync, job_id)
                except Exception as exc:
                    error_msg = str(exc)
                    logger.exception("Job %s failed: %s", job_id, error_msg)
                    self.store.update_status(
                        job_id, JobStatus.FAILED, error_message=error_msg
                    )
                    self.hub.publish_failed(job_id, error_msg)
                finally:
                    self._active_job_id = None
                    self._cancel_flags.pop(job_id, None)

            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                raise
            except Exception:
                logger.exception("Unexpected error in worker loop")

    def _run_job_sync(self, job_id: str) -> None:
        """Run a single inference job synchronously (in thread pool).

        This bridges the existing InferenceRunner with the API layer.
        """
        from alphafold3_mlx.pipeline.cli import CLIArguments
        from alphafold3_mlx.pipeline.input_handler import parse_input_json
        from alphafold3_mlx.pipeline.runner import InferenceRunner
        from alphafold3_mlx.pipeline.errors import (
            InputError,
            ResourceError,
            InferenceError,
            InterruptError,
        )

        job = self.store.get_job(job_id)
        if job is None:
            logger.error("Job %s not found in store", job_id)
            return

        # Set up progress reporter that bridges to WebSocket + DB
        progress = APIProgressReporter(job_id, self.hub, store=self.store, verbose=True)

        output_dir = self.store.job_output_dir(job_id)

        # Write input JSON to a temp file for parse_input_json
        input_file = output_dir / "input.json"
        input_file.write_text(json.dumps(job.input_json))

        try:
            fold_input = parse_input_json(input_file)
        except InputError as e:
            self.store.update_status(job_id, JobStatus.FAILED, error_message=str(e))
            self.hub.publish_failed(job_id, str(e))
            return

        # Build CLIArguments-like object for InferenceRunner
        args = CLIArguments(
            input_path=input_file,
            output_dir=output_dir,
            model_dir=self.config.model_dir,
            num_samples=job.num_samples,
            diffusion_steps=job.diffusion_steps,
            precision=job.precision or self.config.precision,
            seed=(job.input_json.get("modelSeeds") or [42])[0],
            run_data_pipeline=job.run_data_pipeline,
            db_dir=self.config.db_dir,
            msa_cache_dir=self.config.data_dir / "msa_cache" if (job.run_data_pipeline and job.use_cache) else None,
            verbose=True,
            no_overwrite=False,
        )

        # Create runner with the progress bridge
        runner = InferenceRunner(
            args=args,
            input_json=fold_input,
            progress=progress,
        )

        # Monkey-patch _load_model to return cached model singleton
        original_model = self.model_manager.model
        original_config = self.model_manager.model_config

        def patched_load_model() -> tuple[Any, Any]:
            # Apply per-job diffusion settings to the shared model config
            original_config.diffusion.num_samples = args.num_samples
            original_config.diffusion.num_steps = args.diffusion_steps
            # Always reset precision: per-job override or server startup default.
            # Use model_manager.startup_precision (immutable) rather than
            # original_config.global_config.precision which set_precision mutates.
            target_precision = args.precision or self.model_manager.startup_precision
            original_model.set_precision(target_precision)
            return original_model, original_config

        runner._load_model = patched_load_model  # type: ignore[assignment]

        # Hook cancellation into the interrupt handler
        def check_cancel() -> None:
            if self._cancel_flags.get(job_id, False):
                runner.interrupt_handler.interrupted = True

        # Wrap the original interrupt check
        original_check = runner.interrupt_handler.check

        def patched_check() -> None:
            check_cancel()
            original_check()

        runner.interrupt_handler.check = patched_check  # type: ignore[assignment]

        try:
            runner.run()
            self.store.update_status(job_id, JobStatus.COMPLETED, progress=100.0)
            self.hub.publish_completed(job_id)
            logger.info("Job %s completed successfully", job_id)
        except InterruptError:
            self.store.update_status(job_id, JobStatus.CANCELLED)
            self.hub.publish_cancelled(job_id)
        except (InputError, ResourceError, InferenceError) as e:
            error_msg = str(e)
            self.store.update_status(job_id, JobStatus.FAILED, error_message=error_msg)
            self.hub.publish_failed(job_id, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
            self.store.update_status(job_id, JobStatus.FAILED, error_message=error_msg)
            self.hub.publish_failed(job_id, error_msg)
