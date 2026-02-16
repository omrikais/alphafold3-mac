"""Progress bridge: connects InferenceRunner callbacks to WebSocket clients.

APIProgressReporter subclasses ProgressReporter and publishes structured
messages to a ProgressHub, which fans out to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from alphafold3_mlx.api.models import WSMessage
from alphafold3_mlx.pipeline.progress import ProgressReporter

logger = logging.getLogger(__name__)

# Stage weights for percent-complete estimation.
# Must sum to 1.0.
STAGE_WEIGHTS: dict[str, float] = {
    "weight_loading": 0.05,
    "data_pipeline": 0.10,
    "feature_preparation": 0.02,
    "recycling": 0.13,
    "diffusion": 0.55,
    "confidence": 0.05,
    "output_writing": 0.05,
    "ranking": 0.02,
    "startup": 0.03,
}


class ProgressHub:
    """Fan-out hub for WebSocket progress messages.

    Each job has its own hub. WebSocket connections subscribe to a specific
    job by connecting to /api/jobs/{id}/progress.

    Thread safety: subscribe/unsubscribe run on the event loop thread.
    publish() may be called from a worker thread (run_in_executor), so it
    marshals queue operations onto the event loop via call_soon_threadsafe.
    """

    def __init__(self) -> None:
        # job_id -> set of asyncio.Queue (one per connected client)
        self._subscribers: dict[str, set[asyncio.Queue[str]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop reference for thread-safe publishing."""
        self._loop = loop

    def subscribe(self, job_id: str) -> asyncio.Queue[str]:
        """Add a subscriber for a job's progress."""
        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._subscribers[job_id].add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[str]) -> None:
        """Remove a subscriber."""
        if job_id in self._subscribers:
            self._subscribers[job_id].discard(queue)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    def publish(self, job_id: str, message: WSMessage) -> None:
        """Publish a progress message to all subscribers of a job.

        This may be called from the sync worker thread, so we marshal
        onto the event loop via call_soon_threadsafe to avoid racing
        with subscribe/unsubscribe on asyncio.Queue (which is not
        thread-safe).
        """
        if job_id not in self._subscribers:
            return
        msg_str = message.model_dump_json()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._deliver, job_id, msg_str)
        else:
            # Fallback for tests or when no loop is set
            self._deliver(job_id, msg_str)

    def _deliver(self, job_id: str, msg_str: str) -> None:
        """Fan out a serialized message to subscribers. Must run on the event loop thread."""
        if job_id not in self._subscribers:
            return
        dead_queues: list[asyncio.Queue[str]] = []
        for queue in self._subscribers[job_id]:
            try:
                queue.put_nowait(msg_str)
            except asyncio.QueueFull:
                dead_queues.append(queue)
        for q in dead_queues:
            self._subscribers[job_id].discard(q)

    def publish_completed(self, job_id: str) -> None:
        self.publish(job_id, WSMessage(type="completed", percent_complete=100.0))

    def publish_failed(self, job_id: str, error: str) -> None:
        self.publish(job_id, WSMessage(type="failed", error=error))

    def publish_cancelled(self, job_id: str) -> None:
        self.publish(job_id, WSMessage(type="cancelled"))


class APIProgressReporter(ProgressReporter):
    """ProgressReporter subclass that publishes to ProgressHub.

    Used by the job queue worker to bridge InferenceRunner callbacks
    to WebSocket clients.  Also persists stage/progress to the DB so
    REST polling picks up the latest state even when WebSocket messages
    are missed (e.g. client connects after stage already started).
    """

    def __init__(
        self,
        job_id: str,
        hub: ProgressHub,
        store: Any = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose=verbose)
        self._job_id = job_id
        self._hub = hub
        self._store = store
        self._percent: float = 0.0
        self._current_stage_name: str = "startup"

    @property
    def percent_complete(self) -> float:
        return self._percent

    def _persist(self, stage: str | None = None) -> None:
        """Write current progress/stage to DB so REST polling sees it."""
        if self._store is not None:
            try:
                self._store.update_progress(
                    self._job_id, self._percent, stage or self._current_stage_name
                )
            except Exception:
                pass  # best-effort — don't break the pipeline

    def on_stage_start(self, stage: str) -> None:
        super().on_stage_start(stage)
        self._current_stage_name = stage
        self._persist(stage)
        self._hub.publish(
            self._job_id,
            WSMessage(
                type="stage_change",
                stage=stage,
                percent_complete=self._percent,
            ),
        )

    def on_stage_end(self, stage: str) -> None:
        super().on_stage_end(stage)
        # Advance percent to the end of this stage
        weight = STAGE_WEIGHTS.get(stage, 0.0)
        self._percent = min(self._percent + weight * 100, 99.0)
        self._persist(stage)

    def on_diffusion_step(self, step: int, total: int) -> None:
        super().on_diffusion_step(step, total)
        # Diffusion progress: interpolate within the diffusion stage weight
        base = sum(
            STAGE_WEIGHTS[s]
            for s in ("weight_loading", "data_pipeline", "feature_preparation", "recycling", "startup")
        ) * 100
        diffusion_range = STAGE_WEIGHTS["diffusion"] * 100
        self._percent = base + diffusion_range * (step + 1) / max(total, 1)
        # Persist every 10 steps to avoid DB thrashing
        if step % 10 == 0 or step + 1 == total:
            self._persist("diffusion")
        self._hub.publish(
            self._job_id,
            WSMessage(
                type="progress",
                stage="diffusion",
                diffusion_step=step + 1,
                diffusion_total=total,
                percent_complete=round(self._percent, 1),
            ),
        )

    def on_recycling_iteration(self, iteration: int, total: int) -> None:
        super().on_recycling_iteration(iteration, total)
        base = sum(
            STAGE_WEIGHTS[s]
            for s in ("weight_loading", "data_pipeline", "feature_preparation", "startup")
        ) * 100
        recycling_range = STAGE_WEIGHTS["recycling"] * 100
        self._percent = base + recycling_range * (iteration + 1) / max(total, 1)
        self._hub.publish(
            self._job_id,
            WSMessage(
                type="progress",
                stage="recycling",
                recycling_iteration=iteration + 1,
                recycling_total=total,
                percent_complete=round(self._percent, 1),
            ),
        )

    def on_complete(self) -> None:
        super().on_complete()
        self._percent = 100.0
