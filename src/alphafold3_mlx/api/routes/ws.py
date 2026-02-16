"""WebSocket endpoint for real-time job progress."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])


@router.websocket("/api/jobs/{job_id}/progress")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream real-time progress updates for a job.

    The client connects to this endpoint and receives JSON messages
    matching the WSMessage schema until the job completes, fails,
    or is cancelled.
    """
    # M-10: Require token when api_key is configured (HTTP middleware doesn't cover WS)
    api_cfg = getattr(websocket.app.state, "api_config", None)
    if api_cfg and api_cfg.api_key:
        token = websocket.query_params.get("token", "")
        if token != api_cfg.api_key:
            await websocket.close(code=4001)
            return

    # M-09: Validate WebSocket origin (CSWSH protection)
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    _LOCAL_PREFIXES = (
        "http://localhost", "http://127.0.0.1",
        "https://localhost", "https://127.0.0.1",
    )

    def _origin_is_allowed(o: str) -> bool:
        """Check origin is same-origin, local, or in configured cors_origins."""
        # Same-origin: Origin's host:port matches the request Host header
        request_host = websocket.headers.get("host", "")
        if request_host:
            parsed = urlparse(o)
            if parsed.netloc == request_host:
                return True
        for prefix in _LOCAL_PREFIXES:
            if o.startswith(prefix):
                rest = o[len(prefix):]
                if rest == "" or rest[0] in (":", "/"):
                    return True
        # Honor configured CORS origins
        if api_cfg and api_cfg.cors_origins:
            if "*" in api_cfg.cors_origins:
                return True
            normalized = o.rstrip("/")
            for allowed in api_cfg.cors_origins:
                if normalized == allowed.rstrip("/"):
                    return True
        return False

    if origin and not _origin_is_allowed(origin):
        await websocket.close(code=4003)
        return

    await websocket.accept()

    hub = websocket.app.state.progress_hub
    store = websocket.app.state.job_store

    # Verify job exists
    job = store.get_job(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "error": f"Job {job_id} not found"})
        await websocket.close(code=4004)
        return

    # If already terminal, send final state and close
    if job.status.value in ("completed", "failed", "cancelled"):
        await websocket.send_json({
            "type": job.status.value,
            "percent_complete": job.progress,
            "error": job.error_message,
        })
        await websocket.close()
        return

    # Subscribe to progress updates
    queue = hub.subscribe(job_id)
    try:
        while True:
            try:
                # Wait for a message with timeout for keepalive
                msg_str = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(msg_str)

                # Check for terminal messages
                if any(t in msg_str for t in ('"completed"', '"failed"', '"cancelled"')):
                    break
            except asyncio.TimeoutError:
                # Send ping/keepalive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected for job %s", job_id)
    except Exception:
        logger.exception("WebSocket error for job %s", job_id)
    finally:
        hub.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
