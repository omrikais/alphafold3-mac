"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from alphafold3_mlx.api.config import APIConfig
from alphafold3_mlx.api.models import JobStatus
from alphafold3_mlx.api.services.job_queue import JobQueue
from alphafold3_mlx.api.services.job_store import JobStore
from alphafold3_mlx.api.services.model_manager import ModelManager
from alphafold3_mlx.api.services.progress_bridge import ProgressHub

logger = logging.getLogger(__name__)

# H-04: Path to the Next.js static export (env var override for pip installs)
FRONTEND_DIR = Path(os.environ.get(
    "AF3_FRONTEND_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "out"),
))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: load model on startup, cleanup on shutdown."""
    config: APIConfig = app.state.api_config

    # Initialize services
    hub = ProgressHub()
    hub.set_loop(asyncio.get_running_loop())
    model_manager = ModelManager()
    store = JobStore(config.data_dir)

    # Recover jobs stuck as RUNNING from a previous crash or unclean shutdown
    recovered = store.recover_stale_running()
    if recovered:
        logger.warning("Recovered %d stale RUNNING job(s) from previous session", recovered)

    queue = JobQueue(store, model_manager, hub, config)

    # Re-enqueue jobs that were PENDING when the server last stopped
    pending_ids = store.get_pending_job_ids()
    re_enqueued = 0
    for i, job_id in enumerate(pending_ids):
        try:
            await queue.enqueue(job_id)
            re_enqueued += 1
        except asyncio.QueueFull:
            overflow = len(pending_ids) - i
            logger.warning(
                "Queue full — marking %d remaining pending job(s) as failed",
                overflow,
            )
            for overflow_id in pending_ids[i:]:
                store.update_status(
                    overflow_id,
                    JobStatus.FAILED,
                    error_message="Server restarted; queue capacity exceeded",
                )
            break
    if re_enqueued:
        logger.info(
            "Re-enqueued %d pending job(s) from previous session",
            re_enqueued,
        )

    # Store on app state for route access
    app.state.model_manager = model_manager
    app.state.job_store = store
    app.state.job_queue = queue
    app.state.progress_hub = hub

    # Load model weights (blocking at startup)
    logger.info("Loading model weights from %s ...", config.model_dir)
    try:
        await model_manager.load(config)
        logger.info("Model loaded, starting job queue worker")
    except Exception as e:
        logger.warning(
            "Model loading failed: %s\n"
            "  Expected weights at: %s\n"
            "  Ensure af3.bin.zst is in the model directory.\n"
            "  See https://github.com/google-deepmind/alphafold3 for weight access.\n"
            "  Server will start without model — jobs will fail until model is available.",
            e, config.model_dir,
        )

    # Start background job worker
    queue.start()

    yield

    # Shutdown
    await queue.stop()
    model_manager.unload()
    logger.info("Server shut down cleanly")


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = APIConfig()

    app = FastAPI(
        title="AlphaFold 3 MLX",
        description="Local protein structure prediction on Apple Silicon",
        version="3.0.1",
        lifespan=lifespan,
    )

    # Store config for lifespan access
    app.state.api_config = config

    # M-10: Warn when binding to all interfaces without auth
    if config.host == "0.0.0.0":
        logger.warning(
            "Server binding to 0.0.0.0 — accessible on all interfaces. "
            "No authentication is configured. Use --host 127.0.0.1 for local-only access."
        )

    # CORS — always enable user-specified origins
    cors_origins = list(config.cors_origins)
    # H-07: Only add dev origins when binding to localhost
    if config.host in ("127.0.0.1", "localhost", "::1"):
        dev_origins = [
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:3001", "http://127.0.0.1:3001",
            "http://localhost:3002", "http://127.0.0.1:3002",
        ]
        for origin in dev_origins:
            if origin not in cors_origins:
                cors_origins.append(origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # M-10: Simple Bearer token auth middleware (optional)
    if config.api_key:
        @app.middleware("http")
        async def _auth_middleware(request: Request, call_next) -> Response:
            # Skip auth for docs, health, and static files
            path = request.url.path
            if path in ("/docs", "/openapi.json", "/redoc") or not path.startswith("/api/"):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            token = request.query_params.get("token", "")
            if auth != f"Bearer {config.api_key}" and token != config.api_key:
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
            return await call_next(request)

    # Register API routes
    from alphafold3_mlx.api.routes import cache, jobs, results, structure, system, validation, ws
    app.include_router(system.router)
    app.include_router(jobs.router)
    app.include_router(results.router)
    app.include_router(validation.router)
    app.include_router(ws.router)
    app.include_router(cache.router)
    app.include_router(structure.router)

    # Serve frontend static files if the build exists
    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """Mount the Next.js static export as a catch-all for non-API routes."""
    if not FRONTEND_DIR.exists():
        logger.info(
            "Frontend build not found at %s. "
            "Run 'cd frontend && npm run build' to enable the web UI.",
            FRONTEND_DIR,
        )
        return

    # Mount static assets (JS, CSS, images)
    app.mount(
        "/_next",
        StaticFiles(directory=str(FRONTEND_DIR / "_next")),
        name="next-static",
    )

    # SPA fallback: serve index.html for all non-API, non-static routes
    from fastapi import Request
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_index(request: Request) -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(request: Request, path: str) -> FileResponse:
        # Don't intercept API or WebSocket paths
        if path.startswith("api/") or path.startswith("_next/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        # Resolve to prevent path traversal (e.g., "../../etc/passwd")
        resolved_root = FRONTEND_DIR.resolve()

        # Try to serve the exact file first (e.g., favicon.ico)
        file_path = (FRONTEND_DIR / path).resolve()
        if file_path.is_relative_to(resolved_root) and file_path.is_file():
            return FileResponse(str(file_path))

        # Try .html extension (Next.js exports pages as page.html)
        html_path = (FRONTEND_DIR / f"{path}.html").resolve()
        if html_path.is_relative_to(resolved_root) and html_path.is_file():
            return FileResponse(str(html_path))

        # Directory with index.html
        index_path = (FRONTEND_DIR / path / "index.html").resolve()
        if index_path.is_relative_to(resolved_root) and index_path.is_file():
            return FileResponse(str(index_path))

        # Default: serve root index.html for client-side routing
        root_index = FRONTEND_DIR / "index.html"
        if root_index.is_file():
            return FileResponse(str(root_index))

        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    logger.info("Frontend mounted from %s", FRONTEND_DIR)
