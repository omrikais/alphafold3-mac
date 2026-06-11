"""System status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from alphafold3_mlx.api.models import SystemStatus

router = APIRouter(prefix="/api/system", tags=["system"])

# Platform info is static for the lifetime of the process — cache on first call.
_platform_cache: dict | None = None


def _get_platform_info() -> dict:
    """Return cached platform info dict (chip_family, memory_gb, supports_bfloat16)."""
    global _platform_cache
    if _platform_cache is None:
        try:
            from alphafold3_mlx.weights.platform import get_platform_info
            info = get_platform_info()
            _platform_cache = {
                "chip_family": info.chip_family,
                "memory_gb": info.memory_gb,
                "supports_bfloat16": info.supports_bfloat16,
            }
        except Exception:
            _platform_cache = {"chip_family": "Unknown", "memory_gb": 0, "supports_bfloat16": False}
    return _platform_cache


@router.get("/status", response_model=SystemStatus)
async def get_system_status(request: Request) -> SystemStatus:
    """Return system hardware info, model status, and queue state."""
    model_manager = request.app.state.model_manager
    job_queue = request.app.state.job_queue
    config = request.app.state.api_config
    platform = _get_platform_info()

    return SystemStatus(
        model_loaded=model_manager.is_loaded,
        model_loading=model_manager.is_loading,
        chip_family=platform["chip_family"],
        memory_gb=platform["memory_gb"],
        supports_bfloat16=platform["supports_bfloat16"],
        queue_size=job_queue.queue_size,
        active_job_id=job_queue.active_job_id,
        run_data_pipeline=config.run_data_pipeline,
    )
