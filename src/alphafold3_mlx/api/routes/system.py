"""System status endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from alphafold3_mlx.api.models import SystemStatus

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status", response_model=SystemStatus)
async def get_system_status(request: Request) -> SystemStatus:
    """Return system hardware info, model status, and queue state."""
    model_manager = request.app.state.model_manager
    job_queue = request.app.state.job_queue

    # Get platform info (safe to call even if model not loaded)
    chip_family = "Unknown"
    memory_gb = 0
    supports_bfloat16 = False
    try:
        from alphafold3_mlx.weights.platform import get_platform_info
        info = get_platform_info()
        chip_family = info.chip_family
        memory_gb = info.memory_gb
        supports_bfloat16 = info.supports_bfloat16
    except Exception:
        pass

    config = request.app.state.api_config

    return SystemStatus(
        model_loaded=model_manager.is_loaded,
        model_loading=model_manager.is_loading,
        chip_family=chip_family,
        memory_gb=memory_gb,
        supports_bfloat16=supports_bfloat16,
        queue_size=job_queue.queue_size,
        active_job_id=job_queue.active_job_id,
        run_data_pipeline=config.run_data_pipeline,
    )
