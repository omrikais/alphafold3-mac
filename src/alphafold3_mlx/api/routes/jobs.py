"""Job management endpoints: create, list, get, cancel, delete."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from alphafold3_mlx.api.models import (
    JobCreated,
    JobDetail,
    JobStatus,
    JobSubmission,
    PaginatedJobs,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _count_residues_and_chains(sequences: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """Estimate residue and chain count from AlphaFold Server format sequences."""
    try:
        num_chains = 0
        num_residues = 0
        for seq_entry in sequences:
            for entity_key in ("proteinChain", "rnaSequence", "dnaSequence", "ligand", "ion"):
                if entity_key in seq_entry:
                    entity = seq_entry[entity_key]
                    count = entity.get("count", 1)
                    num_chains += count
                    if entity_key in ("proteinChain", "rnaSequence", "dnaSequence"):
                        seq = entity.get("sequence", "")
                        num_residues += len(seq) * count
                    elif entity_key == "ligand":
                        # Ligand atoms ~ rough estimate
                        num_residues += count
                    elif entity_key == "ion":
                        num_residues += count
                    break
        return num_residues or None, num_chains or None
    except Exception:
        return None, None


@router.post("", response_model=JobCreated, status_code=201)
async def create_job(submission: JobSubmission, request: Request) -> JobCreated:
    """Submit a new prediction job."""
    store = request.app.state.job_store
    queue = request.app.state.job_queue
    config = request.app.state.api_config

    num_residues, num_chains = _count_residues_and_chains(submission.sequences)

    # Build the input JSON in AlphaFold Server format
    input_json = {
        "name": submission.name,
        "modelSeeds": submission.modelSeeds,
        "sequences": submission.sequences,
    }
    if submission.dialect:
        input_json["dialect"] = submission.dialect
    if submission.version is not None:
        input_json["version"] = submission.version
    if submission.restraints is not None:
        input_json["restraints"] = submission.restraints
    if submission.guidance is not None:
        input_json["guidance"] = submission.guidance

    num_samples = submission.numSamples or config.num_samples
    diffusion_steps = submission.diffusionSteps or config.diffusion_steps
    precision = submission.precision or config.precision

    run_data_pipeline = (
        submission.runDataPipeline if submission.runDataPipeline is not None
        else config.run_data_pipeline
    )

    job_id = store.create_job(
        name=submission.name,
        input_json=input_json,
        num_residues=num_residues,
        num_chains=num_chains,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        precision=precision,
        run_data_pipeline=run_data_pipeline,
        use_cache=submission.useCache,
    )

    try:
        await queue.enqueue(job_id)
    except asyncio.QueueFull:
        # Roll back the job creation
        store.delete_job(job_id)
        raise HTTPException(status_code=429, detail="Job queue is full. Try again later.")

    return JobCreated(id=job_id)


@router.get("", response_model=PaginatedJobs)
async def list_jobs(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    search: str | None = Query(None, description="Search by name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> PaginatedJobs:
    """List jobs with optional filters and pagination."""
    store = request.app.state.job_store
    return store.list_jobs(status=status, search=search, page=page, page_size=page_size)


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str, request: Request) -> JobDetail:
    """Get full job detail."""
    store = request.app.state.job_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post("/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str, request: Request) -> dict[str, str]:
    """Cancel a pending or running job."""
    store = request.app.state.job_store
    queue = request.app.state.job_queue

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status {job.status.value}",
        )

    cancelled = queue.request_cancel(job_id)
    if not cancelled:
        # Job may have finished between the status check and the cancel attempt.
        # Use conditional update to avoid overwriting a terminal state.
        was_active = store.cancel_if_active(job_id)
        if not was_active:
            # Job already reached a terminal state (completed/failed) — don't mask it.
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} already finished and cannot be cancelled.",
            )

    return {"status": "cancelled"}


@router.delete("/{job_id}", status_code=200)
async def delete_job(job_id: str, request: Request) -> dict[str, str]:
    """Delete a completed, failed, or cancelled job and its outputs."""
    store = request.app.state.job_store

    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job with status {job.status.value}. Cancel it first.",
        )

    store.delete_job(job_id)
    return {"status": "deleted"}
