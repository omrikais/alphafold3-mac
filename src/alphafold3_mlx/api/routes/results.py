"""Results endpoints: confidence scores and structure files."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from alphafold3_mlx.api.models import ConfidenceResult, JobStatus, SampleConfidence

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/jobs", tags=["results"])


def _first_not_none(*values):
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def _load_confidence_json(store, job_id: str) -> dict:
    """Load and return the raw confidence_scores.json for a completed job."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status.value})",
        )

    output_dir = store.job_output_dir(job_id)
    confidence_file = output_dir / "confidence_scores.json"
    if not confidence_file.exists():
        raise HTTPException(status_code=404, detail="Confidence scores not found")

    try:
        return json.loads(confidence_file.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read confidence scores: {e}")


@router.get("/{job_id}/results", response_model=ConfidenceResult)
async def get_results(job_id: str, request: Request) -> ConfidenceResult:
    """Get confidence scores for a completed job."""
    store = request.app.state.job_store
    data = _load_confidence_json(store, job_id)

    # Read per-sample summaries from the samples dict
    samples_dict = data.get("samples", {})
    best_sample_index = data.get("best_sample_index", 0)
    is_complex = data.get("is_complex", False)

    # Build flat sample list ordered by index
    sample_list = []
    for idx in sorted(samples_dict.keys(), key=lambda k: int(k)):
        s = samples_dict[idx]
        entry: dict = {
            "ptm": s.get("ptm"),
            "iptm": s.get("iptm"),
            "mean_plddt": s.get("mean_plddt"),
            "rank": s.get("rank", int(idx) + 1),
        }
        if "restraint_satisfaction" in s:
            entry["restraint_satisfaction"] = s["restraint_satisfaction"]
        sample_list.append(entry)

    # Top-level metrics come from the best sample
    best_key = str(best_sample_index)
    best = samples_dict.get(best_key, {})

    return ConfidenceResult(
        ptm=_first_not_none(best.get("ptm"), data.get("ptm"), data.get("pTM")),
        iptm=_first_not_none(best.get("iptm"), data.get("iptm"), data.get("ipTM")),
        mean_plddt=_first_not_none(best.get("mean_plddt"), data.get("mean_plddt")),
        ranking_metric=data.get("ranking_metric"),
        num_samples=data.get("num_samples", len(sample_list)),
        samples=sample_list,
        best_sample_index=best_sample_index,
        is_complex=is_complex,
    )


@router.get("/{job_id}/results/confidence/{sample_index}", response_model=SampleConfidence)
async def get_sample_confidence(
    job_id: str, sample_index: int, request: Request,
) -> SampleConfidence:
    """Get per-residue pLDDT and PAE matrix for a specific sample."""
    store = request.app.state.job_store
    data = _load_confidence_json(store, job_id)

    samples_dict = data.get("samples", {})
    sample = samples_dict.get(str(sample_index))
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sample {sample_index} not found (available: {list(samples_dict.keys())})",
        )

    plddt = sample.get("plddt", [])
    pae = sample.get("pae", [])

    return SampleConfidence(
        sample_index=sample_index,
        ptm=sample.get("ptm"),
        iptm=sample.get("iptm"),
        mean_plddt=sample.get("mean_plddt"),
        plddt=plddt,
        pae=pae,
        num_residues=len(plddt),
        restraint_satisfaction=sample.get("restraint_satisfaction"),
    )


@router.get("/{job_id}/results/confidence-json")
async def get_confidence_json(job_id: str, request: Request) -> FileResponse:
    """Download the raw confidence_scores.json file."""
    store = request.app.state.job_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status.value})",
        )

    output_dir = store.job_output_dir(job_id)
    confidence_file = output_dir / "confidence_scores.json"
    if not confidence_file.exists():
        raise HTTPException(status_code=404, detail="Confidence scores not found")

    return FileResponse(
        path=str(confidence_file),
        media_type="application/json",
        filename=f"{job.name}_confidence_scores.json",
    )


@router.get("/{job_id}/results/download")
async def download_all(job_id: str, request: Request) -> StreamingResponse:
    """Download a ZIP archive of all output files."""
    store = request.app.state.job_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status.value})",
        )

    output_dir = store.job_output_dir(job_id)
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    # L-09: Use SpooledTemporaryFile to avoid holding large ZIPs in memory
    buf = tempfile.SpooledTemporaryFile(max_size=50 * 1024 * 1024)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(output_dir.iterdir()):
            if fpath.is_file() and not fpath.name.startswith("."):
                zf.write(fpath, fpath.name)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{job.name}_results.zip"',
        },
    )


@router.post("/{job_id}/results/open-directory")
async def open_directory(job_id: str, request: Request) -> JSONResponse:
    """Open the job output directory in Finder (macOS only)."""
    if sys.platform != "darwin":
        raise HTTPException(status_code=400, detail="Only supported on macOS")

    # M-11: Only allow from localhost
    client = request.client
    if client and client.host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="Only available from localhost")

    store = request.app.state.job_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    output_dir = store.job_output_dir(job_id)
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    subprocess.Popen(["open", str(output_dir)])
    return JSONResponse({"status": "opened"})  # No path leaked


@router.get("/{job_id}/results/structure/{rank}")
async def get_structure(job_id: str, rank: int, request: Request) -> FileResponse:
    """Download mmCIF structure file by rank."""
    store = request.app.state.job_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed (status: {job.status.value})",
        )

    output_dir = store.job_output_dir(job_id)
    structure_file = output_dir / f"structure_rank_{rank}.cif"
    if not structure_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Structure file rank {rank} not found",
        )

    return FileResponse(
        path=str(structure_file),
        media_type="chemical/x-mmcif",
        filename=f"{job.name}_rank_{rank}.cif",
    )
