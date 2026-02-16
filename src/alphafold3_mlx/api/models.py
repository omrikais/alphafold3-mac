"""Pydantic request/response models for the API."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, enum.Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EntityType(str, enum.Enum):
    """Molecular entity types supported by the form."""

    PROTEIN = "proteinChain"
    RNA = "rnaSequence"
    DNA = "dnaSequence"
    LIGAND = "ligand"
    ION = "ion"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class JobSubmission(BaseModel):
    """Job submission request body (AlphaFold Server JSON format).

    This is the JSON format accepted by the /api/jobs endpoint.
    It mirrors the AlphaFold Server fold job format so the frontend
    can serialize the entity form directly into this shape.
    """

    name: str = "unnamed"
    modelSeeds: list[int] = Field(default_factory=lambda: [42], min_length=1)
    sequences: list[dict[str, Any]] = Field(min_length=1)
    dialect: str | None = "alphafoldserver"
    version: int | None = 1
    # Optional overrides
    numSamples: int | None = None
    diffusionSteps: int | None = Field(default=None, ge=1)
    precision: Literal["float32", "float16", "bfloat16"] | None = None
    runDataPipeline: bool | None = None
    useCache: bool = True
    # Restraint-guided docking (optional)
    restraints: dict[str, Any] | None = None
    guidance: dict[str, Any] | None = None


class ValidationRequest(BaseModel):
    """Input validation request (no job created)."""

    sequences: list[dict[str, Any]] = Field(min_length=1)
    name: str = "validation"
    modelSeeds: list[int] = Field(default_factory=lambda: [42], min_length=1)
    restraints: dict[str, Any] | None = None
    guidance: dict[str, Any] | None = None
    diffusionSteps: int | None = Field(default=None, ge=1)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JobSummary(BaseModel):
    """Job list item."""

    id: str
    name: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    num_residues: int | None = None
    num_chains: int | None = None
    error_message: str | None = None
    progress: float = 0.0


class JobDetail(BaseModel):
    """Full job detail."""

    id: str
    name: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    input_json: dict[str, Any]
    num_residues: int | None = None
    num_chains: int | None = None
    num_samples: int = 5
    diffusion_steps: int = 200
    precision: Literal["float32", "float16", "bfloat16"] | None = None
    run_data_pipeline: bool = True
    use_cache: bool = True
    error_message: str | None = None
    progress: float = 0.0
    current_stage: str | None = None


class JobCreated(BaseModel):
    """Response after job submission."""

    id: str
    status: JobStatus = JobStatus.PENDING


class ValidationResult(BaseModel):
    """Input validation result."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    estimated_memory_gb: float | None = None
    num_residues: int | None = None
    num_chains: int | None = None


class SystemStatus(BaseModel):
    """System status information."""

    model_loaded: bool = False
    model_loading: bool = False
    chip_family: str = "Unknown"
    memory_gb: int = 0
    supports_bfloat16: bool = False
    queue_size: int = 0
    active_job_id: str | None = None
    version: str = "3.0.1"
    run_data_pipeline: bool = False


class PaginatedJobs(BaseModel):
    """Paginated job list response."""

    jobs: list[JobSummary]
    total: int
    page: int
    page_size: int


class SampleConfidence(BaseModel):
    """Per-sample confidence detail (pLDDT array + PAE matrix)."""

    sample_index: int
    ptm: float | None = None
    iptm: float | None = None
    mean_plddt: float | None = None
    plddt: list[float] = Field(default_factory=list)
    pae: list[list[float]] = Field(default_factory=list)
    num_residues: int = 0
    restraint_satisfaction: dict[str, Any] | None = None


class ConfidenceResult(BaseModel):
    """Confidence scores for a completed job."""

    ptm: float | None = None
    iptm: float | None = None
    mean_plddt: float | None = None
    ranking_metric: str | None = None
    num_samples: int = 0
    samples: list[dict[str, Any]] = Field(default_factory=list)
    best_sample_index: int = 0
    is_complex: bool = False


# ---------------------------------------------------------------------------
# WebSocket message models
# ---------------------------------------------------------------------------


class StructureParseResult(BaseModel):
    """Result of parsing an uploaded PDB/mmCIF file or RCSB fetch."""

    name: str
    sequences: list[dict[str, Any]]
    dialect: str = "alphafoldserver"
    version: int = 1
    source: str  # "upload" or "rcsb"
    pdb_id: str | None = None
    num_chains: int
    num_residues: int
    warnings: list[str] = Field(default_factory=list)


class WSMessage(BaseModel):
    """WebSocket progress message."""

    type: str  # stage_change, progress, completed, failed, cancelled
    stage: str | None = None
    percent_complete: float = 0.0
    recycling_iteration: int | None = None
    recycling_total: int | None = None
    diffusion_step: int | None = None
    diffusion_total: int | None = None
    error: str | None = None
