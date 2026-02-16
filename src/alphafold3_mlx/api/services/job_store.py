"""SQLite-backed job persistence.

Jobs are stored in {data_dir}/jobs.db. Job outputs go into {data_dir}/jobs/{job_id}/.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from alphafold3_mlx.api.models import (
    JobDetail,
    JobStatus,
    JobSummary,
    PaginatedJobs,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    num_residues INTEGER,
    num_chains INTEGER,
    num_samples INTEGER DEFAULT 5,
    diffusion_steps INTEGER DEFAULT 200,
    precision TEXT,
    run_data_pipeline INTEGER DEFAULT 1,
    use_cache INTEGER DEFAULT 1,
    error_message TEXT,
    progress REAL DEFAULT 0.0,
    current_stage TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


class JobStore:
    """SQLite-backed job storage."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._db_path = data_dir / "jobs.db"
        self._jobs_dir = data_dir / "jobs"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # Migration: add columns for existing DBs
            for col, default in [
                ("run_data_pipeline", "INTEGER DEFAULT 1"),
                ("use_cache", "INTEGER DEFAULT 1"),
                ("precision", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {default}")
                except sqlite3.OperationalError:
                    pass  # column already exists

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def job_output_dir(self, job_id: str) -> Path:
        """Get the output directory for a job."""
        d = self._jobs_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -----------------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------------

    def create_job(
        self,
        name: str,
        input_json: dict[str, Any],
        num_residues: int | None = None,
        num_chains: int | None = None,
        num_samples: int = 5,
        diffusion_steps: int = 200,
        precision: Literal["float32", "float16", "bfloat16"] | None = None,
        run_data_pipeline: bool = True,
        use_cache: bool = True,
    ) -> str:
        """Create a new job and return its ID."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, name, status, created_at, updated_at, input_json,
                    num_residues, num_chains, num_samples, diffusion_steps,
                    precision, run_data_pipeline, use_cache)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, name, JobStatus.PENDING.value, now, now,
                    json.dumps(input_json), num_residues, num_chains,
                    num_samples, diffusion_steps, precision,
                    int(run_data_pipeline), int(use_cache),
                ),
            )
        return job_id

    def get_job(self, job_id: str) -> JobDetail | None:
        """Get full job detail."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_detail(row)

    def get_job_summary(self, job_id: str) -> JobSummary | None:
        """Get job summary."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_summary(row)

    def list_jobs(
        self,
        status: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedJobs:
        """List jobs with optional filters and pagination."""
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if search:
            conditions.append("name LIKE ?")
            params.append(f"%{search}%")

        where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM jobs{where}", params
            ).fetchone()[0]

            offset = (page - 1) * page_size
            rows = conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [page_size, offset],
            ).fetchall()

        return PaginatedJobs(
            jobs=[self._row_to_summary(r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
        )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
        progress: float | None = None,
        current_stage: str | None = None,
    ) -> None:
        """Update job status and optional fields."""
        now = datetime.now(timezone.utc).isoformat()
        sets = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, now]

        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if progress is not None:
            sets.append("progress = ?")
            params.append(progress)
        if current_stage is not None:
            sets.append("current_stage = ?")
            params.append(current_stage)

        params.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                params,
            )

    def cancel_if_active(self, job_id: str) -> bool:
        """Atomically set status to CANCELLED only if currently PENDING or RUNNING.

        Returns True if the update was applied (job was still active).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ? AND status IN (?, ?)",
                (JobStatus.CANCELLED.value, now, job_id, JobStatus.PENDING.value, JobStatus.RUNNING.value),
            )
            return cursor.rowcount > 0

    def update_progress(self, job_id: str, progress: float, stage: str | None = None) -> None:
        """Update just the progress percentage and stage."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ?, current_stage = ?, updated_at = ? WHERE id = ?",
                (progress, stage, now, job_id),
            )

    def get_pending_job_ids(self) -> list[str]:
        """Return IDs of all PENDING jobs, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at ASC",
                (JobStatus.PENDING.value,),
            ).fetchall()
        return [r["id"] for r in rows]

    def recover_stale_running(self) -> int:
        """Mark any RUNNING jobs as FAILED.

        Called at startup to clean up jobs left over from an unclean shutdown
        or process crash.  Returns the number of jobs recovered.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? "
                "WHERE status = ?",
                (
                    JobStatus.FAILED.value,
                    "Server restarted while job was running",
                    now,
                    JobStatus.RUNNING.value,
                ),
            )
            return cursor.rowcount

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its output files. Returns True if found."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        # Clean up output directory
        output_dir = self._jobs_dir / job_id
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        return True

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> JobSummary:
        return JobSummary(
            id=row["id"],
            name=row["name"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            num_residues=row["num_residues"],
            num_chains=row["num_chains"],
            error_message=row["error_message"],
            progress=row["progress"] or 0.0,
        )

    @staticmethod
    def _row_to_detail(row: sqlite3.Row) -> JobDetail:
        return JobDetail(
            id=row["id"],
            name=row["name"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            input_json=json.loads(row["input_json"]),
            num_residues=row["num_residues"],
            num_chains=row["num_chains"],
            num_samples=row["num_samples"] or 5,
            diffusion_steps=row["diffusion_steps"] or 200,
            precision=row["precision"],
            run_data_pipeline=bool(row["run_data_pipeline"]) if row["run_data_pipeline"] is not None else True,
            use_cache=bool(row["use_cache"]) if row["use_cache"] is not None else True,
            error_message=row["error_message"],
            progress=row["progress"] or 0.0,
            current_stage=row["current_stage"],
        )
