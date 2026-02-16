"""Structure file upload and RCSB PDB fetch endpoints."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from alphafold3_mlx.api.models import StructureParseResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/structure", tags=["structure"])

_PDB_ID_RE = re.compile(r"^[a-zA-Z0-9]{4}$")
_ALLOWED_EXTENSIONS = {".pdb", ".cif", ".mmcif"}


def _ext_to_fmt(filename: str) -> str:
    """Map a filename extension to a format string."""
    lower = filename.lower()
    if lower.endswith(".pdb"):
        return "pdb"
    if lower.endswith(".cif") or lower.endswith(".mmcif"):
        return "cif"
    raise ValueError(f"Unsupported file type: {filename}")


@router.post("/parse", response_model=StructureParseResult)
async def parse_structure_file(file: UploadFile) -> StructureParseResult:
    """Parse an uploaded PDB or mmCIF file into AlphaFold Server JSON format."""
    filename = file.filename or ""

    # Validate extension
    if not any(filename.lower().endswith(ext) for ext in _ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type. Accepted: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    try:
        fmt = _ext_to_fmt(filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    content = await file.read()
    content_str = content.decode("utf-8", errors="replace")

    try:
        from alphafold3_mlx.pipeline.structure_parser import parse_structure

        result = await run_in_threadpool(parse_structure, content_str, fmt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to parse structure file: %s", filename)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse structure file: {e}",
        ) from e

    return StructureParseResult(**result)


@router.get("/fetch/{pdb_id}", response_model=StructureParseResult)
async def fetch_pdb_structure(pdb_id: str) -> StructureParseResult:
    """Fetch a structure from RCSB by PDB ID and parse it."""
    if not _PDB_ID_RE.match(pdb_id):
        raise HTTPException(
            status_code=422,
            detail="PDB ID must be exactly 4 alphanumeric characters",
        )

    try:
        from alphafold3_mlx.pipeline.structure_parser import fetch_and_parse

        result = await run_in_threadpool(fetch_and_parse, pdb_id)
    except Exception as e:
        from alphafold3_mlx.pipeline.structure_parser import (
            PdbNotFoundError,
            RcsbTimeoutError,
            RcsbUnavailableError,
        )

        if isinstance(e, PdbNotFoundError):
            raise HTTPException(
                status_code=404,
                detail=f"PDB ID '{pdb_id.upper()}' not found in RCSB",
            ) from e
        if isinstance(e, RcsbTimeoutError):
            raise HTTPException(
                status_code=504,
                detail="RCSB request timed out after 30s",
            ) from e
        if isinstance(e, RcsbUnavailableError):
            raise HTTPException(
                status_code=502,
                detail=f"RCSB unavailable: {e.detail}",
            ) from e
        if isinstance(e, ValueError):
            raise HTTPException(status_code=422, detail=str(e)) from e
        logger.exception("Failed to fetch PDB %s", pdb_id)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch PDB: {e}",
        ) from e

    return StructureParseResult(**result)
