"""Cache management endpoints: check MSA cache status for given sequences."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cache", tags=["cache"])


class CacheCheckRequest(BaseModel):
    """Sequences to check against the MSA cache."""

    sequences: list[dict[str, Any]] = Field(min_length=1)


class CacheCheckResponse(BaseModel):
    """Cache check result."""

    cached: bool = False
    cache_key: str | None = None
    cached_at: str | None = None
    size_mb: float | None = None


def _sequences_to_cache_key(sequences: list[dict[str, Any]]) -> str:
    """Compute the same cache key that MSACache uses, from raw submission sequences.

    Mirrors the logic in msa_cache.py:cache_key() but works on the
    raw JSON sequences list (before parsing into folding_input.Input).

    Key format: sorted list of (chain_type, sequence, metadata) 3-tuples
    where chain_type matches ``type(chain).__name__`` from folding_input
    (ProteinChain, RnaChain, DnaChain, Ligand).
    """
    entries: list[tuple[str, str, str]] = []
    for seq_entry in sequences:
        for entity_key, chain_type in (
            ("proteinChain", "ProteinChain"),
            ("rnaSequence", "RnaChain"),
            ("dnaSequence", "DnaChain"),
            ("ligand", "Ligand"),
            ("ion", "Ligand"),
        ):
            if entity_key in seq_entry:
                entity = seq_entry[entity_key]
                count = entity.get("count", 1)
                if entity_key == "proteinChain":
                    seq = entity.get("sequence", "")
                    mods = entity.get("modifications", [])
                    ptms = sorted(
                        (m["ptmType"], m["ptmPosition"]) for m in mods
                    )
                    meta = json.dumps(ptms) if ptms else ""
                elif entity_key in ("rnaSequence", "dnaSequence"):
                    seq = entity.get("sequence", "")
                    mods = entity.get("modifications", [])
                    modifications = sorted(
                        (m["modificationType"], m["basePosition"])
                        for m in mods
                    )
                    meta = json.dumps(modifications) if modifications else ""
                elif entity_key == "ligand":
                    # AF Server: {"ligand": "CCD_HEM"}
                    # mmCIF: {"ccdCodes": ["HEM"]} or {"smiles": "..."}
                    if "ligand" in entity:
                        seq = entity["ligand"].removeprefix("CCD_")
                    elif "ccdCodes" in entity:
                        seq = ",".join(entity["ccdCodes"])
                    elif "smiles" in entity:
                        seq = entity["smiles"]
                    else:
                        seq = ""
                    meta = ""
                elif entity_key == "ion":
                    seq = entity.get("ion", "")
                    meta = ""
                else:
                    seq = ""
                    meta = ""
                for _ in range(count):
                    entries.append((chain_type, seq, meta))
                break

    entries.sort()
    canonical = json.dumps(entries, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@router.post("/check", response_model=CacheCheckResponse)
async def check_cache(body: CacheCheckRequest, request: Request) -> CacheCheckResponse:
    """Check whether MSA cache exists for the given sequences."""
    config = request.app.state.api_config

    cache_dir = config.data_dir / "msa_cache"
    if not cache_dir.exists():
        return CacheCheckResponse(cached=False)

    key = _sequences_to_cache_key(body.sequences)
    pkl_path = cache_dir / f"{key}.pkl"

    if not pkl_path.exists():
        return CacheCheckResponse(cached=False, cache_key=key)

    stat = pkl_path.stat()
    cached_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    size_mb = round(stat.st_size / (1024 * 1024), 1)

    return CacheCheckResponse(
        cached=True,
        cache_key=key,
        cached_at=cached_at,
        size_mb=size_mb,
    )
