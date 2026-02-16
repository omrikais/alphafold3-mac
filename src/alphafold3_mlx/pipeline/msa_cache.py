"""Filesystem cache for processed MSA/template data.

Caches the output of DataPipeline.process() keyed by a hash of the
input sequences.  Re-running the same sequences (even with a different
seed) will reuse the cached MSA/template data, skipping the expensive
HMMER search.

Storage: ``{cache_dir}/{key}.pkl`` where *key* is a truncated SHA-256
of the canonical sequence representation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MSACache:
    """Filesystem-backed MSA/template cache.

    Parameters:
        cache_dir: Directory to store cached pickle files.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cache_key(self, af3_input: Any) -> str:
        """Compute a deterministic cache key from chain types, sequences, and metadata.

        The key is the first 16 hex chars of SHA-256 over a canonical
        JSON representation of ``(chain_type, sequence, metadata)`` tuples
        sorted for order-independence.  Metadata includes PTMs (protein)
        and modifications (RNA/DNA) so that inputs differing only in
        residue modifications do not collide.
        """
        from alphafold3.common import folding_input

        # L-04: Include package version so cache invalidates on upgrades
        from alphafold3_mlx import __version__

        entries: list[tuple[str, str, str]] = []
        for chain in af3_input.chains:
            chain_type = type(chain).__name__
            if isinstance(chain, folding_input.ProteinChain):
                seq = chain.sequence
                meta = json.dumps(sorted(chain.ptms)) if chain.ptms else ""
            elif isinstance(chain, folding_input.RnaChain):
                seq = chain.sequence
                meta = json.dumps(sorted(chain.modifications)) if chain.modifications else ""
            elif isinstance(chain, folding_input.DnaChain):
                seq = chain.sequence
                mods = chain.modifications()
                meta = json.dumps(sorted(mods)) if mods else ""
            elif isinstance(chain, folding_input.Ligand):
                seq = ",".join(chain.ccd_ids) if chain.ccd_ids else (chain.smiles or "")
                meta = ""
            else:
                seq = str(chain)
                meta = ""
            entries.append((chain_type, seq, meta))

        # Sort for determinism regardless of chain order
        entries.sort()
        canonical = json.dumps(entries, sort_keys=True)
        h = hashlib.sha256()
        h.update(__version__.encode())
        h.update(canonical.encode())
        digest = h.hexdigest()[:16]
        return digest

    def get(self, af3_input: Any) -> Any | None:
        """Look up cached processed input.

        Returns:
            The cached ``folding_input.Input`` after pipeline processing,
            or ``None`` on cache miss.
        """
        key = self.cache_key(af3_input)
        path = self._cache_dir / f"{key}.pkl"
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            logger.info("MSA cache hit for key %s", key)
            return cached
        except Exception as exc:
            logger.warning("MSA cache read failed for %s: %s", key, exc)
            return None

    def put(self, af3_input: Any, processed: Any) -> None:
        """Store processed input in the cache.

        Parameters:
            af3_input: The *original* (pre-pipeline) input, used to derive the key.
            processed: The input *after* ``DataPipeline.process()``.
        """
        key = self.cache_key(af3_input)
        path = self._cache_dir / f"{key}.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("MSA cache stored for key %s", key)
        except Exception as exc:
            logger.warning("MSA cache write failed for %s: %s", key, exc)

    def clear(self) -> None:
        """Remove all cached entries."""
        for p in self._cache_dir.glob("*.pkl"):
            try:
                p.unlink()
            except OSError:
                pass
        logger.info("MSA cache cleared")
