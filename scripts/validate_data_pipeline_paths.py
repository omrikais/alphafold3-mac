#!/usr/bin/env python3
"""Validate AF3 data pipeline configuration for MLX runs.

This checks that:
- Required HMMER binaries are resolvable (jackhmmer/hmmsearch/hmmbuild).
- Required database paths are configured and exist.

Configuration sources:
- CLI: --db_dir (passed to build_af3_data_pipeline_config)
- Env: AF3_DB_DIR plus per-path overrides like AF3_UNIREF90_DB, AF3_PDB_MMCIF_DIR

Usage:
  PYTHONPATH=src python scripts/validate_data_pipeline_paths.py
  PYTHONPATH=src python scripts/validate_data_pipeline_paths.py --db_dir /path/to/dbs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_dir",
        type=Path,
        default=None,
        help="Directory containing AF3 databases (overrides AF3_DB_DIR).",
    )
    parser.add_argument(
        "--rna",
        action="store_true",
        help="Also require RNA databases/tools (nhmmer/hmmalign + RNA DBs).",
    )
    args = parser.parse_args()

    from alphafold3_mlx.data.validation import (
        DataPipelineNotConfiguredError,
        build_af3_data_pipeline_config,
    )

    try:
        cfg, resolved, _tried = build_af3_data_pipeline_config(
            db_dir=args.db_dir,
            require_rna=args.rna,
        )
    except DataPipelineNotConfiguredError as e:
        print(str(e), file=sys.stderr)
        return 1

    print("== AF3 Data Pipeline Config ==")
    print(f"max_template_date: {cfg.max_template_date}")
    for k in sorted(resolved.keys()):
        print(f"{k}: {resolved[k]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

