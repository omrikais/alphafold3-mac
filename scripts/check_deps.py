#!/usr/bin/env python3
"""Check external dependencies needed for AF3-style data pipeline runs.

This verifies HMMER binaries used by the AlphaFold 3 data pipeline:
  - jackhmmer (protein MSA search)
  - hmmsearch + hmmbuild (template search)
Optionally:
  - nhmmer + hmmalign (RNA MSA search)

Resolution order for each binary:
  1) Explicit env var override (e.g. AF3_JACKHMMER)
  2) PATH
  3) Common macOS prefixes (/opt/homebrew/bin, /usr/local/bin, ~/hmmer/bin)

Usage:
  PYTHONPATH=src python scripts/check_deps.py
  PYTHONPATH=src python scripts/check_deps.py --rna
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_binary(binary_name: str, env_var: str | None) -> tuple[Path | None, list[str]]:
    tried: list[str] = []
    if env_var:
        tried.append(f"env:{env_var}")
        override = os.environ.get(env_var)
        if override:
            candidate = Path(os.path.expanduser(override))
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate, tried

    tried.append("PATH")
    found = shutil.which(binary_name)
    if found:
        return Path(found), tried

    extra_dirs = [
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path.home() / "hmmer" / "bin",
        Path.home() / ".alphafold3_mlx" / "hmmer" / "bin",  # install.sh default
    ]
    for d in extra_dirs:
        candidate = d / binary_name
        tried.append(str(candidate))
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate, tried

    return None, tried


def _check_seq_limit(jackhmmer_path: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(jackhmmer_path), "-h", "--seq_limit", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return proc.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rna",
        action="store_true",
        help="Also require RNA tools (nhmmer/hmmalign).",
    )
    args = parser.parse_args()

    requirements = [
        ("jackhmmer", "AF3_JACKHMMER"),
        ("hmmsearch", "AF3_HMMSEARCH"),
        ("hmmbuild", "AF3_HMMBUILD"),
    ]
    if args.rna:
        requirements.extend(
            [
                ("nhmmer", "AF3_NHMMER"),
                ("hmmalign", "AF3_HMMALIGN"),
            ]
        )

    ok = True
    resolved: dict[str, Path] = {}

    print("== External Binary Check ==")
    for name, env_var in requirements:
        path, tried = _resolve_binary(name, env_var)
        if path is None:
            ok = False
            print(f"XX {name}: not found (tried: {', '.join(tried)})")
        else:
            resolved[name] = path
            print(f"OK {name}: {path}")

    if "jackhmmer" in resolved:
        seq_limit = _check_seq_limit(resolved["jackhmmer"])
        if seq_limit:
            print("OK jackhmmer: --seq_limit supported")
        else:
            print("!! jackhmmer: --seq_limit NOT supported (optional; build_hmmer_macos.sh enables it)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

