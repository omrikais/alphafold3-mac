#!/usr/bin/env bash
# CLI wrapper for AlphaFold 3 MLX.
# Loads configuration from ~/.alphafold3_mlx/config.env (sets AF3_WEIGHTS_DIR,
# AF3_JACKHMMER, etc.) and forwards all arguments to the alphafold3-mlx CLI.
#
# Usage: ./scripts/alphafold3.sh --input input.json --output_dir output/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load config with safe parser (sets AF3_WEIGHTS_DIR, AF3_JACKHMMER, etc.)
source "$REPO_ROOT/scripts/_load_config.sh"
_load_af3_config || exit 1

# Activate venv
source "$REPO_ROOT/.venv/bin/activate"

# Forward all arguments to the CLI entry point
exec alphafold3-mlx "$@"
