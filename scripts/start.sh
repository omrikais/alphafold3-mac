#!/usr/bin/env bash
# Production server launcher for AlphaFold 3 MLX.
# Loads configuration from ~/.alphafold3_mlx/config.env and starts the
# FastAPI server with the web UI.
#
# Usage: ./scripts/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load config with safe parser
source "$REPO_ROOT/scripts/_load_config.sh"
_load_af3_config || exit 1

# Activate venv
source "$REPO_ROOT/.venv/bin/activate"

# Build server arguments
ARGS=(--port "${AF3_PORT:-8642}")
[[ -n "${AF3_DB_DIR:-}" ]] && ARGS+=(--db-dir "$AF3_DB_DIR")
[[ -n "${AF3_WEIGHTS_DIR:-}" ]] && ARGS+=(--model-dir "$AF3_WEIGHTS_DIR")
[[ -n "${AF3_DATA_DIR:-}" ]] && ARGS+=(--data-dir "$AF3_DATA_DIR")

echo ""
echo "  AlphaFold 3 MLX"
echo "  ────────────────────────────────"
echo "  Web UI:  http://127.0.0.1:${AF3_PORT:-8642}"
echo "  API:     http://127.0.0.1:${AF3_PORT:-8642}/api"
echo ""
echo "  Press Ctrl+C to stop the server."
echo ""

exec python3 -m alphafold3_mlx.api "${ARGS[@]}"
