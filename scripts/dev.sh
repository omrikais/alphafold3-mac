#!/usr/bin/env bash
# Start the AlphaFold 3 MLX development servers (FastAPI + Next.js).
# Usage: ./scripts/dev.sh
#
# Kills any existing instances first, then starts:
#   - FastAPI backend on port 8642
#   - Next.js frontend on port 3001

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_PORT=8642
FRONTEND_PORT=3001
LOG_DIR="$ROOT/.dev-logs"

mkdir -p "$LOG_DIR"

# --- Kill existing servers ---
echo "Stopping existing servers..."

# FastAPI (uvicorn on API_PORT)
if pids=$(lsof -ti:"$API_PORT" 2>/dev/null); then
  echo "  Killing FastAPI on port $API_PORT (PIDs: $(echo $pids | tr '\n' ' '))"
  echo "$pids" | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# Next.js dev server (on FRONTEND_PORT)
if pids=$(lsof -ti:"$FRONTEND_PORT" 2>/dev/null); then
  echo "  Killing Next.js on port $FRONTEND_PORT (PIDs: $(echo $pids | tr '\n' ' '))"
  echo "$pids" | xargs kill -9 2>/dev/null || true
  sleep 1
fi

# Also kill any stray next dev processes
pkill -f "next dev" 2>/dev/null || true
rm -f "$ROOT/frontend/.next/dev/lock" 2>/dev/null || true

sleep 1
echo "Servers stopped."

# --- Resolve database directory (H-01) ---
# Check $AF3_DB_DIR, then standard home path
DB_DIR="${AF3_DB_DIR:-}"
if [ -z "$DB_DIR" ] && [ -d "$HOME/public_databases" ]; then
  DB_DIR="$HOME/public_databases"
fi

DB_ARGS=""
if [ -n "$DB_DIR" ]; then
  DB_ARGS="--db-dir $DB_DIR"
  echo "Using genetic databases: $DB_DIR"
else
  echo "WARNING: No genetic databases found. MSA search will be unavailable."
  echo "  Set AF3_DB_DIR or run: bash fetch_databases.sh ~/public_databases"
fi

# --- Activate virtual environment (H-09) ---
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "  Using active venv: $VIRTUAL_ENV"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
else
  echo "ERROR: No virtual environment found."
  echo "  Create one with: python -m venv .venv && pip install -e '.[mlx,api]'"
  exit 1
fi

# --- Start FastAPI backend ---
echo "Starting FastAPI backend on port $API_PORT..."
cd "$ROOT"
python -m alphafold3_mlx.api --port "$API_PORT" $DB_ARGS > "$LOG_DIR/fastapi.log" 2>&1 &
API_PID=$!
echo "  PID: $API_PID (log: $LOG_DIR/fastapi.log)"

# Wait for FastAPI to be ready
for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:$API_PORT/api/system/status" > /dev/null 2>&1; then
    echo "  FastAPI ready."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "  ERROR: FastAPI failed to start. Check $LOG_DIR/fastapi.log"
    cat "$LOG_DIR/fastapi.log"
    exit 1
  fi
  sleep 1
done

# --- Start Next.js frontend ---
echo "Starting Next.js frontend on port $FRONTEND_PORT..."

# H-10: Install frontend dependencies if missing
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "  Installing frontend dependencies..."
  npm ci --prefix "$ROOT/frontend"
fi

# L-06: Set NEXT_PUBLIC_API_URL in .env.development.local without clobbering other vars
ENV_LOCAL="$ROOT/frontend/.env.development.local"
API_URL_LINE="NEXT_PUBLIC_API_URL=http://localhost:$API_PORT"
if [ -f "$ENV_LOCAL" ] && grep -q '^NEXT_PUBLIC_API_URL=' "$ENV_LOCAL"; then
  sed -i '' "s|^NEXT_PUBLIC_API_URL=.*|$API_URL_LINE|" "$ENV_LOCAL"
else
  echo "$API_URL_LINE" >> "$ENV_LOCAL"
fi

cd "$ROOT/frontend"
npx next dev --port "$FRONTEND_PORT" --hostname 127.0.0.1 > "$LOG_DIR/nextjs.log" 2>&1 &
NEXT_PID=$!
echo "  PID: $NEXT_PID (log: $LOG_DIR/nextjs.log)"

# Wait for Next.js to be ready
for i in $(seq 1 15); do
  if curl -s "http://127.0.0.1:$FRONTEND_PORT/" > /dev/null 2>&1; then
    echo "  Next.js ready."
    break
  fi
  if ! kill -0 "$NEXT_PID" 2>/dev/null; then
    echo "  ERROR: Next.js failed to start. Check $LOG_DIR/nextjs.log"
    cat "$LOG_DIR/nextjs.log"
    exit 1
  fi
  sleep 1
done

echo ""
echo "=== Dev servers running ==="
echo "  Frontend:  http://127.0.0.1:$FRONTEND_PORT"
echo "  API:       http://127.0.0.1:$API_PORT"
echo "  API docs:  http://127.0.0.1:$API_PORT/docs"
echo ""
echo "  Logs: $LOG_DIR/"
echo "  Stop: ./scripts/dev-stop.sh"
echo ""

# Write PIDs for the stop script
echo "$API_PID" > "$LOG_DIR/fastapi.pid"
echo "$NEXT_PID" > "$LOG_DIR/nextjs.pid"
