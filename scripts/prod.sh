#!/usr/bin/env bash
# Restart production servers (FastAPI + static frontend) in detached mode.
# Usage: ./scripts/prod.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT/.prod-logs"

FASTAPI_PID_FILE="$LOG_DIR/fastapi.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"
BACKEND_BUILD_STAMP="$LOG_DIR/backend.build.stamp"
FRONTEND_BUILD_STAMP="$LOG_DIR/frontend.build.stamp"
FRONTEND_API_URL_STAMP="$LOG_DIR/frontend.api_url.stamp"
FRONTEND_DEPS_STAMP="$LOG_DIR/frontend.deps.stamp"

mkdir -p "$LOG_DIR"

CONFIG_FILE="${AF3_CONFIG_FILE:-$HOME/.alphafold3_mlx/config.env}"
if [ -f "$ROOT/scripts/_load_config.sh" ] && [ -f "$CONFIG_FILE" ]; then
  source "$ROOT/scripts/_load_config.sh"
  _load_af3_config "$CONFIG_FILE" || exit 1
else
  echo "Config file not found at $CONFIG_FILE. Using env/default values."
fi

API_PORT="${AF3_PORT:-8642}"
FRONTEND_PORT="${AF3_FRONTEND_PORT:-3001}"
FRONTEND_API_URL="${AF3_FRONTEND_API_URL:-http://127.0.0.1:$API_PORT}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Using active venv: $VIRTUAL_ENV"
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
else
  echo "ERROR: No virtual environment found at $ROOT/.venv" >&2
  exit 1
fi

# Check that required Python packages from [api] extras are installed.
# This catches the case where pyproject.toml added new deps but the venv
# wasn't updated (e.g. after a git pull).
# Uses simple pairs (module:label) to stay compatible with macOS bash 3.2.
check_api_deps() {
  local missing=""
  local pair mod label
  for pair in \
    "fastapi:fastapi" \
    "uvicorn:uvicorn" \
    "websockets:websockets" \
    "aiosqlite:aiosqlite" \
    "pydantic:pydantic" \
    "gemmi:gemmi" \
    "multipart:python-multipart" \
  ; do
    mod="${pair%%:*}"
    label="${pair##*:}"
    if ! python3 -c "import $mod" 2>/dev/null; then
      missing="$missing $label"
    fi
  done

  if [ -n "$missing" ]; then
    echo "  Missing API dependencies:$missing"
    # Ensure pip is available inside the venv (some venvs are created
    # without pip, so the bare `pip` command resolves to /usr/bin/pip
    # which installs to user site-packages instead of the venv).
    if ! python3 -m pip --version >/dev/null 2>&1; then
      echo "  Bootstrapping pip in venv..."
      python3 -m ensurepip --default-pip 2>/dev/null || {
        echo "ERROR: Could not bootstrap pip in venv." >&2
        exit 1
      }
    fi
    echo "  Installing:$missing"
    # Install missing packages directly rather than via editable install,
    # which can fail on unrelated build deps (e.g. rdkit, C extensions).
    # shellcheck disable=SC2086
    python3 -m pip install $missing || {
      echo "ERROR: Failed to install API dependencies." >&2
      exit 1
    }
    echo "  Dependencies installed."
  fi
}

echo "Checking API dependencies..."
check_api_deps

read_stamp() {
  local stamp_file="$1"
  local stamp_value
  if [ -f "$stamp_file" ]; then
    stamp_value="$(tr -dc '0-9' < "$stamp_file")"
    if [ -n "$stamp_value" ]; then
      echo "$stamp_value"
    else
      echo "0"
    fi
  else
    echo "0"
  fi
}

latest_mtime_in_set() {
  local latest=0
  local file_path
  local file_mtime
  while IFS= read -r -d '' file_path; do
    file_mtime=$(stat -f '%m' "$file_path" 2>/dev/null || echo "0")
    if [ "$file_mtime" -gt "$latest" ]; then
      latest="$file_mtime"
    fi
  done
  echo "$latest"
}

backend_source_mtime() {
  local latest=0
  local from_find
  local file_path
  local file_mtime

  from_find=$(latest_mtime_in_set < <(
    find "$ROOT/src" -type f \( -name "*.py" -o -name "*.toml" \) -print0
  ))
  if [ "$from_find" -gt "$latest" ]; then
    latest="$from_find"
  fi

  for file_path in "$ROOT/pyproject.toml" "$ROOT/uv.lock"; do
    if [ -f "$file_path" ]; then
      file_mtime=$(stat -f '%m' "$file_path" 2>/dev/null || echo "0")
      if [ "$file_mtime" -gt "$latest" ]; then
        latest="$file_mtime"
      fi
    fi
  done

  echo "$latest"
}

frontend_source_mtime() {
  latest_mtime_in_set < <(
    find "$ROOT/frontend" \
      \( \
        -path "$ROOT/frontend/node_modules" -o \
        -path "$ROOT/frontend/.next" -o \
        -path "$ROOT/frontend/out" \
      \) -prune -o -type f -print0
  )
}

stop_pid() {
  local service_name="$1"
  local pid="$2"

  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  echo "  Stopping $service_name (PID $pid)"
  kill "$pid" 2>/dev/null || true

  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  Force-killing $service_name (PID $pid)"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_from_pid_file() {
  local service_name="$1"
  local pid_file="$2"

  if [ ! -f "$pid_file" ]; then
    return 0
  fi

  local pid
  pid="$(tr -dc '0-9' < "$pid_file")"
  if [ -n "$pid" ]; then
    stop_pid "$service_name" "$pid"
  fi
  rm -f "$pid_file"
}

echo "Stopping existing production servers..."
stop_from_pid_file "FastAPI" "$FASTAPI_PID_FILE"
stop_from_pid_file "frontend static server" "$FRONTEND_PID_FILE"

for port in "$API_PORT" "$FRONTEND_PORT"; do
  if pids=$(lsof -ti:"$port" 2>/dev/null); then
    echo "  Killing remaining processes on port $port (PIDs: $(echo "$pids" | tr '\n' ' '))"
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
done

echo "Checking whether rebuild is required..."

BACKEND_SRC_MTIME="$(backend_source_mtime)"
BACKEND_LAST_BUILD="$(read_stamp "$BACKEND_BUILD_STAMP")"
if [ "$BACKEND_SRC_MTIME" -gt "$BACKEND_LAST_BUILD" ]; then
  echo "  Backend changes detected. Building Python bytecode..."
  python3 -m compileall -q "$ROOT/src"
  echo "$BACKEND_SRC_MTIME" > "$BACKEND_BUILD_STAMP"
else
  echo "  Backend unchanged; skipping build."
fi

FRONTEND_SRC_MTIME="$(frontend_source_mtime)"
FRONTEND_LAST_BUILD="$(read_stamp "$FRONTEND_BUILD_STAMP")"
FRONTEND_LAST_API_URL=""
if [ -f "$FRONTEND_API_URL_STAMP" ]; then
  FRONTEND_LAST_API_URL="$(cat "$FRONTEND_API_URL_STAMP")"
fi

NEED_FRONTEND_BUILD=false
if [ ! -f "$ROOT/frontend/out/index.html" ]; then
  NEED_FRONTEND_BUILD=true
elif [ "$FRONTEND_SRC_MTIME" -gt "$FRONTEND_LAST_BUILD" ]; then
  NEED_FRONTEND_BUILD=true
elif [ "$FRONTEND_LAST_API_URL" != "$FRONTEND_API_URL" ]; then
  NEED_FRONTEND_BUILD=true
fi

if [ "$NEED_FRONTEND_BUILD" = true ]; then
  LOCK_MTIME=0
  if [ -f "$ROOT/frontend/package-lock.json" ]; then
    LOCK_MTIME="$(stat -f '%m' "$ROOT/frontend/package-lock.json" 2>/dev/null || echo "0")"
  fi
  LAST_DEPS_STAMP="$(read_stamp "$FRONTEND_DEPS_STAMP")"

  if [ ! -d "$ROOT/frontend/node_modules" ] || [ "$LOCK_MTIME" -gt "$LAST_DEPS_STAMP" ]; then
    echo "  Installing frontend dependencies..."
    npm ci --prefix "$ROOT/frontend"
    echo "$LOCK_MTIME" > "$FRONTEND_DEPS_STAMP"
  fi

  echo "  Frontend changes detected. Building static export..."
  (
    cd "$ROOT/frontend"
    NEXT_PUBLIC_API_URL="$FRONTEND_API_URL" npm run build
  )
  echo "$FRONTEND_SRC_MTIME" > "$FRONTEND_BUILD_STAMP"
  echo "$FRONTEND_API_URL" > "$FRONTEND_API_URL_STAMP"
else
  echo "  Frontend unchanged; skipping build."
fi

API_ARGS=(--port "$API_PORT")
if [ -n "${AF3_DB_DIR:-}" ]; then
  API_ARGS+=(--db-dir "$AF3_DB_DIR")
fi
if [ -n "${AF3_WEIGHTS_DIR:-}" ]; then
  API_ARGS+=(--model-dir "$AF3_WEIGHTS_DIR")
fi
if [ -n "${AF3_DATA_DIR:-}" ]; then
  API_ARGS+=(--data-dir "$AF3_DATA_DIR")
fi

echo "Starting FastAPI server on port $API_PORT..."
cd "$ROOT"
nohup python3 -m alphafold3_mlx.api "${API_ARGS[@]}" > "$LOG_DIR/fastapi.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$FASTAPI_PID_FILE"
echo "  PID: $API_PID (log: $LOG_DIR/fastapi.log)"

for _ in $(seq 1 45); do
  if curl -sf "http://127.0.0.1:$API_PORT/api/system/status" >/dev/null; then
    echo "  FastAPI is ready."
    API_READY=true
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "  ERROR: FastAPI failed to start. Check $LOG_DIR/fastapi.log" >&2
    tail -n 80 "$LOG_DIR/fastapi.log" >&2 || true
    exit 1
  fi
  sleep 1
done
if [ "${API_READY:-false}" != "true" ]; then
  echo "  ERROR: FastAPI health check timed out. Check $LOG_DIR/fastapi.log" >&2
  exit 1
fi

echo "Starting frontend static server on port $FRONTEND_PORT..."
cd "$ROOT/frontend/out"
nohup python3 -m http.server "$FRONTEND_PORT" --bind 127.0.0.1 > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
echo "  PID: $FRONTEND_PID (log: $LOG_DIR/frontend.log)"

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$FRONTEND_PORT/" >/dev/null; then
    echo "  Frontend is ready."
    FRONTEND_READY=true
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "  ERROR: Frontend failed to start. Check $LOG_DIR/frontend.log" >&2
    tail -n 80 "$LOG_DIR/frontend.log" >&2 || true
    exit 1
  fi
  sleep 1
done
if [ "${FRONTEND_READY:-false}" != "true" ]; then
  echo "  ERROR: Frontend health check timed out. Check $LOG_DIR/frontend.log" >&2
  exit 1
fi

echo ""
echo "=== Production servers running (detached) ==="
echo "  Frontend:  http://127.0.0.1:$FRONTEND_PORT"
echo "  API:       http://127.0.0.1:$API_PORT"
echo "  API docs:  http://127.0.0.1:$API_PORT/docs"
echo "  Logs:      $LOG_DIR/"
echo ""
