#!/usr/bin/env bash
# Stop the AlphaFold 3 MLX development servers.
# Usage: ./scripts/dev-stop.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/.dev-logs"

echo "Stopping dev servers..."

# Kill by saved PIDs
for svc in fastapi nextjs; do
  pidfile="$LOG_DIR/$svc.pid"
  if [ -f "$pidfile" ]; then
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "  Stopping $svc (PID $pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
done

# Also kill by port as fallback
for port in 8642 3001; do
  if pids=$(lsof -ti:"$port" 2>/dev/null); then
    echo "  Killing remaining processes on port $port"
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
done

pkill -f "next dev" 2>/dev/null || true
rm -f "$ROOT/frontend/.next/dev/lock" 2>/dev/null || true

echo "Done."
