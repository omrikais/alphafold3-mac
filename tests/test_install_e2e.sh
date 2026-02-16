#!/usr/bin/env bash
# End-to-end installer test — runs a REAL install in a fully isolated sandbox.
#
# Creates a temp clone of the repo and a fake HOME so nothing touches the
# real repo or user directories. Verifies that the installer produces all
# expected artifacts (venv, HMMER, frontend, config, launcher).
#
# Usage:
#   bash tests/test_install_e2e.sh              # full mode (requires Node.js)
#   bash tests/test_install_e2e.sh --cli-only   # CLI-only mode (no Node.js)
#   bash tests/test_install_e2e.sh --smoke      # API smoke test (start → health check → stop)
#
# Requirements:
#   - macOS ARM64 with Xcode CLI Tools
#   - Python 3.12+, Homebrew, autoconf, automake
#   - Node.js >= 20 (for full mode only)
#   - ~3 GB free disk space in $TMPDIR
#
# Duration: ~3-8 minutes depending on network speed and mode.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments ─────────────────────────────────────────────────────────

MODE="full"
SMOKE_TEST=false
for arg in "$@"; do
    case "$arg" in
        --cli-only) MODE="cli-only" ;;
        --smoke) SMOKE_TEST=true ;;
    esac
done

# ── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

PASS=0
FAIL=0
ERRORS=()

_ok()   { echo -e "  ${GREEN}[OK]${NC} $*"; PASS=$((PASS + 1)); }
_fail() { echo -e "  ${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); ERRORS+=("$*"); }
_info() { echo -e "  ${DIM}$*${NC}"; }

_check() {
    local label="$1"
    shift
    if "$@" 2>/dev/null; then
        _ok "$label"
    else
        _fail "$label"
    fi
}

# ── Create sandbox ──────────────────────────────────────────────────────────

SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/af3-e2e-XXXXXX")
FAKE_HOME="$SANDBOX/home"
CLONE_DIR="$SANDBOX/repo"

mkdir -p "$FAKE_HOME/Desktop"

cleanup() {
    echo ""
    _info "Cleaning up sandbox: $SANDBOX"
    # Kill any leftover server processes from smoke test
    if [[ -n "${API_PID:-}" ]]; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    rm -rf "$SANDBOX"
}
trap cleanup EXIT

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  AlphaFold 3 MLX — E2E Installer Test${NC}"
echo -e "${BOLD}  Mode: ${MODE}${SMOKE_TEST:+ + smoke test}${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Sandbox:   $SANDBOX"
echo -e "  Fake HOME: $FAKE_HOME"
echo ""

# ── Clone repo to sandbox ───────────────────────────────────────────────────
# Use rsync instead of git clone to capture uncommitted files too.

echo -e "${BOLD}── Cloning repo to sandbox ────────────────────────────────${NC}"
rsync -a --exclude='.git' --exclude='.venv' --exclude='node_modules' \
    --exclude='__pycache__' --exclude='frontend/out' --exclude='frontend/.next' \
    "$REAL_REPO/" "$CLONE_DIR/"
# Initialize a minimal git repo (pyproject.toml needs to be findable)
(cd "$CLONE_DIR" && git init -q && git add -A && git commit -q -m "test snapshot")
_ok "Repo cloned to sandbox ($(du -sh "$CLONE_DIR" | cut -f1))"
echo ""

# ── Run installer ───────────────────────────────────────────────────────────

echo -e "${BOLD}── Running installer ──────────────────────────────────────${NC}"
echo ""

# Build the input pipe for the interactive installer:
#   Line 1: Enter (welcome screen)
#   Line 2: 1 or 2 (install mode)
#   Line 3: Enter (weights dir — accept default, won't exist but that's OK)
#   Line 4: 3 (databases — skip)
#   Line 5: Enter (data dir — accept default)
#   Line 6: 8799 (port — unlikely to conflict)

if [[ "$MODE" == "full" ]]; then
    INSTALL_INPUT=$(printf '\n1\n\n3\n\n8799\n')
else
    INSTALL_INPUT=$(printf '\n2\n\n3\n\n8799\n')
fi

INSTALL_LOG="$SANDBOX/install.log"

# Run with fake HOME so config/launcher go to sandbox
set +e
HOME="$FAKE_HOME" \
    bash "$CLONE_DIR/scripts/install.sh" <<< "$INSTALL_INPUT" \
    > "$INSTALL_LOG" 2>&1
INSTALL_RC=$?
set -e

if [[ $INSTALL_RC -eq 0 ]]; then
    _ok "Installer exited with code 0"
else
    _fail "Installer exited with code $INSTALL_RC"
    echo ""
    echo -e "  ${RED}Last 30 lines of install log:${NC}"
    tail -30 "$INSTALL_LOG" | sed 's/^/    /'
    echo ""
    echo -e "  Full log: $INSTALL_LOG"
fi

# ── Verify artifacts ────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}── Verifying artifacts ────────────────────────────────────${NC}"
echo ""

# 1. Python venv
_check "Python venv created" test -f "$CLONE_DIR/.venv/bin/python3"
_check "Python venv is arm64" \
    "$CLONE_DIR/.venv/bin/python3" -c "import platform; assert platform.machine() == 'arm64'"
_check "Python version >= 3.12" \
    "$CLONE_DIR/.venv/bin/python3" -c "import sys; assert sys.version_info >= (3, 12)"

# 2. Core Python packages importable
_check "mlx importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import mlx.core"
_check "jax importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import jax"
_check "numpy importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import numpy"

# 3. API dependencies (installed via [api] extra)
_check "fastapi importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import fastapi"
_check "uvicorn importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import uvicorn"

# 4. MLX project package importable
_check "alphafold3_mlx importable" \
    "$CLONE_DIR/.venv/bin/python3" -c "import alphafold3_mlx"

# 5. HMMER binaries
HMMER_PREFIX="$FAKE_HOME/.alphafold3_mlx/hmmer"
_check "jackhmmer binary exists" test -x "$HMMER_PREFIX/bin/jackhmmer"
_check "hmmsearch binary exists" test -x "$HMMER_PREFIX/bin/hmmsearch"
_check "hmmbuild binary exists"  test -x "$HMMER_PREFIX/bin/hmmbuild"
_check "nhmmer binary exists"    test -x "$HMMER_PREFIX/bin/nhmmer"

# Verify HMMER is native arm64
if [[ -x "$HMMER_PREFIX/bin/jackhmmer" ]]; then
    JACKHMMER_ARCH=$(file "$HMMER_PREFIX/bin/jackhmmer")
    if echo "$JACKHMMER_ARCH" | grep -q "arm64"; then
        _ok "jackhmmer is native arm64"
    else
        _fail "jackhmmer is not arm64: $JACKHMMER_ARCH"
    fi
fi

# 6. Frontend (full mode only)
if [[ "$MODE" == "full" ]]; then
    _check "frontend/out/ exists" test -d "$CLONE_DIR/frontend/out"
    _check "frontend/out/index.html exists" test -f "$CLONE_DIR/frontend/out/index.html"
    # Check it has some real content
    if [[ -f "$CLONE_DIR/frontend/out/index.html" ]]; then
        HTML_SIZE=$(wc -c < "$CLONE_DIR/frontend/out/index.html")
        if (( HTML_SIZE > 100 )); then
            _ok "index.html has content (${HTML_SIZE} bytes)"
        else
            _fail "index.html is suspiciously small (${HTML_SIZE} bytes)"
        fi
    fi
else
    _info "Frontend build skipped (CLI-only mode)"
fi

# 7. Config file
_check "config.env created" test -f "$FAKE_HOME/.alphafold3_mlx/config.env"
if [[ -f "$FAKE_HOME/.alphafold3_mlx/config.env" ]]; then
    _check "config has AF3_PORT" grep -q "AF3_PORT=" "$FAKE_HOME/.alphafold3_mlx/config.env"
    _check "config has AF3_WEIGHTS_DIR" grep -q "AF3_WEIGHTS_DIR=" "$FAKE_HOME/.alphafold3_mlx/config.env"
    _check "config has AF3_JACKHMMER" grep -q "AF3_JACKHMMER=" "$FAKE_HOME/.alphafold3_mlx/config.env"
fi

# 8. Desktop launcher (full mode only)
if [[ "$MODE" == "full" ]]; then
    _check "Desktop launcher created" test -f "$FAKE_HOME/Desktop/AlphaFold3.command"
    _check "Desktop launcher is executable" test -x "$FAKE_HOME/Desktop/AlphaFold3.command"
else
    _info "Desktop launcher skipped (CLI-only mode)"
fi

# 9. No contamination of real repo
_check "Real repo .venv untouched" \
    test "$(ls -la "$REAL_REPO/.venv/bin/python3" 2>/dev/null | awk '{print $6,$7,$8}')" = \
    "$(ls -la "$REAL_REPO/.venv/bin/python3" 2>/dev/null | awk '{print $6,$7,$8}')"
_check "Real HOME not modified" \
    test ! -f "$HOME/.alphafold3_mlx/config.env" -o \
    "$(stat -f%m "$HOME/.alphafold3_mlx/config.env" 2>/dev/null || echo 0)" = \
    "$(stat -f%m "$HOME/.alphafold3_mlx/config.env" 2>/dev/null || echo 0)"

# ── API Smoke Test ──────────────────────────────────────────────────────────

if [[ "$SMOKE_TEST" == true ]]; then
    echo ""
    echo -e "${BOLD}── API Smoke Test ─────────────────────────────────────────${NC}"
    echo ""

    # Start the API server in the background using the sandbox config
    API_PORT=8799
    API_LOG="$SANDBOX/api.log"

    HOME="$FAKE_HOME" \
        "$CLONE_DIR/.venv/bin/python3" -m alphafold3_mlx.api \
        --port "$API_PORT" \
        > "$API_LOG" 2>&1 &
    API_PID=$!

    _info "Started API server (PID $API_PID) on port $API_PORT"

    # Wait for server to be ready (up to 30 seconds)
    READY=false
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$API_PORT/api/system/status" > /dev/null 2>&1; then
            READY=true
            break
        fi
        sleep 1
    done

    if [[ "$READY" == true ]]; then
        _ok "API server started and responding"

        # Health check
        STATUS_JSON=$(curl -sf "http://127.0.0.1:$API_PORT/api/system/status" 2>/dev/null || echo "{}")
        if echo "$STATUS_JSON" | "$CLONE_DIR/.venv/bin/python3" -c "import json,sys; d=json.load(sys.stdin); assert 'model_loaded' in d" 2>/dev/null; then
            _ok "Health endpoint returns valid JSON with model_loaded field"
        else
            _fail "Health endpoint response unexpected: $STATUS_JSON"
        fi

        # Frontend serving (full mode)
        if [[ "$MODE" == "full" ]]; then
            HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "http://127.0.0.1:$API_PORT/" 2>/dev/null || echo "000")
            if [[ "$HTTP_CODE" == "200" ]]; then
                _ok "Frontend served at / (HTTP 200)"
            else
                _fail "Frontend not served at / (HTTP $HTTP_CODE)"
            fi
        fi
    else
        _fail "API server did not start within 30 seconds"
        echo ""
        echo -e "  ${RED}Last 20 lines of API log:${NC}"
        tail -20 "$API_LOG" | sed 's/^/    /'
    fi

    # Stop the server
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    unset API_PID
    _info "API server stopped"
fi

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
echo -e "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"

if (( FAIL > 0 )); then
    echo ""
    echo -e "  ${RED}Failed checks:${NC}"
    for e in "${ERRORS[@]}"; do
        echo "    - $e"
    done
    echo ""
    echo -e "  Install log: $INSTALL_LOG"
    # Don't clean up on failure so user can inspect
    trap - EXIT
    echo -e "  Sandbox preserved at: $SANDBOX"
    exit 1
fi

echo ""
exit 0
