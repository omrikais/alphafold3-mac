#!/bin/bash
# CI/CD test runner requiring 100% pass rate
#
# This script runs all validation tests and ensures 100% pass rate
# before allowing merge to main branch.
#
# Usage:
# ./scripts/run_validation_tests.sh # Run validation subset
#   ./scripts/run_validation_tests.sh --quick         # Run quick subset (unit only)
#   ./scripts/run_validation_tests.sh --report        # Generate HTML report
#   ./scripts/run_validation_tests.sh --external-deps # Run external deps tests (optional)
#   ./scripts/run_validation_tests.sh --benchmark     # Run benchmark tests (optional)
#   ./scripts/run_validation_tests.sh --optional      # Run external deps + benchmark tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Parse arguments
QUICK_MODE=false
GENERATE_REPORT=false
VERBOSE=false
RUN_EXTERNAL_DEPS=false
RUN_BENCHMARK=false
RUN_OPTIONAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --external-deps)
            RUN_EXTERNAL_DEPS=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --optional)
            RUN_OPTIONAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
echo "=============================================="
echo "AlphaFold 3 MLX Validation Test Runner"
echo "=============================================="
echo ""

# Activate virtual environment
if [[ -d "$VENV_PATH" ]]; then
    echo "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
else
    echo -e "${RED}Error: Virtual environment not found at ${VENV_PATH}${NC}"
    echo "Run: python -m venv .venv && pip install -e .[dev]"
    exit 1
fi

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Build pytest base command (The policy requires -v)
PYTEST_BASE=(python -m pytest -v)

if [[ "$VERBOSE" == "true" ]]; then
    PYTEST_BASE=(python -m pytest -vv)
fi

if [[ "$GENERATE_REPORT" == "true" ]]; then
    PYTEST_BASE+=("--html=test_report.html" "--self-contained-html")
fi

run_pytest() {
    local label="$1"
    local enforce_no_skips="$2"
    shift 2
    local cmd=("$@")

    echo ""
    echo "Running ${label}..."
    echo "Test command: ${cmd[*]}"
    echo "----------------------------------------------"

    local start_time end_time duration
    start_time=$(date +%s)

    set +e
    local output
    output=$("${cmd[@]}" 2>&1)
    local exit_code=$?
    set -e

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "$output"
    echo ""
    echo "----------------------------------------------"

    local passed failed skipped deselected errors total
    passed=$(echo "$output" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
    failed=$(echo "$output" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
    skipped=$(echo "$output" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo "0")
    deselected=$(echo "$output" | grep -oE '[0-9]+ deselected' | grep -oE '[0-9]+' || echo "0")
    errors=$(echo "$output" | grep -oE '[0-9]+ error' | grep -oE '[0-9]+' || echo "0")

    total=$((passed + failed + errors))

    echo ""
    echo "=============================================="
    echo "TEST SUMMARY (${label})"
    echo "=============================================="
    echo "Duration:   ${duration}s"
    echo "Passed:     $passed"
    echo "Failed:     $failed"
    echo "Skipped:    $skipped"
    echo "Deselected: $deselected"
    echo "Errors:     $errors"
    echo "Total:      $total"
    echo ""

    if [[ "$exit_code" -ne 0 ]] || [[ "$failed" -gt 0 ]] || [[ "$errors" -gt 0 ]]; then
        echo -e "${RED}=============================================="
        echo "VALIDATION FAILED (${label})"
        echo "=============================================="
        echo "Please fix all failing tests before merge."
        echo -e "===============================================${NC}"
        exit 1
    fi

    if [[ "$enforce_no_skips" == "true" ]]; then
        if [[ "$skipped" -gt 0 ]] || [[ "$deselected" -gt 0 ]]; then
            echo -e "${RED}=============================================="
            echo "VALIDATION FAILED (${label})"
            echo "=============================================="
            echo "The policy forbids skipped or deselected tests in this run."
            echo -e "===============================================${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}=============================================="
    echo "VALIDATION PASSED (${label})"
    echo "=============================================="
    echo "All $passed tests passed."
    echo -e "===============================================${NC}"
}

# validation run
if [[ "$QUICK_MODE" == "true" ]]; then
    run_pytest "Quick (unit only)" "true" \
        "${PYTEST_BASE[@]}" \
        tests/unit \
        -m "not external_deps and not benchmark and not legacy_parity" \
        -x
else
    run_pytest "Validation" "true" \
        "${PYTEST_BASE[@]}" \
        tests/unit tests/integration \
        -m "not external_deps and not benchmark and not legacy_parity"
fi

# Optional runs (only when dependencies are present)
if [[ "$RUN_OPTIONAL" == "true" ]]; then
    RUN_EXTERNAL_DEPS=true
    RUN_BENCHMARK=true
fi

if [[ "$RUN_EXTERNAL_DEPS" == "true" ]]; then
    run_pytest "External Deps" "false" \
        "${PYTEST_BASE[@]}" \
        tests/unit tests/integration tests/validation \
        -m "external_deps"
fi

if [[ "$RUN_BENCHMARK" == "true" ]]; then
    run_pytest "Benchmarks" "false" \
        "${PYTEST_BASE[@]}" \
        tests/validation \
        -m "benchmark"
fi

if [[ "$GENERATE_REPORT" == "true" ]]; then
    echo ""
    echo "Test report generated: test_report.html"
fi

exit 0
