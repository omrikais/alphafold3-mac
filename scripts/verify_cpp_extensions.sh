#!/bin/bash
# Verify C++ extensions compile and work on macOS ARM64
#
# This script satisfies C++ extensions compile with Apple Clang.
# It builds the package and verifies all C++ submodules can be imported.
#
# Usage:
#   ./scripts/verify_cpp_extensions.sh
#
# Requirements:
#   - macOS ARM64 (Apple Silicon)
#   - Python 3.12+
#   - CMake 3.28+
#   - Xcode Command Line Tools (provides Apple Clang)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=== C++ Extension Verification for macOS ARM64 ==="
echo ""

# Check platform
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: This script only runs on macOS"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Error: This script requires Apple Silicon (ARM64)"
    exit 1
fi

echo "Platform: $(uname -s) $(uname -m)"

# Check for required tools
echo ""
echo "Checking required tools..."

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Install with: brew install cmake"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+')
echo "  CMake: ${CMAKE_VERSION}"

if ! command -v clang++ &> /dev/null; then
    echo "Error: clang++ not found. Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi
CLANG_VERSION=$(clang++ --version | head -n1)
echo "  Clang: ${CLANG_VERSION}"

# Build the package
echo ""
echo "Building package with C++ extensions..."
cd "${PROJECT_DIR}"

# Activate venv if present
if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# Determine Python command (prefer venv python)
if [[ -f "${PROJECT_DIR}/.venv/bin/python3" ]]; then
    PYTHON="${PROJECT_DIR}/.venv/bin/python3"
elif command -v python3.12 &> /dev/null; then
    PYTHON="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo "Error: Python 3 not found"
    exit 1
fi

PYTHON_VERSION=$("${PYTHON}" --version)
echo "  Python: ${PYTHON_VERSION}"

# Use uv if available, otherwise pip
if command -v uv &> /dev/null; then
    echo "Using uv for build..."
    uv pip install -e . --quiet
else
    echo "Using pip for build..."
    "${PYTHON}" -m pip install -e . --quiet
fi

# Verify imports
echo ""
echo "Verifying C++ extension imports..."

"${PYTHON}" << 'VERIFY_SCRIPT'
import sys

# List of expected submodules
submodules = [
    "cif_dict",
    "fasta_iterator",
    "msa_conversion",
    "mmcif_layout",
    "mmcif_struct_conn",
    "membership",
    "mmcif_utils",
    "aggregation",
    "string_array",
    "mmcif_atom_site",
    "mkdssp",
    "msa_profile",
]

print("Importing alphafold3.cpp module...")
try:
    from alphafold3 import cpp
    print("  ✓ alphafold3.cpp imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import alphafold3.cpp: {e}")
    sys.exit(1)

# Check each submodule
failed = []
for name in submodules:
    try:
        submodule = getattr(cpp, name)
        print(f"  ✓ cpp.{name}")
    except AttributeError as e:
        print(f"  ✗ cpp.{name}: {e}")
        failed.append(name)

if failed:
    print(f"\nFailed to load {len(failed)} submodule(s): {', '.join(failed)}")
    sys.exit(1)

print(f"\nAll {len(submodules)} submodules loaded successfully!")
VERIFY_SCRIPT

echo ""
echo "=== Verification Complete ==="
echo ""
echo "C++ extensions compiled and imported successfully on macOS ARM64."
echo "requirement satisfied."
