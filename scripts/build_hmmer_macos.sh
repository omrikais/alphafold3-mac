#!/usr/bin/env bash
# Build HMMER 3.4 for macOS with AlphaFold 3 seq_limit patch
#
# This script downloads, patches, and builds HMMER 3.4 for macOS ARM64.
# The patch adds a --seq_limit flag to jackhmmer for limiting sequence hits.
#
# Usage: ./scripts/build_hmmer_macos.sh [--prefix=<install_dir>]
#
# Requirements:
#   - macOS with Xcode Command Line Tools
#   - Homebrew with autoconf and automake
#
# Output:
#   - HMMER binaries installed to ~/hmmer (or custom prefix)
#   - jackhmmer will have the --seq_limit option

set -euo pipefail

# Configuration
HMMER_VERSION="3.4"
HMMER_URL="https://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz"
HMMER_MIRROR_URL="https://distfiles.macports.org/hmmer/hmmer-${HMMER_VERSION}.tar.gz"
# SHA256 checksum for supply-chain security verification
# From: http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
HMMER_SHA256="ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_PATH="${REPO_ROOT}/docker/jackhmmer_seq_limit.patch"
BUILD_DIR="${TMPDIR:-/tmp}/hmmer-build-$$"
INSTALL_PREFIX="${HOME}/hmmer"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --prefix=*)
            INSTALL_PREFIX="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--prefix=<install_dir>]"
            echo ""
            echo "Build HMMER 3.4 for macOS with AlphaFold 3 seq_limit patch."
            echo ""
            echo "Options:"
            echo "  --prefix=DIR  Install to DIR (default: ~/hmmer)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Building HMMER ${HMMER_VERSION} for macOS ===${NC}"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo -e "${RED}Error: Xcode Command Line Tools not found.${NC}"
    echo "Install with: xcode-select --install"
    exit 1
fi
echo "  - Xcode Command Line Tools: OK"

# Check for clang
if ! command -v clang &>/dev/null; then
    echo -e "${RED}Error: clang not found.${NC}"
    echo "Install Xcode Command Line Tools with: xcode-select --install"
    exit 1
fi
CLANG_VERSION=$(clang --version | head -1)
echo "  - Compiler: ${CLANG_VERSION}"

# Check for autoconf
if ! command -v autoconf &>/dev/null; then
    echo -e "${RED}Error: autoconf not found.${NC}"
    echo "Install with: brew install autoconf"
    exit 1
fi
echo "  - autoconf: $(autoconf --version | head -1)"

# Check for automake
if ! command -v automake &>/dev/null; then
    echo -e "${RED}Error: automake not found.${NC}"
    echo "Install with: brew install automake"
    exit 1
fi
echo "  - automake: $(automake --version | head -1)"

# Check for patch file
if [[ ! -f "${PATCH_PATH}" ]]; then
    echo -e "${RED}Error: Patch file not found at ${PATCH_PATH}${NC}"
    echo "Make sure you're running this script from the repository root."
    exit 1
fi
echo "  - Patch file: OK"

# Check architecture
ARCH=$(uname -m)
if [[ "${ARCH}" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Building on ${ARCH}, optimized for ARM64${NC}"
fi
echo "  - Architecture: ${ARCH}"

echo ""

# Step 2: Download HMMER
echo -e "${YELLOW}[2/7] Downloading HMMER ${HMMER_VERSION}...${NC}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [[ -f "hmmer-${HMMER_VERSION}.tar.gz" ]]; then
    echo "  - Using cached download"
else
    if ! curl -fSL --connect-timeout 30 -o "hmmer-${HMMER_VERSION}.tar.gz" "${HMMER_URL}" 2>/dev/null; then
        echo "  - Primary URL unreachable, trying mirror..."
        curl -fSL --connect-timeout 30 -o "hmmer-${HMMER_VERSION}.tar.gz" "${HMMER_MIRROR_URL}"
    fi
fi
echo "  - Downloaded: hmmer-${HMMER_VERSION}.tar.gz"

# Verify SHA256 checksum for supply-chain security
echo "  - Verifying SHA256 checksum..."
COMPUTED_SHA256=$(shasum -a 256 "hmmer-${HMMER_VERSION}.tar.gz" | cut -d ' ' -f 1)
if [[ "${COMPUTED_SHA256}" != "${HMMER_SHA256}" ]]; then
    echo -e "${RED}Error: SHA256 checksum mismatch!${NC}"
    echo "  Expected: ${HMMER_SHA256}"
    echo "  Got:      ${COMPUTED_SHA256}"
    echo "  The download may be corrupted or tampered with."
    rm -f "hmmer-${HMMER_VERSION}.tar.gz"
    exit 1
fi
echo "  - SHA256 checksum: OK"

echo ""

# Step 3: Extract
echo -e "${YELLOW}[3/7] Extracting...${NC}"

tar xzf "hmmer-${HMMER_VERSION}.tar.gz"
cd "hmmer-${HMMER_VERSION}"
echo "  - Extracted to: ${BUILD_DIR}/hmmer-${HMMER_VERSION}"

echo ""

# Step 4: Apply patch
echo -e "${YELLOW}[4/7] Applying seq_limit patch...${NC}"

# Check if patch has already been applied
if grep -q "seq_limit" src/jackhmmer.c 2>/dev/null; then
    echo "  - Patch already applied, skipping"
else
    patch -p1 < "${PATCH_PATH}"
    echo "  - Patch applied successfully"
fi

echo ""

# Step 5: Configure and build
echo -e "${YELLOW}[5/7] Configuring and building...${NC}"

# Configure with ARM64 optimizations
./configure \
    --prefix="${INSTALL_PREFIX}" \
    CC=clang \
    CFLAGS="-O3 -arch $(uname -m)"

# Build using all available cores
NCPU=$(sysctl -n hw.ncpu)
echo "  - Building with ${NCPU} parallel jobs..."
make -j"${NCPU}"

echo ""

# Step 6: Install
echo -e "${YELLOW}[6/7] Installing to ${INSTALL_PREFIX}...${NC}"

make install
echo "  - Installation complete"

echo ""

# Step 7: Verify installation
echo -e "${YELLOW}[7/7] Verifying installation...${NC}"

JACKHMMER="${INSTALL_PREFIX}/bin/jackhmmer"

if [[ ! -x "${JACKHMMER}" ]]; then
    echo -e "${RED}Error: jackhmmer not found at ${JACKHMMER}${NC}"
    exit 1
fi
echo "  - jackhmmer binary: OK"

# Check for --seq_limit option
if "${JACKHMMER}" -h 2>&1 | grep -q "seq_limit"; then
    echo "  - --seq_limit option: OK"
else
    echo -e "${RED}Error: --seq_limit option not found in jackhmmer${NC}"
    echo "The patch may not have been applied correctly."
    exit 1
fi

# Clean up build directory
echo ""
echo -e "${YELLOW}Cleaning up build directory...${NC}"
rm -rf "${BUILD_DIR}"

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "HMMER ${HMMER_VERSION} has been installed to: ${INSTALL_PREFIX}"
echo ""
echo "To use HMMER, add it to your PATH:"
echo ""
echo "  export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""
echo ""
echo "Or add this line to your ~/.zshrc:"
echo ""
echo "  echo 'export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\"' >> ~/.zshrc"
echo ""
echo "Verify installation with:"
echo ""
echo "  ${INSTALL_PREFIX}/bin/jackhmmer --help | grep seq_limit"
echo ""
