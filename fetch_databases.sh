#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Downloads AlphaFold 3 genetic databases from Google Cloud Storage.
#
# Features:
#   - Per-file progress bars (curl --progress-bar)
#   - Resume interrupted downloads (curl -C -); partial archives kept on failure
#   - Atomic writes: download to .zst.tmp, decompress to .tmp, mv on success
#   - mmCIF: extract into staging dir, promote on success, marker-based completion
#   - Skip files that already exist and are non-empty
#   - Disk space check at start
#   - Structured exit code: non-zero if any required protein DB failed
#
# Usage:
#   bash fetch_databases.sh [target_directory]
#   bash fetch_databases.sh ~/public_databases
#   bash fetch_databases.sh ~/public_databases --non-interactive

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

readonly SOURCE=https://storage.googleapis.com/alphafold-databases/v3.0
readonly MIN_FREE_GB=650

# Required protein databases (failure = non-zero exit)
readonly REQUIRED_DATASETS=(
    mgy_clusters_2022_05.fa
    bfd-first_non_consensus_sequences.fasta
    uniref90_2022_05.fa
    uniprot_all_2021_04.fa
    pdb_seqres_2022_09_28.fasta
)

# Optional RNA databases (failure = warning only)
readonly OPTIONAL_DATASETS=(
    rnacentral_active_seq_id_90_cov_80_linclust.fasta
    nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
    rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
)

# Special case: tar archive (not a single .zst file)
readonly MMCIF_ARCHIVE="pdb_2022_09_28_mmcif_files.tar.zst"
readonly MMCIF_ARCHIVE_DIR="pdb_2022_09_28_mmcif_files"  # top-level dir inside archive
readonly MMCIF_CANONICAL="mmcif_files"                    # canonical install target

# Minimum .cif files to consider a manually-managed mmCIF dir complete
readonly MMCIF_MIN_FILES=100

# ── Parse arguments ──────────────────────────────────────────────────────────

NON_INTERACTIVE=false
db_dir=""

for arg in "$@"; do
    case "$arg" in
        --non-interactive) NON_INTERACTIVE=true ;;
        --*) echo "Unknown flag: $arg"; exit 1 ;;
        *) db_dir="$arg" ;;
    esac
done

readonly db_dir="${db_dir:-$HOME/public_databases}"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Utility functions ────────────────────────────────────────────────────────

_info() { echo -e "  ${GREEN}[OK]${NC} $*"; }
_warn() { echo -e "  ${YELLOW}[!!]${NC} $*"; }
_fail() { echo -e "  ${RED}[XX]${NC} $*"; }

# Set by download helpers: "downloaded", "skipped", or "failed"
LAST_DL_STATUS=""

# Returns 0 if the dataset name is in the REQUIRED list.
_is_required() {
    local name="$1"
    for req in "${REQUIRED_DATASETS[@]}"; do
        if [[ "$req" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

# ── Preflight checks ────────────────────────────────────────────────────────

# H-13: Check all required tools upfront
missing=()
for cmd in curl tar zstd; do
    if ! command -v "${cmd}" > /dev/null 2>&1; then
        missing+=("$cmd")
    fi
done
if (( ${#missing[@]} > 0 )); then
    _fail "Missing required tools: ${missing[*]}"
    echo "  Install with: brew install ${missing[*]}"
    exit 1
fi

# Disk space check on target volume
mkdir -p "${db_dir}"
free_kb=$(df -k "${db_dir}" | tail -1 | awk '{print $4}')
free_gb=$(( free_kb / 1048576 ))

if (( free_gb < MIN_FREE_GB )); then
    _warn "Low disk space: ${free_gb} GB free on target volume (recommend >= ${MIN_FREE_GB} GB)"
    echo "       Databases require approximately 630 GB when fully unpacked."
    if [[ "$NON_INTERACTIVE" != true ]]; then
        echo ""
        read -rp "  Continue anyway? [y/N]: " continue_choice
        continue_choice=$(printf '%s' "$continue_choice" | tr '[:upper:]' '[:lower:]')
        if [[ "$continue_choice" != "y" ]]; then
            echo "  Aborted."
            exit 1
        fi
    else
        echo "       Continuing anyway (non-interactive mode)..."
    fi
fi

# ── Download helpers ─────────────────────────────────────────────────────────

# Download a .zst-compressed flat file with resume support.
# Downloads compressed archive to .zst.tmp, then decompresses to .tmp,
# then atomically renames to final path.
#
# Sets LAST_DL_STATUS to "downloaded", "skipped", or "failed".
# Returns 0 on success (including skip), 1 on failure.
_download_flat() {
    local name="$1"
    local idx="$2"
    local total="$3"
    local final_path="${db_dir}/${name}"
    local archive_tmp="${db_dir}/${name}.zst.tmp"
    local decomp_tmp="${db_dir}/${name}.tmp"
    local url="${SOURCE}/${name}.zst"

    # Skip if final file already exists and is non-empty
    if [[ -f "$final_path" ]] && [[ -s "$final_path" ]]; then
        LAST_DL_STATUS="skipped"
        echo -e "  ${BLUE}[${idx}/${total}]${NC} ${name}  ${DIM}(already exists, skipping)${NC}"
        return 0
    fi

    LAST_DL_STATUS="failed"
    echo -e "  ${BLUE}[${idx}/${total}]${NC} Downloading ${BOLD}${name}${NC}"

    # Download compressed file with resume support.
    # Partial archive is kept on failure so curl -C - can resume next run.
    if ! curl -fSL --progress-bar -C - -o "$archive_tmp" "$url"; then
        _fail "Download failed: ${name}"
        return 1
    fi

    # Decompress to temporary file
    echo -e "         Decompressing..."
    if ! zstd -d -f -o "$decomp_tmp" "$archive_tmp"; then
        _fail "Decompression failed: ${name}"
        rm -f "$decomp_tmp"
        return 1
    fi

    # Atomic rename
    mv "$decomp_tmp" "$final_path"
    rm -f "$archive_tmp"
    LAST_DL_STATUS="downloaded"
    _info "${name}"
    return 0
}

# Count .cif/.cif.gz files in a directory.
# Returns the count (capped at MMCIF_MIN_FILES for efficiency).
_count_cif_files() {
    local dir="$1"
    find "$dir" \( -name '*.cif' -o -name '*.cif.gz' \) -print 2>/dev/null \
        | head -${MMCIF_MIN_FILES} | wc -l | tr -d '[:space:]'
}

# Download and extract the mmCIF tar archive.
# Downloads to .tar.zst.tmp with resume, extracts into a staging dir,
# then promotes to the canonical location: ${db_dir}/mmcif_files/
#
# Completion is tracked by a marker file AND valid content (>= MMCIF_MIN_FILES
# .cif files). Marker alone with empty/missing dir triggers re-download.
#
# After a fresh download, stale alternative layouts are removed so the
# runtime path resolver always finds the canonical location first.
#
# Sets LAST_DL_STATUS to "downloaded", "skipped", or "failed".
# Returns 0 on success (including skip), 1 on failure.
_download_mmcif() {
    local idx="$1"
    local total="$2"
    local archive_tmp="${db_dir}/${MMCIF_ARCHIVE}.tmp"
    local url="${SOURCE}/${MMCIF_ARCHIVE}"
    local marker="${db_dir}/.mmcif_download_complete"
    local staging="${db_dir}/${MMCIF_ARCHIVE_DIR}.staging"
    local canonical="${db_dir}/${MMCIF_CANONICAL}"

    # Check for completion marker AND valid content in any accepted layout
    if [[ -f "$marker" ]]; then
        for candidate in \
            "${db_dir}/mmcif_files" \
            "${db_dir}/${MMCIF_ARCHIVE_DIR}/mmcif_files" \
            "${db_dir}/${MMCIF_ARCHIVE_DIR}"; do
            if [[ -d "$candidate" ]]; then
                local cif_count
                cif_count=$(_count_cif_files "$candidate")
                if (( cif_count >= MMCIF_MIN_FILES )); then
                    LAST_DL_STATUS="skipped"
                    local layout="${candidate#${db_dir}/}"
                    echo -e "  ${BLUE}[${idx}/${total}]${NC} ${layout}/  ${DIM}(already complete, skipping)${NC}"
                    return 0
                fi
            fi
        done
        # Marker exists but no valid content — remove stale marker
        rm -f "$marker"
    fi

    # Fallback: check for existing layout with substantial content (manually managed)
    for candidate in \
        "${db_dir}/mmcif_files" \
        "${db_dir}/${MMCIF_ARCHIVE_DIR}/mmcif_files" \
        "${db_dir}/${MMCIF_ARCHIVE_DIR}"; do
        if [[ -d "$candidate" ]]; then
            local file_count
            file_count=$(_count_cif_files "$candidate")
            if (( file_count >= MMCIF_MIN_FILES )); then
                LAST_DL_STATUS="skipped"
                local layout="${candidate#${db_dir}/}"
                echo -e "  ${BLUE}[${idx}/${total}]${NC} ${layout}/  ${DIM}(already exists, skipping)${NC}"
                return 0
            fi
        fi
    done

    LAST_DL_STATUS="failed"
    echo -e "  ${BLUE}[${idx}/${total}]${NC} Downloading ${BOLD}${MMCIF_ARCHIVE}${NC}"

    # Download tar.zst archive with resume support.
    # Partial archive is kept on failure so curl -C - can resume next run.
    if ! curl -fSL --progress-bar -C - -o "$archive_tmp" "$url"; then
        _fail "Download failed: ${MMCIF_ARCHIVE}"
        return 1
    fi

    # Extract atomically: decompress+untar into staging dir, then promote
    rm -rf "$staging"
    mkdir -p "$staging"
    echo -e "         Extracting (this may take a while)..."
    if ! (zstd -d < "$archive_tmp" | tar xf - -C "$staging"); then
        _fail "Extraction failed: ${MMCIF_ARCHIVE}"
        rm -rf "$staging"
        return 1
    fi

    # Verify expected directory structure in staging.
    # The archive may contain either:
    #   - pdb_2022_09_28_mmcif_files/  (with or without nested mmcif_files/)
    #   - mmcif_files/  (directly)
    local source_dir=""
    if [[ -d "$staging/$MMCIF_ARCHIVE_DIR/mmcif_files" ]]; then
        source_dir="$staging/$MMCIF_ARCHIVE_DIR/mmcif_files"
    elif [[ -d "$staging/$MMCIF_ARCHIVE_DIR" ]]; then
        source_dir="$staging/$MMCIF_ARCHIVE_DIR"
    elif [[ -d "$staging/$MMCIF_CANONICAL" ]]; then
        source_dir="$staging/$MMCIF_CANONICAL"
    else
        _fail "Unexpected archive structure: neither ${MMCIF_ARCHIVE_DIR}/ nor ${MMCIF_CANONICAL}/ found"
        rm -rf "$staging"
        return 1
    fi

    rm -rf "$canonical"
    mv "$source_dir" "$canonical"
    rm -rf "$staging"

    # Remove stale alternative layouts that would shadow canonical path
    if [[ -d "${db_dir}/${MMCIF_ARCHIVE_DIR}" ]]; then
        rm -rf "${db_dir}/${MMCIF_ARCHIVE_DIR}"
    fi

    # Mark complete and clean up archive
    touch "$marker"
    rm -f "$archive_tmp"
    LAST_DL_STATUS="downloaded"
    _info "${MMCIF_CANONICAL}/"
    return 0
}

# ── Main download loop ──────────────────────────────────────────────────────

ALL_DATASETS=("${REQUIRED_DATASETS[@]}" "${OPTIONAL_DATASETS[@]}")
# +1 for mmCIF archive
TOTAL=$(( ${#ALL_DATASETS[@]} + 1 ))

echo ""
echo -e "${BOLD}─── Downloading AlphaFold 3 Databases ─────────────────────${NC}"
echo ""
echo "  Source:      ${SOURCE}"
echo "  Destination: ${db_dir}"
echo "  Datasets:    ${TOTAL} files (~252 GB download, ~630 GB unpacked)"
echo ""

downloaded=0
skipped=0
failed_required=0
failed_optional=0
idx=0

# Download flat datasets sequentially with progress
for name in "${ALL_DATASETS[@]}"; do
    idx=$((idx + 1))
    LAST_DL_STATUS=""
    if _download_flat "$name" "$idx" "$TOTAL"; then
        case "$LAST_DL_STATUS" in
            downloaded) downloaded=$((downloaded + 1)) ;;
            skipped)    skipped=$((skipped + 1)) ;;
        esac
    else
        if _is_required "$name"; then
            failed_required=$((failed_required + 1))
        else
            failed_optional=$((failed_optional + 1))
        fi
    fi
done

# Download mmCIF archive
idx=$((idx + 1))
LAST_DL_STATUS=""
if _download_mmcif "$idx" "$TOTAL"; then
    case "$LAST_DL_STATUS" in
        downloaded) downloaded=$((downloaded + 1)) ;;
        skipped)    skipped=$((skipped + 1)) ;;
    esac
else
    failed_required=$((failed_required + 1))
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}─── Download Summary ──────────────────────────────────────${NC}"
echo ""

failed_total=$((failed_required + failed_optional))

if (( failed_total == 0 )); then
    if (( skipped == TOTAL )); then
        _info "All ${TOTAL} datasets already present (nothing to download)"
    elif (( skipped > 0 )); then
        _info "All ${TOTAL} datasets complete (${downloaded} downloaded, ${skipped} already present)"
    else
        _info "All ${TOTAL} datasets downloaded successfully"
    fi
else
    if (( downloaded > 0 || skipped > 0 )); then
        _info "${downloaded} downloaded, ${skipped} already present"
    fi
    if (( failed_required > 0 )); then
        _fail "${failed_required} required dataset(s) failed"
    fi
    if (( failed_optional > 0 )); then
        _warn "${failed_optional} optional dataset(s) failed"
    fi
fi
echo ""

# Exit non-zero only if required datasets failed
if (( failed_required > 0 )); then
    exit 1
fi
exit 0
