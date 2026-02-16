#!/usr/bin/env bash
# AlphaFold 3 MLX — Interactive Installer
#
# Installs the full stack (model inference + CLI + optional web UI) on
# Apple Silicon Macs. Creates a persistent configuration at
# ~/.alphafold3_mlx/config.env and a Desktop launcher.
#
# Usage:
#   ./scripts/install.sh              # interactive
#   ./scripts/install.sh --dry-run    # skip real installs (for testing)
#   Double-click install.command       # from Finder

set -euo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$HOME/.alphafold3_mlx"
CONFIG_FILE="$CONFIG_DIR/config.env"
HMMER_PREFIX="$CONFIG_DIR/hmmer"

# ── Dry-run mode ─────────────────────────────────────────────────────────────
# Skips real installations (uv, Python deps, HMMER, frontend) and destructive
# filesystem writes (config.env, Desktop launcher) while exercising all
# interactive prompts, system checks, validation logic, and phase flow.

DRY_RUN=false
for _arg in "$@"; do
    [[ "$_arg" == "--dry-run" ]] && DRY_RUN=true
done

# Database download state
DOWNLOAD_DATABASES=false
DB_DOWNLOAD_SUCCEEDED=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Utility functions ────────────────────────────────────────────────────────

_info()    { echo -e "  ${GREEN}[OK]${NC} $*"; }
_warn()    { echo -e "  ${YELLOW}[!!]${NC} $*"; }
_fail()    { echo -e "  ${RED}[XX]${NC} $*"; }
_step()    { printf "  ${BLUE}[%s/%s]${NC} %-45s" "$1" "$2" "$3"; }
_tolower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }  # Bash 3.2 portable

# Safe path expansion — handles ~ prefix and spaces, no eval/shell execution.
# macOS-portable: does NOT use realpath -m (GNU-only flag).
_expand_path() {
    local input="$1"
    # Strip surrounding quotes (single or double) — common when users paste
    # paths from macOS Finder's "Copy as Pathname" which wraps in quotes.
    input="${input#\'}"
    input="${input%\'}"
    input="${input#\"}"
    input="${input%\"}"
    # Only expand leading ~ (not arbitrary shell expressions)
    if [[ "$input" == "~/"* ]]; then
        input="$HOME/${input#\~/}"
    elif [[ "$input" == "~" ]]; then
        input="$HOME"
    fi
    # Resolve to absolute path
    if [[ -d "$input" ]]; then
        # Existing directory: cd+pwd (universally portable)
        (cd "$input" && pwd)
    elif [[ -f "$input" ]]; then
        # Existing file: resolve parent dir, append filename
        local dir
        dir="$(cd "$(dirname "$input")" && pwd)"
        echo "$dir/$(basename "$input")"
    else
        # Non-existent path: use Python for safe normalization
        # (Python 3 is guaranteed available — checked in Phase 2)
        python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$input"
    fi
}

# Read user input with a default value.
# Prompt text goes to stderr so it's visible on the terminal but not captured
# when called via VAR=$(_prompt ...) command substitution.
_prompt() {
    local prompt_text="$1"
    local default_value="$2"
    local result

    if [[ -n "$default_value" ]]; then
        printf "%s [%s]: " "$prompt_text" "$default_value" >&2
    else
        printf "%s: " "$prompt_text" >&2
    fi
    read -r result
    echo "${result:-$default_value}"
}

# Run a step with status display. Usage: _run_step NUM TOTAL LABEL COMMAND...
_run_step() {
    local num="$1" total="$2" label="$3"
    shift 3
    _step "$num" "$total" "$label"

    local start_time
    start_time=$(date +%s)
    local logfile
    logfile=$(mktemp "${TMPDIR:-/tmp}/af3-install-step-XXXXXX.log")

    if "$@" > "$logfile" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        if (( elapsed > 5 )); then
            echo -e "${GREEN}[OK]${NC}  ${DIM}(${elapsed}s)${NC}"
        else
            echo -e "${GREEN}[OK]${NC}"
        fi
        rm -f "$logfile"
        return 0
    else
        local exit_code=$?
        echo -e "${RED}[FAIL]${NC}"
        echo ""
        echo -e "  ${RED}Error output:${NC}"
        tail -20 "$logfile" | sed 's/^/    /'
        echo ""
        rm -f "$logfile"
        return $exit_code
    fi
}

# Retry a step up to N times.
_run_step_with_retry() {
    local retries=2
    local attempt=0
    while (( attempt <= retries )); do
        if _run_step "$@"; then
            return 0
        fi
        attempt=$((attempt + 1))
        if (( attempt <= retries )); then
            echo ""
            read -rp "  Retry this step? [Y/n]: " retry_choice
            if [[ "$(_tolower "$retry_choice")" == "n" ]]; then
                return 1
            fi
        fi
    done
    echo ""
    echo -e "  ${RED}Step failed after $((retries + 1)) attempts.${NC}"
    return 1
}

# Dry-run-aware wrapper: prints [DRY RUN] instead of executing the step.
_maybe_run_step() {
    if [[ "$DRY_RUN" == true ]]; then
        _step "$1" "$2" "$3"
        echo -e "${DIM}[DRY RUN]${NC}"
        return 0
    fi
    _run_step_with_retry "$@"
}

# ── Phase 1: Welcome Screen ─────────────────────────────────────────────────

_phase_welcome() {
    local version
    version=$(grep '^version' "$REPO_ROOT/pyproject.toml" | head -1 | cut -d'"' -f2)

    clear 2>/dev/null || true
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                                                              ║${NC}"
    echo -e "${BOLD}║            AlphaFold 3 for Mac — Installer                   ║${NC}"
    echo -e "${BOLD}║            Protein Structure Prediction on Apple Silicon      ║${NC}"
    echo -e "${BOLD}║                                                              ║${NC}"
    echo -e "${BOLD}║            Version ${version}                                      ║${NC}"
    echo -e "${BOLD}║                                                              ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  This installer will:"
    echo "    1. Check system requirements"
    echo "    2. Configure paths for model weights and databases"
    echo "    3. Install Python dependencies"
    echo "    4. Build HMMER 3.4 (genetic search tool)"
    echo "    5. Build the web interface (optional)"
    echo "    6. Create a Desktop launcher"
    echo ""
    read -rp "  Press Enter to continue or Ctrl+C to cancel... "
    echo ""
}

# ── Phase 1b: Install Mode Selection ────────────────────────────────────────

_phase_install_mode() {
    echo -e "${BOLD}─── Install Mode ──────────────────────────────────────────${NC}"
    echo ""
    echo "  [1] Full install (Recommended)"
    echo "      Model + CLI + Web UI — everything you need"
    echo "      (Requires Node.js >= 20)"
    echo ""
    echo "  [2] CLI-only install"
    echo "      Model + CLI only — no web interface"
    echo "      (No Node.js required; skips frontend build)"
    echo ""
    local choice
    choice=$(_prompt "  Choose" "1")
    echo ""

    case "$choice" in
        2) INSTALL_MODE="cli-only" ;;
        *) INSTALL_MODE="full" ;;
    esac
}

# ── Phase 2: System Checks ──────────────────────────────────────────────────

_check_with_retry() {
    local label="$1"
    local check_fn="$2"
    local fix_msg="${3:-}"
    local retries=3
    local attempt=0

    while (( attempt < retries )); do
        if $check_fn; then
            return 0
        fi
        attempt=$((attempt + 1))
        if (( attempt < retries )) && [[ -n "$fix_msg" ]]; then
            echo -e "       ${DIM}Fix: ${fix_msg}${NC}"
            echo ""
            read -rp "  Press Enter after installing, or Ctrl+C to cancel... "
            echo ""
        fi
    done
    return 1
}

_check_macos_arm() {
    local os_name arch chip_info mem_gb
    os_name=$(uname -s)
    arch=$(uname -m)
    if [[ "$os_name" != "Darwin" || "$arch" != "arm64" ]]; then
        _fail "Requires macOS on Apple Silicon (got: $os_name $arch)"
        return 1
    fi
    chip_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
    _info "macOS ARM64 ($chip_info, ${mem_gb} GB)"
    return 0
}

_check_xcode_cli() {
    if xcode-select -p &>/dev/null; then
        local xcode_ver
        xcode_ver=$(xcode-select --version 2>/dev/null | head -1 || echo "installed")
        _info "Xcode Command Line Tools ($xcode_ver)"
        return 0
    fi
    _warn "Xcode Command Line Tools not found"
    if [[ "$DRY_RUN" == true ]]; then
        _info "Xcode Command Line Tools (dry run — would install)"
        return 0
    fi
    echo "       Installing (this may take a few minutes)..."
    xcode-select --install 2>/dev/null || true
    # Wait for installation — xcode-select --install is async
    echo "       Waiting for installation to complete..."
    until xcode-select -p &>/dev/null; do
        sleep 5
    done
    _info "Xcode Command Line Tools installed"
    return 0
}

_check_python() {
    # Prefer python3.12 explicitly, then fall back to python3
    local py_cmd=""
    if command -v python3.12 &>/dev/null; then
        py_cmd="python3.12"
    elif command -v python3 &>/dev/null; then
        py_cmd="python3"
    fi

    if [[ -z "$py_cmd" ]]; then
        _fail "Python 3.12+ not found"
        return 1
    fi

    local py_version
    py_version=$($py_cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    local py_major py_minor
    py_major=$(echo "$py_version" | cut -d. -f1)
    py_minor=$(echo "$py_version" | cut -d. -f2)

    if (( py_major < 3 || (py_major == 3 && py_minor < 12) )); then
        _fail "Python 3.12+ required (found $py_version)"
        return 1
    fi

    _info "Python $py_version ($py_cmd)"
    return 0
}

_check_node() {
    if ! command -v node &>/dev/null; then
        _fail "Node.js not found"
        return 1
    fi

    local node_version
    node_version=$(node --version 2>/dev/null | grep -oE '[0-9]+' | head -1)
    if (( node_version < 20 )); then
        _fail "Node.js >= 20 required (found v$node_version)"
        return 1
    fi

    _info "Node.js $(node --version)"
    return 0
}

_check_homebrew() {
    if command -v brew &>/dev/null; then
        _info "Homebrew $(brew --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo 'installed')"
        return 0
    fi
    _warn "Homebrew not found"
    echo "       Homebrew is needed to install build tools (autoconf, automake)."
    echo ""
    read -rp "  Install Homebrew now? [Y/n]: " install_brew
    if [[ "$(_tolower "$install_brew")" == "n" ]]; then
        return 1
    fi
    if [[ "$DRY_RUN" == true ]]; then
        _info "Homebrew (dry run — would install)"
        return 0
    fi
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Ensure brew is on PATH for current session
    eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
    if command -v brew &>/dev/null; then
        _info "Homebrew installed"
        return 0
    fi
    _fail "Homebrew installation failed"
    return 1
}

_check_autotools() {
    local missing=()
    command -v autoconf &>/dev/null || missing+=("autoconf")
    command -v automake &>/dev/null || missing+=("automake")

    if (( ${#missing[@]} == 0 )); then
        _info "autoconf + automake"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        _info "autoconf + automake (dry run — would install ${missing[*]})"
        return 0
    fi
    echo -e "  ${YELLOW}[..]${NC} Installing ${missing[*]} via Homebrew..."
    if brew install "${missing[@]}" &>/dev/null; then
        _info "autoconf + automake (installed via Homebrew)"
        return 0
    fi
    _fail "Failed to install ${missing[*]}"
    return 1
}

_check_disk_space() {
    # Check free space in home directory (GB)
    local free_kb
    free_kb=$(df -k "$HOME" | tail -1 | awk '{print $4}')
    local free_gb=$(( free_kb / 1048576 ))

    if (( free_gb < 5 )); then
        _warn "Low disk space: ${free_gb} GB free (recommend >= 5 GB)"
    else
        _info "Disk space: ${free_gb} GB free"
    fi
    return 0
}

_phase_system_checks() {
    echo -e "${BOLD}─── System Checks ─────────────────────────────────────────${NC}"
    echo ""

    # Hard requirements
    _check_macos_arm || { echo ""; echo -e "  ${RED}Cannot continue: macOS ARM64 is required.${NC}"; exit 1; }
    _check_xcode_cli || { echo ""; echo -e "  ${RED}Cannot continue: Xcode CLI Tools are required.${NC}"; exit 1; }
    _check_with_retry "Python" _check_python "brew install python@3.12" || {
        echo ""; echo -e "  ${RED}Cannot continue: Python 3.12+ is required.${NC}"; exit 1;
    }

    # Node.js — only in full mode
    if [[ "$INSTALL_MODE" == "full" ]]; then
        _check_with_retry "Node.js" _check_node "brew install node@20" || {
            echo ""; echo -e "  ${RED}Cannot continue: Node.js >= 20 is required for full install.${NC}";
            echo -e "  ${DIM}Tip: Re-run the installer and choose CLI-only mode to skip this.${NC}";
            exit 1;
        }
    else
        echo -e "  ${DIM}[--]${NC} Node.js (skipped — CLI-only mode)"
    fi

    _check_homebrew || { echo ""; echo -e "  ${RED}Cannot continue: Homebrew is required for building HMMER.${NC}"; exit 1; }
    _check_autotools || { echo ""; echo -e "  ${RED}Cannot continue: autoconf/automake are required.${NC}"; exit 1; }
    _check_disk_space

    echo ""
}

# ── Phase 3: Hardcoded Path Scan ────────────────────────────────────────────

_phase_path_scan() {
    echo -e "${BOLD}─── Preflight Scan ────────────────────────────────────────${NC}"
    echo ""

    # Regex stored in a variable so the grep calls below don't contain the
    # literal patterns — avoids self-matching when the scanner reads this file.
    # The variable definition line is filtered by the inline PATH_SCAN_REGEX marker.
    local scan_re='/Volumes/|/Users/[a-z][a-z0-9_-]*/|/home/[a-z][a-z0-9_-]*/' # PATH_SCAN_REGEX

    local leaked
    leaked=$(
        grep -rn --include='*.py' --include='*.ts' --include='*.tsx' \
            -E "$scan_re" \
            "$REPO_ROOT/src/" "$REPO_ROOT/frontend/src/" "$REPO_ROOT/run_alphafold_mlx.py" 2>/dev/null;
        grep -rn --include='*.sh' \
            -E "$scan_re" \
            "$REPO_ROOT/scripts/" 2>/dev/null \
        || true
    )
    # Filter known-safe patterns (including this file's regex definition line)
    leaked=$(echo "$leaked" \
        | grep -v 'PATH_SCAN_REGEX' \
        | grep -v 'CLAUDE\.md' \
        | grep -v 'AGENTS\.md' \
        | grep -v 'MEMORY\.md' \
        | grep -v '__pycache__' \
        | grep -v '\.pyc' \
        | grep -v 'node_modules' \
        | grep -v 'expanduser' \
        | grep -v 'Path\.home()' \
        | grep -v '/\.' || true)

    if [[ -n "$leaked" ]]; then
        _warn "Found developer-specific paths in production code:"
        echo "$leaked" | head -10 | sed 's/^/       /'
        echo "       (Continuing anyway...)"
    else
        _info "No developer-specific paths in production code"
    fi
    echo ""
}

# ── Phase 4: Configuration Prompts ──────────────────────────────────────────

_validate_weights_dir() {
    local dir="$1"
    # Check for AF3 weight file patterns (matching run_alphafold_mlx.py:validate_weights_directory)
    local patterns=(
        "af3.bin.zst"
        "af3.bin"
        "af3.0.bin.zst"
        "af3.bin.zst.0"
    )
    for pat in "${patterns[@]}"; do
        if [[ -f "$dir/$pat" ]]; then
            return 0
        fi
    done
    # Check glob patterns for sharded files (af3. prefix required to match runtime validator)
    if compgen -G "$dir/af3.*.bin.zst" >/dev/null 2>&1 || \
       compgen -G "$dir/af3.bin.zst.*" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Validate database layout and print status. Returns 0 if all required files
# are present, 1 if any are missing. Prompts user to continue on missing files.
_validate_db_layout() {
    local dir="$1"
    _info "Databases directory: $dir"
    local expected_files=(
        "uniref90_2022_05.fa"
        "mgy_clusters_2022_05.fa"
        "bfd-first_non_consensus_sequences.fasta"
        "uniprot_all_2021_04.fa"
        "pdb_seqres_2022_09_28.fasta"
    )
    local missing_files=()
    for f in "${expected_files[@]}"; do
        if [[ -f "$dir/$f" ]]; then
            _info "$f"
        else
            missing_files+=("$f")
        fi
    done
    # mmCIF directory — check alternatives, require at least 1 .cif file
    local mmcif_ok=false
    for mmcif_candidate in \
        "$dir/mmcif_files" \
        "$dir/pdb_2022_09_28_mmcif_files/mmcif_files" \
        "$dir/pdb_2022_09_28_mmcif_files"; do
        if [[ -d "$mmcif_candidate" ]]; then
            local cif_count
            cif_count=$(find "$mmcif_candidate" \( -name '*.cif' -o -name '*.cif.gz' \) -print 2>/dev/null | head -1 | wc -l | tr -d '[:space:]')
            if (( cif_count > 0 )); then
                local mmcif_label="${mmcif_candidate#$dir/}"
                _info "${mmcif_label}/"
                mmcif_ok=true
                break
            fi
        fi
    done
    if [[ "$mmcif_ok" != true ]]; then
        missing_files+=("mmcif_files/ (or pdb_2022_09_28_mmcif_files/) with .cif content")
    fi
    # RNA databases (optional — report but don't add to missing)
    local rna_files=(
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    )
    local rna_found=0
    for f in "${rna_files[@]}"; do
        [[ -f "$dir/$f" ]] && rna_found=$((rna_found + 1))
    done
    if (( rna_found == ${#rna_files[@]} )); then
        _info "RNA databases (${rna_found}/${#rna_files[@]})"
    else
        echo -e "  ${DIM}[--]${NC} RNA databases (${rna_found}/${#rna_files[@]} — optional)"
    fi
    if (( ${#missing_files[@]} > 0 )); then
        echo ""
        _warn "Required database files not found:"
        for f in "${missing_files[@]}"; do
            echo "       - $f"
        done
        return 1
    fi
    return 0
}

_phase_config_prompts() {
    echo -e "${BOLD}─── Model Weights ─────────────────────────────────────────${NC}"
    echo ""
    echo "  AlphaFold 3 requires model weights from Google DeepMind."
    echo ""
    echo "    How to obtain weights:"
    echo "    1. Visit: https://forms.gle/svvpY4u2jsHEwWYS6"
    echo "    2. Accept the license terms"
    echo "    3. Download the weight files (~973 MB compressed)"
    echo "    4. Place them in a directory of your choice"
    echo ""
    WEIGHTS_DIR=$(_prompt "  Enter weights directory" "~/.alphafold3/weights/model")
    WEIGHTS_DIR=$(_expand_path "$WEIGHTS_DIR")
    echo ""

    WEIGHTS_FOUND=false
    if _validate_weights_dir "$WEIGHTS_DIR"; then
        _info "Weight files found at $WEIGHTS_DIR"
        WEIGHTS_FOUND=true
    else
        _warn "No weight files found at $WEIGHTS_DIR"
        echo "       Recognized formats: af3.bin.zst, af3.bin, or sharded variants"
        echo "       (af3.0.bin.zst, af3.bin.zst.0, etc.)"
        echo ""
        echo "       The server will start but jobs will fail until weights are installed."
        echo "       You can download weights later and place them in this directory."
        echo ""
        echo "       Continuing installation..."
    fi
    echo ""

    # Databases (optional)
    echo -e "${BOLD}─── Genetic Databases (Optional) ──────────────────────────${NC}"
    echo ""
    echo "  For MSA-enhanced predictions, AlphaFold 3 can search genetic"
    echo "  databases (~252 GB download, ~630 GB on disk)."
    echo ""
    echo "  Without databases, the server runs in sequence-only mode"
    echo "  which still produces high-quality predictions."
    echo ""
    echo "  [1] I already have the databases"
    echo "      Enter the path to your existing databases directory"
    echo ""
    echo "  [2] Download databases now (~252 GB download, ~630 GB on disk)"
    echo "      Downloads all databases from Google Cloud Storage"
    echo ""
    echo "  [3] Skip for now (sequence-only mode)"
    echo "      You can download databases later"
    echo ""
    local db_choice
    db_choice=$(_prompt "  Choose" "3")

    case "$db_choice" in
        1)
            # Option 1: Existing path — validate layout
            DB_DIR=$(_prompt "  Enter databases directory" "~/public_databases")
            DB_DIR=$(_expand_path "$DB_DIR")
            if [[ -d "$DB_DIR" ]]; then
                if ! _validate_db_layout "$DB_DIR"; then
                    echo ""
                    read -rp "  Continue anyway? [Y/n]: " continue_choice
                    if [[ "$(_tolower "$continue_choice")" == "n" ]]; then
                        echo "  Set up databases and re-run the installer."
                        exit 1
                    fi
                fi
            else
                _warn "Directory does not exist: $DB_DIR"
                echo "       It will be used when the directory is created."
            fi
            ;;
        2)
            # Option 2: Download now
            DB_DIR=$(_prompt "  Enter target directory" "~/public_databases")
            DB_DIR=$(_expand_path "$DB_DIR")
            echo ""

            # Disk space check on target volume
            if [[ "$DRY_RUN" != true ]]; then
                mkdir -p "$DB_DIR" 2>/dev/null || true
            fi
            local target_vol_dir="${DB_DIR}"
            # Use parent dir for space check if target doesn't exist yet
            if [[ ! -d "$target_vol_dir" ]]; then
                target_vol_dir=$(dirname "$target_vol_dir")
            fi
            if [[ -d "$target_vol_dir" ]]; then
                local free_kb
                free_kb=$(df -k "$target_vol_dir" | tail -1 | awk '{print $4}')
                local free_gb=$(( free_kb / 1048576 ))
                if (( free_gb < 650 )); then
                    _warn "Low disk space: ${free_gb} GB free (recommend >= 650 GB)"
                    echo "       Databases require approximately 630 GB when fully unpacked."
                    echo ""
                    read -rp "  Continue anyway? [Y/n]: " space_choice
                    if [[ "$(_tolower "$space_choice")" == "n" ]]; then
                        echo "  Skipping database download."
                        DB_DIR=""
                        DOWNLOAD_DATABASES=false
                    fi
                else
                    _info "Disk space: ${free_gb} GB free"
                fi
            fi

            # Tool checks (only if download still planned)
            if [[ -n "${DB_DIR:-}" ]]; then
                local dl_missing=()
                command -v curl &>/dev/null || dl_missing+=("curl")
                command -v tar &>/dev/null  || dl_missing+=("tar")
                if ! command -v zstd &>/dev/null; then
                    echo -e "  ${YELLOW}[..]${NC} zstd not found — required for decompression"
                    if command -v brew &>/dev/null; then
                        echo ""
                        read -rp "  Install zstd via Homebrew? [Y/n]: " zstd_choice
                        if [[ "$(_tolower "$zstd_choice")" != "n" ]]; then
                            if [[ "$DRY_RUN" == true ]]; then
                                _info "zstd (dry run — would install)"
                            else
                                if brew install zstd &>/dev/null; then
                                    _info "zstd installed"
                                else
                                    dl_missing+=("zstd")
                                fi
                            fi
                        else
                            dl_missing+=("zstd")
                        fi
                    else
                        dl_missing+=("zstd")
                    fi
                fi
                if (( ${#dl_missing[@]} > 0 )); then
                    _fail "Missing required tools: ${dl_missing[*]}"
                    echo "       Install with: brew install ${dl_missing[*]}"
                    echo "       Skipping database download."
                    DB_DIR=""
                    DOWNLOAD_DATABASES=false
                else
                    DOWNLOAD_DATABASES=true
                    _info "Database download will start after installation completes"
                fi
            fi
            ;;
        *)
            # Option 3: Skip
            DB_DIR=""
            echo -e "  ${DIM}[--]${NC} Databases skipped (sequence-only mode)"
            ;;
    esac
    echo ""

    # Data directory
    echo -e "${BOLD}─── Data Directory ────────────────────────────────────────${NC}"
    echo ""
    echo "  Job history and results are stored in a persistent directory."
    echo ""
    DATA_DIR=$(_prompt "  Enter data directory" "~/.alphafold3_mlx/data")
    DATA_DIR=$(_expand_path "$DATA_DIR")
    if [[ "$DRY_RUN" != true ]]; then
        mkdir -p "$DATA_DIR" 2>/dev/null || true
    fi
    _info "Data directory: $DATA_DIR"
    echo ""

    # Server port
    echo -e "${BOLD}─── Server Port ───────────────────────────────────────────${NC}"
    echo ""
    echo "  The web server will listen on this port."
    echo ""
    while true; do
        PORT=$(_prompt "  Enter port number" "8642")
        # Validate: numeric
        if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
            _warn "Port must be a number"
            continue
        fi
        # Validate: range
        if (( PORT < 1024 || PORT > 65535 )); then
            _warn "Port must be between 1024 and 65535"
            continue
        fi
        # Validate: not in use (skip in dry-run — port won't actually be bound)
        if [[ "$DRY_RUN" != true ]] && lsof -ti:"$PORT" &>/dev/null; then
            _warn "Port $PORT is already in use"
            continue
        fi
        break
    done
    _info "Port: $PORT"
    echo ""
}

# ── Phase 5: Installation Steps ─────────────────────────────────────────────

_install_uv() {
    if command -v uv &>/dev/null; then
        return 0
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure uv is on PATH for current session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null
}

_create_venv() {
    if [[ -f "$REPO_ROOT/.venv/bin/python3" ]]; then
        # Check if existing venv is Python 3.12+ AND native arm64
        local existing_ver
        existing_ver=$("$REPO_ROOT/.venv/bin/python3" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        local existing_minor
        existing_minor=$(echo "$existing_ver" | cut -d. -f2)
        local existing_arch
        existing_arch=$("$REPO_ROOT/.venv/bin/python3" -c 'import platform; print(platform.machine())' 2>/dev/null || echo "unknown")
        if (( existing_minor >= 12 )) && [[ "$existing_arch" == "arm64" ]]; then
            return 0
        fi
        # Wrong version or architecture — recreate
        rm -rf "$REPO_ROOT/.venv"
    fi
    # Use only-managed to ensure we get a native arm64 Python, avoiding any
    # x86_64 Homebrew Python that might be installed under /usr/local/.
    cd "$REPO_ROOT" && uv venv --python 3.12 --python-preference only-managed .venv
}

_clean_resource_forks() {
    # ExFAT/NTFS volumes create ._* resource fork files that break C++ builds.
    # CMake GLOB picks up ._*.cc files as source, causing compilation failures.
    cd "$REPO_ROOT" && find src/ -name '._*' -delete 2>/dev/null || true
}

_install_python_deps() {
    cd "$REPO_ROOT" && uv sync --frozen --all-extras
}

_build_ccd() {
    cd "$REPO_ROOT" && .venv/bin/python3 -c 'from alphafold3.build_data import build_data; build_data()'
}

_build_hmmer() {
    "$REPO_ROOT/scripts/build_hmmer_macos.sh" "--prefix=$HMMER_PREFIX"
}

_install_frontend_deps() {
    npm ci --prefix "$REPO_ROOT/frontend"
}

_build_frontend() {
    npm run build --prefix "$REPO_ROOT/frontend"
}

_phase_install() {
    local total_steps
    if [[ "$INSTALL_MODE" == "full" ]]; then
        total_steps=7
    else
        total_steps=5
    fi

    echo -e "${BOLD}─── Installing ────────────────────────────────────────────${NC}"
    echo ""

    _maybe_run_step 1 "$total_steps" "Installing uv package manager..." _install_uv || {
        echo -e "  ${RED}Failed to install uv. Try manually: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        exit 1
    }

    _maybe_run_step 2 "$total_steps" "Creating Python environment..." _create_venv || {
        echo -e "  ${RED}Failed to create Python environment.${NC}"
        exit 1
    }

    # Clean macOS resource fork files that break C++ builds on ExFAT/NTFS volumes
    if [[ "$DRY_RUN" != true ]]; then
        _clean_resource_forks
    fi

    _maybe_run_step 3 "$total_steps" "Installing Python dependencies..." _install_python_deps || {
        echo -e "  ${RED}Failed to install Python dependencies.${NC}"
        exit 1
    }

    _maybe_run_step 4 "$total_steps" "Building chemical components database..." _build_ccd || {
        echo -e "  ${RED}Failed to build chemical components database.${NC}"
        exit 1
    }

    _maybe_run_step 5 "$total_steps" "Building HMMER 3.4 with seq_limit patch..." _build_hmmer || {
        echo -e "  ${RED}Failed to build HMMER. Try manually: ./scripts/build_hmmer_macos.sh${NC}"
        exit 1
    }

    if [[ "$INSTALL_MODE" == "full" ]]; then
        _maybe_run_step 6 "$total_steps" "Installing frontend dependencies..." _install_frontend_deps || {
            echo -e "  ${RED}Failed to install frontend dependencies.${NC}"
            exit 1
        }

        _maybe_run_step 7 "$total_steps" "Building web interface..." _build_frontend || {
            echo -e "  ${RED}Failed to build web interface.${NC}"
            exit 1
        }
    fi

    echo ""
}

# ── Phase 5b: Database Download ─────────────────────────────────────────────

_phase_database_download() {
    if [[ "$DOWNLOAD_DATABASES" != true ]]; then
        return 0
    fi

    echo -e "${BOLD}─── Downloading Genetic Databases ─────────────────────────${NC}"
    echo ""
    echo "  This will download ~252 GB and unpack to ~630 GB."
    echo "  The download can be resumed if interrupted."
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        _info "Database download (dry run — would run):"
        echo "       bash $REPO_ROOT/fetch_databases.sh $DB_DIR --non-interactive"
        echo ""
        DB_DOWNLOAD_SUCCEEDED=true
        return 0
    fi

    # Run the fetch script; capture exit code without aborting (set -e safe)
    if bash "$REPO_ROOT/fetch_databases.sh" "$DB_DIR" --non-interactive; then
        DB_DOWNLOAD_SUCCEEDED=true
    else
        DB_DOWNLOAD_SUCCEEDED=false
    fi

    # Validate the downloaded layout
    echo ""
    if [[ -d "$DB_DIR" ]] && _validate_db_layout "$DB_DIR"; then
        _info "All required databases are in place"
    else
        echo ""
        _warn "Database download is incomplete."
        echo "       The server will default to sequence-only mode."
        echo ""
        echo "       To finish downloading later, run:"
        echo "         bash fetch_databases.sh $DB_DIR"
        echo ""
        read -rp "  Keep partial database path in config anyway? [y/N]: " keep_choice
        keep_choice=$(_tolower "$keep_choice")
        if [[ "$keep_choice" != "y" ]]; then
            DB_DIR=""
            _info "Database path cleared (sequence-only mode)"
        else
            _warn "Keeping partial database path — jobs may fail until download completes"
        fi
    fi
    echo ""
}

# ── Phase 6: Write Configuration ────────────────────────────────────────────

_phase_write_config() {
    echo -e "${BOLD}─── Saving Configuration ──────────────────────────────────${NC}"
    echo ""

    local install_dir
    install_dir=$(_expand_path "$REPO_ROOT")
    local gen_date
    gen_date=$(date '+%Y-%m-%d %H:%M:%S')

    local config_body
    config_body=$(cat <<CONFIGEOF
# AlphaFold 3 MLX Configuration
# Generated by install.sh on $gen_date
# Edit this file to change settings, then restart the server.

AF3_WEIGHTS_DIR="$WEIGHTS_DIR"
AF3_DATA_DIR="$DATA_DIR"
AF3_DB_DIR="${DB_DIR:-}"
AF3_PORT="$PORT"
AF3_INSTALL_DIR="$install_dir"
AF3_JACKHMMER="$HMMER_PREFIX/bin/jackhmmer"
AF3_HMMSEARCH="$HMMER_PREFIX/bin/hmmsearch"
AF3_HMMBUILD="$HMMER_PREFIX/bin/hmmbuild"
AF3_NHMMER="$HMMER_PREFIX/bin/nhmmer"
AF3_HMMALIGN="$HMMER_PREFIX/bin/hmmalign"
CONFIGEOF
    )

    if [[ "$DRY_RUN" == true ]]; then
        _info "Configuration (dry run — not written):"
        echo "$config_body" | sed 's/^/       /'
    else
        mkdir -p "$CONFIG_DIR"
        echo "$config_body" > "$CONFIG_FILE"
        _info "Configuration saved to $CONFIG_FILE"
    fi
    echo ""
}

# ── Phase 7: Generate Launch Artifacts ──────────────────────────────────────

_phase_generate_launchers() {
    echo -e "${BOLD}─── Creating Launchers ────────────────────────────────────${NC}"
    echo ""

    local install_dir
    install_dir=$(_expand_path "$REPO_ROOT")

    # Desktop launcher
    if [[ "$INSTALL_MODE" == "full" ]]; then
        local desktop_launcher="$HOME/Desktop/AlphaFold3.command"
        if [[ "$DRY_RUN" == true ]]; then
            _info "Desktop launcher (dry run — not written): $desktop_launcher"
        else
            cat > "$desktop_launcher" <<LAUNCHEREOF
#!/usr/bin/env bash
# Double-click this file to start AlphaFold 3 MLX.
# Close this terminal window to stop the server.
cd "$install_dir"
exec ./scripts/start.sh
LAUNCHEREOF
            chmod +x "$desktop_launcher"
            _info "Desktop launcher: $desktop_launcher"
        fi
    else
        echo -e "  ${DIM}[--]${NC} Desktop launcher skipped (CLI-only mode)"
    fi

    echo ""
}

# ── Phase 8: Completion Screen ──────────────────────────────────────────────

_phase_complete() {
    local install_dir
    install_dir=$(_expand_path "$REPO_ROOT")

    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                                                              ║${NC}"
    echo -e "${BOLD}║            Installation Complete!                             ║${NC}"
    echo -e "${BOLD}║                                                              ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  Configuration:  $CONFIG_FILE"
    echo "  Weights:        $WEIGHTS_DIR"
    echo "  Data:           $DATA_DIR"
    if [[ -n "${DB_DIR:-}" ]]; then
        echo "  Databases:      $DB_DIR"
    else
        echo "  Databases:      (not configured — sequence-only mode)"
    fi
    echo "  HMMER:          $HMMER_PREFIX/bin/jackhmmer"
    echo "  Port:           $PORT"
    echo ""

    # Weights reminder
    if [[ "$WEIGHTS_FOUND" != "true" ]]; then
        echo -e "  ${BOLD}─── Action Required: Model Weights ──────────────────────${NC}"
        echo ""
        echo -e "  ${YELLOW}[!!]${NC} No weight files found."
        echo "       The server will start, but prediction jobs will fail"
        echo "       until weights are installed."
        echo ""
        echo "       Download af3.bin.zst from Google DeepMind:"
        echo "         https://forms.gle/svvpY4u2jsHEwWYS6"
        echo ""
        echo "       Then place it in:"
        echo "         $WEIGHTS_DIR/"
        echo ""
        echo "       Recognized formats: af3.bin.zst, af3.bin,"
        echo "       or sharded variants (af3.0.bin.zst, af3.bin.zst.0)"
        echo ""
    fi

    # Database download warning (if user kept partial path)
    if [[ -n "${DB_DIR:-}" ]] && [[ "$DOWNLOAD_DATABASES" == true ]] && [[ "$DB_DOWNLOAD_SUCCEEDED" != true ]]; then
        echo -e "  ${BOLD}─── Warning: Incomplete Databases ───────────────────────${NC}"
        echo ""
        echo -e "  ${YELLOW}[!!]${NC} Database download did not complete successfully."
        echo "       Jobs requiring MSA search may fail."
        echo ""
        echo "       To finish downloading, run:"
        echo "         bash fetch_databases.sh $DB_DIR"
        echo ""
    fi

    echo -e "  ${BOLD}─── How to Start ────────────────────────────────────────${NC}"
    echo ""

    if [[ "$INSTALL_MODE" == "full" ]]; then
        echo "  Option 1:  Double-click \"AlphaFold3\" on your Desktop"
        echo ""
        echo "  Option 2:  From Terminal:"
        echo "             cd $install_dir && ./scripts/start.sh"
        echo ""
        echo "  Option 3:  CLI only (no web UI):"
        echo "             cd $install_dir && ./scripts/alphafold3.sh \\"
        echo "                 --input input.json --output_dir output/"
        echo ""
        echo "  Then open:  http://127.0.0.1:$PORT"
    else
        echo "  Option 1:  Start API server (no web UI):"
        echo "             cd $install_dir && ./scripts/start.sh"
        echo ""
        echo "  Option 2:  CLI only:"
        echo "             cd $install_dir && ./scripts/alphafold3.sh \\"
        echo "                 --input input.json --output_dir output/"
        echo ""
        echo "  API available at:  http://127.0.0.1:$PORT/api"
    fi
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    # Ensure we're running from the repo root
    if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
        echo "Error: Cannot find pyproject.toml. Are you running from the project directory?" >&2
        exit 1
    fi

    _phase_welcome
    _phase_install_mode
    _phase_system_checks
    _phase_path_scan
    _phase_config_prompts
    _phase_install
    _phase_database_download
    _phase_write_config
    _phase_generate_launchers
    _phase_complete
}

# Only run main when executed directly, not when sourced (for testing).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
