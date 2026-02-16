#!/usr/bin/env bash
# Unit tests for scripts/install.sh functions and scripts/_load_config.sh.
#
# Runs individual functions in isolation without triggering real installs.
# All tests use temp directories and clean up after themselves.
#
# Usage:
#   bash tests/test_install.sh                  # run all tests
#   bash tests/test_install.sh test_expand_path # run one suite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Test framework ───────────────────────────────────────────────────────────

PASS=0
FAIL=0
ERRORS=()

_assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        echo "  [OK] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        echo "         expected: '$expected'"
        echo "         actual:   '$actual'"
        FAIL=$((FAIL + 1))
        ERRORS+=("$label")
    fi
}

_assert_ne() {
    local label="$1" not_expected="$2" actual="$3"
    if [[ "$not_expected" != "$actual" ]]; then
        echo "  [OK] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label: should NOT be '$not_expected'"
        FAIL=$((FAIL + 1))
        ERRORS+=("$label")
    fi
}

_assert_match() {
    local label="$1" pattern="$2" actual="$3"
    if [[ "$actual" =~ $pattern ]]; then
        echo "  [OK] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label: '$actual' does not match /$pattern/"
        FAIL=$((FAIL + 1))
        ERRORS+=("$label")
    fi
}

_assert_exit() {
    local label="$1" expected_code="$2"
    shift 2
    local actual_code=0
    "$@" >/dev/null 2>&1 || actual_code=$?
    _assert_eq "$label" "$expected_code" "$actual_code"
}

# ── Source install.sh functions without running main() ───────────────────────

# install.sh uses a BASH_SOURCE guard, so sourcing only defines functions.
# shellcheck source=../scripts/install.sh
source "$REPO_ROOT/scripts/install.sh"
# Also source the config loader.
# shellcheck source=../scripts/_load_config.sh
source "$REPO_ROOT/scripts/_load_config.sh"

# ── Tests: _expand_path ─────────────────────────────────────────────────────

test_expand_path() {
    echo ""
    echo "=== _expand_path ==="

    _assert_eq "tilde/foo expands to \$HOME/foo" \
        "$HOME/foo" "$(_expand_path "~/foo")"

    _assert_eq "bare tilde expands to \$HOME" \
        "$HOME" "$(_expand_path "~")"

    _assert_eq "tilde/deeply/nested expands" \
        "$HOME/a/b/c" "$(_expand_path "~/a/b/c")"

    _assert_eq "absolute path stays absolute" \
        "/tmp" "$(_expand_path "/tmp")"

    _assert_eq "existing dir resolves via cd+pwd" \
        "/tmp" "$(_expand_path "/tmp")"

    _assert_eq "non-existent path resolves via python3" \
        "/tmp/nonexistent_af3_test" "$(_expand_path "/tmp/nonexistent_af3_test")"

    # Security: shell injection must not execute.
    local injected
    injected=$(_expand_path '$(whoami)')
    _assert_match "no eval injection — literal \$(whoami)" \
        'whoami' "$injected"
    # Should be a path like /cwd/$(whoami), NOT the output of whoami.
    _assert_match "injection returns path not username" \
        '^\/' "$injected"

    # Path with spaces
    local tmpsp
    tmpsp=$(mktemp -d "/tmp/af3 test space.XXXXXX")
    _assert_eq "dir with spaces resolves" \
        "$tmpsp" "$(_expand_path "$tmpsp")"
    rmdir "$tmpsp"
}

# ── Tests: _validate_weights_dir ─────────────────────────────────────────────

test_validate_weights_dir() {
    echo ""
    echo "=== _validate_weights_dir ==="

    local tmpw
    tmpw=$(mktemp -d)

    # Single file: af3.bin.zst
    touch "$tmpw/af3.bin.zst"
    _assert_exit "detects af3.bin.zst" 0 _validate_weights_dir "$tmpw"
    rm "$tmpw/af3.bin.zst"

    # Single file: af3.bin (uncompressed)
    touch "$tmpw/af3.bin"
    _assert_exit "detects af3.bin" 0 _validate_weights_dir "$tmpw"
    rm "$tmpw/af3.bin"

    # Sharded: af3.0.bin.zst
    touch "$tmpw/af3.0.bin.zst"
    _assert_exit "detects sharded af3.0.bin.zst" 0 _validate_weights_dir "$tmpw"
    rm "$tmpw/af3.0.bin.zst"

    # Sharded: af3.bin.zst.0
    touch "$tmpw/af3.bin.zst.0"
    _assert_exit "detects sharded af3.bin.zst.0" 0 _validate_weights_dir "$tmpw"
    rm "$tmpw/af3.bin.zst.0"

    # Multiple shards
    touch "$tmpw/af3.0.bin.zst" "$tmpw/af3.1.bin.zst"
    _assert_exit "detects multiple shards" 0 _validate_weights_dir "$tmpw"
    rm "$tmpw/af3.0.bin.zst" "$tmpw/af3.1.bin.zst"

    # Empty directory
    _assert_exit "empty dir fails" 1 _validate_weights_dir "$tmpw"

    # Unrelated files
    touch "$tmpw/readme.txt"
    _assert_exit "unrelated file fails" 1 _validate_weights_dir "$tmpw"
    rm "$tmpw/readme.txt"

    # Non-af3 binary files (must not match — runtime validator requires af3. prefix)
    touch "$tmpw/firmware.bin"
    _assert_exit "non-af3 .bin fails" 1 _validate_weights_dir "$tmpw"
    rm "$tmpw/firmware.bin"

    touch "$tmpw/other_model.bin.zst"
    _assert_exit "non-af3 .bin.zst fails" 1 _validate_weights_dir "$tmpw"
    rm "$tmpw/other_model.bin.zst"

    # Non-existent directory
    _assert_exit "non-existent dir fails" 1 _validate_weights_dir "$tmpw/nope"

    rm -rf "$tmpw"
}

# ── Tests: _load_af3_config ─────────────────────────────────────────────────

test_load_af3_config() {
    echo ""
    echo "=== _load_af3_config ==="

    local cfg
    cfg=$(mktemp)

    # Basic key=value
    cat > "$cfg" <<'EOF'
AF3_PORT="9999"
AF3_WEIGHTS_DIR="/some/path"
EOF
    unset AF3_PORT AF3_WEIGHTS_DIR 2>/dev/null || true
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "loads AF3_PORT" "9999" "${AF3_PORT:-}"
    _assert_eq "loads AF3_WEIGHTS_DIR" "/some/path" "${AF3_WEIGHTS_DIR:-}"
    unset AF3_PORT AF3_WEIGHTS_DIR

    # Blocked (non-allowlisted) key
    cat > "$cfg" <<'EOF'
AF3_PORT="1234"
EVIL_KEY="bad"
PATH="/hijacked"
EOF
    unset AF3_PORT EVIL_KEY 2>/dev/null || true
    local saved_path="$PATH"
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "blocked key EVIL_KEY not exported" "" "${EVIL_KEY:-}"
    _assert_eq "PATH not hijacked" "$saved_path" "$PATH"
    _assert_eq "allowed key still loaded" "1234" "${AF3_PORT:-}"
    unset AF3_PORT EVIL_KEY 2>/dev/null || true

    # Comments and blank lines
    cat > "$cfg" <<'EOF'
# This is a comment
AF3_DATA_DIR="/data"

  # Indented comment
AF3_DB_DIR="/db"
EOF
    unset AF3_DATA_DIR AF3_DB_DIR 2>/dev/null || true
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "skips comments, loads AF3_DATA_DIR" "/data" "${AF3_DATA_DIR:-}"
    _assert_eq "skips blank lines, loads AF3_DB_DIR" "/db" "${AF3_DB_DIR:-}"
    unset AF3_DATA_DIR AF3_DB_DIR

    # Unquoted values
    cat > "$cfg" <<'EOF'
AF3_PORT=5555
EOF
    unset AF3_PORT 2>/dev/null || true
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "unquoted value loaded" "5555" "${AF3_PORT:-}"
    unset AF3_PORT

    # Path with spaces
    cat > "$cfg" <<'EOF'
AF3_WEIGHTS_DIR="/Users/J Smith/weights"
EOF
    unset AF3_WEIGHTS_DIR 2>/dev/null || true
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "path with spaces" "/Users/J Smith/weights" "${AF3_WEIGHTS_DIR:-}"
    unset AF3_WEIGHTS_DIR

    # Missing config file
    local missing_code=0
    _load_af3_config "/nonexistent/config.env" 2>/dev/null || missing_code=$?
    _assert_eq "missing file returns error" "1" "$missing_code"

    # All HMMER keys
    cat > "$cfg" <<'EOF'
AF3_JACKHMMER="/usr/bin/jackhmmer"
AF3_HMMSEARCH="/usr/bin/hmmsearch"
AF3_HMMBUILD="/usr/bin/hmmbuild"
AF3_NHMMER="/usr/bin/nhmmer"
AF3_HMMALIGN="/usr/bin/hmmalign"
EOF
    unset AF3_JACKHMMER AF3_HMMSEARCH AF3_HMMBUILD AF3_NHMMER AF3_HMMALIGN 2>/dev/null || true
    _load_af3_config "$cfg" 2>/dev/null
    _assert_eq "AF3_JACKHMMER loaded" "/usr/bin/jackhmmer" "${AF3_JACKHMMER:-}"
    _assert_eq "AF3_HMMALIGN loaded" "/usr/bin/hmmalign" "${AF3_HMMALIGN:-}"
    unset AF3_JACKHMMER AF3_HMMSEARCH AF3_HMMBUILD AF3_NHMMER AF3_HMMALIGN

    rm "$cfg"
}

# ── Tests: _prompt ───────────────────────────────────────────────────────────

test_prompt() {
    echo ""
    echo "=== _prompt ==="

    # With default, empty input → returns default
    local result
    result=$(echo "" | _prompt "Pick a value" "mydefault")
    _assert_eq "empty input returns default" "mydefault" "$result"

    # With input → returns input
    result=$(echo "custom" | _prompt "Pick a value" "mydefault")
    _assert_eq "input overrides default" "custom" "$result"

    # No default, with input
    result=$(echo "something" | _prompt "Enter value" "")
    _assert_eq "no-default with input" "something" "$result"

    # Prompt text must NOT appear in captured output
    result=$(echo "" | _prompt "Enter path" "~/default")
    _assert_eq "prompt text not in output" "~/default" "$result"
    # Specifically check the prompt label is absent:
    if [[ "$result" == *"Enter path"* ]]; then
        _assert_eq "prompt label leaked into output (BUG)" "no leak" "leaked"
    else
        _assert_eq "prompt label not in captured value" "true" "true"
    fi
}

# ── Tests: _tolower ───────────────────────────────────────────────────────────

test_tolower() {
    echo ""
    echo "=== _tolower ==="

    _assert_eq "lowercase stays lowercase" "n" "$(_tolower "n")"
    _assert_eq "uppercase N becomes n" "n" "$(_tolower "N")"
    _assert_eq "mixed case" "yes" "$(_tolower "YeS")"
    _assert_eq "empty string" "" "$(_tolower "")"
    _assert_eq "all uppercase" "hello" "$(_tolower "HELLO")"
}

# ── Tests: dry-run non-mutating checks ───────────────────────────────────────
# Uses shell function shims to force "missing tool" branches deterministically,
# regardless of what's installed on the host. Function shims take precedence
# over PATH lookups, so we don't rely on PATH manipulation.

test_dry_run_non_mutating() {
    echo ""
    echo "=== dry-run non-mutating ==="

    DRY_RUN=true
    local output

    # ── _check_autotools: shim autoconf/automake as missing ──────────────
    # Override `command` builtin so -v autoconf/automake fails.
    command() {
        if [[ "${1:-}" == "-v" ]]; then
            case "${2:-}" in
                autoconf|automake) return 1 ;;
            esac
        fi
        builtin command "$@"
    }
    output=$(_check_autotools 2>&1) || true
    unset -f command
    _assert_match "autotools dry-run says would install" "dry run.*would install" "$output"

    # ── _check_homebrew: shim brew as missing ────────────────────────────
    command() {
        if [[ "${1:-}" == "-v" && "${2:-}" == "brew" ]]; then return 1; fi
        builtin command "$@"
    }
    output=$(echo "Y" | _check_homebrew 2>&1) || true
    unset -f command
    _assert_match "homebrew dry-run says would install" "dry run.*would install" "$output"

    # ── _check_xcode_cli: shim xcode-select as missing ──────────────────
    # Shim xcode-select: -p returns failure; --install records a flag file.
    local _install_flag
    _install_flag=$(mktemp "${TMPDIR:-/tmp}/xcode-install-flag.XXXXXX")
    rm -f "$_install_flag"  # absent = not called; touch = called

    xcode-select() {
        case "${1:-}" in
            -p)        return 1 ;;
            --install) touch "$_install_flag"; return 0 ;;
            --version) echo "mock-xcode 1.0"; return 0 ;;
        esac
        return 0
    }
    output=$(_check_xcode_cli 2>&1) || true
    unset -f xcode-select

    local rc=0
    _check_xcode_cli_rc=$?

    _assert_match "xcode dry-run says would install" "dry run.*would install" "$output"
    # Verify xcode-select --install was NOT called (file must be absent).
    local install_was_called
    install_was_called=$([[ -f "$_install_flag" ]] && echo "true" || echo "false")
    _assert_eq "xcode-select --install not called in dry-run" "false" "$install_was_called"
    # Verify the real install message ("Installing (this may take") is absent.
    if [[ "$output" == *"this may take"* ]]; then
        _assert_eq "real xcode install path not reached" "absent" "present"
    else
        _assert_eq "real xcode install path not reached" "true" "true"
    fi
    rm -f "$_install_flag"

    DRY_RUN=false
}

# ── Tests: dry-run E2E ───────────────────────────────────────────────────────

test_dry_run_full() {
    echo ""
    echo "=== dry-run E2E (full mode) ==="

    local output
    # Inputs: Enter(welcome), 1(full), Enter(weights default),
    #         Enter(db choice default=3=skip), Enter(data default), 8642(port)
    output=$(printf '\n1\n\n\n\n8642\n' | bash "$REPO_ROOT/scripts/install.sh" --dry-run 2>&1)
    local rc=$?

    _assert_eq "dry-run exits 0" "0" "$rc"
    _assert_match "shows DRY RUN labels" "DRY RUN" "$output"
    _assert_match "shows 7 steps for full mode" '\[7/7\]' "$output"
    _assert_match "shows Installation Complete" "Installation Complete" "$output"
    _assert_match "config not written" "dry run.*not written" "$output"
    _assert_match "shows Web UI option" "Double-click" "$output"
}

test_dry_run_cli_only() {
    echo ""
    echo "=== dry-run E2E (CLI-only mode) ==="

    local output
    # Inputs: Enter(welcome), 2(cli-only), Enter(weights default),
    #         Enter(db choice default=3=skip), Enter(data default), 8642(port)
    output=$(printf '\n2\n\n\n\n8642\n' | bash "$REPO_ROOT/scripts/install.sh" --dry-run 2>&1)
    local rc=$?

    _assert_eq "cli-only dry-run exits 0" "0" "$rc"
    _assert_match "shows 5 steps for CLI-only" '\[5/5\]' "$output"
    _assert_match "Node.js skipped" "Node\.js.*skipped" "$output"
    _assert_match "desktop launcher skipped" "Desktop launcher skipped" "$output"
    _assert_match "shows API-only URL" "API available at" "$output"
}

# ── Tests: _validate_db_layout ────────────────────────────────────────────────

test_validate_db_layout() {
    echo ""
    echo "=== _validate_db_layout ==="

    local tmpdb
    tmpdb=$(mktemp -d)

    # Helper: set up required protein files
    _setup_protein_files() {
        touch "$tmpdb/uniref90_2022_05.fa"
        touch "$tmpdb/mgy_clusters_2022_05.fa"
        touch "$tmpdb/bfd-first_non_consensus_sequences.fasta"
        touch "$tmpdb/uniprot_all_2021_04.fa"
        touch "$tmpdb/pdb_seqres_2022_09_28.fasta"
    }

    _setup_protein_files

    # Layout 1: mmcif_files/ with .cif content
    mkdir -p "$tmpdb/mmcif_files"
    touch "$tmpdb/mmcif_files/1abc.cif"
    local rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "mmcif_files/ with content passes" "0" "$rc"
    rm -rf "$tmpdb/mmcif_files"

    # Layout 2: pdb_2022_09_28_mmcif_files/mmcif_files/ with .cif content
    mkdir -p "$tmpdb/pdb_2022_09_28_mmcif_files/mmcif_files"
    touch "$tmpdb/pdb_2022_09_28_mmcif_files/mmcif_files/2def.cif"
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "nested mmcif layout with content passes" "0" "$rc"
    rm -rf "$tmpdb/pdb_2022_09_28_mmcif_files"

    # Layout 3: pdb_2022_09_28_mmcif_files/ (flat) with .cif content
    mkdir -p "$tmpdb/pdb_2022_09_28_mmcif_files"
    touch "$tmpdb/pdb_2022_09_28_mmcif_files/3ghi.cif"
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "flat pdb mmcif layout with content passes" "0" "$rc"
    rm -rf "$tmpdb/pdb_2022_09_28_mmcif_files"

    # Layout 4: .cif.gz files also count
    mkdir -p "$tmpdb/mmcif_files"
    touch "$tmpdb/mmcif_files/4jkl.cif.gz"
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "mmcif_files/ with .cif.gz passes" "0" "$rc"
    rm -rf "$tmpdb/mmcif_files"

    # Empty mmcif dir fails (no .cif content)
    mkdir -p "$tmpdb/mmcif_files"
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "empty mmcif_files/ dir fails" "1" "$rc"
    rm -rf "$tmpdb/mmcif_files"

    # Missing mmcif entirely
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "missing mmcif fails" "1" "$rc"

    # Missing a required protein file
    rm "$tmpdb/uniref90_2022_05.fa"
    mkdir -p "$tmpdb/mmcif_files"
    touch "$tmpdb/mmcif_files/5mno.cif"
    rc=0
    _validate_db_layout "$tmpdb" >/dev/null 2>&1 || rc=$?
    _assert_eq "missing protein file fails" "1" "$rc"

    # Stale non-canonical layout when canonical also exists (canonical wins)
    rm -rf "$tmpdb/mmcif_files" "$tmpdb/pdb_2022_09_28_mmcif_files"
    _setup_protein_files
    mkdir -p "$tmpdb/mmcif_files"
    touch "$tmpdb/mmcif_files/fresh.cif"
    mkdir -p "$tmpdb/pdb_2022_09_28_mmcif_files"
    touch "$tmpdb/pdb_2022_09_28_mmcif_files/stale.cif"
    rc=0
    local output
    output=$(_validate_db_layout "$tmpdb" 2>&1) || rc=$?
    _assert_eq "canonical+stale both present passes" "0" "$rc"
    _assert_match "prefers canonical mmcif_files/" "mmcif_files/" "$output"

    rm -rf "$tmpdb"
}

# ── Tests: dry-run DB download option ────────────────────────────────────────

test_dry_run_db_download() {
    echo ""
    echo "=== dry-run E2E (DB download option) ==="

    local output
    # Inputs: Enter(welcome), 1(full), Enter(weights default),
    #         2(db download), /tmp/af3_test_db(target dir),
    #         Y(disk space continue — may or may not appear),
    #         Enter(data default), 8642(port)
    # Extra Y is harmlessly consumed by data dir prompt if space check
    # doesn't fire.
    output=$(printf '\n1\n\n2\n/tmp/af3_test_db\nY\n\n8642\n' | bash "$REPO_ROOT/scripts/install.sh" --dry-run 2>&1)
    local rc=$?

    _assert_eq "db-download dry-run exits 0" "0" "$rc"
    _assert_match "shows download will start" "Database download will start" "$output"
    _assert_match "download phase dry-run label" "dry run.*would run" "$output"
    _assert_match "shows fetch command" "fetch_databases.sh" "$output"
    _assert_match "shows Installation Complete" "Installation Complete" "$output"
}

# ── Run tests ────────────────────────────────────────────────────────────────

# If a test name is given as argument, run only that suite.
if [[ $# -gt 0 ]]; then
    for suite in "$@"; do
        "$suite"
    done
else
    test_expand_path
    test_validate_weights_dir
    test_validate_db_layout
    test_load_af3_config
    test_prompt
    test_tolower
    test_dry_run_non_mutating
    test_dry_run_full
    test_dry_run_cli_only
    test_dry_run_db_download
fi

echo ""
echo "════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════"
if (( FAIL > 0 )); then
    echo ""
    echo "  Failed tests:"
    for e in "${ERRORS[@]}"; do
        echo "    - $e"
    done
    exit 1
fi
