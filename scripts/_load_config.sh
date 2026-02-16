#!/usr/bin/env bash
# Safe config.env loader — only exports allowlisted keys.
# Rejects unknown keys, handles quoting/escaping for paths with spaces.
#
# Usage:
#   source scripts/_load_config.sh
#   _load_af3_config                         # uses default path
#   _load_af3_config /path/to/config.env     # uses custom path

_load_af3_config() {
    local config_file="${1:-$HOME/.alphafold3_mlx/config.env}"

    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file not found: $config_file" >&2
        echo "       Run ./scripts/install.sh to create it." >&2
        return 1
    fi

    # Strict allowlist of recognized keys
    local -a allowed_keys=(
        AF3_WEIGHTS_DIR
        AF3_DATA_DIR
        AF3_DB_DIR
        AF3_PORT
        AF3_INSTALL_DIR
        AF3_JACKHMMER
        AF3_HMMSEARCH
        AF3_HMMBUILD
        AF3_NHMMER
        AF3_HMMALIGN
    )

    local allowed_re
    allowed_re=$(printf '%s|' "${allowed_keys[@]}")
    allowed_re="^(${allowed_re%|})$"

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue

        # Extract key and value (handle KEY="value" and KEY=value)
        if [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            # Strip surrounding quotes if present
            value="${value#\"}"
            value="${value%\"}"

            # Only export if key is in allowlist
            if [[ "$key" =~ $allowed_re ]]; then
                export "$key=$value"
            else
                echo "Warning: Ignoring unknown config key: $key" >&2
            fi
        fi
    done < "$config_file"
}
