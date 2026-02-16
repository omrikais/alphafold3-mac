# Troubleshooting

This guide covers common failures for the macOS/MLX workflow.

## Quick Triage Checklist

1. Confirm you are on Apple Silicon.
2. Confirm Python version is supported (`3.12` recommended).
3. Confirm weights are installed and readable.
4. Run a small smoke test (`--num_samples 1 --diffusion_steps 20`).
5. Enable verbose logging to identify the failing stage.

## Environment and Startup

### "Unsupported platform" or startup platform errors

Cause:

- Running on unsupported hardware or architecture.

Fix:

- Use Apple Silicon hardware (M2/M3/M4 family recommended).

### Dependency install failures

Cause:

- Wrong Python interpreter or stale virtual environment.

Fix:

```bash
rm -rf .venv
./scripts/install.sh
```

## Weights and Model Loading

### Missing weight file (`af3.bin.zst`) or model load errors

Cause:

- Weights are not in configured location or file permissions are wrong.

Fix:

- Confirm configured directory and filename.
- Confirm readable permissions for current user.

## Input and Validation

### Validation errors on sequences/entities

Cause:

- Invalid sequence alphabet, malformed JSON, or inconsistent chain/entity definitions.

Fix:

- Re-check input format and entity IDs.
- Use API preflight validation:

```bash
curl -X POST http://127.0.0.1:8642/api/validate \
  -H 'Content-Type: application/json' \
  -d @your_input.json
```

### Restraint validation errors

Cause:

- Invalid atom identifiers, impossible bounds, or invalid guidance step ranges.

Fix:

- Ensure residue and atom references are resolvable.
- Ensure guidance start/end diffusion steps are inside your configured step count.

## Data Pipeline and Databases

### HMMER tools not found

Cause:

- `jackhmmer`/`hmmsearch`/`hmmbuild` not installed or not discoverable.

Fix:

- Re-run installer or setup from pipeline guide.
- Verify binaries on `PATH`.

### Database path errors

Cause:

- Missing required files under `AF3_DB_DIR` (or bad per-database overrides).

Fix:

- Validate directory contents against required filenames.
- Temporarily run sequence-only mode until databases are complete.

## Runtime and Performance

### Out-of-memory errors

Cause:

- Input size + settings exceed unified memory.

Fix:

- Lower `--num_samples`.
- Lower `--diffusion_steps` during exploratory runs.
- Use `--precision float16` or `--precision bfloat16`.

### Very slow jobs

Cause:

- Full data pipeline I/O, high sample counts, high diffusion step counts, or queue contention.

Fix:

- Confirm whether `--run_data_pipeline` is enabled.
- Start with one sample and fewer steps.
- Use fast local SSD for large database directories.

## API and Web UI

### Web UI cannot connect to backend

Cause:

- API not running, wrong port, or stale frontend process.

Fix:

- Restart with `./scripts/start.sh` or `./scripts/dev.sh`.
- Verify health endpoint:

```bash
curl http://127.0.0.1:8642/api/system/status
```

### WebSocket progress not updating

Cause:

- Auth token/origin mismatch or network interruption.

Fix:

- If API key is enabled, ensure token is passed.
- Confirm local-origin access policy when using browser clients.

## When to Escalate

Collect these artifacts before filing an issue:

- Exact command used
- Input JSON (or minimal redacted reproducer)
- Full terminal logs with `--verbose`
- `api/system/status` output
- Hardware and macOS version
