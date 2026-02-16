# Performance Tuning (macOS + MLX)

This guide documents practical performance tuning for the Mac/MLX port.

## Quick Wins

For faster iteration during setup and debugging:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/smoke \
  --num_samples 1 \
  --diffusion_steps 20 \
  --precision float16 \
  --verbose
```

Then increase quality settings for production runs.

## Main Runtime Levers

### 1) Diffusion Steps

- Default: `200`
- Lower values reduce latency substantially.
- Typical workflow:
  - `20-50` for smoke tests and debugging
  - `200` for final production-quality runs

### 2) Number of Samples

- Default: `5`
- Runtime scales roughly linearly with sample count.
- Use `1` during iteration and increase for final ranking confidence.

### 3) Precision Mode

Supported modes:

- `float32`: most conservative numerically, highest memory/time cost.
- `float16`: best default speed-memory tradeoff on Apple Silicon.
- `bfloat16`: recommended to test on M3/M4 systems where supported.

Example:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/bf16 \
  --precision bfloat16
```

## Sequence-Only vs Full Data Pipeline

### Sequence-only mode (default)

- Skips MSA/template search.
- Fastest way to run end-to-end inference.
- Useful for iteration and many practical workloads.

### Full pipeline mode (`--run_data_pipeline`)

- Runs HMMER search and template retrieval first.
- Improves quality for harder targets but adds CPU, disk, and I/O cost.
- Requires database setup and HMMER binaries.

## Database and I/O Considerations

When using full pipeline mode:

- Keep databases on fast local SSD when possible.
- Avoid network mounts with high latency.
- Ensure enough free space and memory headroom for search tools.

## API/Web UI Throughput Notes

- Job queue is asynchronous; large jobs can block smaller jobs behind them.
- For interactive use, keep one API instance for UI jobs and run batch CLI jobs separately.
- Reusing identical sequences benefits from built-in MSA cache and avoids repeated search work.

## Recommended Presets

### Development preset

- `--num_samples 1`
- `--diffusion_steps 20-50`
- `--precision float16`
- Sequence-only mode

### Production preset

- `--num_samples 5`
- `--diffusion_steps 200`
- `--precision float16` or `bfloat16` (after validation)
- Full pipeline mode if databases are available

## Diagnosing Slow Runs

1. Confirm whether you are in sequence-only or full pipeline mode.
2. Check sample count and diffusion step count first.
3. Lower precision from `float32` to `float16` where acceptable.
4. Inspect API/UI logs for queue delays and per-stage timing.
5. Run a small known-good example to isolate environment issues.
