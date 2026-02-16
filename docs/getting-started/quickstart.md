# Quickstart

This guide gets a first MLX run working on macOS with minimal setup.

## Prerequisites

- Apple Silicon Mac (M2/M3/M4 preferred)
- Python 3.12
- Weights at `weights/model/af3.bin.zst`
- Repository dependencies installed

## 1. Create a minimal input

Use one of the included examples:

```bash
cp examples/desi1_monomer.json /tmp/fold_input.json
```

## 2. Run a quick inference smoke test

```bash
source .venv/bin/activate
PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input /tmp/fold_input.json \
  --output_dir output/quickstart \
  --num_samples 1 \
  --diffusion_steps 20 \
  --verbose
```

## 3. Check outputs

Expected files include:

- `*.cif` structure output(s)
- confidence JSON output(s)
- timing JSON output(s)

## 4. Move to production settings

For better quality:

- Increase `--diffusion_steps` back to `200`
- Use `--num_samples 5` (or more)
- Configure full MSA/templates via [Pipeline Setup (macOS)](pipeline-setup.md)

## Useful links

- [Installation](installation.md)
- [Input Format](../user-guide/input-format.md)
- [Output Format](../user-guide/output-format.md)
- [CLI Reference](../reference/cli.md)
