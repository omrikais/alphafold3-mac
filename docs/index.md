<div class="doc-eyebrow">GETTING STARTED</div>

# AlphaFold3 Mac Port overview

Run AlphaFold 3 on Apple Silicon with an MLX-native inference stack while keeping AF3-compatible data pipeline behavior.

## Get started in 10 minutes

Prerequisites:

- Apple Silicon Mac (M2/M3/M4 recommended)
- Python 3.12 environment
- AF3 model weights in `weights/model/af3.bin.zst`

Install and run:

```bash
source .venv/bin/activate
PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/quickstart \
  --num_samples 1 \
  --diffusion_steps 20 \
  --verbose
```

!!! info
    Continue with [Quickstart](getting-started/quickstart.md) for the full walkthrough and expected outputs.

!!! warning
    AlphaFold 3 weights are required for real inference and are subject to usage terms.
    Review the [weights terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
    before running production workloads.

## Where to go next

- [Installation](getting-started/installation.md) for complete machine and dependency setup
- [Pipeline Setup (macOS)](getting-started/pipeline-setup.md) for full AF3-style MSA/templates
- [Input Format](user-guide/input-format.md) and [Output Format](user-guide/output-format.md)
- [Web Interface Guide](user-guide/web-interface.md) for all UI flows and controls
- [Troubleshooting](user-guide/troubleshooting.md) for common failures
- [CLI Reference](reference/cli.md) for all runtime flags
- [API Reference](reference/api.md) for all REST and WebSocket endpoints

## Project context

This repository is a macOS-focused port of AlphaFold 3 inference using MLX while preserving AF3 data and output contracts.

For the upstream project and legal terms, see
[AlphaFold 3 upstream](https://github.com/google-deepmind/alphafold3).
