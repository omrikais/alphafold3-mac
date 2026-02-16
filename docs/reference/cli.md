# CLI Reference

Primary entrypoint:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input <input.json> \
  --output_dir <output_dir>
```

## Required Arguments

- `--input PATH`: Input JSON file.
- `--output_dir DIR`: Output directory for generated artifacts.

## Optional Arguments

- `--model_dir DIR`: Weights directory. Default resolution order:
  - `AF3_WEIGHTS_DIR`
  - `~/.alphafold3/weights/model`
  - `weights/model`
- `--num_samples N`: Number of samples (default: `5`).
- `--diffusion_steps N`: Diffusion steps (default: `200`).
- `--seed N`: Random seed (optional).
- `--precision {float32,float16,bfloat16}`: Override precision.
- `--verbose`, `-v`: Enable stage-level logging and timing output.
- `--no-overwrite`: Prevent output overwrite.
- `--run_data_pipeline`: Enable MSA/template search before inference.
- `--db_dir DIR`: Database root override (`AF3_DB_DIR` fallback otherwise).
- `--msa_cache_dir DIR`: Cache directory for MSA/template reuse.
- `--max_template_date YYYY-MM-DD`: Template date cutoff (default: `2021-09-30`).
- `--max_tokens N`: Limit max token bucket for memory control.
- `--restraints PATH`: Standalone restraints JSON file (mutually exclusive with inline restraints in input JSON).

## Examples

### Fast smoke test

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/smoke \
  --num_samples 1 \
  --diffusion_steps 20 \
  --verbose
```

### Full pipeline mode

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/full \
  --run_data_pipeline \
  --db_dir /path/to/public_databases
```

### Restraint-guided run

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_dimer.json \
  --restraints examples/restraints/dimer_contacts.json \
  --output_dir output/restrained
```

## Exit Codes

- `0`: Success
- `1`: Input/resource/inference/runtime error
- `130`: Interrupted (`SIGINT`, typically Ctrl+C)
