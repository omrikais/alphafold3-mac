# Full AF3-Style Data Pipeline (MSA + Templates) on macOS

This repository’s default MLX runner can operate on sequence-only inputs using
`fill_missing_fields()` placeholders. For accuracy comparisons against AlphaFold
Server, you typically want to run the **real AlphaFold 3 data pipeline** first:
MSA search and template search, then featurisation.

This doc explains how to configure and run that path locally.

## 1) Install HMMER (Required)

AlphaFold 3’s data pipeline requires HMMER tools:
- `jackhmmer` (protein MSA search)
- `hmmsearch` and `hmmbuild` (template search)

On macOS, Homebrew is the simplest:

```bash
brew install hmmer
```

Verify:

```bash
command -v jackhmmer hmmsearch hmmbuild
jackhmmer -h | head -n 2
```

Optional (performance): build a `jackhmmer` with the `--seq_limit` patch:

```bash
./scripts/build_hmmer_macos.sh
```

If you use the build script, binaries install to `~/hmmer/bin` by default.

## 2) Configure Database Paths (Required)

The full AF3 data pipeline needs large genetic databases and PDB template
resources (hundreds of GB). See upstream AF3 `docs/installation.md`.

### Recommended: set `AF3_DB_DIR`

If you downloaded databases using `./fetch_databases.sh <DB_DIR>`, set:

```bash
export AF3_DB_DIR="<DB_DIR>"
```

The runner will look for these standard names inside `AF3_DB_DIR`:
- `uniref90_2022_05.fa`
- `mgy_clusters_2022_05.fa`
- `bfd-first_non_consensus_sequences.fasta`
- `uniprot_all_2021_04.fa`
- `pdb_seqres_2022_09_28.fasta`
- `mmcif_files/` (directory containing PDB mmCIF files)
  - Some installs extract into `pdb_2022_09_28_mmcif_files/mmcif_files/` instead.

### Overrides (if your filenames differ)

Set explicit env vars (paths can be absolute or `~`-expanded):
- `AF3_UNIREF90_DB`
- `AF3_MGNIFY_DB`
- `AF3_SMALL_BFD_DB`
- `AF3_UNIPROT_DB`
- `AF3_PDB_SEQRES_DB`
- `AF3_PDB_MMCIF_DIR`

Optional:
- `AF3_MAX_TEMPLATE_DATE` (default: `2021-09-30`)

## 3) Validate Configuration

Check binaries:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 scripts/check_deps.py
```

Check full data pipeline configuration (binaries + databases):

```bash
source .venv/bin/activate && PYTHONPATH=src python3 scripts/validate_data_pipeline_paths.py
```

If anything is missing, the script prints exactly what was missing and where it
looked (env vars, PATH, and db_dir-derived locations).

## 4) Run DeSI1 with the Full Pipeline

Monomer:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir /private/tmp/test_out/desi1_monomer_full_pipeline/ \
  --num_samples 1 --diffusion_steps 50 --seed 42 \
  --precision float16 \
  --run_data_pipeline
```

Dimer (compare to AlphaFold Server reference):

```bash
source .venv/bin/activate && PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_dimer.json \
  --output_dir /private/tmp/test_out/desi1_dimer_full_pipeline/ \
  --num_samples 1 --diffusion_steps 50 --seed 42 \
  --precision float16 \
  --run_data_pipeline
```

If you don’t want to use `AF3_DB_DIR`, you can pass:

```bash
  --db_dir "<DB_DIR>"
```

## 5) Evaluate Baseline vs Full Pipeline

Run both conditions and print a summary table:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 scripts/eval_desi1_pipeline_vs_placeholders.py --diffusion_steps 50 --seed 42 --precision float16
```

To skip the full pipeline (baseline only):

```bash
source .venv/bin/activate && PYTHONPATH=src python3 scripts/eval_desi1_pipeline_vs_placeholders.py --skip_full
```
