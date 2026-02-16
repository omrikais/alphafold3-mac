# Golden Server Tests

Reference inputs and outputs from the official AlphaFold Server (alphafoldserver.com).
Used to validate our MLX port against real AF3 predictions.

## Test Cases

| # | File | Protein | PDB | Residues | What it tests |
|---|------|---------|-----|----------|---------------|
| 1 | `01_trp_cage` | Trp-cage TC5b | 1L2Y | 20 | Minimal single-chain baseline |
| 2 | `02_ubiquitin` | Human ubiquitin | 1UBQ | 76 | Classic single-chain benchmark |
| 3 | `03_insulin` | Human insulin (A+B) | 3I40 | 21+30 | Hetero-dimer complex |
| 4 | `04_myoglobin_heme` | Sperm whale myoglobin | 1MBN | 153 + HEM + FE | Protein + ligand + ion |
| 5 | `05_calmodulin_calcium` | Human calmodulin | 1CLL | 148 + 4xCA | Protein + multiple ions |
| 6 | `06_desi1_homodimer` | Human DeSI1 | 2WP7 | 2x168 | Homo-dimer (count=2) |

## Directory Structure

```
golden_server/
├── inputs/          # JSON files to upload to alphafoldserver.com
├── outputs/         # Downloaded result zips (extracted)
├── converted/       # alphafold3-dialect JSONs with MSA/template paths
└── README.md
```

## Workflow

### 1. Generate golden outputs (already done)

Upload `inputs/*.json` to alphafoldserver.com, download results into `outputs/`.

### 2. Convert to alphafold3 format

```bash
python scripts/convert_server_golden.py
```

Produces `converted/*.json` — alphafold3-dialect inputs with pre-computed
MSA and template paths pointing to the server's output files.

### 3. Run our MLX model

```bash
python run_alphafold_mlx.py \
    --input tests/fixtures/golden_server/converted/02_ubiquitin.json \
    --output_dir output/golden_runs/02_ubiquitin/ \
    --model_dir weights/model
```

### 4. Compare outputs

```bash
# Baseline report (golden reference quality)
python scripts/compare_golden.py --baseline

# Compare one case
python scripts/compare_golden.py --compare \
    --case 02_ubiquitin \
    --predicted output/golden_runs/02_ubiquitin/

# Compare all cases
python scripts/compare_golden.py --compare --all \
    --predicted-root output/golden_runs/
```

## Parity Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| pLDDT Pearson r | > 0.90 | Per-atom confidence correlation |
| pLDDT MAE | < 10.0 | Mean absolute pLDDT error |
| PAE relative Frobenius | < 0.30 | Relative PAE matrix difference |
| pTM absolute diff | < 0.10 | Global TM-score difference |
| ipTM absolute diff | < 0.10 | Interface TM-score difference |

## JSON Format

- `inputs/` uses `alphafoldserver` dialect
- `converted/` uses `alphafold3` dialect (version 2, with MSA/template paths)
- See: https://github.com/google-deepmind/alphafold/blob/main/server/README.md
