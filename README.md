# AlphaFold 3 for Mac

Run [AlphaFold 3](https://github.com/google-deepmind/alphafold3) protein
structure prediction natively on Apple Silicon Macs (M1, M2, M3, M4). The
model inference layer is rewritten in Apple's
[MLX](https://github.com/ml-explore/mlx) framework while the data pipeline
and output format remain fully compatible with the original. No NVIDIA GPU
or Linux required.

## Highlights

- **Native Apple Silicon** -- M1, M2, M3, and M4 (Max/Ultra) with unified memory
- **Web interface** -- Submit jobs, track progress, and visualize 3D structures
  in the browser (Next.js + Mol\*)
- **CLI** -- Single-command predictions from the terminal
- **Restraint-guided docking** -- Specify distance and contact restraints to
  guide multi-chain docking during diffusion
- **MSA caching** -- Content-addressed cache skips redundant HMMER searches
- **Sequence-only mode** -- Run without genetic databases when they are
  unavailable

## Supported Hardware

Requires an Apple Silicon Mac with a **Max** or **Ultra** chip (M1 through M4).
Minimum **36 GB unified memory** recommended. Larger proteins and multi-chain
complexes require more RAM -- as a rough guide, a single-chain protein of ~500
residues fits comfortably in 64 GB, while complexes with thousands of residues
benefit from 128 GB or more.

## Quick Start

### 1. Install

```bash
git clone https://github.com/omrikais/alphafold3-mac.git
cd alphafold3-mac
./scripts/install.sh
```

The interactive installer sets up Python, MLX, HMMER, the web UI, and
optionally downloads genetic databases (~500 GB). See the full
[Installation guide](docs/getting-started/installation.md) for details.

### 2. Obtain model weights

Request access to the AlphaFold 3 model parameters from Google DeepMind via
[this form](https://forms.gle/svvpY4u2jsHEwWYS6). Place the downloaded
`af3.bin.zst` in the weights directory configured during installation (default
`~/.alphafold3/weights/model/`).

### 3. Run a prediction

**Web interface:**

```bash
./scripts/start.sh
# Open http://127.0.0.1:8642
```

**CLI:**

```bash
source .venv/bin/activate
PYTHONPATH=src python3 run_alphafold_mlx.py \
  --input examples/desi1_monomer.json \
  --output_dir output/my_prediction
```

## Documentation

All documentation lives in the [`docs/`](docs/) directory as plain Markdown,
readable directly on GitHub or in any text editor.

For a searchable local site with navigation, you can optionally serve it
with MkDocs:

```bash
source .venv/bin/activate
uv sync --group docs
uv run mkdocs serve     # http://127.0.0.1:8000
```

Key pages:

- [Quickstart](docs/getting-started/quickstart.md)
- [Input Format](docs/user-guide/input-format.md)
- [Output Format](docs/user-guide/output-format.md)
- [Web Interface](docs/user-guide/web-interface.md)
- [Restraint-Guided Docking](docs/user-guide/restraint-guided-docking.md)
- [CLI Reference](docs/reference/cli.md)
- [API Reference](docs/reference/api.md)
- [Performance Tuning](docs/user-guide/performance.md)
- [Troubleshooting](docs/user-guide/troubleshooting.md)

## Architecture

```
Web UI + REST API Next.js 15 + FastAPI
        ↓
Data Pipeline (unchanged) HMMER / MSA / Templates
        ↓
Model Inference (MLX) Evoformer → Diffusion → Confidence
        ↓
Post-processing mmCIF output, confidence scores
```

The original `src/alphafold3/` data pipeline is preserved. Model inference lives
in `src/alphafold3_mlx/` and runs entirely on Apple GPU via MLX.

## Citing This Work

Any publication that discloses findings arising from using this source code, the
model parameters, or outputs produced by those should cite:

> Abramson, J. et al. "Accurate structure prediction of biomolecular
> interactions with AlphaFold 3." *Nature* **630**, 493--500 (2024).
> [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)

<details>
<summary>BibTeX</summary>

```bibtex
@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  title   = {Accurate structure prediction of biomolecular interactions
             with {AlphaFold} 3},
  journal = {Nature},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493--500},
  doi     = {10.1038/s41586-024-07487-w}
}
```

</details>

## License

The AlphaFold 3 source code is licensed under
[CC-BY-NC-SA 4.0](LICENSE).
Model parameters are subject to the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Based on [AlphaFold 3](https://github.com/google-deepmind/alphafold3) by
Google DeepMind. This is not an officially supported Google product.
