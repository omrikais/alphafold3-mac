# NOTICE

This project is a derivative work based on AlphaFold 3 by Google DeepMind
and Isomorphic Labs. It ports the AlphaFold 3 model inference pipeline to
run natively on Apple Silicon Macs using Apple's MLX framework, replacing
the original JAX/CUDA backend. The sections below identify the origin of
each major component.

## 1. Original Work (Google DeepMind)

The following files and directories originate from the AlphaFold 3 source
code published by Google DeepMind:

- `src/alphafold3/` -- Data pipeline, model definitions, parsers,
  structure handling, constants, JAX geometry
- `run_alphafold.py` -- Original Linux/CUDA entry point
- `run_alphafold_data_test.py`, `run_alphafold_test.py` -- Original test entry points
- `docker/Dockerfile` -- Original Docker configuration
- `CMakeLists.txt` -- C++ extension build configuration
- `fetch_databases.sh` -- Genetic database download script

**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC-BY-NC-SA 4.0)

Model parameters are subject to separate terms; see
[WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
and
[OUTPUT_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md).

**Original repository:** <https://github.com/google-deepmind/alphafold3>

**Citation:**

> Abramson, J. et al. "Accurate structure prediction of biomolecular
> interactions with AlphaFold 3." *Nature* 630, 493--500 (2024).
> <https://doi.org/10.1038/s41586-024-07487-w>

## 2. MLX Port (New Work)

The following files and directories are new work created for the Apple
Silicon port:

- `src/alphafold3_mlx/` -- Complete MLX inference implementation
  (Evoformer, diffusion head, confidence head, geometry modules,
  restraint-guided docking, REST API, pipeline orchestration)
- `frontend/` -- Next.js web interface for job submission,
  progress tracking, and results visualization
- `run_alphafold_mlx.py` -- MLX entry point for macOS
- `scripts/` -- macOS build, install, and development scripts
- `docs/` -- User-facing documentation (MkDocs)
- `tests/` -- Unit, integration, and parity test suites
- `benchmarks/` -- Attention and geometry performance benchmarks

This new work is licensed under CC-BY-NC-SA 4.0 in compliance with the
ShareAlike requirement of the original license.

## 3. Third-Party Tools, Data, and Influences

| Component | License | URL |
|-----------|---------|-----|
| HMMER (Sean Eddy, HHMI / Harvard) | BSD 3-Clause | <http://hmmer.org/> |
| MLX (Apple Inc.) | MIT | <https://github.com/ml-explore/mlx> |
| RDKit | BSD 3-Clause | <https://www.rdkit.org/> |

**Restraint-guided docking -- prior work:**

The restraint type taxonomy (distance, contact, repulsive) used in this
project's docking feature is inspired by
[ColabDock](https://github.com/JeffSHF/ColabDock) (Shihao Feng et al.,
*Nature Communications*, 2024). ColabDock demonstrated restraint-guided
protein docking using AlphaFold 2's recycling architecture. Our
implementation uses a fundamentally different mechanism -- classifier-guided
diffusion (Dhariwal & Nichol, 2021) applied to AlphaFold 3's denoising
loop -- but ColabDock's restraint categories informed our design.

**Genetic databases** (used at runtime, not distributed with this project):

| Database | License | URL |
|----------|---------|-----|
| UniProt / UniRef90 | CC-BY 4.0 | <https://www.uniprot.org/> |
| Protein Data Bank (PDB) | CC0 1.0 | <https://www.rcsb.org/> |
| MGnify | CC-BY 4.0 | <https://www.ebi.ac.uk/metagenomics/> |
| BFD | CC-BY 4.0 | |
| Rfam | CC0 1.0 | <https://rfam.org/> |
| RNACentral | CC0 1.0 | <https://rnacentral.org/> |
