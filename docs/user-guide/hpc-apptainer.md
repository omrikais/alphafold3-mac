# Running AlphaFold 3 Reference Generation on HPC with Apptainer

This guide explains how to generate Linux reference outputs on HPC clusters that don't support Docker but provide Apptainer (formerly Singularity).

!!! info "Advanced and optional workflow"
    This is **not required** for normal macOS/MLX prediction usage.
    Use this only if you need Linux reference artifacts for cross-platform parity checks.

## Overview

The reference image generates canonical Linux outputs for cross-platform parity validation. These outputs are compared against macOS outputs to ensure the C++ extensions produce identical results.

## Prerequisites

- Apptainer 1.1+ (or Singularity 3.8+)
- Access to the AlphaFold 3 repository

## Building the Image Locally

### Option 1: Build with Docker, Convert to SIF

On a machine with Docker:

```bash
# Build the reference image
cd /path/to/alphafold3
docker build -f docker/Dockerfile.reference -t alphafold3-reference .

# Save as tarball for transfer
docker save alphafold3-reference -o alphafold3-reference.tar

# Transfer to HPC, then convert
apptainer build alphafold3-reference.sif docker-archive://alphafold3-reference.tar
```

### Option 2: Build Directly with Apptainer

If your HPC has network access during build:

```bash
# Build from Dockerfile (requires root or fakeroot)
apptainer build --fakeroot alphafold3-reference.sif \
    docker-daemon://alphafold3-reference:latest
```

## Generating Reference Outputs

```bash
# Create output directory
mkdir -p reference_outputs

# Run reference generation
apptainer run \
    --bind ./reference_outputs:/output \
    alphafold3-reference.sif

# Outputs will be in ./reference_outputs/
ls reference_outputs/*.npz
```

## Running with Slurm

```bash
#!/bin/bash
#SBATCH --job-name=af3-reference
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

module load apptainer

apptainer run \
    --bind $PWD/reference_outputs:/output \
    $PWD/alphafold3-reference.sif
```

## Verifying Outputs

The generated NPZ files should have `platform=Linux` metadata:

```python
import numpy as np

files = [
    'reference_outputs/fasta_parsing_reference.npz',
    'reference_outputs/msa_profile_reference.npz',
    'reference_outputs/string_array_reference.npz',
]

for f in files:
    data = np.load(f, allow_pickle=True)
    print(f"{f}: platform={data['platform']}")
    # Should print: platform=Linux
```

## Troubleshooting

### Permission Errors on Bind Mount

```bash
apptainer run --bind ./reference_outputs:/output:rw alphafold3-reference.sif
```

### Fakeroot Not Available

Contact your HPC admin to enable fakeroot, or build the image on a machine with Docker and transfer the SIF file.

### Network Issues During Build

Pre-download the base image on a login node with network access:

```bash
apptainer pull docker://python:3.12-slim-bookworm
```

## What Gets Generated

| File | Description | C++ Module |
|------|-------------|------------|
| `fasta_parsing_reference.npz` | FASTA parsing outputs | `fasta_iterator` |
| `msa_profile_reference.npz` | MSA profile computation | `msa_profile` |
| `string_array_reference.npz` | String array operations | `string_array` |

Each NPZ file contains:
- Input data or checksums
- Output arrays
- `platform` metadata (should be "Linux")
- `python_version` metadata
