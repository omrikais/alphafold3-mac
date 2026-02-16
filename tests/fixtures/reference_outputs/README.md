# Reference Outputs for Cross-Platform Validation

This directory contains reference outputs generated for validating
that the macOS data pipeline produces identical results to the reference
platform.

## Purpose

The policy requires that the data pipeline produces identical outputs on macOS
compared to Linux. Since the data pipeline code (Python + C++ extensions) is
unchanged between platforms, this validation ensures:

1. C++ extensions compile correctly on macOS
2. C++ extensions produce bitwise-identical outputs
3. No platform-specific numerical differences

## Reference Files

### Current Files

- `fasta_parsing_reference.npz` - FASTA parsing outputs from `fasta_iterator.parse_fasta()` and `fasta_iterator.parse_fasta_include_descriptions()`
- `msa_profile_reference.npz` - MSA profile computation from `msa_profile.compute_msa_profile()` with deterministic test data (seed=42)
- `string_array_reference.npz` - String array operations from `string_array.isin()` and `string_array.remap()`

### File Contents

Each `.npz` file contains:
- Input parameters used (with checksums where applicable)
- Expected output arrays/values
- Platform and Python version metadata for traceability

## Generating Reference Outputs

### Using Docker (Recommended)

The Docker-based approach ensures outputs are generated on Linux with a reproducible environment:

```bash
# Build the reference image (use linux/amd64 on Apple Silicon for x86_64 Linux)
docker build --platform linux/amd64 -f docker/Dockerfile.reference -t alphafold3-reference .

# Generate reference outputs
mkdir -p tests/fixtures/reference_outputs
docker run --platform linux/amd64 --rm \
    -v $(pwd)/tests/fixtures/reference_outputs:/output \
    alphafold3-reference

# Verify platform metadata
python -c "
import numpy as np
for f in ['fasta_parsing_reference.npz', 'msa_profile_reference.npz', 'string_array_reference.npz']:
    data = np.load(f'tests/fixtures/reference_outputs/{f}', allow_pickle=True)
    print(f'{f}: platform={data[\"platform\"]}')
"
```

### Using Apptainer/Singularity (HPC)

For HPC environments without Docker, see `docs/hpc-apptainer-guide.md`.

### Direct Script (Development Only)

For development/testing, you can run the script directly:

```bash
# On Linux (preferred for true cross-platform validation)
python scripts/generate_reference_outputs.py \
    --input tests/fixtures/test_sequence.fasta \
    --output tests/fixtures/reference_outputs/

# On macOS (for development/testing only)
python scripts/generate_reference_outputs.py --force \
    --input tests/fixtures/test_sequence.fasta \
    --output tests/fixtures/reference_outputs/
```

## Running Cross-Platform Tests

On macOS:

```bash
# Run only parity tests
pytest tests/integration/test_data_pipeline.py -v -k "Parity"

# Run all cross-platform tests
pytest tests/integration/test_data_pipeline.py -v -k "cross_platform or Parity"
```

Tests will:
- Skip if reference outputs are not present (with instructions to generate)
- Compare macOS outputs against stored references
- Fail if any output differs from the reference

## Test Details

### FASTA Parsing Parity
- Verifies `parse_fasta()` returns identical sequence lists
- Verifies `parse_fasta_include_descriptions()` returns identical sequences and descriptions
- Input file validated by SHA256 checksum

### MSA Profile Parity
- Uses deterministic MSA data (numpy seed=42)
- Verifies `compute_msa_profile()` produces bitwise-identical output
- Shape: (100, 50) MSA → (50, 22) profile

### String Array Parity
- Verifies `isin()` produces identical boolean arrays
- Verifies `remap()` produces identical remapped arrays

## Notes

- Reference outputs should be committed to the repository
- Re-generate when C++ extension code changes
- For true cross-platform validation, generate references on Linux using Docker
- All numeric arrays use appropriate dtypes for exact comparison
- See `MANIFEST.json` for checksums and generation metadata
