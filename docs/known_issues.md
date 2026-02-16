# Known Issues

This page tracks user-impacting issues in the macOS/MLX port.

## `trp_cage` Non-determinism Across Runs

In some configurations, `trp_cage` can show run-to-run variation beyond expected stochasticity.

Workarounds:

- Use a fixed `--seed`.
- Keep precision and diffusion settings constant between comparisons.
- Prefer relative comparisons on identical hardware/software environments.

## No Apple Neural Engine (ANE) Backend

MLX in this project runs on CPU/GPU. ANE acceleration is not currently used.

Impact:

- Performance expectations should be based on GPU execution, not ANE offload.

## Large Inputs Can Exhaust Unified Memory

Very large complexes can run out of memory, especially with high sample counts.

Workarounds:

- Reduce `--num_samples`.
- Reduce `--diffusion_steps` for exploratory runs.
- Use `--precision float16` or `--precision bfloat16` when appropriate.

## Full Data Pipeline Requires External Databases

`--run_data_pipeline` requires correctly configured HMMER binaries and large sequence/template databases.

Symptoms:

- Validation errors for missing DB files.
- Jobs failing before inference starts.

Workarounds:

- Confirm `AF3_DB_DIR` or per-database environment variables.
- Run in sequence-only mode until pipeline dependencies are ready.

## Reference Generation Workflows Are Linux-Oriented

Cross-platform parity reference generation (Docker/Apptainer) is a separate workflow from normal Mac inference.

Impact:

- Most users do not need Linux reference generation for everyday prediction use.

## Weights Availability and Terms

Predictions require official AF3 weights and compliance with their terms of use.

Impact:

- Installation can complete without weights, but inference jobs will fail until weights are placed correctly.
