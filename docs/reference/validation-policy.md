# Validation Policy

The validation policy defines the **validation subset** as unit + integration tests that do not
require external dependencies or benchmarking infrastructure.

## Required validation run

Run the validation subset exactly as:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 -m pytest tests/unit tests/integration -m "not external_deps and not benchmark" -v
```

The policy requires **zero skipped and zero deselected** tests in this run.

## Optional runs (when dependencies are present)

External dependency tests:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 -m pytest tests/unit tests/integration tests/validation -m "external_deps" -v
```

Benchmark tests:

```bash
source .venv/bin/activate && PYTHONPATH=src python3 -m pytest tests/validation -m "benchmark" -v
```

## Script helper

Use the helper script for the validation subset and optional runs:

```bash
./scripts/run_validation_tests.sh
./scripts/run_validation_tests.sh --external-deps
./scripts/run_validation_tests.sh --benchmark
./scripts/run_validation_tests.sh --optional
```
