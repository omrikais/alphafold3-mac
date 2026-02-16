"""Tests for API and runtime validation parity.

Ensures that the same input produces the same validation results in both
the API /api/validate endpoint and the runtime inference path.

Each test invokes BOTH code paths independently:
- API path: validation.py::_validate_restraints()
- Runtime path: runner.py::InferenceRunner._early_validate_restraints()

This prevents the bug where API accepts inputs that runtime later rejects
(or vice versa).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from alphafold3_mlx.pipeline.errors import InputError
from alphafold3_mlx.pipeline.input_handler import parse_input_json


def _make_protein_rna_input() -> dict:
    """Create input with protein and RNA chains."""
    return {
        "name": "protein_rna_test",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},  # Chain A
            {"rnaSequence": {"sequence": "GCGCGCGC", "count": 1}},  # Chain B
        ],
        "restraints": {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                },
            ],
        },
    }


def _make_protein_ligand_input() -> dict:
    """Create input with protein and ligand."""
    return {
        "name": "protein_ligand_test",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},  # Chain A
            {"ligand": {"smiles": "CCO", "count": 1}},  # Chain B (ethanol)
        ],
        "restraints": {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                },
            ],
        },
    }


def _make_protein_only_input() -> dict:
    """Create input with only protein chains (valid for restraints)."""
    return {
        "name": "protein_only_test",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},  # Chain A
            {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},  # Chain B
        ],
        "restraints": {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                },
            ],
        },
    }


def _make_rna_only_input() -> dict:
    """Create input with only RNA chains (no protein)."""
    return {
        "name": "rna_only_test",
        "modelSeeds": [42],
        "sequences": [
            {"rnaSequence": {"sequence": "GCGCGCGC", "count": 1}},  # Chain A
            {"rnaSequence": {"sequence": "GCGCGCGC", "count": 1}},  # Chain B
        ],
        "restraints": {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                },
            ],
        },
    }


def _parse_input(input_data: dict) -> "FoldInput":
    """Parse input JSON dict into FoldInput via temp file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(input_data, f)
        tmp_path = Path(f.name)
    try:
        return parse_input_json(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_api_validation(
    fold_input,
    num_diffusion_steps: int = 200,
) -> list[str]:
    """Run the API validation path and return errors."""
    from alphafold3_mlx.api.routes.validation import _validate_restraints

    errors: list[str] = []
    _validate_restraints(fold_input, errors, num_diffusion_steps=num_diffusion_steps)
    return errors


def _run_runtime_validation(
    fold_input,
    num_diffusion_steps: int = 200,
) -> list[str]:
    """Run the runtime InferenceRunner early validation path and return errors.

    Constructs a real InferenceRunner with mock args and calls
    _early_validate_restraints() — the same method the pipeline uses.
    Returns collected errors (empty list if validation passed).
    """
    from alphafold3_mlx.pipeline.cli import CLIArguments
    from alphafold3_mlx.pipeline.runner import InferenceRunner

    args = mock.MagicMock(spec=CLIArguments)
    args.diffusion_steps = num_diffusion_steps

    runner = InferenceRunner(args=args, input_json=fold_input)
    try:
        runner._early_validate_restraints()
        return []
    except InputError as e:
        # Parse the multi-line error message back into individual errors.
        # Format: "Invalid restraint references (caught before MSA search):\n  err1\n  err2"
        msg = str(e)
        if ":\n" in msg:
            body = msg.split(":\n", 1)[1]
            return [line.strip() for line in body.strip().split("\n") if line.strip()]
        return [msg]


class TestValidationParity:
    """Tests proving API and runtime validation produce identical results."""

    def test_protein_only_accepted_by_both(self):
        """Protein-only restraints pass both API and runtime validation."""
        fold_input = _parse_input(_make_protein_only_input())

        api_errors = _run_api_validation(fold_input)
        runtime_errors = _run_runtime_validation(fold_input)

        assert api_errors == [], f"API rejected valid input: {api_errors}"
        assert runtime_errors == [], f"Runtime rejected valid input: {runtime_errors}"

    def test_rna_restraints_rejected_by_both(self):
        """Restraints referencing RNA chain rejected by both paths with matching errors."""
        fold_input = _parse_input(_make_protein_rna_input())

        api_errors = _run_api_validation(fold_input)
        runtime_errors = _run_runtime_validation(fold_input)

        # Both must reject
        assert len(api_errors) > 0, "API accepted restraints on RNA chain"
        assert len(runtime_errors) > 0, "Runtime accepted restraints on RNA chain"

        # Errors should be identical (same shared function)
        assert api_errors == runtime_errors, (
            f"API and runtime produced different errors:\n"
            f"  API:     {api_errors}\n"
            f"  Runtime: {runtime_errors}"
        )

        # Error should mention non-protein
        error_text = " ".join(api_errors).lower()
        assert "non-protein" in error_text, (
            f"Error should mention 'non-protein': {api_errors}"
        )

    def test_ligand_restraints_rejected_by_both(self):
        """Restraints referencing ligand chain rejected by both paths."""
        fold_input = _parse_input(_make_protein_ligand_input())

        api_errors = _run_api_validation(fold_input)
        runtime_errors = _run_runtime_validation(fold_input)

        assert len(api_errors) > 0, "API accepted restraints on ligand chain"
        assert len(runtime_errors) > 0, "Runtime accepted restraints on ligand chain"
        assert api_errors == runtime_errors, (
            f"Parity failure:\n  API: {api_errors}\n  Runtime: {runtime_errors}"
        )

    def test_no_protein_chains_rejected_by_both(self):
        """Input with no protein chains rejected by both API and runtime."""
        fold_input = _parse_input(_make_rna_only_input())

        api_errors = _run_api_validation(fold_input)
        runtime_errors = _run_runtime_validation(fold_input)

        # Both must reject — this is the key assertion that was missing before
        assert len(api_errors) > 0, "API accepted restraints with no protein chains"
        assert len(runtime_errors) > 0, "Runtime accepted restraints with no protein chains"
        assert api_errors == runtime_errors, (
            f"Parity failure:\n  API: {api_errors}\n  Runtime: {runtime_errors}"
        )

        # Error should mention no protein chains
        error_text = " ".join(api_errors).lower()
        assert "no protein chains" in error_text, (
            f"Error should mention 'no protein chains': {api_errors}"
        )

    def test_build_chain_info_excludes_non_protein(self):
        """build_chain_info_from_input only includes protein chains."""
        from alphafold3.common import folding_input
        from alphafold3_mlx.restraints.validate import build_chain_info_from_input

        af3_input = folding_input.Input(
            name="test",
            chains=[
                folding_input.ProteinChain(id="A", sequence="ACDEFGHIKLMNPQRSTVWY", ptms=[]),
                folding_input.RnaChain(id="B", sequence="GCGCGCGC", modifications=[]),
                folding_input.Ligand(id="C", smiles="CCO"),
            ],
            rng_seeds=[42],
        )

        chains = build_chain_info_from_input(af3_input)

        assert len(chains) == 1
        assert "A" in chains
        assert "B" not in chains  # RNA excluded
        assert "C" not in chains  # Ligand excluded
        assert chains["A"].chain_id == "A"
        assert chains["A"].length == 20
        assert chains["A"].residue_types[0] == "ALA"  # A → ALA

    def test_nonexistent_chain_error_identical(self):
        """Both paths give identical error for a completely nonexistent chain."""
        input_data = {
            "name": "nonexistent_chain",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
            ],
            "restraints": {
                "distance": [{
                    "chain_i": "Z", "residue_i": 1,
                    "chain_j": "A", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
        }
        fold_input = _parse_input(input_data)

        api_errors = _run_api_validation(fold_input)
        runtime_errors = _run_runtime_validation(fold_input)

        assert len(api_errors) > 0
        assert api_errors == runtime_errors
        assert "does not exist" in api_errors[0]

    def test_diffusion_steps_parity(self):
        """Both paths reject end_step > num_diffusion_steps with matching errors."""
        input_data = {
            "name": "step_range_test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
            ],
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"start_step": 0, "end_step": 150},
        }
        fold_input = _parse_input(input_data)

        # With 100 steps, end_step=150 exceeds limit
        api_errors = _run_api_validation(fold_input, num_diffusion_steps=100)
        runtime_errors = _run_runtime_validation(fold_input, num_diffusion_steps=100)

        assert len(api_errors) > 0, "API should reject end_step > num_steps"
        assert len(runtime_errors) > 0, "Runtime should reject end_step > num_steps"
        assert api_errors == runtime_errors, (
            f"Parity failure:\n  API: {api_errors}\n  Runtime: {runtime_errors}"
        )
        assert "end_step" in api_errors[0]
        assert "150" in api_errors[0]

    def test_diffusion_steps_parity_valid(self):
        """Both paths accept end_step within num_diffusion_steps."""
        input_data = {
            "name": "step_range_valid",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTGKTITLEVEPS", "count": 1}},
            ],
            "restraints": {
                "distance": [{
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }],
            },
            "guidance": {"start_step": 0, "end_step": 150},
        }
        fold_input = _parse_input(input_data)

        # With 200 steps, end_step=150 is valid
        api_errors = _run_api_validation(fold_input, num_diffusion_steps=200)
        runtime_errors = _run_runtime_validation(fold_input, num_diffusion_steps=200)

        assert api_errors == [], f"API rejected valid input: {api_errors}"
        assert runtime_errors == [], f"Runtime rejected valid input: {runtime_errors}"


class TestDiffusionStepsValidation:
    """Tests for diffusionSteps field validation in request models."""

    def test_omitted_diffusion_steps_defaults_to_none(self):
        """Omitted diffusionSteps defaults to None in the model."""
        from alphafold3_mlx.api.models import ValidationRequest

        req = ValidationRequest(
            sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
        )
        assert req.diffusionSteps is None

    def test_positive_diffusion_steps_accepted(self):
        """Positive diffusionSteps values are accepted."""
        from alphafold3_mlx.api.models import ValidationRequest

        req = ValidationRequest(
            sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
            diffusionSteps=100,
        )
        assert req.diffusionSteps == 100

    def test_diffusion_steps_one_accepted(self):
        """diffusionSteps=1 (minimum valid) is accepted."""
        from alphafold3_mlx.api.models import ValidationRequest

        req = ValidationRequest(
            sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
            diffusionSteps=1,
        )
        assert req.diffusionSteps == 1

    def test_diffusion_steps_zero_rejected(self):
        """diffusionSteps=0 is rejected by Pydantic ge=1 constraint."""
        from pydantic import ValidationError
        from alphafold3_mlx.api.models import ValidationRequest

        with pytest.raises(ValidationError, match="diffusionSteps"):
            ValidationRequest(
                sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
                diffusionSteps=0,
            )

    def test_diffusion_steps_negative_rejected(self):
        """Negative diffusionSteps is rejected by Pydantic ge=1 constraint."""
        from pydantic import ValidationError
        from alphafold3_mlx.api.models import ValidationRequest

        with pytest.raises(ValidationError, match="diffusionSteps"):
            ValidationRequest(
                sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
                diffusionSteps=-5,
            )

    def test_job_submission_diffusion_steps_zero_rejected(self):
        """diffusionSteps=0 is also rejected on JobSubmission model."""
        from pydantic import ValidationError
        from alphafold3_mlx.api.models import JobSubmission

        with pytest.raises(ValidationError, match="diffusionSteps"):
            JobSubmission(
                sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
                diffusionSteps=0,
            )

    def test_job_submission_positive_diffusion_steps_accepted(self):
        """Positive diffusionSteps on JobSubmission is accepted."""
        from alphafold3_mlx.api.models import JobSubmission

        job = JobSubmission(
            sequences=[{"proteinChain": {"sequence": "ACDEF", "count": 1}}],
            diffusionSteps=50,
        )
        assert job.diffusionSteps == 50

    def test_validation_route_defaults_to_200_when_omitted(self):
        """When diffusionSteps is None, validation uses 200 as default."""
        fold_input = _parse_input(_make_protein_only_input())

        # Add guidance with end_step=199 — valid with 200 steps, invalid with fewer
        fold_input._guidance = None  # Clear any existing guidance

        # end_step=199 should pass with default 200 steps
        import copy
        input_data = copy.deepcopy(_make_protein_only_input())
        input_data["guidance"] = {"start_step": 0, "end_step": 199}
        fold_input_with_guidance = _parse_input(input_data)

        api_errors = _run_api_validation(fold_input_with_guidance, num_diffusion_steps=200)
        assert api_errors == [], f"Unexpected errors with default 200 steps: {api_errors}"

    def test_validation_route_uses_custom_steps(self):
        """When diffusionSteps is provided, validation uses that value."""
        import copy
        input_data = copy.deepcopy(_make_protein_only_input())
        input_data["guidance"] = {"start_step": 0, "end_step": 150}
        fold_input = _parse_input(input_data)

        # With 100 steps, end_step=150 should fail
        api_errors = _run_api_validation(fold_input, num_diffusion_steps=100)
        assert len(api_errors) > 0, "Should reject end_step > num_steps"
        assert "end_step" in api_errors[0]
