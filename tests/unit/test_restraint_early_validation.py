"""Unit tests for early fail-fast restraint validation timing.

Tests that invalid restraint references are caught BEFORE expensive operations
like MSA search and feature preparation, not after.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from alphafold3_mlx.pipeline.errors import InputError
from alphafold3_mlx.pipeline.input_handler import parse_input_json
from alphafold3_mlx.pipeline.runner import InferenceRunner
from alphafold3_mlx.restraints.types import (
    DistanceRestraint,
    GuidanceConfig,
    RestraintConfig,
)
from alphafold3_mlx.restraints.validate import ChainInfo, validate_restraints


def _make_test_input_json(restraints: dict | None = None) -> Path:
    """Create a minimal test input JSON file with optional restraints.

    Args:
        restraints: Optional restraints dict to include inline.

    Returns:
        Path to temporary JSON file (caller must unlink).
    """
    input_data = {
        "name": "test",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": {"sequence": "MQIFVKTLTG" * 10, "count": 1}},  # 100 residues
            {"proteinChain": {"sequence": "MQIFVKTLTG" * 10, "count": 1}},  # 100 residues
        ],
    }
    if restraints is not None:
        input_data["restraints"] = restraints

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(input_data, tmp)
    tmp.close()
    return Path(tmp.name)


class TestEarlyValidationTiming:
    """Tests proving early validation catches errors before expensive operations."""

    def test_invalid_chain_caught_before_msa_search(self):
        """Invalid chain reference caught before _run_data_pipeline is called."""
        # Create input with invalid restraint (chain Z does not exist)
        input_path = _make_test_input_json(
            restraints={
                "distance": [
                    {
                        "chain_i": "Z", "residue_i": 1,
                        "chain_j": "A", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        )

        try:
            fold_input = parse_input_json(input_path)

            # Mock args to trigger data pipeline
            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = True  # Would trigger expensive MSA search
            args.db_dir = None
            args.msa_cache_dir = None
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            # Mock expensive operations to verify they're NOT called
            with mock.patch.object(runner, "_run_data_pipeline") as mock_msa, \
                 mock.patch.object(runner, "_prepare_features") as mock_features, \
                 mock.patch.object(runner, "_load_model") as mock_load:

                # Early validation should raise BEFORE any expensive ops
                with pytest.raises(InputError, match="Invalid restraint references.*before MSA"):
                    runner.run()

                # Verify expensive operations were NEVER called
                mock_msa.assert_not_called()
                mock_features.assert_not_called()
                # Model loading is also skipped (happens after early validation)
                mock_load.assert_not_called()

        finally:
            input_path.unlink(missing_ok=True)

    def test_out_of_range_residue_caught_before_featurization(self):
        """Out-of-range residue caught before _prepare_features is called."""
        # Chain A and B have 100 residues each, but restraint references residue 9999
        input_path = _make_test_input_json(
            restraints={
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 9999,
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        )

        try:
            fold_input = parse_input_json(input_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = False  # Skip MSA search
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            with mock.patch.object(runner, "_prepare_features") as mock_features, \
                 mock.patch.object(runner, "_load_model") as mock_load:

                # Error message changed to include "before MSA search"
                with pytest.raises(InputError, match="out of range"):
                    runner.run()

                # Featurization should NOT be called
                mock_features.assert_not_called()
                mock_load.assert_not_called()

        finally:
            input_path.unlink(missing_ok=True)

    def test_invalid_atom_name_caught_before_featurization(self):
        """Invalid atom name (NZ on GLY) caught before featurization."""
        # Create input with GLY at position 1 (sequence starts with M but let's use different seq)
        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                # GLY is G, so "GGGGGGGGGG" = all glycines
                {"proteinChain": {"sequence": "GGGGGGGGGG", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
            ],
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 1, "atom_i": "NZ",  # GLY has no NZ
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        }

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(input_data, tmp)
        tmp.close()
        input_path = Path(tmp.name)

        try:
            fold_input = parse_input_json(input_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = False
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            with mock.patch.object(runner, "_prepare_features") as mock_features, \
                 mock.patch.object(runner, "_load_model") as mock_load:

                with pytest.raises(InputError, match="NZ.*not valid.*GLY"):
                    runner.run()

                mock_features.assert_not_called()
                mock_load.assert_not_called()

        finally:
            input_path.unlink(missing_ok=True)

    def test_bad_guidance_config_caught_before_featurization(self):
        """Invalid guidance config (end_step > num_steps) caught early."""
        # Create input with restraints but NO guidance inline
        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG" * 10, "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTG" * 10, "count": 1}},
            ],
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(input_data, tmp)
        tmp.close()
        input_path = Path(tmp.name)

        # Add BOTH restraints AND guidance in the file (not inline)
        guidance_data = {
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 1,
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
            "guidance": {
                "scale": 1.0,
                "annealing": "linear",
                "start_step": 0,
                "end_step": 300,  # Exceeds num_diffusion_steps=200
            },
        }
        tmp_guidance = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(guidance_data, tmp_guidance)
        tmp_guidance.close()
        guidance_path = Path(tmp_guidance.name)

        try:
            fold_input = parse_input_json(input_path)

            # Apply guidance config
            from alphafold3_mlx.pipeline.input_handler import apply_restraints_file
            fold_input = apply_restraints_file(fold_input, guidance_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200  # Guidance end_step=300 exceeds this
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = False
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            with mock.patch.object(runner, "_prepare_features") as mock_features, \
                 mock.patch.object(runner, "_load_model") as mock_load:

                with pytest.raises(InputError, match="end_step.*300"):
                    runner.run()

                mock_features.assert_not_called()
                mock_load.assert_not_called()

        finally:
            input_path.unlink(missing_ok=True)
            guidance_path.unlink(missing_ok=True)

    def test_valid_restraints_pass_early_validation(self):
        """Valid restraints pass early validation and proceed to later stages."""
        input_path = _make_test_input_json(
            restraints={
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 10, "atom_i": "CA",
                        "chain_j": "B", "residue_j": 20, "atom_j": "CA",
                        "target_distance": 8.0,
                    },
                ],
            },
        )

        try:
            fold_input = parse_input_json(input_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            from alphafold3_mlx.pipeline.output_handler import (
                create_output_directory,
                handle_existing_outputs,
            )
            from alphafold3_mlx.pipeline.input_handler import check_memory_available

            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = False
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            # Mock all later stages to verify early validation passes
            with mock.patch.object(runner, "_load_model") as mock_load, \
                 mock.patch.object(runner, "_prepare_features") as mock_features, \
                 mock.patch.object(runner, "_run_inference") as mock_inference, \
                 mock.patch("alphafold3_mlx.pipeline.output_handler.create_output_directory"), \
                 mock.patch("alphafold3_mlx.pipeline.output_handler.handle_existing_outputs"), \
                 mock.patch("alphafold3_mlx.pipeline.input_handler.check_memory_available"):

                # Set up mock returns
                mock_model = mock.MagicMock()
                mock_config = mock.MagicMock()
                mock_config.diffusion.num_steps = 200
                mock_load.return_value = (mock_model, mock_config)
                mock_features.return_value = mock.MagicMock()
                mock_result = mock.MagicMock()
                mock_result.num_samples = 1
                mock_inference.return_value = (mock_result, None)

                # Early validation should pass and reach later stages
                # (we'll catch an error at _rank_samples since we didn't mock everything)
                try:
                    runner.run()
                except (AttributeError, KeyError, TypeError):
                    # Expected - we didn't fully mock the pipeline
                    pass

                # Verify early validation passed and later stages were called
                mock_load.assert_called_once()
                mock_features.assert_called_once()

        finally:
            input_path.unlink(missing_ok=True)


class TestEarlyValidationErrorMessages:
    """Test that early validation errors have clear, actionable messages."""

    def test_error_message_indicates_early_stage(self):
        """Error message clearly states validation happened before MSA search."""
        input_path = _make_test_input_json(
            restraints={
                "distance": [
                    {
                        "chain_i": "Z", "residue_i": 1,
                        "chain_j": "A", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        )

        try:
            fold_input = parse_input_json(input_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = True
            args.db_dir = None
            args.msa_cache_dir = None
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            try:
                runner.run()
                pytest.fail("Expected InputError to be raised")
            except InputError as e:
                # Error message should indicate this was caught EARLY
                assert "before MSA search" in str(e)
                # Error message should contain the actual validation error
                assert "chain 'Z'" in str(e)
                assert "does not exist" in str(e)

        finally:
            input_path.unlink(missing_ok=True)

    def test_multiple_errors_reported_together(self):
        """Multiple validation errors are reported together in early stage."""
        input_path = _make_test_input_json(
            restraints={
                "distance": [
                    {
                        "chain_i": "Z", "residue_i": 1,  # Invalid chain
                        "chain_j": "A", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                    {
                        "chain_i": "A", "residue_i": 9999,  # Out of range
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        )

        try:
            fold_input = parse_input_json(input_path)

            from alphafold3_mlx.pipeline.cli import CLIArguments
            args = mock.MagicMock(spec=CLIArguments)
            args.output_dir = Path("/tmp/test_output")
            args.model_dir = Path("weights/model")
            args.num_samples = 1
            args.diffusion_steps = 200
            args.precision = None
            args.seed = 42
            args.run_data_pipeline = False
            args.verbose = False
            args.no_overwrite = False
            args.max_tokens = None
            args.max_template_date = "2021-09-30"

            runner = InferenceRunner(args=args, input_json=fold_input)

            try:
                runner.run()
                pytest.fail("Expected InputError to be raised")
            except InputError as e:
                error_msg = str(e)
                # Both errors should be present
                assert "chain 'Z'" in error_msg
                assert "9999" in error_msg
                assert "out of range" in error_msg

        finally:
            input_path.unlink(missing_ok=True)


class TestValidateRestraintsDirectBehavior:
    """Test validate_restraints() behavior with synthetic ChainInfo (no I/O)."""

    def _two_chain_info(self) -> dict[str, ChainInfo]:
        """Two 100-residue alanine chains (A, B)."""
        return {
            "A": ChainInfo(chain_id="A", length=100, residue_types=["ALA"] * 100),
            "B": ChainInfo(chain_id="B", length=100, residue_types=["ALA"] * 100),
        }

    def test_valid_distance_restraint_returns_no_errors(self):
        """Valid CA-CA restraint on existing chains/residues passes."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="A", residue_i=10, chain_j="B", residue_j=20,
                target_distance=8.0, atom_i="CA", atom_j="CA",
            ),
        ])
        errors = validate_restraints(config, self._two_chain_info())
        assert errors == []

    def test_nonexistent_chain_returns_error(self):
        """Referencing chain 'Z' produces an error mentioning that chain."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="Z", residue_i=1, chain_j="A", residue_j=1,
                target_distance=5.0,
            ),
        ])
        errors = validate_restraints(config, self._two_chain_info())
        assert len(errors) >= 1
        assert any("Z" in e and "does not exist" in e for e in errors)

    def test_out_of_range_residue_returns_error(self):
        """Residue 9999 on a 100-residue chain produces an error."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="A", residue_i=9999, chain_j="B", residue_j=1,
                target_distance=5.0,
            ),
        ])
        errors = validate_restraints(config, self._two_chain_info())
        assert len(errors) >= 1
        assert any("9999" in e and "out of range" in e for e in errors)

    def test_invalid_atom_for_residue_type_returns_error(self):
        """NZ on GLY produces an error (GLY has no NZ sidechain atom)."""
        chains = {
            "A": ChainInfo(chain_id="A", length=10, residue_types=["GLY"] * 10),
            "B": ChainInfo(chain_id="B", length=10, residue_types=["ALA"] * 10),
        }
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="A", residue_i=1, chain_j="B", residue_j=1,
                target_distance=5.0, atom_i="NZ",
            ),
        ])
        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("NZ" in e and "GLY" in e for e in errors)

    def test_guidance_end_step_exceeds_num_steps_returns_error(self):
        """end_step=300 with num_diffusion_steps=200 produces an error."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="A", residue_i=1, chain_j="B", residue_j=1,
                target_distance=5.0,
            ),
        ])
        guidance = GuidanceConfig(scale=1.0, end_step=300)
        errors = validate_restraints(
            config, self._two_chain_info(),
            guidance=guidance, num_diffusion_steps=200,
        )
        assert len(errors) >= 1
        assert any("end_step" in e and "300" in e for e in errors)

    def test_non_protein_chain_distinguished_from_missing(self):
        """When all_chain_ids includes 'C' but chains doesn't, error says non-protein."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="C", residue_i=1, chain_j="A", residue_j=1,
                target_distance=5.0,
            ),
        ])
        errors = validate_restraints(
            config, self._two_chain_info(),
            all_chain_ids={"A", "B", "C"},
        )
        assert len(errors) >= 1
        assert any("non-protein" in e.lower() for e in errors)

    def test_multiple_errors_collected(self):
        """Multiple invalid restraints produce multiple errors (not just the first)."""
        config = RestraintConfig(distance=[
            DistanceRestraint(
                chain_i="Z", residue_i=1, chain_j="A", residue_j=1,
                target_distance=5.0,
            ),
            DistanceRestraint(
                chain_i="A", residue_i=9999, chain_j="B", residue_j=1,
                target_distance=5.0,
            ),
        ])
        errors = validate_restraints(config, self._two_chain_info())
        assert len(errors) >= 2
        assert any("Z" in e for e in errors)
        assert any("9999" in e for e in errors)

    def test_empty_restraints_returns_no_errors(self):
        """Empty RestraintConfig produces no errors."""
        config = RestraintConfig()
        errors = validate_restraints(config, self._two_chain_info())
        assert errors == []
