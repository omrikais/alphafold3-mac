"""Unit tests for restraint reference validation.

Tests that invalid restraint references (nonexistent chain, out-of-range
residue, invalid atom name) are caught during input validation with
actionable error messages across all 3 input paths:
1. Inline JSON (via validate_restraints)
2. --restraints file (via load_restraints_file + validate_restraints)
3. API /api/validate (tested in integration tests)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from alphafold3_mlx.restraints.types import (
    CandidateResidue,
    ContactRestraint,
    DistanceRestraint,
    GuidanceConfig,
    RepulsiveRestraint,
    RestraintConfig,
    guidance_config_from_dict,
    restraint_config_from_dict,
)
from alphafold3_mlx.restraints.validate import ChainInfo, validate_restraints


def _make_ubiquitin_chains() -> dict[str, ChainInfo]:
    """Create chain info for two 76-residue ubiquitin chains.

    Chain A: 76 residues, all ALA except residue 7 = GLY, residue 48 = LYS
    Chain B: 76 residues, all ALA except residue 7 = GLY, residue 48 = LYS
    """
    residue_types_a = ["ALA"] * 76
    residue_types_a[6] = "GLY"   # residue 7 (0-indexed → 6)
    residue_types_a[47] = "LYS"  # residue 48 (0-indexed → 47)
    residue_types_b = list(residue_types_a)  # same for chain B

    return {
        "A": ChainInfo(chain_id="A", length=76, residue_types=residue_types_a),
        "B": ChainInfo(chain_id="B", length=76, residue_types=residue_types_b),
    }


# ── Invalid Chain Tests ────────────────────────────────────────────────────


class TestInvalidChain:
    """Tests for nonexistent chain references."""

    def test_distance_invalid_chain(self):
        """Distance restraint with nonexistent chain 'Z' produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="Z", residue_i=1,
                    chain_j="A", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "chain 'Z'" in errors[0]
        assert "does not exist" in errors[0]
        assert "A, B" in errors[0]  # Lists available chains

    def test_contact_invalid_chain(self):
        """Contact restraint with nonexistent chain produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="Z", residue_i=1,
                    candidates=[CandidateResidue(chain_j="A", residue_j=1)],
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("chain 'Z'" in e for e in errors)

    def test_repulsive_invalid_chain(self):
        """Repulsive restraint with nonexistent chain produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="Z", residue_j=1,
                    min_distance=10.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("chain 'Z'" in e for e in errors)

    def test_contact_candidate_invalid_chain(self):
        """Contact candidate with nonexistent chain produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=1,
                    candidates=[CandidateResidue(chain_j="Z", residue_j=1)],
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("chain 'Z'" in e for e in errors)


# ── Out-of-Range Residue Tests ─────────────────────────────────────────────


class TestOutOfRangeResidue:
    """Tests for out-of-range residue numbers."""

    def test_distance_residue_too_high(self):
        """Residue 9999 on a 76-residue chain produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=9999,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "9999" in errors[0]
        assert "out of range" in errors[0]
        assert "1-76" in errors[0]

    def test_distance_residue_zero(self):
        """Residue 0 (below 1-indexed minimum) produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=0,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_repulsive_residue_out_of_range(self):
        """Repulsive restraint with out-of-range residue produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=9999,
                    min_distance=10.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("9999" in e for e in errors)

    def test_contact_residue_out_of_range(self):
        """Contact restraint with out-of-range source residue produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=9999,
                    candidates=[CandidateResidue(chain_j="B", residue_j=1)],
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("9999" in e for e in errors)


# ── Invalid Atom Name Tests ────────────────────────────────────────────────


class TestInvalidAtomName:
    """Tests for invalid atom names."""

    def test_nz_on_glycine(self):
        """Atom 'NZ' on GLY (residue 7) produces error with valid atom list."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=7, atom_i="NZ",  # GLY has no NZ
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "NZ" in errors[0]
        assert "not valid" in errors[0]
        assert "GLY" in errors[0]
        # Should list valid atoms for GLY (N, CA, C, O)
        assert "N" in errors[0]
        assert "CA" in errors[0]

    def test_nz_on_lysine_is_valid(self):
        """Atom 'NZ' on LYS (residue 48) does NOT produce error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=48, atom_i="NZ",  # LYS has NZ
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 0

    def test_completely_invalid_atom_name(self):
        """Completely invalid atom name produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1, atom_i="FAKE",
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) >= 1
        assert any("FAKE" in e for e in errors)


# ── Multiple Errors ────────────────────────────────────────────────────────


class TestMultipleErrors:
    """Tests for collecting multiple validation errors at once."""

    def test_multiple_errors_collected(self):
        """Multiple invalid restraints produce multiple errors."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="Z", residue_i=1,  # bad chain
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
                DistanceRestraint(
                    chain_i="A", residue_i=9999,  # bad residue
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
                DistanceRestraint(
                    chain_i="A", residue_i=7, atom_i="NZ",  # bad atom (GLY)
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        # All three errors should be collected
        assert len(errors) == 3
        assert any("chain 'Z'" in e for e in errors)
        assert any("9999" in e for e in errors)
        assert any("NZ" in e and "GLY" in e for e in errors)


# ── Inline JSON Path Tests ─────────────────────────────────────────────────


class TestInlineJsonValidation:
    """Tests that inline JSON with invalid restraints produces errors (path 1)."""

    def test_inline_json_parsing_and_validation(self):
        """Parse inline JSON with invalid restraints and validate."""
        raw = {
            "distance": [
                {
                    "chain_i": "Z", "residue_i": 1,
                    "chain_j": "A", "residue_j": 1,
                    "target_distance": 5.0,
                },
            ],
        }
        config = restraint_config_from_dict(raw)
        chains = _make_ubiquitin_chains()
        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "chain 'Z'" in errors[0]


# ── Restraints File Path Tests ─────────────────────────────────────────────


class TestRestraintsFileValidation:
    """Tests that --restraints file with invalid references produces errors (path 2)."""

    def test_restraints_file_with_invalid_references(self):
        """Load restraints from file, then validate against chains."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file

        restraints_data = {
            "restraints": {
                "distance": [
                    {
                        "chain_i": "Z", "residue_i": 1,
                        "chain_j": "A", "residue_j": 9999,
                        "target_distance": 5.0,
                    },
                ],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(restraints_data, f)
            tmp_path = Path(f.name)

        try:
            config, guidance = load_restraints_file(tmp_path)
            assert config is not None

            chains = _make_ubiquitin_chains()
            errors = validate_restraints(config, chains)
            assert len(errors) >= 2
            assert any("chain 'Z'" in e for e in errors)
            assert any("9999" in e for e in errors)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_restraints_file_with_nz_on_glycine(self):
        """Restraints file with NZ on GLY produces atom validation error."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file

        restraints_data = {
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 7, "atom_i": "NZ",
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(restraints_data, f)
            tmp_path = Path(f.name)

        try:
            config, guidance = load_restraints_file(tmp_path)
            assert config is not None

            chains = _make_ubiquitin_chains()
            errors = validate_restraints(config, chains)
            assert len(errors) == 1
            assert "NZ" in errors[0]
            assert "GLY" in errors[0]
        finally:
            tmp_path.unlink(missing_ok=True)


# ── Guidance Validation Tests ──────────────────────────────────────────────


class TestGuidanceValidation:
    """Tests for guidance parameter validation."""

    def test_start_step_gt_end_step_error(self):
        """start_step > end_step is caught at GuidanceConfig construction."""
        # GuidanceConfig __post_init__ raises ValueError for end_step < start_step
        with pytest.raises(ValueError, match="end_step.*must be >= start_step"):
            GuidanceConfig(start_step=150, end_step=50)

    def test_end_step_exceeds_num_steps_error(self):
        """end_step > num_diffusion_steps produces error."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig()
        guidance = GuidanceConfig(start_step=0, end_step=300)

        errors = validate_restraints(config, chains, guidance=guidance, num_diffusion_steps=200)
        assert len(errors) == 1
        assert "end_step" in errors[0]
        assert "300" in errors[0]


# ── Warning Path Tests ──────────────────────────────────────────────────────


class TestWarningPaths:
    """Tests for warning-only validation paths."""

    def test_high_guidance_scale_logs_warning(self, caplog):
        """Guidance scale > 10 logs warning without creating validation errors."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig()
        guidance = GuidanceConfig(scale=20.0, start_step=0, end_step=100)

        with caplog.at_level("WARNING", logger="alphafold3_mlx.restraints.validate"):
            errors = validate_restraints(config, chains, guidance=guidance)

        assert len(errors) == 0
        assert any("Guidance scale" in msg and "very high" in msg for msg in caplog.messages)

    def test_restraint_count_over_100_logs_warning(self, caplog):
        """More than 100 restraints logs warning without creating validation errors."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A",
                    residue_i=1,
                    chain_j="B",
                    residue_j=1,
                    target_distance=5.0,
                )
                for _ in range(101)
            ],
        )

        with caplog.at_level("WARNING", logger="alphafold3_mlx.restraints.validate"):
            errors = validate_restraints(config, chains)

        assert len(errors) == 0
        assert any("exceeds 100" in msg for msg in caplog.messages)


# ── Actionable Error Message Tests ─────────────────────────────────────────


class TestActionableErrorMessages:
    """Verify error messages are specific and actionable."""

    def test_error_identifies_restraint_index(self):
        """Error message identifies which restraint (by index) is invalid."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
                DistanceRestraint(
                    chain_i="Z", residue_i=1,  # second restraint is bad
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "distance[1]" in errors[0]  # 0-indexed, second restraint

    def test_error_lists_available_chains(self):
        """Error message lists available chain IDs."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="X", residue_i=1,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert "A, B" in errors[0]

    def test_error_lists_valid_residue_range(self):
        """Error message includes valid residue range."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=100,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        assert "1-76" in errors[0]

    def test_error_lists_valid_atoms(self):
        """Error message lists valid atoms for the residue type."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=7, atom_i="NZ",  # GLY
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        errors = validate_restraints(config, chains)
        # Should mention the valid atoms for GLY
        assert "N" in errors[0]
        assert "CA" in errors[0]
        assert "C" in errors[0]
        assert "O" in errors[0]


# ── Atom-Pair Conflict Detection ────────────────────────────────


class TestAtomPairConflictDetection:
    """Tests for atom-pair-level conflict detection."""

    def test_same_atom_pair_detected(self, caplog):
        """Distance CA-CA and repulsive on same pair logs warning."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1, atom_i="CA",
                    chain_j="B", residue_j=1, atom_j="CA",
                    target_distance=5.0,
                ),
            ],
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    min_distance=10.0,
                ),
            ],
        )

        with caplog.at_level("WARNING", logger="alphafold3_mlx.restraints.validate"):
            errors = validate_restraints(config, chains)
        assert len(errors) == 0  # Conflicts are warnings, not errors
        assert any("Conflict" in msg for msg in caplog.messages)

    def test_different_atoms_same_residue_no_conflict(self):
        """Distance NZ-C and repulsive CA-CA on same residues: no conflict."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=48, atom_i="NZ",
                    chain_j="B", residue_j=1, atom_j="C",
                    target_distance=1.5,
                ),
            ],
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=48,
                    chain_j="B", residue_j=1,
                    min_distance=10.0,
                ),
            ],
        )

        # This should NOT produce a conflict warning because:
        # - Distance restraint is on NZ-C atoms
        # - Repulsive restraint is implicitly on CA-CA atoms
        # These are different atom pairs even though they share residue pairs
        errors = validate_restraints(config, chains)
        assert len(errors) == 0  # No validation errors

    def test_reversed_pair_still_detected(self):
        """Conflict detected even when atom pairs are in reversed order."""
        chains = _make_ubiquitin_chains()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="B", residue_i=1, atom_i="CA",
                    chain_j="A", residue_j=1, atom_j="CA",
                    target_distance=5.0,
                ),
            ],
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    min_distance=10.0,
                ),
            ],
        )
        # Both target (A:1:CA, B:1:CA) after normalization
        errors = validate_restraints(config, chains)
        assert len(errors) == 0  # No validation errors (conflict is a warning, not error)


# ── Mutual Exclusion Tests ───────────────────────────────────────


class TestFR011MutualExclusion:
    """Inline restraints + --restraints file is rejected."""

    def _make_fold_input_with_inline_restraints(self):
        """Create a FoldInput that has inline restraints set."""
        from alphafold3_mlx.pipeline.input_handler import parse_input_json

        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
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
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(input_data, tmp)
        tmp.close()
        try:
            return parse_input_json(Path(tmp.name))
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _make_fold_input_with_inline_guidance(self):
        """Create a FoldInput that has inline guidance set (no restraints)."""
        from alphafold3_mlx.pipeline.input_handler import parse_input_json

        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
            ],
            "guidance": {"scale": 2.0, "annealing": "cosine"},
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(input_data, tmp)
        tmp.close()
        try:
            return parse_input_json(Path(tmp.name))
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _make_fold_input_no_restraints(self):
        """Create a FoldInput with no inline restraints or guidance."""
        from alphafold3_mlx.pipeline.input_handler import parse_input_json

        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
            ],
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(input_data, tmp)
        tmp.close()
        try:
            return parse_input_json(Path(tmp.name))
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _write_restraints_file(self, data: dict) -> Path:
        """Write restraints JSON to a temp file and return its path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        json.dump(data, f)
        f.close()
        return Path(f.name)

    def test_inline_restraints_plus_file_raises(self):
        """Inline restraints + --restraints file => clear error."""
        from alphafold3_mlx.pipeline.input_handler import apply_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        fold_input = self._make_fold_input_with_inline_restraints()
        restraints_path = self._write_restraints_file({
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 2,
                        "chain_j": "B", "residue_j": 2,
                        "target_distance": 8.0,
                    },
                ],
            },
        })

        try:
            with pytest.raises(InputError, match="both.*input JSON.*--restraints"):
                apply_restraints_file(fold_input, restraints_path)
        finally:
            restraints_path.unlink(missing_ok=True)

    def test_inline_guidance_plus_file_guidance_raises(self):
        """Inline guidance + file guidance => clear error."""
        from alphafold3_mlx.pipeline.input_handler import apply_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        fold_input = self._make_fold_input_with_inline_guidance()
        restraints_path = self._write_restraints_file({
            "guidance": {"scale": 3.0, "annealing": "constant"},
        })

        try:
            with pytest.raises(InputError, match="both.*input JSON.*--restraints"):
                apply_restraints_file(fold_input, restraints_path)
        finally:
            restraints_path.unlink(missing_ok=True)

    def test_valid_file_only_restraints_succeeds(self):
        """Valid --restraints file with no inline restraints works."""
        from alphafold3_mlx.pipeline.input_handler import apply_restraints_file

        fold_input = self._make_fold_input_no_restraints()
        restraints_path = self._write_restraints_file({
            "restraints": {
                "distance": [
                    {
                        "chain_i": "A", "residue_i": 1,
                        "chain_j": "B", "residue_j": 1,
                        "target_distance": 5.0,
                    },
                ],
            },
            "guidance": {"scale": 1.0, "annealing": "linear"},
        })

        try:
            result = apply_restraints_file(fold_input, restraints_path)
            assert result._restraints is not None
            assert result._guidance is not None
        finally:
            restraints_path.unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        """Nonexistent --restraints file raises clear error."""
        from alphafold3_mlx.pipeline.input_handler import apply_restraints_file, load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        with pytest.raises(InputError, match="not found"):
            load_restraints_file(Path("/nonexistent/restraints.json"))

    def test_invalid_json_file_raises(self):
        """Malformed JSON in --restraints file raises clear error."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        )
        f.write("{invalid json")
        f.close()
        tmp_path = Path(f.name)

        try:
            with pytest.raises(InputError, match="Invalid JSON"):
                load_restraints_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_empty_restraints_file_raises(self):
        """Restraints file with no 'restraints' or 'guidance' key raises."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        restraints_path = self._write_restraints_file({"other_key": "value"})
        try:
            with pytest.raises(InputError, match="at least one"):
                load_restraints_file(restraints_path)
        finally:
            restraints_path.unlink(missing_ok=True)


# ── Unknown Key Rejection Tests ──────────────────────────────────────────


class TestUnknownKeyRejection:
    """Tests that unknown keys in restraint/guidance dicts are rejected."""

    def test_unknown_top_level_restraint_key(self):
        """Unknown key in restraints dict raises ValueError."""
        data = {
            "distance": [],
            "bogus_key": "value",
        }
        with pytest.raises(ValueError, match="Unknown keys in restraints.*bogus_key"):
            restraint_config_from_dict(data)

    def test_unknown_guidance_key(self):
        """Unknown key in guidance dict raises ValueError."""
        data = {
            "scale": 1.0,
            "unknown_param": 42,
        }
        with pytest.raises(ValueError, match="Unknown keys in guidance.*unknown_param"):
            guidance_config_from_dict(data)

    def test_unknown_contact_entry_key(self):
        """Unknown key inside a contact entry raises ValueError."""
        data = {
            "contact": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "candidates": [{"chain_j": "B", "residue_j": 1}],
                    "extra_field": True,
                },
            ],
        }
        with pytest.raises(ValueError, match="Unknown keys in contact restraint.*extra_field"):
            restraint_config_from_dict(data)

    def test_unknown_distance_entry_key(self):
        """Unknown key inside a distance entry raises ValueError (not TypeError)."""
        data = {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                    "bogus": 99,
                },
            ],
        }
        with pytest.raises(ValueError, match="Invalid distance restraint"):
            restraint_config_from_dict(data)

    def test_valid_keys_accepted(self):
        """All valid keys are accepted without error."""
        data = {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                    "atom_i": "CA", "atom_j": "CA",
                    "sigma": 1.0, "weight": 1.0,
                },
            ],
            "contact": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "candidates": [{"chain_j": "B", "residue_j": 1}],
                    "threshold": 8.0, "weight": 1.0,
                },
            ],
            "repulsive": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "min_distance": 10.0, "weight": 1.0,
                },
            ],
        }
        config = restraint_config_from_dict(data)
        assert len(config.distance) == 1
        assert len(config.contact) == 1
        assert len(config.repulsive) == 1

    def test_valid_guidance_keys_accepted(self):
        """All valid guidance keys are accepted without error."""
        data = {"scale": 2.0, "annealing": "cosine", "start_step": 10, "end_step": 100}
        config = guidance_config_from_dict(data)
        assert config.scale == 2.0
        assert config.annealing == "cosine"


# ── InputError Normalization Tests ───────────────────────────────────────


class TestInputErrorNormalization:
    """Tests that parse failures are normalized to InputError."""

    def test_parse_input_json_unknown_restraint_key_raises_input_error(self):
        """parse_input_json wraps unknown-key ValueError as InputError."""
        from alphafold3_mlx.pipeline.input_handler import parse_input_json
        from alphafold3_mlx.pipeline.errors import InputError

        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
            ],
            "restraints": {"distance": [], "bogus": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(InputError, match="Invalid restraints.*bogus"):
                parse_input_json(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_input_json_unknown_guidance_key_raises_input_error(self):
        """parse_input_json wraps unknown-key ValueError for guidance as InputError."""
        from alphafold3_mlx.pipeline.input_handler import parse_input_json
        from alphafold3_mlx.pipeline.errors import InputError

        input_data = {
            "name": "test",
            "modelSeeds": [42],
            "sequences": [
                {"proteinChain": {"sequence": "MQIFVKTLTG", "count": 1}},
            ],
            "guidance": {"scale": 1.0, "warmup": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(InputError, match="Invalid guidance.*warmup"):
                parse_input_json(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_restraints_file_unknown_key_raises_input_error(self):
        """load_restraints_file wraps unknown-key ValueError as InputError."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        data = {
            "restraints": {"distance": [], "invalid_key": 1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(InputError, match="Invalid restraints.*invalid_key"):
                load_restraints_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_restraints_file_unexpected_top_level_key(self):
        """load_restraints_file rejects unexpected top-level keys (schema)."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        data = {
            "restraints": {"distance": []},
            "guidanc": {"scale": 0.5},  # typo
            "extra": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(InputError, match="unexpected top-level keys.*extra.*guidanc"):
                load_restraints_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_restraints_file_bad_guidance_raises_input_error(self):
        """load_restraints_file wraps invalid guidance as InputError."""
        from alphafold3_mlx.pipeline.input_handler import load_restraints_file
        from alphafold3_mlx.pipeline.errors import InputError

        data = {
            "guidance": {"scale": 1.0, "bad_param": "x"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(InputError, match="Invalid guidance.*bad_param"):
                load_restraints_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)


# ── UNK Residue Early Rejection Tests ────────────────────────────────────


def _make_chains_with_unk() -> dict[str, ChainInfo]:
    """Chain A with 10 residues; position 5 is UNK."""
    res_types = ["ALA"] * 10
    res_types[4] = "UNK"  # residue 5 (1-indexed)
    return {
        "A": ChainInfo(chain_id="A", length=10, residue_types=res_types),
        "B": ChainInfo(chain_id="B", length=10, residue_types=["ALA"] * 10),
    }


class TestUNKResidueEarlyRejection:
    """UNK/X residues must be rejected during early validation (not deferred
    to late resolve_restraints), since they have no dense-24 atom mapping."""

    def test_distance_restraint_on_unk_residue_rejected(self):
        """Distance restraint targeting a UNK residue produces early error."""
        chains = _make_chains_with_unk()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=5, atom_i="CA",
                    chain_j="B", residue_j=1, atom_j="CA",
                    target_distance=8.0,
                ),
            ],
        )
        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "unknown residue type" in errors[0].lower()
        assert "UNK" in errors[0]

    def test_contact_restraint_on_unk_source_rejected(self):
        """Contact restraint with UNK source residue produces early error."""
        chains = _make_chains_with_unk()
        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=5,
                    candidates=[CandidateResidue(chain_j="B", residue_j=1)],
                ),
            ],
        )
        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "unknown residue type" in errors[0].lower()

    def test_repulsive_restraint_on_unk_residue_rejected(self):
        """Repulsive restraint with UNK residue produces early error."""
        chains = _make_chains_with_unk()
        config = RestraintConfig(
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=5,
                    chain_j="B", residue_j=1,
                    min_distance=4.0,
                ),
            ],
        )
        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "unknown residue type" in errors[0].lower()

    def test_x_residue_type_also_rejected(self):
        """Residue type 'X' is rejected the same as UNK."""
        res_types = ["ALA"] * 10
        res_types[2] = "X"
        chains = {
            "A": ChainInfo(chain_id="A", length=10, residue_types=res_types),
            "B": ChainInfo(chain_id="B", length=10, residue_types=["ALA"] * 10),
        }
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=3, atom_i="CA",
                    chain_j="B", residue_j=1, atom_j="CA",
                    target_distance=8.0,
                ),
            ],
        )
        errors = validate_restraints(config, chains)
        assert len(errors) == 1
        assert "unknown residue type" in errors[0].lower()

    def test_valid_residue_next_to_unk_still_passes(self):
        """Restraints on valid residues adjacent to UNK are unaffected."""
        chains = _make_chains_with_unk()
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=4, atom_i="CA",  # ALA, not UNK
                    chain_j="B", residue_j=1, atom_j="CA",
                    target_distance=8.0,
                ),
            ],
        )
        errors = validate_restraints(config, chains)
        assert len(errors) == 0

