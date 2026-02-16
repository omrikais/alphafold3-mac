"""Unit tests for restraint type dataclasses.

Tests construction, defaults, validation constraints, and parsing helpers.
"""

import pytest

from alphafold3_mlx.restraints.types import (
    CandidateResidue,
    ContactRestraint,
    ContactSatisfaction,
    DistanceRestraint,
    DistanceSatisfaction,
    GuidanceConfig,
    RepulsiveRestraint,
    RepulsiveSatisfaction,
    ResolvedContactRestraint,
    ResolvedDistanceRestraint,
    ResolvedRepulsiveRestraint,
    RestraintConfig,
    guidance_config_from_dict,
    restraint_config_from_dict,
)


# ── DistanceRestraint ────────────────────────────────────────────────────────


class TestDistanceRestraint:
    def test_construction_with_defaults(self):
        r = DistanceRestraint(
            chain_i="A", residue_i=48,
            chain_j="B", residue_j=76,
            target_distance=1.5,
        )
        assert r.atom_i == "CA"
        assert r.atom_j == "CA"
        assert r.sigma == 1.0
        assert r.weight == 1.0

    def test_construction_with_all_fields(self):
        r = DistanceRestraint(
            chain_i="A", residue_i=48, atom_i="NZ",
            chain_j="B", residue_j=76, atom_j="C",
            target_distance=1.5, sigma=0.5, weight=2.0,
        )
        assert r.atom_i == "NZ"
        assert r.atom_j == "C"
        assert r.sigma == 0.5
        assert r.weight == 2.0
        assert r.target_distance == 1.5

    def test_negative_target_distance_raises(self):
        with pytest.raises(ValueError, match="target_distance must be > 0"):
            DistanceRestraint(
                chain_i="A", residue_i=1,
                chain_j="B", residue_j=1,
                target_distance=-1.0,
            )

    def test_zero_target_distance_raises(self):
        with pytest.raises(ValueError, match="target_distance must be > 0"):
            DistanceRestraint(
                chain_i="A", residue_i=1,
                chain_j="B", residue_j=1,
                target_distance=0.0,
            )

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            DistanceRestraint(
                chain_i="A", residue_i=1,
                chain_j="B", residue_j=1,
                target_distance=5.0, sigma=-1.0,
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight must be > 0"):
            DistanceRestraint(
                chain_i="A", residue_i=1,
                chain_j="B", residue_j=1,
                target_distance=5.0, weight=-0.5,
            )

    def test_frozen(self):
        r = DistanceRestraint(
            chain_i="A", residue_i=1,
            chain_j="B", residue_j=1,
            target_distance=5.0,
        )
        with pytest.raises(AttributeError):
            r.target_distance = 10.0  # type: ignore


# ── ContactRestraint ─────────────────────────────────────────────────────────


class TestContactRestraint:
    def test_construction_with_defaults(self):
        r = ContactRestraint(
            chain_i="A", residue_i=11,
            candidates=[CandidateResidue(chain_j="B", residue_j=42)],
        )
        assert r.threshold == 8.0
        assert r.weight == 1.0
        assert len(r.candidates) == 1

    def test_multiple_candidates(self):
        r = ContactRestraint(
            chain_i="A", residue_i=11,
            candidates=[
                CandidateResidue(chain_j="B", residue_j=42),
                CandidateResidue(chain_j="B", residue_j=43),
                CandidateResidue(chain_j="B", residue_j=44),
            ],
        )
        assert len(r.candidates) == 3

    def test_empty_candidates_raises(self):
        with pytest.raises(ValueError, match="at least 1 candidate"):
            ContactRestraint(
                chain_i="A", residue_i=11,
                candidates=[],
            )

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be > 0"):
            ContactRestraint(
                chain_i="A", residue_i=11,
                candidates=[CandidateResidue(chain_j="B", residue_j=42)],
                threshold=-1.0,
            )


# ── RepulsiveRestraint ───────────────────────────────────────────────────────


class TestRepulsiveRestraint:
    def test_construction_with_defaults(self):
        r = RepulsiveRestraint(
            chain_i="A", residue_i=1,
            chain_j="B", residue_j=1,
            min_distance=15.0,
        )
        assert r.weight == 1.0

    def test_negative_min_distance_raises(self):
        with pytest.raises(ValueError, match="min_distance must be > 0"):
            RepulsiveRestraint(
                chain_i="A", residue_i=1,
                chain_j="B", residue_j=1,
                min_distance=-5.0,
            )


# ── GuidanceConfig ───────────────────────────────────────────────────────────


class TestGuidanceConfig:
    def test_defaults(self):
        g = GuidanceConfig()
        assert g.scale == 1.0
        assert g.annealing == "linear"
        assert g.start_step == 0
        assert g.end_step is None

    def test_valid_annealing_values(self):
        for val in ("linear", "cosine", "constant"):
            g = GuidanceConfig(annealing=val)
            assert g.annealing == val

    def test_invalid_annealing_raises(self):
        with pytest.raises(ValueError, match="annealing must be"):
            GuidanceConfig(annealing="exponential")

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be >= 0"):
            GuidanceConfig(scale=-1.0)

    def test_zero_scale_allowed(self):
        g = GuidanceConfig(scale=0.0)
        assert g.scale == 0.0

    def test_negative_start_step_raises(self):
        with pytest.raises(ValueError, match="start_step must be >= 0"):
            GuidanceConfig(start_step=-1)

    def test_end_step_must_be_after_start(self):
        with pytest.raises(ValueError, match="end_step.*must be >= start_step"):
            GuidanceConfig(start_step=100, end_step=50)

    def test_single_step_guidance_window(self):
        """end_step == start_step is valid (single-step window)."""
        g = GuidanceConfig(start_step=5, end_step=5)
        assert g.start_step == 5
        assert g.end_step == 5


# ── RestraintConfig ──────────────────────────────────────────────────────────


class TestRestraintConfig:
    def test_empty_config(self):
        c = RestraintConfig()
        assert c.total_count == 0
        assert c.is_empty is True

    def test_total_count(self):
        c = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    min_distance=15.0,
                ),
            ],
        )
        assert c.total_count == 2
        assert c.is_empty is False


# ── Resolved types ───────────────────────────────────────────────────────────


class TestResolvedTypes:
    def test_resolved_distance(self):
        r = ResolvedDistanceRestraint(
            atom_i_idx=(0, 1), atom_j_idx=(75, 2),
            target_distance=1.5, sigma=0.5, weight=2.0,
        )
        assert r.atom_i_idx == (0, 1)
        assert r.atom_j_idx == (75, 2)

    def test_resolved_contact(self):
        r = ResolvedContactRestraint(
            source_atom_idx=(10, 1),
            candidate_atom_idxs=[(41, 1), (42, 1)],
            threshold=8.0, weight=1.0,
        )
        assert len(r.candidate_atom_idxs) == 2

    def test_resolved_repulsive(self):
        r = ResolvedRepulsiveRestraint(
            atom_i_idx=(0, 1), atom_j_idx=(75, 1),
            min_distance=15.0, weight=1.0,
        )
        assert r.min_distance == 15.0


# ── Satisfaction types ───────────────────────────────────────────────────────


class TestSatisfactionTypes:
    def test_distance_satisfaction(self):
        s = DistanceSatisfaction(
            chain_i="A", residue_i=48, atom_i="NZ",
            chain_j="B", residue_j=76, atom_j="C",
            target_distance=1.5, actual_distance=1.72,
            satisfied=True,
        )
        assert s.satisfied is True
        assert s.actual_distance == 1.72

    def test_contact_satisfaction(self):
        s = ContactSatisfaction(
            chain_i="A", residue_i=11,
            closest_candidate_chain="B", closest_candidate_residue=42,
            threshold=8.0, actual_distance=6.3,
            satisfied=True,
        )
        assert s.satisfied is True

    def test_repulsive_satisfaction(self):
        s = RepulsiveSatisfaction(
            chain_i="A", residue_i=1,
            chain_j="B", residue_j=1,
            min_distance=15.0, actual_distance=22.4,
            satisfied=True,
        )
        assert s.satisfied is True


# ── Parsing helpers ──────────────────────────────────────────────────────────


class TestParsingHelpers:
    def test_restraint_config_from_dict(self):
        data = {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 48, "atom_i": "NZ",
                    "chain_j": "B", "residue_j": 76, "atom_j": "C",
                    "target_distance": 1.5, "sigma": 0.5, "weight": 2.0,
                }
            ],
            "contact": [
                {
                    "chain_i": "A", "residue_i": 11,
                    "candidates": [{"chain_j": "B", "residue_j": 42}],
                    "threshold": 8.0,
                }
            ],
            "repulsive": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "min_distance": 15.0,
                }
            ],
        }
        config = restraint_config_from_dict(data)
        assert len(config.distance) == 1
        assert len(config.contact) == 1
        assert len(config.repulsive) == 1
        assert config.distance[0].atom_i == "NZ"
        assert config.contact[0].candidates[0].chain_j == "B"
        assert config.repulsive[0].min_distance == 15.0

    def test_restraint_config_from_dict_defaults(self):
        data = {
            "distance": [
                {
                    "chain_i": "A", "residue_i": 1,
                    "chain_j": "B", "residue_j": 1,
                    "target_distance": 5.0,
                }
            ],
        }
        config = restraint_config_from_dict(data)
        assert config.distance[0].atom_i == "CA"
        assert config.distance[0].sigma == 1.0
        assert config.distance[0].weight == 1.0

    def test_guidance_config_from_dict_defaults(self):
        config = guidance_config_from_dict({})
        assert config.scale == 1.0
        assert config.annealing == "linear"
        assert config.start_step == 0
        assert config.end_step is None

    def test_guidance_config_from_dict_custom(self):
        config = guidance_config_from_dict({
            "scale": 0.5,
            "annealing": "cosine",
            "start_step": 10,
            "end_step": 150,
        })
        assert config.scale == 0.5
        assert config.annealing == "cosine"
        assert config.start_step == 10
        assert config.end_step == 150

    def test_restraint_config_from_dict_rejects_empty_object(self):
        with pytest.raises(
            ValueError,
            match="'restraints' must define at least one of: distance, contact, repulsive",
        ):
            restraint_config_from_dict({})

    def test_restraint_config_from_dict_rejects_non_object_contact_entry(self):
        with pytest.raises(
            ValueError,
            match=r"Invalid contact restraint \[0\]: must be a JSON object, got int",
        ):
            restraint_config_from_dict({"contact": [123]})

    # ── Malformed type rejection ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "bad_input,expected_type_name",
        [
            (["distance"], "list"),
            ("not a dict", "str"),
            (42, "int"),
            (True, "bool"),
            (None, "NoneType"),
        ],
    )
    def test_restraint_config_from_dict_rejects_non_dict(self, bad_input, expected_type_name):
        with pytest.raises(ValueError, match=f"must be a JSON object, got {expected_type_name}"):
            restraint_config_from_dict(bad_input)

    @pytest.mark.parametrize(
        "bad_input,expected_type_name",
        [
            (["scale"], "list"),
            ("not a dict", "str"),
            (42, "int"),
            (True, "bool"),
            (None, "NoneType"),
        ],
    )
    def test_guidance_config_from_dict_rejects_non_dict(self, bad_input, expected_type_name):
        with pytest.raises(ValueError, match=f"must be a JSON object, got {expected_type_name}"):
            guidance_config_from_dict(bad_input)
