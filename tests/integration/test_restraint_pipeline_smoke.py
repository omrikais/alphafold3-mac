"""Fast smoke tests for the restraint pipeline plumbing.

Validates the full restraint pipeline end-to-end WITHOUT model weights or
databases by exercising each stage with synthetic data:

1. Validation: validate_restraints() catches invalid references
2. Resolution: resolve_restraints() maps (chain, residue, atom) → token indices
3. Guidance: build_guidance_fn() produces a callable that returns gradients
4. Satisfaction: satisfaction metrics are computed from positions + restraints

These tests run in <1 second and prove the pipeline wiring is correct.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from alphafold3_mlx.restraints.guidance import build_guidance_fn
from alphafold3_mlx.restraints.loss import combined_restraint_loss
from alphafold3_mlx.restraints.resolve import resolve_restraints, RestraintResolutionError
from alphafold3_mlx.restraints.types import (
    CandidateResidue,
    ContactRestraint,
    DistanceRestraint,
    GuidanceConfig,
    RepulsiveRestraint,
    ResolvedContactRestraint,
    ResolvedDistanceRestraint,
    ResolvedRepulsiveRestraint,
    RestraintConfig,
    restraint_config_from_dict,
    guidance_config_from_dict,
)
from alphafold3_mlx.restraints.validate import ChainInfo, validate_restraints


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_token_layout(
    chain_lengths: dict[str, int],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic token arrays mimicking the featurized batch layout.

    All residues are alanine (aatype=0).  Token ordering follows chain_id
    alphabetical order, matching AF3 convention.

    Returns:
        (chain_ids, asym_id, residue_index, aatype, mask) arrays.
    """
    chain_ids = sorted(chain_lengths.keys())
    asym_list, resid_list, aa_list, mask_list = [], [], [], []
    for i, cid in enumerate(chain_ids):
        length = chain_lengths[cid]
        asym_list.extend([i + 1] * length)        # 1-based
        resid_list.extend(range(1, length + 1))     # 1-based PDB numbering
        aa_list.extend([0] * length)                # 0 = ALA
        mask_list.extend([1] * length)

    return (
        chain_ids,
        np.array(asym_list, dtype=np.int32),
        np.array(resid_list, dtype=np.int32),
        np.array(aa_list, dtype=np.int32),
        np.array(mask_list, dtype=np.float32),
    )


def _make_positions(num_tokens: int, num_atoms: int = 24, spacing: float = 3.8) -> mx.array:
    """Create linearly spaced positions along x-axis (protein-like CA spacing)."""
    pos = mx.zeros((num_tokens, num_atoms, 3))
    for t in range(num_tokens):
        # Place CA (atom 1) at (t*spacing, 0, 0)
        pos = pos.at[t, 1].add(mx.array([float(t) * spacing, 0.0, 0.0]))
    mx.eval(pos)
    return pos


# ── Stage 1: Validation ─────────────────────────────────────────────────────


class TestValidationStage:
    """validate_restraints blocks bad input before expensive work."""

    def test_valid_config_passes(self):
        """Well-formed restraints produce zero errors."""
        chains = {
            "A": ChainInfo("A", 50, ["ALA"] * 50),
            "B": ChainInfo("B", 50, ["ALA"] * 50),
        }
        config = RestraintConfig(
            distance=[DistanceRestraint("A", 10, "B", 20, target_distance=8.0)],
            contact=[ContactRestraint("A", 5, [CandidateResidue("B", 15)], threshold=10.0)],
            repulsive=[RepulsiveRestraint("A", 1, "B", 50, min_distance=12.0)],
        )
        errors = validate_restraints(config, chains)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_invalid_chain_blocked(self):
        """Nonexistent chain reference produces an actionable error."""
        chains = {"A": ChainInfo("A", 50, ["ALA"] * 50)}
        config = RestraintConfig(
            distance=[DistanceRestraint("A", 1, "X", 1, target_distance=5.0)],
        )
        errors = validate_restraints(config, chains)
        assert any("X" in e and "does not exist" in e for e in errors)

    def test_bad_guidance_blocked(self):
        """Guidance with end_step > num_steps produces an error."""
        chains = {
            "A": ChainInfo("A", 50, ["ALA"] * 50),
            "B": ChainInfo("B", 50, ["ALA"] * 50),
        }
        config = RestraintConfig(
            distance=[DistanceRestraint("A", 1, "B", 1, target_distance=5.0)],
        )
        guidance = GuidanceConfig(scale=1.0, end_step=300)
        errors = validate_restraints(config, chains, guidance=guidance, num_diffusion_steps=200)
        assert any("end_step" in e for e in errors)


# ── Stage 2: Resolution ─────────────────────────────────────────────────────


class TestResolutionStage:
    """resolve_restraints maps user references to internal indices."""

    def test_distance_resolution_produces_valid_indices(self):
        """Distance restraint resolves to (token_idx, dense_atom_idx) tuples."""
        config = RestraintConfig(
            distance=[DistanceRestraint("A", 5, "B", 10, target_distance=8.0)],
        )
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})

        resolved_dist, resolved_contact, resolved_repulsive = resolve_restraints(
            config, chain_ids, asym_id, resid, aatype, mask,
        )

        assert len(resolved_dist) == 1
        assert len(resolved_contact) == 0
        assert len(resolved_repulsive) == 0

        rd = resolved_dist[0]
        # Token for chain A, residue 5 should be index 4 (0-based)
        assert rd.atom_i_idx[0] == 4
        # Token for chain B, residue 10 should be index 20+9=29 (0-based)
        assert rd.atom_j_idx[0] == 29
        # CA dense index is 1
        assert rd.atom_i_idx[1] == 1
        assert rd.atom_j_idx[1] == 1

    def test_contact_resolution_multiple_candidates(self):
        """Contact restraint with 3 candidates produces 3 candidate indices."""
        config = RestraintConfig(
            contact=[ContactRestraint("A", 5, [
                CandidateResidue("B", 10),
                CandidateResidue("B", 15),
                CandidateResidue("B", 20),
            ])],
        )
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})

        _, resolved_contact, _ = resolve_restraints(
            config, chain_ids, asym_id, resid, aatype, mask,
        )

        assert len(resolved_contact) == 1
        rc = resolved_contact[0]
        assert len(rc.candidate_atom_idxs) == 3

    def test_repulsive_resolution_uses_ca(self):
        """Repulsive restraint resolves to CA atom indices."""
        config = RestraintConfig(
            repulsive=[RepulsiveRestraint("A", 1, "B", 1, min_distance=10.0)],
        )
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})

        _, _, resolved_repulsive = resolve_restraints(
            config, chain_ids, asym_id, resid, aatype, mask,
        )

        assert len(resolved_repulsive) == 1
        rr = resolved_repulsive[0]
        assert rr.atom_i_idx[1] == 1  # CA
        assert rr.atom_j_idx[1] == 1  # CA

    def test_invalid_chain_raises_resolution_error(self):
        """Referencing a chain not in the token layout raises RestraintResolutionError."""
        config = RestraintConfig(
            distance=[DistanceRestraint("A", 1, "Z", 1, target_distance=5.0)],
        )
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})

        with pytest.raises(RestraintResolutionError, match="chain 'Z'"):
            resolve_restraints(config, chain_ids, asym_id, resid, aatype, mask)


# ── Stage 3: Guidance function ───────────────────────────────────────────────


class TestGuidanceFunctionStage:
    """build_guidance_fn produces a callable that returns valid gradients."""

    def test_guidance_fn_returns_nonzero_gradient(self):
        """Guidance function produces nonzero gradient for unsatisfied restraint."""
        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
        ]
        fn = build_guidance_fn(
            resolved, None, None,
            GuidanceConfig(scale=1.0, annealing="constant"),
            num_steps=200,
        )

        # Positions where CA atoms are 20A apart (much larger than 5A target)
        positions = _make_positions(10, num_atoms=24, spacing=4.0)
        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)

        assert grad.shape == positions.shape
        assert not mx.any(mx.isnan(grad)).item()
        assert float(mx.sum(mx.abs(grad))) > 0

    def test_guidance_fn_zero_outside_step_range(self):
        """Guidance is zero before start_step and after end_step."""
        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
        ]
        fn = build_guidance_fn(
            resolved, None, None,
            GuidanceConfig(scale=1.0, annealing="constant", start_step=50, end_step=150),
            num_steps=200,
        )
        positions = _make_positions(10)

        # Before start
        grad_before = fn(positions, mx.array(10.0), step=10)
        mx.eval(grad_before)
        assert float(mx.sum(mx.abs(grad_before))) == 0.0

        # After end
        grad_after = fn(positions, mx.array(10.0), step=180)
        mx.eval(grad_after)
        assert float(mx.sum(mx.abs(grad_after))) == 0.0

    def test_guidance_fn_with_all_restraint_types(self):
        """Guidance function handles distance + contact + repulsive together."""
        rd = [ResolvedDistanceRestraint((0, 1), (5, 1), 5.0, 1.0, 1.0)]
        rc = [ResolvedContactRestraint((1, 1), [(6, 1)], 8.0, 1.0)]
        rr = [ResolvedRepulsiveRestraint((2, 1), (7, 1), 15.0, 1.0)]

        fn = build_guidance_fn(
            rd, rc, rr,
            GuidanceConfig(scale=1.0, annealing="constant"),
            num_steps=200,
        )

        positions = _make_positions(10, spacing=4.0)
        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)).item()
        assert float(mx.sum(mx.abs(grad))) > 0


# ── Stage 4: Full pipeline plumbing (validate → resolve → guidance) ──────────


class TestFullPipelinePlumbing:
    """End-to-end plumbing: parse → validate → resolve → build guidance → loss.

    Proves the data flows correctly through all stages without model weights.
    """

    def test_distance_pipeline_from_dict_to_gradient(self):
        """Dict → RestraintConfig → validate → resolve → guidance → gradient."""
        # Stage 0: Parse from dict (as would come from JSON)
        raw = {
            "distance": [{
                "chain_i": "A", "residue_i": 5, "atom_i": "CA",
                "chain_j": "B", "residue_j": 10, "atom_j": "CA",
                "target_distance": 8.0, "sigma": 2.0, "weight": 1.0,
            }],
        }
        config = restraint_config_from_dict(raw)
        assert config.total_count == 1

        # Stage 1: Validate
        chains = {
            "A": ChainInfo("A", 20, ["ALA"] * 20),
            "B": ChainInfo("B", 20, ["ALA"] * 20),
        }
        errors = validate_restraints(config, chains)
        assert errors == [], f"Validation failed: {errors}"

        # Stage 2: Resolve
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})
        resolved_dist, resolved_contact, resolved_repulsive = resolve_restraints(
            config, chain_ids, asym_id, resid, aatype, mask,
        )
        assert len(resolved_dist) == 1

        # Stage 3: Build guidance
        guidance = GuidanceConfig(scale=1.0, annealing="linear")
        fn = build_guidance_fn(
            resolved_dist, resolved_contact, resolved_repulsive,
            guidance, num_steps=200,
        )

        # Stage 4: Compute gradient
        positions = _make_positions(40, num_atoms=24, spacing=3.8)
        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)

        assert grad.shape == positions.shape
        assert not mx.any(mx.isnan(grad)).item()
        # Gradient should be nonzero (restraint is unsatisfied at 3.8*25=95A apart)
        assert float(mx.sum(mx.abs(grad))) > 0

    def test_mixed_restraints_pipeline(self):
        """All three restraint types flow through the full pipeline."""
        raw = {
            "distance": [{
                "chain_i": "A", "residue_i": 5,
                "chain_j": "B", "residue_j": 10,
                "target_distance": 8.0,
            }],
            "contact": [{
                "chain_i": "A", "residue_i": 3,
                "candidates": [
                    {"chain_j": "B", "residue_j": 8},
                    {"chain_j": "B", "residue_j": 12},
                ],
                "threshold": 10.0,
            }],
            "repulsive": [{
                "chain_i": "A", "residue_i": 1,
                "chain_j": "B", "residue_j": 1,
                "min_distance": 15.0,
            }],
        }
        config = restraint_config_from_dict(raw)
        assert config.total_count == 3

        chains = {
            "A": ChainInfo("A", 20, ["ALA"] * 20),
            "B": ChainInfo("B", 20, ["ALA"] * 20),
        }
        errors = validate_restraints(config, chains)
        assert errors == []

        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 20, "B": 20})
        rd, rc, rr = resolve_restraints(
            config, chain_ids, asym_id, resid, aatype, mask,
        )
        assert len(rd) == 1
        assert len(rc) == 1
        assert len(rr) == 1

        fn = build_guidance_fn(rd, rc, rr, GuidanceConfig(), num_steps=200)
        positions = _make_positions(40, num_atoms=24)
        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)

        assert not mx.any(mx.isnan(grad)).item()
        assert float(mx.sum(mx.abs(grad))) > 0

    def test_invalid_restraints_blocked_before_resolution(self):
        """Validation catches bad references, preventing resolution from running."""
        config = RestraintConfig(
            distance=[DistanceRestraint("Z", 1, "A", 1, target_distance=5.0)],
        )
        chains = {"A": ChainInfo("A", 50, ["ALA"] * 50)}

        # Validation catches the error
        errors = validate_restraints(config, chains)
        assert len(errors) >= 1

        # If we somehow reached resolution, it would also fail
        chain_ids, asym_id, resid, aatype, mask = _make_token_layout({"A": 50})
        with pytest.raises(RestraintResolutionError):
            resolve_restraints(config, chain_ids, asym_id, resid, aatype, mask)


# ── Loss correctness smoke tests ────────────────────────────────────────────


class TestLossCorrectness:
    """Sanity checks that loss functions produce expected values."""

    def test_satisfied_distance_has_near_zero_loss(self):
        """When atoms are at the target distance, loss is ~0."""
        # Place atoms exactly 8.0A apart
        pos = mx.zeros((10, 24, 3))
        pos = pos.at[0, 1].add(mx.array([0.0, 0.0, 0.0]))
        pos = pos.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(pos)

        resolved = [ResolvedDistanceRestraint((0, 1), (5, 1), 8.0, 1.0, 1.0)]
        loss = combined_restraint_loss(pos, resolved)
        mx.eval(loss)
        assert float(loss) < 0.01  # ~epsilon from sqrt safety term

    def test_violated_distance_has_positive_loss(self):
        """When atoms are far from target, loss is large."""
        pos = mx.zeros((10, 24, 3))
        pos = pos.at[0, 1].add(mx.array([0.0, 0.0, 0.0]))
        pos = pos.at[5, 1].add(mx.array([20.0, 0.0, 0.0]))
        mx.eval(pos)

        resolved = [ResolvedDistanceRestraint((0, 1), (5, 1), 5.0, 1.0, 1.0)]
        loss = combined_restraint_loss(pos, resolved)
        mx.eval(loss)
        # (20 - 5)^2 / 1^2 = 225
        assert float(loss) > 100.0

    def test_repulsive_zero_when_far(self):
        """Repulsive loss is zero when atoms are beyond min_distance."""
        pos = mx.zeros((10, 24, 3))
        pos = pos.at[0, 1].add(mx.array([0.0, 0.0, 0.0]))
        pos = pos.at[5, 1].add(mx.array([20.0, 0.0, 0.0]))
        mx.eval(pos)

        resolved_rep = [ResolvedRepulsiveRestraint((0, 1), (5, 1), 10.0, 1.0)]
        loss = combined_restraint_loss(pos, [], resolved_repulsive=resolved_rep)
        mx.eval(loss)
        assert float(loss) < 1e-6


# ── Parsing round-trip tests ────────────────────────────────────────────────


class TestParsingRoundTrip:
    """restraint_config_from_dict and guidance_config_from_dict validation."""

    def test_unknown_restraint_key_raises(self):
        """Unknown key in restraints dict raises ValueError."""
        with pytest.raises(ValueError, match="Unknown keys.*bogus"):
            restraint_config_from_dict({"distance": [], "bogus": True})

    def test_unknown_guidance_key_raises(self):
        """Unknown key in guidance dict raises ValueError."""
        with pytest.raises(ValueError, match="Unknown keys.*warmup"):
            guidance_config_from_dict({"scale": 1.0, "warmup": True})

    def test_valid_guidance_round_trip(self):
        """Valid guidance dict parses correctly."""
        g = guidance_config_from_dict({
            "scale": 2.0, "annealing": "cosine",
            "start_step": 10, "end_step": 150,
        })
        assert g.scale == 2.0
        assert g.annealing == "cosine"
        assert g.start_step == 10
        assert g.end_step == 150

    def test_distance_restraint_rejects_negative_target(self):
        """target_distance <= 0 raises ValueError at construction time."""
        with pytest.raises(ValueError, match="target_distance"):
            DistanceRestraint("A", 1, "B", 1, target_distance=-1.0)

    def test_contact_restraint_rejects_empty_candidates(self):
        """Contact with 0 candidates raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            ContactRestraint("A", 1, candidates=[])
