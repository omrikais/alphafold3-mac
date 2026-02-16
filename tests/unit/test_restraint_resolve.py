"""Unit tests for restraint index resolution.

Tests valid references resolve correctly, invalid references produce clear errors,
and CA default when atom is omitted.
"""

import numpy as np
import pytest

from alphafold3_mlx.restraints.resolve import (
    RestraintResolutionError,
    resolve_restraints,
)
from alphafold3_mlx.restraints.types import (
    CandidateResidue,
    ContactRestraint,
    DistanceRestraint,
    RepulsiveRestraint,
    RestraintConfig,
)


def _make_two_chain_layout():
    """Create a minimal two-chain layout for testing.

    Chain A: 10 residues (ALA), asym_id=1, residue_index=1..10
    Chain B: 10 residues (LYS), asym_id=2, residue_index=1..10
    Total: 20 tokens.
    """
    chain_ids = ["A", "B"]
    n = 20  # 10 per chain
    asym_id = np.array([1]*10 + [2]*10, dtype=np.int32)
    residue_index = np.array(list(range(1, 11)) + list(range(1, 11)), dtype=np.int32)
    # aatype: 0=ALA for chain A, 11=LYS for chain B
    aatype = np.array([0]*10 + [11]*10, dtype=np.int32)
    mask = np.ones(n, dtype=np.float32)
    return chain_ids, asym_id, residue_index, aatype, mask


class TestResolveDistanceRestraint:
    """Tests for resolving distance restraints."""

    def test_valid_ca_to_ca(self):
        """Default CA atoms resolve correctly."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=5,
                    chain_j="B", residue_j=3,
                    target_distance=5.0,
                    # atom_i and atom_j default to "CA"
                ),
            ],
        )

        resolved_d, resolved_c, resolved_r = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert len(resolved_d) == 1
        r = resolved_d[0]
        # Chain A, residue 5 → token index 4 (0-based)
        assert r.atom_i_idx[0] == 4
        assert r.atom_i_idx[1] == 1  # CA is atom37 index 1
        # Chain B, residue 3 → token index 12 (10 + 2)
        assert r.atom_j_idx[0] == 12
        assert r.atom_j_idx[1] == 1  # CA
        assert r.target_distance == 5.0
        assert r.sigma == 1.0
        assert r.weight == 1.0

    def test_valid_specific_atoms(self):
        """Specific atom names resolve correctly."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="B", residue_i=1, atom_i="NZ",  # LYS has NZ
                    chain_j="A", residue_j=1, atom_j="N",   # ALA has N
                    target_distance=3.0,
                ),
            ],
        )

        resolved_d, _, _ = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert len(resolved_d) == 1
        r = resolved_d[0]
        # Chain B residue 1 → token 10
        assert r.atom_i_idx[0] == 10
        # Chain A residue 1 → token 0
        assert r.atom_j_idx[0] == 0

    def test_invalid_chain_raises(self):
        """Nonexistent chain ID raises RestraintResolutionError."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="Z", residue_i=1,
                    chain_j="A", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="chain 'Z' not found"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_invalid_residue_raises(self):
        """Out-of-range residue number raises RestraintResolutionError."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=9999,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="residue 9999"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_invalid_atom_for_residue_type(self):
        """Atom not valid for residue type raises error."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        # ALA doesn't have NZ (only LYS does)
        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1, atom_i="NZ",  # ALA has no NZ
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="not valid for ALA"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_invalid_atom37_name_raises(self):
        """Completely invalid atom name raises error."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1, atom_i="FAKE",
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="not a valid atom37"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )


class TestResolveContactRestraint:
    """Tests for resolving contact restraints."""

    def test_valid_contact(self):
        """Contact restraint resolves source and candidates to CA atoms."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=5,
                    candidates=[
                        CandidateResidue(chain_j="B", residue_j=3),
                        CandidateResidue(chain_j="B", residue_j=7),
                    ],
                ),
            ],
        )

        _, resolved_c, _ = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert len(resolved_c) == 1
        r = resolved_c[0]
        assert r.source_atom_idx[0] == 4  # Chain A, residue 5
        assert r.source_atom_idx[1] == 1  # CA
        assert len(r.candidate_atom_idxs) == 2
        assert r.candidate_atom_idxs[0][0] == 12  # Chain B, residue 3
        assert r.candidate_atom_idxs[1][0] == 16  # Chain B, residue 7


class TestResolveRepulsiveRestraint:
    """Tests for resolving repulsive restraints."""

    def test_valid_repulsive(self):
        """Repulsive restraint resolves both residues to CA atoms."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    min_distance=15.0,
                ),
            ],
        )

        _, _, resolved_r = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert len(resolved_r) == 1
        r = resolved_r[0]
        assert r.atom_i_idx[0] == 0   # Chain A, residue 1
        assert r.atom_i_idx[1] == 1   # CA
        assert r.atom_j_idx[0] == 10  # Chain B, residue 1
        assert r.atom_j_idx[1] == 1   # CA
        assert r.min_distance == 15.0


class TestResolveWithMasking:
    """Tests for resolution with masked/padded tokens."""

    def test_masked_tokens_excluded(self):
        """Padded tokens (mask=0) are not resolved to."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()
        # Mask out last 5 tokens of chain B (residues 6-10)
        mask[15:] = 0.0

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=8,  # This is masked
                    target_distance=5.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="residue 8"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )


class TestUNKResidueRejection:
    """Tests that UNK/X residues are rejected for CA-based restraints.

    UNK residues have no CA in the dense-24 layout (all slots map to N).
    Contact and repulsive restraints must raise RestraintResolutionError
    instead of silently using wrong coordinates.
    """

    def _make_layout_with_unk(self):
        """Two-chain layout where chain B residue 5 is UNK (aatype=20)."""
        chain_ids = ["A", "B"]
        n = 20
        asym_id = np.array([1]*10 + [2]*10, dtype=np.int32)
        residue_index = np.array(list(range(1, 11)) + list(range(1, 11)), dtype=np.int32)
        aatype = np.array([0]*10 + [11]*10, dtype=np.int32)  # ALA + LYS
        aatype[14] = 20  # Chain B, residue 5 → UNK
        mask = np.ones(n, dtype=np.float32)
        return chain_ids, asym_id, residue_index, aatype, mask

    def test_contact_on_unk_source_raises(self):
        """Contact restraint with UNK source residue raises resolution error."""
        chain_ids, asym_id, residue_index, aatype, mask = self._make_layout_with_unk()

        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="B", residue_i=5,  # UNK residue
                    candidates=[
                        CandidateResidue(chain_j="A", residue_j=1),
                    ],
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="no dense-24 mapping"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_contact_on_unk_candidate_raises(self):
        """Contact restraint with UNK candidate residue raises resolution error."""
        chain_ids, asym_id, residue_index, aatype, mask = self._make_layout_with_unk()

        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=1,
                    candidates=[
                        CandidateResidue(chain_j="B", residue_j=5),  # UNK
                    ],
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="no dense-24 mapping"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_repulsive_on_unk_raises(self):
        """Repulsive restraint on UNK residue raises resolution error."""
        chain_ids, asym_id, residue_index, aatype, mask = self._make_layout_with_unk()

        config = RestraintConfig(
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=5,  # UNK residue
                    min_distance=10.0,
                ),
            ],
        )

        with pytest.raises(RestraintResolutionError, match="no dense-24 mapping"):
            resolve_restraints(
                config, chain_ids, asym_id, residue_index, aatype, mask,
            )

    def test_standard_residues_still_resolve(self):
        """Standard residues in same layout still resolve correctly."""
        chain_ids, asym_id, residue_index, aatype, mask = self._make_layout_with_unk()

        config = RestraintConfig(
            contact=[
                ContactRestraint(
                    chain_i="A", residue_i=1,  # ALA — has CA
                    candidates=[
                        CandidateResidue(chain_j="B", residue_j=1),  # LYS — has CA
                    ],
                ),
            ],
            repulsive=[
                RepulsiveRestraint(
                    chain_i="A", residue_i=2,
                    chain_j="B", residue_j=2,
                    min_distance=10.0,
                ),
            ],
        )

        _, resolved_c, resolved_r = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert len(resolved_c) == 1
        assert resolved_c[0].source_atom_idx[1] == 1  # CA dense idx
        assert len(resolved_r) == 1
        assert resolved_r[0].atom_i_idx[1] == 1  # CA dense idx


class TestCADefault:
    """Tests for atom defaults to CA when omitted."""

    def test_ca_default_distance(self):
        """Distance restraint without atom fields uses CA."""
        chain_ids, asym_id, residue_index, aatype, mask = _make_two_chain_layout()

        config = RestraintConfig(
            distance=[
                DistanceRestraint(
                    chain_i="A", residue_i=1,
                    chain_j="B", residue_j=1,
                    target_distance=5.0,
                    # atom_i and atom_j omitted → "CA"
                ),
            ],
        )

        resolved_d, _, _ = resolve_restraints(
            config, chain_ids, asym_id, residue_index, aatype, mask,
        )

        assert resolved_d[0].atom_i_idx[1] == 1  # CA index in atom37
        assert resolved_d[0].atom_j_idx[1] == 1
