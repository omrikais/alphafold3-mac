"""Unit tests for restraint loss functions.

Tests distance_loss gradient correctness, harmonic shape, sigma/weight scaling,
numerical stability, and contact_loss 1vN semantics (non-negativity, threshold
satisfaction, gradient direction).
"""

import math

import mlx.core as mx
import pytest

from alphafold3_mlx.restraints.loss import combined_restraint_loss, contact_loss, distance_loss
from alphafold3_mlx.restraints.types import ResolvedContactRestraint, ResolvedDistanceRestraint


class TestDistanceLoss:
    """Tests for the distance_loss function."""

    def test_harmonic_shape(self):
        """Loss is zero when distance equals target."""
        # Place atoms exactly 5.0 apart along x-axis
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[0, 1].add(mx.array([0.0, 0.0, 0.0]))
        positions = positions.at[5, 1].add(mx.array([5.0, 0.0, 0.0]))
        mx.eval(positions)

        loss = distance_loss(
            positions,
            atom_i_idx=(0, 1),
            atom_j_idx=(5, 1),
            target_distance=5.0,
            sigma=1.0,
            weight=1.0,
        )
        mx.eval(loss)
        # Should be very close to zero (only 1e-8 stability offset)
        assert float(loss) < 1e-6

    def test_nonzero_loss_when_not_at_target(self):
        """Loss is positive when distance differs from target."""
        positions = mx.zeros((10, 37, 3))
        # Place atom j at (8, 0, 0) — distance = 8.0
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        loss = distance_loss(
            positions,
            atom_i_idx=(0, 1),
            atom_j_idx=(5, 1),
            target_distance=5.0,
            sigma=1.0,
            weight=1.0,
        )
        mx.eval(loss)
        # Expected: ((8 - 5) / 1)^2 = 9.0
        assert abs(float(loss) - 9.0) < 0.1

    def test_sigma_scaling(self):
        """Larger sigma reduces loss proportionally."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        loss_sigma1 = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=1.0, weight=1.0,
        )
        loss_sigma2 = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=2.0, weight=1.0,
        )
        mx.eval(loss_sigma1, loss_sigma2)
        # sigma=2 should give 1/4 the loss of sigma=1
        ratio = float(loss_sigma1) / float(loss_sigma2)
        assert abs(ratio - 4.0) < 0.1

    def test_weight_scaling(self):
        """Weight scales loss linearly."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        loss_w1 = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=1.0, weight=1.0,
        )
        loss_w3 = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=1.0, weight=3.0,
        )
        mx.eval(loss_w1, loss_w3)
        ratio = float(loss_w3) / float(loss_w1)
        assert abs(ratio - 3.0) < 0.1

    def test_gradient_correctness(self):
        """Gradient points from current position toward target distance."""
        positions = mx.zeros((10, 37, 3))
        # Atom j at (8, 0, 0), target distance = 5.0
        # Gradient should pull atoms closer (reduce distance)
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        def loss_fn(pos):
            return distance_loss(
                pos,
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            )

        grad = mx.grad(loss_fn)(positions)
        mx.eval(grad)

        # Gradient of atom i (at origin) should point away from j (positive x)
        # since moving i away from j increases distance toward target (distance > target)
        # Actually: distance = 8 > target = 5, so loss increases with distance
        # Gradient of pos_i should point toward j (to reduce distance)
        # d(loss)/d(pos_i) = 2 * weight * (dist - target) / sigma^2 * d(dist)/d(pos_i)
        # d(dist)/d(pos_i) = -(pos_j - pos_i) / dist = -(8,0,0)/8 = (-1,0,0)
        # So gradient at pos_i points in -x direction (toward j)
        grad_i = grad[0, 1]  # (3,)
        assert float(grad_i[0]) < 0  # x-gradient is negative (toward j)

        # Gradient of atom j should point away from i (positive x)
        grad_j = grad[5, 1]  # (3,)
        assert float(grad_j[0]) > 0  # x-gradient is positive (away from i)

    def test_numerical_stability_same_position(self):
        """Loss doesn't produce NaN when atoms are at the same position."""
        positions = mx.zeros((10, 37, 3))
        mx.eval(positions)
        # Both atoms at origin

        loss = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=1.0, weight=1.0,
        )
        mx.eval(loss)
        assert not mx.isnan(loss).item()
        assert float(loss) > 0  # Should be ~25.0 = (5/1)^2

    def test_gradient_no_nan_same_position(self):
        """Gradient doesn't produce NaN when atoms are at the same position."""
        positions = mx.zeros((10, 37, 3))
        mx.eval(positions)

        def loss_fn(pos):
            return distance_loss(
                pos,
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            )

        grad = mx.grad(loss_fn)(positions)
        mx.eval(grad)
        assert not mx.any(mx.isnan(grad)).item()


class TestContactLoss:
    """Tests for contact_loss 1vN semantics.

    Validates non-negativity, near-zero loss when any candidate is within
    threshold, gradient direction toward nearest violating candidate, and
    weight/temperature scaling.
    """

    def _make_positions(self, source_pos, candidate_positions):
        """Create positions array with source at token 0 and candidates at tokens 5+."""
        n_tokens = 5 + len(candidate_positions)
        positions = mx.zeros((n_tokens, 37, 3))
        positions = positions.at[0, 1].add(mx.array(source_pos))
        for i, cand_pos in enumerate(candidate_positions):
            positions = positions.at[5 + i, 1].add(mx.array(cand_pos))
        mx.eval(positions)
        return positions

    def test_non_negativity_all_violated(self):
        """Loss is non-negative when all candidates are beyond threshold."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[15.0, 0.0, 0.0], [20.0, 0.0, 0.0], [25.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        assert float(loss) >= 0.0

    def test_non_negativity_one_satisfied(self):
        """Loss is non-negative even when one candidate is within threshold."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[5.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        assert float(loss) >= 0.0

    def test_non_negativity_all_satisfied(self):
        """Loss is non-negative (and near zero) when all candidates are within threshold."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        assert float(loss) >= 0.0
        assert float(loss) < 1e-6  # All within threshold → near zero

    def test_near_zero_when_one_candidate_within_threshold(self):
        """Loss is near zero when at least one candidate satisfies threshold.

        This is the core 1vN semantic: source is satisfied if ANY candidate
        is within threshold.
        """
        # One candidate at 5Å (within 8Å threshold), others far away
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[5.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        # With Boltzmann softmin, weight concentrates on the zero-penalty
        # candidate, so loss should be very small
        assert float(loss) < 0.01

    def test_positive_loss_when_all_beyond_threshold(self):
        """Loss is positive when all candidates are beyond threshold."""
        # Single candidate at 15Å, threshold 8Å → penalty = (15-8)^2 = 49
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[15.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        assert float(loss) > 1.0
        # Single candidate: softmin = penalty exactly = 49
        assert abs(float(loss) - 49.0) < 1.0

    def test_single_candidate_equals_penalty(self):
        """With one candidate, loss equals the penalty directly."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[12.0, 0.0, 0.0]],
        )
        loss = contact_loss(
            positions,
            source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1)],
            threshold=8.0, weight=1.0,
        )
        mx.eval(loss)
        # penalty = (12 - 8)^2 = 16; single candidate → softmin = 16
        assert abs(float(loss) - 16.0) < 0.5

    def test_gradient_toward_nearest_violating_candidate(self):
        """Gradient pulls source toward the nearest violating candidate."""
        # Source at origin, candidates at (12, 0, 0), (12, 10, 0), (12, 20, 0)
        # Nearest candidate is at (12, 0, 0) with distance 12
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[12.0, 0.0, 0.0], [12.0, 10.0, 0.0], [12.0, 20.0, 0.0]],
        )

        def loss_fn(pos):
            return contact_loss(
                pos,
                source_atom_idx=(0, 1),
                candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
                threshold=8.0, weight=1.0,
            )

        grad = mx.grad(loss_fn)(positions)
        mx.eval(grad)

        grad_source = grad[0, 1]
        # Source at x=0, candidates at x=12: moving source in +x reduces loss
        # So d(loss)/d(source_x) < 0 (gradient points in -x, uphill direction)
        assert float(grad_source[0]) < 0, (
            "Source gradient should be negative in x (loss decreases toward candidates)"
        )

        # x-component should dominate for nearest candidate (12, 0, 0)
        assert abs(float(grad_source[0])) > abs(float(grad_source[1]))

    def test_gradient_zero_when_satisfied(self):
        """Gradient is near-zero when the closest candidate satisfies threshold."""
        # Source at origin, one candidate at (3, 0, 0) — well within 8Å threshold
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[3.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        )

        def loss_fn(pos):
            return contact_loss(
                pos,
                source_atom_idx=(0, 1),
                candidate_atom_idxs=[(5, 1), (6, 1)],
                threshold=8.0, weight=1.0,
            )

        grad = mx.grad(loss_fn)(positions)
        mx.eval(grad)

        grad_source = grad[0, 1]
        # Penalty for candidate at 3Å is 0 (within threshold), so gradient ≈ 0
        assert float(mx.sum(mx.abs(grad_source))) < 0.01

    def test_weight_scaling(self):
        """Weight scales contact loss linearly."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[15.0, 0.0, 0.0]],
        )
        loss_w1 = contact_loss(
            positions, source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1)],
            threshold=8.0, weight=1.0,
        )
        loss_w3 = contact_loss(
            positions, source_atom_idx=(0, 1),
            candidate_atom_idxs=[(5, 1)],
            threshold=8.0, weight=3.0,
        )
        mx.eval(loss_w1, loss_w3)
        ratio = float(loss_w3) / float(loss_w1)
        assert abs(ratio - 3.0) < 0.1

    def test_gradient_no_nan(self):
        """Gradient doesn't produce NaN for contact loss."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],  # One at same position
        )

        def loss_fn(pos):
            return contact_loss(
                pos,
                source_atom_idx=(0, 1),
                candidate_atom_idxs=[(5, 1), (6, 1)],
                threshold=8.0, weight=1.0,
            )

        grad = mx.grad(loss_fn)(positions)
        mx.eval(grad)
        assert not mx.any(mx.isnan(grad)).item()

    def test_combined_loss_contact_non_negative(self):
        """Contact loss via combined_restraint_loss is also non-negative."""
        positions = self._make_positions(
            [0.0, 0.0, 0.0],
            [[5.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        )
        resolved_contact = [
            ResolvedContactRestraint(
                source_atom_idx=(0, 1),
                candidate_atom_idxs=[(5, 1), (6, 1), (7, 1)],
                threshold=8.0, weight=1.0,
            ),
        ]
        loss = combined_restraint_loss(
            positions, resolved_distance=[], resolved_contact=resolved_contact,
        )
        mx.eval(loss)
        assert float(loss) >= 0.0


class TestCombinedRestraintLoss:
    """Tests for combined_restraint_loss."""

    def test_single_restraint(self):
        """Combined loss with one distance restraint matches distance_loss."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
        ]

        combined = combined_restraint_loss(positions, resolved)
        individual = distance_loss(
            positions,
            atom_i_idx=(0, 1), atom_j_idx=(5, 1),
            target_distance=5.0, sigma=1.0, weight=1.0,
        )
        mx.eval(combined, individual)
        assert abs(float(combined) - float(individual)) < 1e-6

    def test_multiple_restraints_sum(self):
        """Combined loss sums over all restraints."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        positions = positions.at[3, 1].add(mx.array([0.0, 4.0, 0.0]))
        mx.eval(positions)

        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(3, 1),
                target_distance=4.0, sigma=1.0, weight=1.0,
            ),
        ]

        combined = combined_restraint_loss(positions, resolved)
        mx.eval(combined)
        # First: ((8-5)/1)^2 = 9, Second: ((4-4)/1)^2 ≈ 0
        assert float(combined) > 8.0  # At least the first restraint contributes

    def test_empty_restraints(self):
        """Combined loss with no restraints returns zero."""
        positions = mx.zeros((10, 37, 3))
        mx.eval(positions)

        combined = combined_restraint_loss(positions, [])
        mx.eval(combined)
        assert float(combined) == 0.0

    def test_gradient_with_mx_grad(self):
        """mx.grad works on combined_restraint_loss."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
        ]

        grad_fn = mx.grad(lambda pos: combined_restraint_loss(pos, resolved))
        grad = grad_fn(positions)
        mx.eval(grad)

        # Gradient should be non-zero for the restrained atoms
        assert float(mx.sum(mx.abs(grad[0, 1]))) > 0
        assert float(mx.sum(mx.abs(grad[5, 1]))) > 0

        # Gradient should be zero for non-restrained atoms
        assert float(mx.sum(mx.abs(grad[2, 0]))) == 0.0
