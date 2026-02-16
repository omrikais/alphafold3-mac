"""Unit tests for annealing schedules, guidance weight, and build_guidance_fn.

Tests linear, cosine, constant envelopes, compute_guidance_weight at various
sigma levels, step range boundaries, and NaN safety.
"""

import math

import mlx.core as mx
import pytest

from alphafold3_mlx.restraints.guidance import (
    SIGMA_DATA,
    build_guidance_fn,
    compute_guidance_weight,
    constant_envelope,
    cosine_envelope,
    get_envelope_fn,
    linear_envelope,
)

assert SIGMA_DATA == 16.0, "Tests assume SIGMA_DATA == 16.0"
from alphafold3_mlx.restraints.types import (
    GuidanceConfig,
    ResolvedDistanceRestraint,
)


# ── Linear Envelope ─────────────────────────────────────────────────────────


class TestLinearEnvelope:
    def test_start_is_one(self):
        assert linear_envelope(0, 100) == 1.0

    def test_end_is_zero(self):
        assert linear_envelope(100, 100) == 0.0

    def test_midpoint(self):
        assert abs(linear_envelope(50, 100) - 0.5) < 1e-10

    def test_quarter(self):
        assert abs(linear_envelope(25, 100) - 0.75) < 1e-10

    def test_zero_range_returns_one(self):
        assert linear_envelope(0, 0) == 1.0


# ── Cosine Envelope ─────────────────────────────────────────────────────────


class TestCosineEnvelope:
    def test_start_is_one(self):
        assert abs(cosine_envelope(0, 100) - 1.0) < 1e-10

    def test_end_is_zero(self):
        assert abs(cosine_envelope(100, 100)) < 1e-10

    def test_midpoint(self):
        assert abs(cosine_envelope(50, 100) - 0.5) < 1e-10

    def test_quarter(self):
        # cos(pi/4) = sqrt(2)/2 ≈ 0.707
        expected = 0.5 * (1.0 + math.cos(math.pi * 0.25))
        assert abs(cosine_envelope(25, 100) - expected) < 1e-10

    def test_zero_range_returns_one(self):
        assert cosine_envelope(0, 0) == 1.0


# ── Constant Envelope ───────────────────────────────────────────────────────


class TestConstantEnvelope:
    def test_always_one(self):
        assert constant_envelope(0, 100) == 1.0
        assert constant_envelope(50, 100) == 1.0
        assert constant_envelope(100, 100) == 1.0
        assert constant_envelope(0, 0) == 1.0


# ── get_envelope_fn ─────────────────────────────────────────────────────────


class TestGetEnvelopeFn:
    def test_linear(self):
        fn = get_envelope_fn("linear")
        assert fn is linear_envelope

    def test_cosine(self):
        fn = get_envelope_fn("cosine")
        assert fn is cosine_envelope

    def test_constant(self):
        fn = get_envelope_fn("constant")
        assert fn is constant_envelope

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown annealing"):
            get_envelope_fn("exponential")


# ── compute_guidance_weight ─────────────────────────────────────────────────


class TestComputeGuidanceWeight:
    def test_basic_computation(self):
        # weight = scale * envelope * (sigma_data / sigma)
        # sigma_data=16, sigma=16 → factor=1.0
        w = compute_guidance_weight(scale=1.0, envelope=1.0, sigma=16.0)
        assert abs(w - 1.0) < 1e-6

    def test_scale_factor(self):
        w = compute_guidance_weight(scale=2.0, envelope=1.0, sigma=16.0)
        assert abs(w - 2.0) < 1e-6

    def test_envelope_factor(self):
        w = compute_guidance_weight(scale=1.0, envelope=0.5, sigma=16.0)
        assert abs(w - 0.5) < 1e-6

    def test_sigma_scaling(self):
        # Weight depends on sigma via sigma_data/sigma, clamped to
        # MAX_SIGMA_FACTOR=10 to prevent ODE instability at low noise.
        w_low = compute_guidance_weight(scale=1.0, envelope=1.0, sigma=1.0)
        w_high = compute_guidance_weight(scale=1.0, envelope=1.0, sigma=16.0)
        # At low sigma, weight should be larger (clamped: min(16/1, 10) = 10 vs 16/16 = 1)
        assert w_low > w_high
        assert abs(w_low - 10.0) < 1e-6  # clamped to MAX_SIGMA_FACTOR
        assert abs(w_high - 1.0) < 1e-6

    def test_near_zero_sigma_is_safe(self):
        # sigma near 0 is clamped to 1e-6 to avoid division by zero,
        # then the sigma_data/sigma factor is clamped to MAX_SIGMA_FACTOR=10.
        w = compute_guidance_weight(scale=1.0, envelope=1.0, sigma=0.0)
        # Without MAX_SIGMA_FACTOR it would be 16/1e-6 = 16e6, but with
        # the clamp it saturates at MAX_SIGMA_FACTOR.
        assert abs(w - 10.0) < 1e-6
        assert math.isfinite(w)

    def test_negative_sigma_uses_abs(self):
        # Negative sigma uses abs for safety
        w = compute_guidance_weight(scale=1.0, envelope=1.0, sigma=-8.0)
        expected = SIGMA_DATA / 8.0  # = 2.0
        assert abs(w - expected) < 1e-6

    def test_zero_scale_returns_zero(self):
        w = compute_guidance_weight(scale=0.0, envelope=1.0, sigma=16.0)
        assert w == 0.0

    def test_combined_factors(self):
        # scale=0.5, envelope=0.8, sigma=8.0 → 0.5 * 0.8 * (16/8) = 0.8
        w = compute_guidance_weight(scale=0.5, envelope=0.8, sigma=8.0)
        assert abs(w - 0.8) < 1e-6


# ── build_guidance_fn ───────────────────────────────────────────────────────


class TestBuildGuidanceFn:
    def _make_positions_and_restraints(self):
        """Create test positions with atoms 8A apart and a 5A target."""
        positions = mx.zeros((10, 37, 3))
        positions = positions.at[5, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)
        resolved = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(5, 1),
                target_distance=5.0, sigma=1.0, weight=1.0,
            ),
        ]
        return positions, resolved

    def test_returns_callable(self):
        _, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=1.0)
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)
        assert callable(fn)

    def test_produces_nonzero_gradient_in_range(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=1.0, annealing="constant")
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)
        # Gradient should be nonzero for restrained atoms
        assert float(mx.sum(mx.abs(grad))) > 0

    def test_zero_gradient_outside_step_range(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(
            scale=1.0, start_step=50, end_step=150,
        )
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        # Step 10 is before start_step
        grad = fn(positions, mx.array(10.0), step=10)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) == 0.0

        # Step 180 is after end_step
        grad = fn(positions, mx.array(10.0), step=180)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) == 0.0

    def test_boundary_steps_are_active(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(
            scale=1.0, start_step=50, end_step=150, annealing="constant",
        )
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        # Exactly at start_step should be active
        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) > 0

        # Exactly at end_step should be active
        grad = fn(positions, mx.array(10.0), step=150)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) > 0

    def test_gradient_scales_with_guidance_scale(self):
        positions, resolved = self._make_positions_and_restraints()

        guidance_low = GuidanceConfig(scale=0.1, annealing="constant")
        guidance_high = GuidanceConfig(scale=1.0, annealing="constant")

        fn_low = build_guidance_fn(resolved, None, None, guidance_low, num_steps=200)
        fn_high = build_guidance_fn(resolved, None, None, guidance_high, num_steps=200)

        grad_low = fn_low(positions, mx.array(10.0), step=50)
        grad_high = fn_high(positions, mx.array(10.0), step=50)
        mx.eval(grad_low, grad_high)

        mag_low = float(mx.sum(mx.abs(grad_low)))
        mag_high = float(mx.sum(mx.abs(grad_high)))
        # Higher scale produces proportionally larger gradient (no hard cap)
        assert mag_high > mag_low

    def test_linear_annealing_decays(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=1.0, annealing="linear")
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        # Early step (envelope near 1.0)
        grad_early = fn(positions, mx.array(10.0), step=10)
        # Late step (envelope near 0.0)
        grad_late = fn(positions, mx.array(10.0), step=190)
        mx.eval(grad_early, grad_late)

        mag_early = float(mx.sum(mx.abs(grad_early)))
        mag_late = float(mx.sum(mx.abs(grad_late)))
        assert mag_early > mag_late

    def test_zero_scale_produces_zero_gradient(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=0.0, annealing="constant")
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)
        assert float(mx.sum(mx.abs(grad))) == 0.0

    def test_gradient_shape_matches_positions(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=1.0, annealing="constant")
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)
        assert grad.shape == positions.shape

    def test_no_nan_in_gradient(self):
        positions, resolved = self._make_positions_and_restraints()
        guidance = GuidanceConfig(scale=1.0, annealing="constant")
        fn = build_guidance_fn(resolved, None, None, guidance, num_steps=200)

        grad = fn(positions, mx.array(10.0), step=50)
        mx.eval(grad)
        assert not mx.any(mx.isnan(grad)).item()

    def test_higher_scale_increases_gradient(self):
        """Higher guidance scale produces larger gradients.

        Raw gradients are clipped per-atom (_RAW_GRAD_LIMIT=10).  The final
        gradient cap is scale-proportional (BASE_GUIDANCE_NORM * max(scale, 1)),
        so doubling the scale always doubles the effective limit.  Verify
        proportional scaling at 2x and monotonicity at 100x.
        """
        positions, resolved = self._make_positions_and_restraints()
        fn_base = build_guidance_fn(
            resolved, None, None,
            GuidanceConfig(scale=1.0, annealing="constant"),
            num_steps=200,
        )
        fn_double = build_guidance_fn(
            resolved, None, None,
            GuidanceConfig(scale=2.0, annealing="constant"),
            num_steps=200,
        )
        fn_high = build_guidance_fn(
            resolved, None, None,
            GuidanceConfig(scale=100.0, annealing="constant"),
            num_steps=200,
        )

        grad_base = fn_base(positions, mx.array(10.0), step=50)
        grad_double = fn_double(positions, mx.array(10.0), step=50)
        grad_high = fn_high(positions, mx.array(10.0), step=50)
        mx.eval(grad_base, grad_double, grad_high)

        mag_base = float(mx.sum(mx.abs(grad_base)))
        mag_double = float(mx.sum(mx.abs(grad_double)))
        mag_high = float(mx.sum(mx.abs(grad_high)))

        # 2x scale should give ~2x magnitude (within linear regime)
        assert mag_double > mag_base * 1.5, (
            f"2x scale should produce ~2x gradient: "
            f"base={mag_base:.4f}, double={mag_double:.4f}"
        )
        # Very high scale still exceeds base (monotonicity, even with caps)
        assert mag_high > mag_base, (
            f"100x scale should exceed 1x: "
            f"base={mag_base:.4f}, high={mag_high:.4f}"
        )

    def test_smaller_sigma_produces_stronger_gradient(self):
        """Smaller sigma produces stronger gradient for same distance deviation."""
        positions = mx.zeros((10, 37, 3))
        # Atom 0 at origin, atom 3 at (8, 0, 0), atom 6 at (8, 0, 0)
        # Use moderate distance for clear gradient magnitude comparison
        positions = positions.at[3, 1].add(mx.array([8.0, 0.0, 0.0]))
        positions = positions.at[6, 1].add(mx.array([8.0, 0.0, 0.0]))
        mx.eval(positions)

        # Tight restraint (sigma=0.5) should produce stronger gradient
        # than loose restraint (sigma=5.0) for the same distance deviation.
        # Loss = weight * ((dist - target) / sigma)^2
        # grad ∝ 2 * weight * (dist - target) / sigma^2
        resolved_tight = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(3, 1),
                target_distance=5.0, sigma=0.5, weight=1.0,
            ),
        ]
        resolved_loose = [
            ResolvedDistanceRestraint(
                atom_i_idx=(0, 1), atom_j_idx=(6, 1),
                target_distance=5.0, sigma=5.0, weight=1.0,
            ),
        ]

        # Use low scale to avoid per-atom clipping saturating both
        guidance = GuidanceConfig(scale=0.01, annealing="constant")
        fn_tight = build_guidance_fn(resolved_tight, None, None, guidance, num_steps=200)
        fn_loose = build_guidance_fn(resolved_loose, None, None, guidance, num_steps=200)

        grad_tight = fn_tight(positions, mx.array(10.0), step=50)
        grad_loose = fn_loose(positions, mx.array(10.0), step=50)
        mx.eval(grad_tight, grad_loose)

        norm_tight = float(mx.sqrt(mx.sum(grad_tight[3, 1] ** 2)))
        norm_loose = float(mx.sqrt(mx.sum(grad_loose[6, 1] ** 2)))
        assert norm_tight > 0
        assert norm_loose > 0
        # sigma=0.5 vs sigma=5.0: gradient ratio should be (5.0/0.5)^2 = 100x
        # (or close, modulo sqrt in distance formula)
        assert norm_tight > norm_loose * 10, (
            f"Tight sigma should produce much stronger gradient: "
            f"tight={norm_tight:.6f}, loose={norm_loose:.6f}"
        )
