"""Guidance module for restraint-guided diffusion.

Implements annealing schedules, guidance weight computation, and the
build_guidance_fn closure that produces the gradient function injected
into the diffusion sampling loop.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Callable

import mlx.core as mx

from alphafold3_mlx.restraints.loss import contact_loss, repulsive_loss

if TYPE_CHECKING:
    from alphafold3_mlx.restraints.types import (
        GuidanceConfig,
        ResolvedContactRestraint,
        ResolvedDistanceRestraint,
        ResolvedRepulsiveRestraint,
    )

logger = logging.getLogger(__name__)

# AF3 constant from the diffusion schedule
SIGMA_DATA = 16.0

# Full-chain inter-chain coupling parameters.
# When restraints exist between atoms on different chains, a FULL-CHAIN
# center-of-mass distance penalty is added to the loss.  The gradient
# distributes equally among all CA atoms in each chain, producing pure
# rigid-body translation.  Kabsch alignment removes rigid-body motion,
# so full-chain coupling has ZERO impact on Kabsch-aligned RMSD while
# effectively controlling inter-chain distance.
#
# ONE-SIDED penalty: only activates when chains are far apart
# (CoM distance > COUPLING_MAX_DISTANCE).  When chains are within range,
# coupling produces zero force, letting the Evoformer + per-atom restraint
# gradients handle fine interface placement.  This avoids compressing
# well-folded structures when MSA provides good evolutionary signal.
#
#   Loss = w * max(0, chain_com_dist - COUPLING_MAX_DISTANCE)^2
#
# Coupling is only applied when inter-chain distance restraints are sparse
# (≤ MAX_INTERCHAIN_FOR_COUPLING).  With many distance restraints, per-atom
# gradients already provide adequate chain-level force.
COUPLING_MAX_DISTANCE = 30.0  # Angstroms – coupling only activates beyond this
CA_ATOM_IDX = 1  # Dense atom index for alpha-carbon (N=0, CA=1, C=2)
MAX_INTERCHAIN_FOR_COUPLING = 10  # Enable coupling for typical protein-protein cases

# Cap the sigma_data/sigma factor to prevent ODE instability at low noise.
# At sigma=1.6 the factor is 10; below that it's clamped.  This prevents
# the guidance weight from exploding (e.g., sigma=0.01 → factor=1600)
# which causes coordinate overflow and NaN gradients.
MAX_SIGMA_FACTOR = 10.0

# Base per-atom guidance gradient norm limit (per unit of scale).
# The effective cap is BASE_GUIDANCE_NORM * max(scale, 1.0), so doubling
# the scale doubles the maximum allowed correction.  This preserves ODE
# stability while keeping the scale parameter monotonically effective.
# Typical ODE tangent norms are O(1-10) Å per step.
BASE_GUIDANCE_NORM = 50.0


def linear_envelope(t: float, T: float) -> float:
    """Linear annealing: 1.0 → 0.0 over active range."""
    if T <= 0:
        return 1.0
    return 1.0 - t / T


def cosine_envelope(t: float, T: float) -> float:
    """Cosine annealing: 1.0 → 0.0 over active range."""
    if T <= 0:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * t / T))


def constant_envelope(t: float, T: float) -> float:
    """Constant envelope: always 1.0."""
    return 1.0


def get_envelope_fn(annealing: str) -> Callable[[float, float], float]:
    """Get the annealing envelope function by name.

    Args:
        annealing: One of "linear", "cosine", "constant".

    Returns:
        Envelope function (t, T) -> float.
    """
    if annealing == "linear":
        return linear_envelope
    elif annealing == "cosine":
        return cosine_envelope
    elif annealing == "constant":
        return constant_envelope
    else:
        raise ValueError(f"Unknown annealing schedule: {annealing}")


def compute_guidance_weight(
    scale: float,
    envelope: float,
    sigma: float,
) -> float:
    """Compute the guidance weight for a diffusion step.

    weight = scale * envelope * min(sigma_data / sigma, MAX_SIGMA_FACTOR)

    The noise-dependent factor ``sigma_data / sigma`` anneals guidance
    naturally: at high noise (sigma >> sigma_data) the weight is small,
    allowing the denoiser to explore freely; at low noise (sigma <<
    sigma_data) the weight is larger, enforcing restraints in the
    refinement steps.  The factor is clamped to MAX_SIGMA_FACTOR to
    prevent ODE instability at very low noise levels (sigma << 1).

    Args:
        scale: User-specified guidance scale.
        envelope: Annealing envelope value at current step (0-1).
        sigma: Current noise level from the diffusion schedule.

    Returns:
        Scalar guidance weight.
    """
    sigma_safe = max(abs(sigma), 1e-6)
    sigma_factor = min(SIGMA_DATA / sigma_safe, MAX_SIGMA_FACTOR)
    return scale * envelope * sigma_factor


def _find_coupling_chain_pairs(
    resolved_distance: list["ResolvedDistanceRestraint"],
    chain_token_ranges: dict[int, tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Identify FULL-CHAIN coupling pairs from inter-chain distance restraints.

    For each inter-chain DISTANCE restraint, returns the FULL chain ranges
    for both chains.  Using full-chain ranges produces pure rigid-body
    translation (gradient distributes equally among all CA atoms), which
    Kabsch alignment removes completely — zero RMSD impact.

    Contact restraints are intentionally excluded: their loss function
    produces adequate per-atom gradients without additional coupling, and
    coupling can interfere with contact satisfaction.

    Returns:
        List of (chain_range_i, chain_range_j, weight) tuples where each
        range is (start_token, end_token) for the full chain.
    """

    def _chain_of(token_idx: int) -> int | None:
        for cid, (s, e) in chain_token_ranges.items():
            if s <= token_idx < e:
                return cid
        return None

    # Collect unique inter-chain pairs (deduplicate by chain pair).
    # When multiple restraints span the same chain pair with different
    # weights, use the max weight so coupling strength is independent
    # of restraint list order.
    pair_map: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], float]] = {}

    for r in resolved_distance:
        token_i = r.atom_i_idx[0]
        token_j = r.atom_j_idx[0]
        ci = _chain_of(token_i)
        cj = _chain_of(token_j)
        if ci is not None and cj is not None and ci != cj:
            key = (min(ci, cj), max(ci, cj))
            if key not in pair_map or r.weight > pair_map[key][2]:
                pair_map[key] = (
                    chain_token_ranges[key[0]],
                    chain_token_ranges[key[1]],
                    r.weight,
                )

    pairs = list(pair_map.values())

    # When many distance restraints exist, their per-atom gradients already
    # provide adequate chain-level force.  Coupling is only needed to
    # supplement sparse cases (1-2 restraints).  With 3+ inter-chain
    # distances, coupling interferes with contact satisfaction.
    n_interchain = sum(
        1 for r in resolved_distance
        if _chain_of(r.atom_i_idx[0]) != _chain_of(r.atom_j_idx[0])
        and _chain_of(r.atom_i_idx[0]) is not None
    )
    if n_interchain > MAX_INTERCHAIN_FOR_COUPLING:
        logger.info(
            "Skipping coupling: %d inter-chain distance restraints "
            "(threshold %d) — per-atom gradients sufficient",
            n_interchain, MAX_INTERCHAIN_FOR_COUPLING,
        )
        return []

    return pairs


def build_guidance_fn(
    resolved_distance: list["ResolvedDistanceRestraint"],
    resolved_contact: list["ResolvedContactRestraint"] | None,
    resolved_repulsive: list["ResolvedRepulsiveRestraint"] | None,
    guidance: "GuidanceConfig",
    num_steps: int,
    chain_token_ranges: dict[int, tuple[int, int]] | None = None,
) -> Callable[[mx.array, mx.array, int], mx.array]:
    """Build the guidance function closure for the diffusion loop.

    Returns a function that takes (positions_denoised, noise_level, step)
    and returns the restraint loss gradient (w·∇L) to be ADDED to the
    Karras/EDM ODE tangent direction.  In the EDM formulation the tangent
    ``d = (x - D(x))/σ`` is the negative of ``σ×score``, so adding the
    positive loss gradient to the tangent corresponds to *subtracting*
    it from the score, which is the standard classifier-guidance update::

      d = (x_noisy - D(x_noisy; σ)) / σ       # ODE tangent
      d_guided = d + guidance_fn(D(x), σ, step) # add w·∇L
      x_next = x_noisy + Δσ × d_guided          # Euler step

    The guidance weight includes noise-dependent scaling:
      weight = scale * envelope * (sigma_data / sigma)

    Full-chain inter-chain coupling: when restraints exist between atoms
    on different chains, a full-chain center-of-mass distance penalty is
    added.  The gradient distributes equally among all CA atoms in each
    chain, producing pure rigid-body translation.  Kabsch alignment
    removes rigid-body motion, so coupling has zero RMSD impact.

    A scale-proportional gradient limit prevents ODE instability without
    capping the user-visible scale: the effective limit is
    ``BASE_GUIDANCE_NORM * max(scale, 1.0)``, so doubling the scale
    always doubles the maximum allowed correction.

    NaN safety: if any gradient component is NaN, a zero gradient
    is substituted and a warning is logged.

    Args:
        resolved_distance: Resolved distance restraints.
        resolved_contact: Resolved contact restraints (may be None for US1).
        resolved_repulsive: Resolved repulsive restraints (may be None for US1).
        guidance: Guidance configuration.
        num_steps: Total number of diffusion steps.
        chain_token_ranges: Optional mapping from chain asym_id (int) to
            (start_token, end_token) ranges.  Required for inter-chain CoM
            coupling.  When None, only per-atom gradients are computed.

    Returns:
        Callable (positions, noise_level, step) -> gradient array.
    """
    envelope_fn = get_envelope_fn(guidance.annealing)
    end_step = guidance.end_step if guidance.end_step is not None else num_steps
    start_step = guidance.start_step
    # Diffusion steps are 1-indexed (range(1, num_steps+1)), so start_step=0
    # never actually occurs.  Use the effective first active step for envelope
    # math so that t=0 (full envelope) on the first step that runs.
    _effective_start = max(start_step, 1)
    active_range = end_step - _effective_start

    # Pre-compute full-chain inter-chain coupling pairs.
    coupling_pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    if chain_token_ranges is not None and len(chain_token_ranges) > 1:
        coupling_pairs = _find_coupling_chain_pairs(
            resolved_distance, chain_token_ranges,
        )
        if coupling_pairs:
            logger.info(
                "Full-chain CoM coupling: %d chain pair(s) detected",
                len(coupling_pairs),
            )

    # Separate loss functions for atom-level restraints and coupling.
    def atom_loss_fn(positions: mx.array) -> mx.array:
        total = mx.array(0.0)

        for r in resolved_distance:
            pos_i = positions[r.atom_i_idx[0], r.atom_i_idx[1]]
            pos_j = positions[r.atom_j_idx[0], r.atom_j_idx[1]]
            diff = pos_i - pos_j
            dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
            total = total + r.weight * ((dist - r.target_distance) / r.sigma) ** 2

        if resolved_contact:
            for r in resolved_contact:
                total = total + contact_loss(
                    positions,
                    source_atom_idx=r.source_atom_idx,
                    candidate_atom_idxs=r.candidate_atom_idxs,
                    threshold=r.threshold,
                    weight=r.weight,
                )

        if resolved_repulsive:
            for r in resolved_repulsive:
                total = total + repulsive_loss(
                    positions,
                    atom_i_idx=r.atom_i_idx,
                    atom_j_idx=r.atom_j_idx,
                    min_distance=r.min_distance,
                    weight=r.weight,
                )

        return total

    atom_grad_fn = mx.grad(atom_loss_fn)

    # Coupling gradient is computed MANUALLY (not via mx.grad) so that
    # it distributes to ALL atoms in each token, not just CA.  When
    # mx.grad computes the coupling gradient, it only flows to CA atoms
    # (the ones used in the CoM computation).  This causes CA atoms to
    # drift away from backbone N/C atoms, destroying the chain fold.
    # By distributing the gradient uniformly to ALL atoms, every atom
    # in the chain moves by the same vector → true rigid-body translation.

    def guidance_fn(
        positions_denoised: mx.array,
        noise_level: mx.array,
        step: int,
    ) -> mx.array:
        """Compute guidance gradient for a single diffusion step.

        Args:
            positions_denoised: Denoised positions, shape (num_tokens, num_atoms, 3).
            noise_level: Current noise level (scalar mx.array).
            step: Current step index (1-indexed).

        Returns:
            Gradient array, same shape as positions_denoised.
        """
        # Check step range
        if step < start_step or step > end_step:
            return mx.zeros_like(positions_denoised)

        sigma_val = float(noise_level)

        # Compute envelope
        t = step - _effective_start
        envelope = envelope_fn(float(t), float(max(active_range, 1)))

        # Compute guidance weight (includes natural sigma_data/sigma annealing)
        weight = compute_guidance_weight(guidance.scale, envelope, sigma_val)

        if weight == 0.0:
            return mx.zeros_like(positions_denoised)

        # Compute atom-level gradient (distance + contact + repulsive).
        atom_grad = atom_grad_fn(positions_denoised)

        # NaN safety on atom-level gradient
        has_nan = mx.any(mx.isnan(atom_grad))
        mx.eval(has_nan)
        if has_nan.item():
            logger.warning(
                "NaN detected in restraint gradient at step %d; "
                "substituting zero gradient",
                step,
            )
            return mx.zeros_like(positions_denoised)

        # Limit the raw loss gradient per atom for ODE stability.  The
        # quadratic loss gradient can be very large for tight restraints
        # far from target (grad ∝ deviation/sigma²), which can cause the
        # ODE to overshoot and oscillate.  Limiting the raw gradient to
        # _RAW_GRAD_LIMIT preserves the relative magnitude between tight
        # and loose restraints while preventing divergence.  The guidance
        # weight (which includes scale, envelope, and sigma_data/sigma)
        # is applied AFTER this limit — higher scale always produces
        # proportionally larger corrections (no fixed hard cap).
        _RAW_GRAD_LIMIT = 10.0
        atom_norms = mx.sqrt(
            mx.sum(atom_grad * atom_grad, axis=-1, keepdims=True) + 1e-8
        )
        clip_factor = mx.minimum(
            mx.array(1.0), _RAW_GRAD_LIMIT / atom_norms
        )
        atom_grad = atom_grad * clip_factor

        # Start with weighted atom gradient
        grad = weight * atom_grad

        # Add coupling: manual rigid-body gradient distributed to ALL atoms.
        # Using mx.grad would route gradient only to CA atoms (the ones
        # in the CoM computation), causing CA to drift from backbone.
        # Manual computation gives the SAME gradient vector to every atom
        # in each chain → true rigid-body translation → zero Kabsch RMSD.
        if coupling_pairs:
            n_tokens = positions_denoised.shape[0]
            token_idx = mx.arange(n_tokens)
            for (ci_start, ci_end), (cj_start, cj_end), w_c in coupling_pairs:
                ca_i = positions_denoised[ci_start:ci_end, CA_ATOM_IDX, :]
                ca_j = positions_denoised[cj_start:cj_end, CA_ATOM_IDX, :]
                com_i = mx.mean(ca_i, axis=0)
                com_j = mx.mean(ca_j, axis=0)
                c_diff = com_i - com_j
                c_dist = mx.sqrt(mx.sum(c_diff * c_diff) + 1e-8)

                # One-sided penalty: only pull when chains too far apart
                excess = mx.maximum(mx.array(0.0), c_dist - COUPLING_MAX_DISTANCE)
                mx.eval(excess)
                if float(excess.item()) <= 0:
                    continue  # Chains within range, no coupling needed

                # d(loss)/d(com_i) = 2*w*excess * direction
                grad_com = (
                    2.0 * w_c
                    * excess
                    * (c_diff / c_dist)
                )

                # Per-token gradient (same magnitude/direction for all atoms)
                n_i = ci_end - ci_start
                n_j = cj_end - cj_start
                per_token_i = grad_com / n_i    # (3,)
                per_token_j = -(grad_com / n_j)  # opposite direction

                # Broadcast to all tokens × all atoms via boolean mask
                mask_i = (
                    (token_idx >= ci_start) & (token_idx < ci_end)
                )[:, None, None]
                mask_j = (
                    (token_idx >= cj_start) & (token_idx < cj_end)
                )[:, None, None]
                grad = grad + weight * (
                    mask_i * per_token_i + mask_j * per_token_j
                )

        # Scale-proportional cap on final gradient norm for ODE stability.
        # The effective limit grows with scale so that doubling the scale
        # always doubles the maximum correction (no fixed saturation).
        effective_limit = BASE_GUIDANCE_NORM * max(guidance.scale, 1.0)
        final_norms = mx.sqrt(
            mx.sum(grad * grad, axis=-1, keepdims=True) + 1e-8
        )
        clip = mx.minimum(mx.array(1.0), effective_limit / final_norms)
        grad = grad * clip

        return grad

    return guidance_fn
