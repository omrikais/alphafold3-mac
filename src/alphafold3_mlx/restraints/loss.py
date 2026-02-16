"""Restraint loss functions for gradient-guided diffusion.

Implements differentiable loss functions for distance, contact, and repulsive
restraints. All functions operate on MLX arrays and are compatible with mx.grad.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    from alphafold3_mlx.restraints.types import (
        ResolvedContactRestraint,
        ResolvedDistanceRestraint,
        ResolvedRepulsiveRestraint,
    )


def distance_loss(
    positions: mx.array,
    atom_i_idx: tuple[int, int],
    atom_j_idx: tuple[int, int],
    target_distance: float,
    sigma: float,
    weight: float,
) -> mx.array:
    """Harmonic distance restraint loss.

    Loss = weight * ((distance - target) / sigma)^2

    Args:
        positions: Atom positions, shape (num_tokens, num_atoms, 3).
        atom_i_idx: (token_index, atom37_index) for atom i.
        atom_j_idx: (token_index, atom37_index) for atom j.
        target_distance: Target distance in Angstroms.
        sigma: Tolerance in Angstroms.
        weight: Restraint weight.

    Returns:
        Scalar loss value.
    """
    pos_i = positions[atom_i_idx[0], atom_i_idx[1]]  # (3,)
    pos_j = positions[atom_j_idx[0], atom_j_idx[1]]  # (3,)
    diff = pos_i - pos_j
    dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
    return weight * ((dist - target_distance) / sigma) ** 2


def contact_loss(
    positions: mx.array,
    source_atom_idx: tuple[int, int],
    candidate_atom_idxs: list[tuple[int, int]],
    threshold: float,
    weight: float,
    temperature: float = 1.0,
) -> mx.array:
    """Contact restraint loss with 1vN semantics via Boltzmann softmin.

    Penalizes when the source atom is beyond threshold distance from ALL
    candidate atoms. Uses Boltzmann softmin (weighted average with softmax
    weights) for a differentiable, non-negative smooth minimum.

    Loss = weight * sum_j(penalty_j * softmax(-penalty / temperature)_j)
    where penalty_j = max(0, dist_j - threshold)^2

    Properties:
    - Always non-negative (weighted average of non-negative penalties).
    - Near zero when ANY candidate is within threshold (softmax concentrates
      weight on the zero-penalty candidate).
    - Gradients point toward the nearest violating candidate.

    Args:
        positions: Atom positions, shape (num_tokens, num_atoms, 3).
        source_atom_idx: (token_index, atom37_index) for source CA.
        candidate_atom_idxs: List of (token_index, atom37_index) for candidate CAs.
        threshold: Contact distance threshold in Angstroms.
        weight: Restraint weight.
        temperature: Smoothing temperature for softmin (lower = sharper min).

    Returns:
        Scalar non-negative loss value.
    """
    pos_source = positions[source_atom_idx[0], source_atom_idx[1]]  # (3,)

    penalties = []
    for cand_idx in candidate_atom_idxs:
        pos_cand = positions[cand_idx[0], cand_idx[1]]  # (3,)
        diff = pos_source - pos_cand
        dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
        penalty = mx.maximum(mx.array(0.0), dist - threshold) ** 2
        penalties.append(penalty)

    # Boltzmann softmin: weighted average using softmax weights on negated penalties.
    # softmin(p) = sum_j(p_j * w_j) where w_j = softmax(-p / temperature)_j
    # This is always in [min(p), max(p)] — non-negative since all p_j >= 0.
    # When one candidate satisfies threshold (p_j=0), softmax concentrates
    # weight there, pulling the result toward zero.
    penalty_stack = mx.stack(penalties)  # (num_candidates,)
    softmin_weights = mx.softmax(-penalty_stack / temperature)
    soft_min_penalty = mx.sum(penalty_stack * softmin_weights)

    return weight * soft_min_penalty


def repulsive_loss(
    positions: mx.array,
    atom_i_idx: tuple[int, int],
    atom_j_idx: tuple[int, int],
    min_distance: float,
    weight: float,
) -> mx.array:
    """Repulsive restraint loss using one-sided harmonic.

    Penalizes when atoms are closer than min_distance.

    Loss = weight * max(0, min_distance - distance)^2

    Args:
        positions: Atom positions, shape (num_tokens, num_atoms, 3).
        atom_i_idx: (token_index, atom37_index) for CA of residue i.
        atom_j_idx: (token_index, atom37_index) for CA of residue j.
        min_distance: Minimum allowed distance in Angstroms.
        weight: Restraint weight.

    Returns:
        Scalar loss value.
    """
    pos_i = positions[atom_i_idx[0], atom_i_idx[1]]  # (3,)
    pos_j = positions[atom_j_idx[0], atom_j_idx[1]]  # (3,)
    diff = pos_i - pos_j
    dist = mx.sqrt(mx.sum(diff * diff) + 1e-8)
    return weight * mx.maximum(mx.array(0.0), min_distance - dist) ** 2


def combined_restraint_loss(
    positions: mx.array,
    resolved_distance: list["ResolvedDistanceRestraint"],
    resolved_contact: list["ResolvedContactRestraint"] | None = None,
    resolved_repulsive: list["ResolvedRepulsiveRestraint"] | None = None,
) -> mx.array:
    """Combined loss over all resolved restraints.

    Sums distance, contact, and repulsive losses over their respective
    resolved restraint lists.

    Args:
        positions: Atom positions, shape (num_tokens, num_atoms, 3).
        resolved_distance: List of resolved distance restraints.
        resolved_contact: List of resolved contact restraints.
        resolved_repulsive: List of resolved repulsive restraints.

    Returns:
        Scalar total loss value.
    """
    total = mx.array(0.0)

    for r in resolved_distance:
        total = total + distance_loss(
            positions,
            atom_i_idx=r.atom_i_idx,
            atom_j_idx=r.atom_j_idx,
            target_distance=r.target_distance,
            sigma=r.sigma,
            weight=r.weight,
        )

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
