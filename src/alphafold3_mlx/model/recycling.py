"""Recycling loop for AlphaFold 3 MLX.

This module implements the recycling mechanism which
iteratively refines Evoformer embeddings by feeding back the output
as input for multiple iterations.

Key implementation details:
- Python for-loop (no jax.lax.fori_loop in MLX)
- mx.eval at the end of each iteration to prevent graph explosion
- Convergence tracking for stability testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import mlx.core as mx

if TYPE_CHECKING:
    from alphafold3_mlx.core.entities import Embeddings


@dataclass
class RecyclingState:
    """State for recycling iterations.

    Tracks embeddings across iterations and optionally monitors convergence.
    """

    single: mx.array
    """Single representation. Shape: [batch, seq, seq_channel]"""

    pair: mx.array
    """Pair representation. Shape: [batch, seq, seq, pair_channel]"""

    iteration: int = 0
    """Current iteration number."""

    prev_single: mx.array | None = None
    """Previous single representation (for convergence tracking)."""

    prev_pair: mx.array | None = None
    """Previous pair representation (for convergence tracking)."""

    differences: list[float] = field(default_factory=list)
    """Per-iteration difference metrics."""


def compute_embedding_difference(
    current: mx.array,
    previous: mx.array,
    mask: mx.array | None = None,
) -> float:
    """Compute normalized difference between embeddings.

    Used for convergence tracking.

    Args:
        current: Current embedding.
        previous: Previous embedding.
        mask: Optional mask for valid positions.

    Returns:
        Normalized L2 difference.
    """
    diff = current - previous
    diff_sq = mx.sum(diff ** 2)

    if mask is not None:
        # Normalize by number of valid elements
        num_valid = mx.sum(mask) * current.shape[-1]
        return float((diff_sq / mx.maximum(num_valid, 1.0)).item())
    else:
        num_elements = diff.size
        return float((diff_sq / num_elements).item())


def run_recycling_loop(
    evoformer_fn,
    initial_single: mx.array,
    initial_pair: mx.array,
    residue_index: mx.array,
    asym_id: mx.array,
    num_recycles: int,
    seq_mask: mx.array | None = None,
    pair_mask: mx.array | None = None,
    track_convergence: bool = False,
    return_intermediates: bool = False,
    iteration_callback: "Callable[[int, int], None] | None" = None,
    **evoformer_kwargs,
) -> tuple[mx.array, mx.array, RecyclingState | None] | tuple[mx.array, mx.array, RecyclingState | None, dict[str, mx.array]]:
    """Run recycling loop.

    Iteratively applies the Evoformer, feeding back output as input.
    Uses explicit mx.eval() at the end of each iteration to prevent
    graph explosion.

    Args:
        evoformer_fn: Evoformer forward function.
        initial_single: Initial single representation.
        initial_pair: Initial pair representation.
        residue_index: Residue indices.
        asym_id: Chain IDs.
        num_recycles: Number of recycling iterations.
        seq_mask: Optional sequence mask.
        pair_mask: Optional pair mask.
        track_convergence: Whether to track iteration differences.
        return_intermediates: If True, return intermediate checkpoints from
            the final Evoformer iteration.
        iteration_callback: Optional callback called after each iteration with
            (iteration, total_iterations) for progress reporting.
        **evoformer_kwargs: Additional arguments for evoformer_fn.

    Returns:
        If return_intermediates is False:
            Tuple of (final_single, final_pair, optional_state).
        If return_intermediates is True:
            Tuple of (final_single, final_pair, optional_state, intermediates_dict).
    """
    single = initial_single
    pair = initial_pair
    intermediates: dict[str, mx.array] | None = None

    state = None
    if track_convergence:
        state = RecyclingState(single=single, pair=pair)

    # num_recycles is the number of additional forward passes where output is recycled.
    # Total iterations = 1 (initial) + num_recycles (recycled), matching JAX AF3 semantics.
    total_iterations = num_recycles + 1

    # Recycling loop using Python for (per research.md Section 3)
    for i in range(total_iterations):
        # Store previous for convergence tracking
        if track_convergence:
            state.prev_single = single
            state.prev_pair = pair

        # Request intermediates only on final iteration
        is_final_iteration = (i == total_iterations - 1)
        request_intermediates = return_intermediates and is_final_iteration

        # Run Evoformer
        result = evoformer_fn(
            single=single,
            pair=pair,
            residue_index=residue_index,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pair_mask=pair_mask,
            return_intermediates=request_intermediates,
            **evoformer_kwargs,
        )

        # Handle return value based on whether intermediates were requested
        if request_intermediates:
            single, pair, intermediates = result
        else:
            single, pair = result

        # JAX AF3 parity: recycle_body casts embeddings to float32 after each
        # iteration (model.py lines 286-287).  This ensures recycled inputs and
        # final outputs fed to diffusion/confidence heads are always float32.
        single = single.astype(mx.float32)
        pair = pair.astype(mx.float32)

        # mx.eval at end of each iteration
        mx.eval(single, pair)

        # Progress callback
        if iteration_callback is not None:
            iteration_callback(i, total_iterations)

        # Track convergence
        if track_convergence and state.prev_single is not None:
            single_diff = compute_embedding_difference(single, state.prev_single, seq_mask)
            pair_diff = compute_embedding_difference(pair, state.prev_pair, pair_mask)
            total_diff = single_diff + pair_diff
            state.differences.append(total_diff)
            state.iteration = i

    if track_convergence:
        state.single = single
        state.pair = pair

    if return_intermediates:
        return single, pair, state, intermediates
    return single, pair, state


def check_convergence(
    state: RecyclingState,
    threshold: float = 1e-4,
    require_decreasing: bool = True,
) -> tuple[bool, str]:
    """Check if recycling has converged.

    Args:
        state: RecyclingState with tracked differences.
        threshold: Convergence threshold.
        require_decreasing: Whether differences must be monotonically decreasing.

    Returns:
        Tuple of (converged: bool, message: str).
    """
    if not state.differences:
        return False, "No differences tracked"

    final_diff = state.differences[-1]

    # Check if below threshold
    if final_diff > threshold:
        return False, f"Final difference {final_diff:.2e} above threshold {threshold:.2e}"

    # Check if decreasing
    if require_decreasing and len(state.differences) > 1:
        for i in range(1, len(state.differences)):
            if state.differences[i] > state.differences[i - 1] * 1.1:  # 10% tolerance
                return False, (
                    f"Differences not monotonically decreasing: "
                    f"iteration {i-1}={state.differences[i-1]:.2e} < "
                    f"iteration {i}={state.differences[i]:.2e}"
                )

    return True, f"Converged after {state.iteration + 1} iterations (diff={final_diff:.2e})"
