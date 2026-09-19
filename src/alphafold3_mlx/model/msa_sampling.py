"""JAX-compatible, padding-stable MSA sampling without a JAX dependency.

AlphaFold 3 uses Threefry keys and per-row folded Gumbel samples when it
shuffles the MSA.  Keeping this small implementation local lets MLX inference
select the exact canonical rows before expanding them into 34-channel MSA
features.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from alphafold3_mlx.jax_rng import _threefry2x32, fold_in, split


def split_key(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a legacy JAX Threefry key into two keys."""

    keys = split(key, 2)
    return keys[0], keys[1]


def _fold_in(key: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Fold uint32 row indices into a key, as JAX ``random.fold_in`` does."""

    return fold_in(key, data)


def _padding_consistent_gumbel(key: np.ndarray, size: int) -> np.ndarray:
    """Return one canonical float32 Gumbel variate for every MSA row."""

    row_keys = _fold_in(key, np.arange(size, dtype=np.uint32))
    zeros = np.zeros((size,), dtype=np.uint32)
    bits_hi, bits_lo = _threefry2x32(row_keys.T, zeros, zeros)
    bits = bits_hi ^ bits_lo

    # JAX random.uniform<float32>: populate the 23 mantissa bits of 1.x,
    # bit-cast to float32, then subtract one.
    float_bits = (bits >> np.uint32(9)) | np.uint32(0x3F800000)
    uniform = float_bits.view(np.float32) - np.float32(1.0)
    uniform = np.maximum(uniform, np.finfo(np.float32).tiny)
    return -np.log(-np.log(uniform))


def sample_msa_for_recycle(
    model_key: np.ndarray,
    msa_mask: np.ndarray,
    num_msa: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select canonical MSA row indices for one AF3 recycle.

    This reproduces the three nested canonical key operations: the model split
    that supplies the Evoformer key, the template split, and the MSA shuffle
    split.  The returned key is the model key for the next recycle.
    """

    next_model_key, evoformer_key = split_key(np.asarray(model_key, np.uint32))
    indices = sample_msa_for_evoformer(evoformer_key, msa_mask, num_msa)
    return next_model_key, indices


def sample_msa_for_evoformer(
    evoformer_key: np.ndarray,
    msa_mask: np.ndarray,
    num_msa: int,
) -> np.ndarray:
    """Select rows after consuming the canonical template and shuffle splits."""

    shuffle_key, _template_key = split_key(evoformer_key)
    _unused_key, sample_key = split_key(shuffle_key)

    mask = np.asarray(msa_mask)
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError("canonical MSA sampling currently supports batch size 1")
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"MSA mask must have shape [msa, tokens], got {mask.shape}")

    active = np.clip(np.sum(mask, axis=-1), 0.0, 1.0)
    logits = (active - 1.0) * 1e6
    scores = logits.astype(np.float32) + _padding_consistent_gumbel(
        sample_key, mask.shape[0]
    )
    # Canonical JAX explicitly requests an unstable sort. Equal float32
    # Gumbel scores have platform-dependent order, so only the selected set is
    # portable across backends; quicksort preserves that unstable contract.
    order = np.argsort(scores, kind="quicksort")[::-1]
    limit = min(int(num_msa), int(order.shape[0]))
    return order[:limit]


def create_msa_features(
    msa_rows: mx.array | np.ndarray,
    msa_mask: mx.array | np.ndarray,
    deletion_matrix: mx.array | np.ndarray | None,
    indices: np.ndarray,
) -> tuple[mx.array, mx.array]:
    """Gather synchronized raw MSA fields and expand only selected rows."""

    rows = mx.array(msa_rows)
    mask = mx.array(msa_mask)
    deletion = None if deletion_matrix is None else mx.array(deletion_matrix)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise ValueError("MSA feature construction currently supports batch size 1")
        rows = rows[0]
        mask = mask[0]
        if deletion is not None:
            deletion = deletion[0]

    row_indices = mx.array(np.asarray(indices, dtype=np.int32))
    rows = rows[row_indices]
    mask = mask[row_indices]
    if deletion is None:
        deletion = mx.zeros(rows.shape, dtype=mx.float32)
    else:
        deletion = deletion[row_indices].astype(mx.float32)

    # 31 canonical polymer/gap classes plus the extra unknown MSA class.
    msa_vocab = 32
    rows = mx.clip(rows, 0, msa_vocab - 1)
    one_hot = (rows[..., None] == mx.arange(msa_vocab)).astype(mx.float32)
    has_deletion = mx.clip(deletion, 0.0, 1.0)[..., None]
    deletion_value = (
        mx.arctan(deletion / 3.0) * (2.0 / mx.pi)
    )[..., None]
    features = mx.concatenate((one_hot, has_deletion, deletion_value), axis=-1)
    return features[None, ...], mask[None, ...]
