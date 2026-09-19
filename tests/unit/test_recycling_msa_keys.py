"""Recycling integration tests for canonical MSA key advancement."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from alphafold3_mlx.model.msa_sampling import sample_msa_for_recycle, split_key
from alphafold3_mlx.model.recycling import run_recycling_loop


def test_recycling_passes_a_distinct_canonical_evoformer_key_each_iteration() -> None:
    """The trunk receives the second model split key on every recycle."""

    observed: list[np.ndarray] = []

    def evoformer_fn(*, single, pair, key, **_kwargs):
        observed.append(np.asarray(key, dtype=np.uint32))
        return single, pair

    initial_key = np.array([0, 42], dtype=np.uint32)
    run_recycling_loop(
        evoformer_fn=evoformer_fn,
        initial_single=mx.zeros((1, 2, 2)),
        initial_pair=mx.zeros((1, 2, 2, 2)),
        residue_index=mx.arange(2)[None],
        asym_id=mx.zeros((1, 2), dtype=mx.int32),
        num_recycles=1,
        recycling_key=initial_key,
    )

    expected = []
    key = initial_key
    for _ in range(2):
        key, evoformer_key = split_key(key)
        expected.append(evoformer_key)
    assert len(observed) == 2
    for actual, wanted in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_recycling_samples_raw_rows_before_feature_expansion() -> None:
    """Recycles select new synchronized rows and expand only that subset."""

    observed: list[tuple[np.ndarray, np.ndarray]] = []

    def evoformer_fn(*, single, pair, msa_features, msa_mask, **_kwargs):
        observed.append((np.asarray(msa_features), np.asarray(msa_mask)))
        return single, pair

    rows = np.broadcast_to(np.arange(24, dtype=np.int32)[:, None], (24, 3))
    mask = np.ones_like(rows, dtype=np.float32)
    deletion = rows.astype(np.float32) + 1.0
    initial_key = np.array([0, 42], dtype=np.uint32)
    run_recycling_loop(
        evoformer_fn=evoformer_fn,
        initial_single=mx.zeros((1, 3, 2)),
        initial_pair=mx.zeros((1, 3, 3, 2)),
        residue_index=mx.arange(3)[None],
        asym_id=mx.zeros((1, 3), dtype=mx.int32),
        num_recycles=1,
        recycling_key=initial_key,
        msa_rows=mx.array(rows),
        msa_mask=mx.array(mask),
        msa_deletion_matrix=mx.array(deletion),
        num_msa=6,
    )

    key = initial_key
    expected_subsets = []
    for _ in range(2):
        key, indices = sample_msa_for_recycle(key, mask, 6)
        expected_subsets.append(indices)

    assert len(observed) == 2
    selected_subsets = []
    for (features, sampled_mask), indices in zip(
        observed, expected_subsets, strict=True
    ):
        assert features.shape == (1, 6, 3, 34)
        np.testing.assert_array_equal(sampled_mask, mask[indices][None])
        selected_rows = np.argmax(features[0, :, 0, :32], axis=-1)
        np.testing.assert_array_equal(selected_rows, rows[indices, 0])
        np.testing.assert_allclose(
            features[0, :, 0, 33],
            np.arctan(deletion[indices, 0] / 3.0) * (2.0 / np.pi),
        )
        selected_subsets.append(selected_rows)
    assert not np.array_equal(selected_subsets[0], selected_subsets[1])
