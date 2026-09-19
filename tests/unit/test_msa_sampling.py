"""Canonical AF3 MSA sampling parity tests."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphafold3.model.network import featurization as jax_featurization


def _canonical_recycle_sample(
    key: jax.Array,
    msa_mask: np.ndarray,
    num_msa: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the canonical model/template/shuffle key schedule for one recycle."""

    next_key, evoformer_key = jax.random.split(key)
    shuffle_key, _template_key = jax.random.split(evoformer_key)
    _unused_key, sample_key = jax.random.split(shuffle_key)
    logits = (
        jnp.clip(jnp.sum(jnp.asarray(msa_mask), axis=-1), 0.0, 1.0) - 1.0
    ) * 1e6
    order = jax_featurization.gumbel_argsort_sample_idx(sample_key, logits)
    return np.asarray(next_key, dtype=np.uint32), np.asarray(order[:num_msa])


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("depth", [9, 17])
def test_sampler_and_recycle_keys_match_canonical_jax(seed: int, depth: int) -> None:
    """Pure MLX/NumPy sampling must exactly match canonical JAX recycles."""

    try:
        sampling = importlib.import_module("alphafold3_mlx.model.msa_sampling")
    except ModuleNotFoundError:
        pytest.fail("canonical MSA sampler is not implemented")

    # The first seven rows are identical across padding depths. Include a
    # partially masked active row and inactive padding rows.
    mask = np.zeros((depth, 5), dtype=np.float32)
    mask[:6] = 1.0
    mask[6, 2] = 1.0
    model_key = np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)

    for _ in range(2):
        expected_key, expected_indices = _canonical_recycle_sample(
            jnp.asarray(model_key), mask, num_msa=7
        )
        actual_key, actual_indices = sampling.sample_msa_for_recycle(
            model_key, mask, num_msa=7
        )

        np.testing.assert_array_equal(actual_key, expected_key)
        np.testing.assert_array_equal(actual_indices, expected_indices)
        assert set(actual_indices.tolist()) == set(range(7))
        model_key = actual_key


def test_padding_does_not_change_active_row_order() -> None:
    """Canonical per-row randomness must be stable when padding depth changes."""

    try:
        sampling = importlib.import_module("alphafold3_mlx.model.msa_sampling")
    except ModuleNotFoundError:
        pytest.fail("canonical MSA sampler is not implemented")

    short_mask = np.ones((7, 3), dtype=np.float32)
    padded_mask = np.concatenate(
        [short_mask, np.zeros((10, 3), dtype=np.float32)], axis=0
    )
    key = np.array([0, 42], dtype=np.uint32)

    _, short_indices = sampling.sample_msa_for_recycle(key, short_mask, 7)
    _, padded_indices = sampling.sample_msa_for_recycle(key, padded_mask, 7)

    np.testing.assert_array_equal(padded_indices, short_indices)


def test_successive_recycles_can_select_different_active_subsets() -> None:
    """Each recycle must advance the key instead of reusing one fixed subset."""

    try:
        sampling = importlib.import_module("alphafold3_mlx.model.msa_sampling")
    except ModuleNotFoundError:
        pytest.fail("canonical MSA sampler is not implemented")

    mask = np.ones((32, 4), dtype=np.float32)
    key = np.array([0, 42], dtype=np.uint32)
    key, first = sampling.sample_msa_for_recycle(key, mask, 8)
    _, second = sampling.sample_msa_for_recycle(key, mask, 8)

    assert not np.array_equal(first, second)
    assert np.all(mask[first].any(axis=-1))
    assert np.all(mask[second].any(axis=-1))
    # Row zero is the query row. Across canonical seeds it remains eligible,
    # rather than being discarded before sampling.
    selected = set(first.tolist()) | set(second.tolist())
    for seed in range(64):
        _, indices = sampling.sample_msa_for_recycle(
            np.array([0, seed], dtype=np.uint32), mask, 8
        )
        selected.update(indices.tolist())
    assert 0 in selected


def test_selected_rows_masks_and_deletions_stay_synchronized() -> None:
    """Feature expansion must happen after one shared row selection."""

    sampling = importlib.import_module("alphafold3_mlx.model.msa_sampling")
    rows = np.arange(12, dtype=np.int32).reshape(4, 3)
    mask = np.array(
        [[1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0]], dtype=np.float32
    )
    deletion = rows.astype(np.float32) + 10.0
    indices = np.array([3, 1], dtype=np.int32)

    features, selected_mask = sampling.create_msa_features(
        rows, mask, deletion, indices
    )

    assert features.shape == (1, 2, 3, 34)
    np.testing.assert_array_equal(selected_mask, mask[indices][None])
    selected = np.asarray(features)[0]
    np.testing.assert_array_equal(np.argmax(selected[..., :32], axis=-1), rows[indices])
    np.testing.assert_allclose(selected[..., 32], np.clip(deletion[indices], 0, 1))
    np.testing.assert_allclose(
        selected[..., 33], np.arctan(deletion[indices] / 3.0) * (2.0 / np.pi)
    )


def test_full_af3_depth_matches_canonical_selected_subsets_across_recycles() -> None:
    """The 16,384-row production depth selects the canonical 1,024-row sets."""

    sampling = importlib.import_module("alphafold3_mlx.model.msa_sampling")
    mask = np.ones((16_384, 1), dtype=np.float32)
    model_key = np.asarray(jax.random.PRNGKey(42), dtype=np.uint32)

    for _ in range(2):
        expected_key, expected = _canonical_recycle_sample(
            jnp.asarray(model_key), mask, num_msa=1024
        )
        actual_key, actual = sampling.sample_msa_for_recycle(
            model_key, mask, num_msa=1024
        )
        np.testing.assert_array_equal(actual_key, expected_key)
        # Canonical requests an unstable sort, so equal float32 Gumbel values
        # can swap order across backends while selecting the same rows.
        assert set(actual.tolist()) == set(expected.tolist())
        model_key = actual_key
