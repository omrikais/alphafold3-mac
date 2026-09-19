"""Fixed-vector tests for the JAX-compatible MLX RNG layer.

These tests intentionally contain no JAX import. The expected values were
captured from JAX 0.8.1 with ``jax_threefry_partitionable=True``.
"""

from __future__ import annotations

import numpy as np


def test_general_split_matches_embedded_jax_vectors() -> None:
    from alphafold3_mlx import jax_rng

    key = np.array([1705926158, 899080142], dtype=np.uint32)
    expected = np.array(
        [
            [2829437526, 3258650986],
            [2451885785, 2215112154],
            [4179084902, 2188378661],
            [435435197, 103253076],
            [2907409616, 1437230654],
        ],
        dtype=np.uint32,
    )

    np.testing.assert_array_equal(jax_rng.split(key, 5), expected)


def test_float32_normals_match_embedded_jax_vectors() -> None:
    from alphafold3_mlx import jax_rng

    key = np.array([1705926158, 899080142], dtype=np.uint32)
    expected = np.array(
        [
            [-0.21089035, -1.3627948, -0.04500385],
            [-1.1536394, 1.9141139, -0.47701314],
        ],
        dtype=np.float32,
    )

    actual = np.asarray(jax_rng.normal(key, (2, 3)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.2e-6)


def test_sequential_haiku_keys_match_embedded_seed_42_vectors() -> None:
    from alphafold3_mlx import jax_rng

    model_key, second_key = jax_rng.haiku_next_rng_keys(
        jax_rng.key(42), 2
    )
    np.testing.assert_array_equal(
        model_key, np.array([64467757, 2916123636], dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        second_key, np.array([1705926158, 899080142], dtype=np.uint32)
    )

    _, evoformer_key = jax_rng.split(model_key, 2)
    np.testing.assert_array_equal(
        evoformer_key, np.array([2350016172, 1168365246], dtype=np.uint32)
    )


def test_official_af3_post_trunk_diffusion_keys_match_embedded_vectors() -> None:
    """Official trunk internals consume 12 Haiku keys per extra recycle."""
    from alphafold3_mlx import jax_rng

    expected = {
        0: [1364423604, 2995594396],
        1: [2305633233, 858993948],
        10: [301858529, 85087857],
    }
    root_key = jax_rng.key(42)

    for num_recycles, expected_diffusion_key in expected.items():
        model_key, diffusion_key = (
            jax_rng.official_af3_model_and_diffusion_keys(
                root_key, num_recycles=num_recycles
            )
        )
        np.testing.assert_array_equal(
            model_key, np.array([64467757, 2916123636], dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            diffusion_key,
            np.array(expected_diffusion_key, dtype=np.uint32),
        )
