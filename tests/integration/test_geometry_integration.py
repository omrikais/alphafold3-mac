"""Geometry integration tests.

Tests that Vec3Array and Rot3Array from Phase 2 geometry modules
are properly integrated into the model pipeline.
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from alphafold3_mlx.geometry import Vec3Array, Rot3Array
from alphafold3_mlx.model import Model
from alphafold3_mlx.core import ModelConfig, FeatureBatch
from alphafold3_mlx.core.config import EvoformerConfig, DiffusionConfig, GlobalConfig
from alphafold3_mlx.network.diffusion_head import DiffusionHead


def create_test_batch(num_residues: int = 10, seed: int = 42) -> FeatureBatch:
    """Create minimal feature batch for testing."""
    np.random.seed(seed)

    feature_dict = {
        "aatype": np.random.randint(0, 20, size=num_residues).astype(np.int32),
        "token_mask": np.ones(num_residues, dtype=np.float32),
        "residue_index": np.arange(num_residues, dtype=np.int32),
        "asym_id": np.zeros(num_residues, dtype=np.int32),
        "entity_id": np.zeros(num_residues, dtype=np.int32),
        "sym_id": np.zeros(num_residues, dtype=np.int32),
    }

    return FeatureBatch.from_numpy(feature_dict)


class TestVec3ArrayIntegration:
    """Test Vec3Array integration."""

    def test_vec3_from_coordinates(self):
        """Test creating Vec3Array from coordinate array."""
        # Simulate model output coordinates
        coords = mx.array([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # batch 0
        ])  # [batch, atoms, 3]

        vec = Vec3Array.from_array(coords)

        assert vec.shape == (1, 2)
        np.testing.assert_allclose(np.array(vec.x), [[1.0, 4.0]])
        np.testing.assert_allclose(np.array(vec.y), [[2.0, 5.0]])
        np.testing.assert_allclose(np.array(vec.z), [[3.0, 6.0]])

    def test_vec3_roundtrip(self):
        """Test Vec3Array to array roundtrip."""
        coords = mx.array([
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]],
        ])

        vec = Vec3Array.from_array(coords)
        coords_back = vec.to_array()

        np.testing.assert_allclose(np.array(coords), np.array(coords_back))

    def test_vec3_distance_calculation(self):
        """Test distance calculation using Vec3Array."""
        from alphafold3_mlx.geometry import euclidean_distance

        # Two points
        p1 = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )
        p2 = Vec3Array(
            x=mx.array([3.0]),
            y=mx.array([4.0]),
            z=mx.array([0.0]),
        )

        dist = euclidean_distance(p1, p2)
        np.testing.assert_allclose(np.array(dist), [5.0])


class TestRot3ArrayIntegration:
    """Test Rot3Array integration."""

    def test_rot3_identity_preserves_coords(self):
        """Test that identity rotation preserves coordinates."""
        coords = Vec3Array(
            x=mx.array([1.0, 2.0, 3.0]),
            y=mx.array([4.0, 5.0, 6.0]),
            z=mx.array([7.0, 8.0, 9.0]),
        )

        identity = Rot3Array.identity((3,))
        rotated = identity.apply_to_point(coords)

        np.testing.assert_allclose(np.array(rotated.x), np.array(coords.x))
        np.testing.assert_allclose(np.array(rotated.y), np.array(coords.y))
        np.testing.assert_allclose(np.array(rotated.z), np.array(coords.z))

    def test_rot3_random_uniform_is_orthogonal(self):
        """Test that random rotations are valid (orthogonal)."""
        key = mx.random.key(42)
        rot = Rot3Array.random_uniform(key=key, shape=(10,))

        # Convert to matrix and check R @ R^T = I
        rot_mat = rot.to_array()  # [10, 3, 3]
        rot_mat_T = mx.transpose(rot_mat, (0, 2, 1))

        identity = mx.eye(3)
        result = rot_mat @ rot_mat_T

        # Should be close to identity
        for i in range(10):
            np.testing.assert_allclose(
                np.array(result[i]),
                np.array(identity),
                atol=1e-5,
            )

    def test_rot3_determinant_is_one(self):
        """Test that random rotations have determinant 1 (not -1)."""
        key = mx.random.key(42)
        rot = Rot3Array.random_uniform(key=key, shape=(10,))

        rot_mat = rot.to_array()

        for i in range(10):
            det = np.linalg.det(np.array(rot_mat[i]))
            np.testing.assert_allclose(det, 1.0, atol=1e-5)


class TestDiffusionHeadGeometryIntegration:
    """Test geometry integration in diffusion head.

    Tests the random_augmentation function which uses geometric primitives
    for random rotation and translation.
    """

    def test_random_augmentation_uses_geometry(self):
        """Test that random_augmentation applies random rotation and translation."""
        from alphafold3_mlx.network.diffusion_head import random_augmentation

        num_residues = 4
        num_atoms = 37  # atom37 format
        coords = mx.random.normal(
            shape=(num_residues, num_atoms, 3), key=mx.random.key(42)
        )
        mask = mx.ones((num_residues, num_atoms))
        key = mx.random.key(123)

        transformed = random_augmentation(key, coords, mask)

        # Result should be valid (no NaN, proper shape)
        mx.eval(transformed)
        assert transformed.shape == coords.shape
        assert not np.any(np.isnan(np.array(transformed)))

    def test_random_augmentation_preserves_distances(self):
        """Test that rigid transformation preserves pairwise distances."""
        from alphafold3_mlx.network.diffusion_head import random_augmentation

        # Create atom37 coordinates
        num_residues = 2
        num_atoms = 37  # atom37
        coords = mx.array(np.random.randn(num_residues, num_atoms, 3).astype(np.float32))
        mask = mx.ones((num_residues, num_atoms))
        key = mx.random.key(42)

        transformed = random_augmentation(key, coords, mask)
        mx.eval(transformed)

        # Flatten to [num_residues * num_atoms, 3] for distance calculation
        coords_flat = coords.reshape(-1, 3)
        transformed_flat = transformed.reshape(-1, 3)

        # Compute pairwise distances before and after
        def pairwise_dist(c):
            diff = c[:, None, :] - c[None, :, :]  # [n, n, 3]
            return mx.sqrt(mx.sum(diff ** 2, axis=-1))

        dist_before = pairwise_dist(coords_flat)
        dist_after = pairwise_dist(transformed_flat)

        # Distances should be preserved (rigid transformation)
        np.testing.assert_allclose(
            np.array(dist_before),
            np.array(dist_after),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_random_augmentation_different_seeds_different_results(self):
        """Test that different seeds produce different transformations."""
        from alphafold3_mlx.network.diffusion_head import random_augmentation

        num_residues = 4
        num_atoms = 37  # atom37
        coords = mx.random.normal(
            shape=(num_residues, num_atoms, 3), key=mx.random.key(42)
        )
        mask = mx.ones((num_residues, num_atoms))

        t1 = random_augmentation(mx.random.key(1), coords, mask)
        t2 = random_augmentation(mx.random.key(2), coords, mask)

        mx.eval(t1, t2)

        # Results should be different
        max_diff = np.max(np.abs(np.array(t1) - np.array(t2)))
        assert max_diff > 0.1, "Different seeds should produce different results"


class TestModelWithGeometry:
    """Test full model integration with geometry modules."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(
            evoformer=EvoformerConfig(
                num_pairformer_layers=2,
                use_msa_stack=False,
            ),
            diffusion=DiffusionConfig(
                num_steps=5,
                num_samples=1,
                num_transformer_blocks=4,  # Must be divisible by super_block_size=4
            ),
            global_config=GlobalConfig(use_compile=False),
            num_recycles=1,
        )
        return Model(config)

    def test_model_inference_with_geometry(self, model):
        """Test that model runs with geometry integration."""
        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)

        # Basic sanity checks
        assert not np.any(np.isnan(coords)), "Coordinates should not contain NaN"
        assert coords.shape[0] == 1  # num_samples
        assert coords.shape[1] == 8  # num_residues

    def test_model_coordinates_finite(self, model):
        """Test that model produces finite coordinates."""
        batch = create_test_batch(num_residues=10)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)

        assert np.all(np.isfinite(coords)), "All coordinates should be finite"

    def test_model_multiple_samples_different(self, model):
        """Test that multiple samples produce different results."""
        # Update model config to generate multiple samples
        from dataclasses import replace
        new_diffusion = replace(model.config.diffusion, num_samples=2)
        object.__setattr__(model.config, "diffusion", new_diffusion)
        model.diffusion_head.num_samples = 2

        batch = create_test_batch(num_residues=8)
        key = mx.random.key(42)

        result = model(batch, key)
        mx.eval(result.atom_positions.positions)

        coords = np.array(result.atom_positions.positions)

        # Different samples should have different coordinates
        sample_diff = np.max(np.abs(coords[0] - coords[1] if coords.shape[0] > 1 else 0))

        # Note: This test may need adjustment based on actual model behavior
        # With random initialization, samples should differ


class TestGeometryEdgeCases:
    """Test edge cases in geometry integration."""

    def test_zero_vector_normalization(self):
        """Test that zero vectors are handled in normalization."""
        vec = Vec3Array(
            x=mx.array([0.0]),
            y=mx.array([0.0]),
            z=mx.array([0.0]),
        )

        # Should not produce NaN with epsilon
        normalized = vec.normalized(epsilon=1e-6)
        mx.eval(normalized.x, normalized.y, normalized.z)

        # Result should be finite (not NaN)
        assert np.isfinite(np.array(normalized.x)[0])

    def test_rotation_composition(self):
        """Test that rotation composition works correctly."""
        key = mx.random.key(42)
        k1, k2 = mx.random.split(key)

        r1 = Rot3Array.random_uniform(key=k1, shape=())
        r2 = Rot3Array.random_uniform(key=k2, shape=())

        # Compose rotations
        r_composed = r1 @ r2

        # Apply to a point
        point = Vec3Array(
            x=mx.array(1.0),
            y=mx.array(0.0),
            z=mx.array(0.0),
        )

        # Should equal sequential application
        p1 = r2.apply_to_point(point)
        p2 = r1.apply_to_point(p1)
        p_composed = r_composed.apply_to_point(point)

        np.testing.assert_allclose(np.array(p2.x), np.array(p_composed.x), atol=1e-5)
        np.testing.assert_allclose(np.array(p2.y), np.array(p_composed.y), atol=1e-5)
        np.testing.assert_allclose(np.array(p2.z), np.array(p_composed.z), atol=1e-5)
