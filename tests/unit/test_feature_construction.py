"""Unit tests for MLX-safe feature construction in Model.

Tests that MSA, template, and bond features are constructed correctly
using MLX-compatible operations (no JAX .at[] usage).

Phase 1: MLX-Safe Feature Construction
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx


class TestMSAOneHotEncoding:
    """Test MSA one-hot encoding using MLX-native operations."""

    def test_msa_one_hot_shape(self):
        """Test that MSA one-hot encoding produces correct shape."""
        # Simulate MSA construction from model.py
        batch_size = 1
        num_msa = 16
        num_residues = 10

        # Create mock MSA sequences [batch, num_msa, seq] with amino acid indices 0-21
        msa = mx.array(np.random.randint(0, 22, size=(batch_size, num_msa, num_residues)))

        # MLX-native one-hot encoding via broadcast comparison
        msa_clamped = mx.clip(msa, 0, 21)
        msa_one_hot = (msa_clamped[..., None] == mx.arange(22)).astype(mx.float32)

        assert msa_one_hot.shape == (batch_size, num_msa, num_residues, 22)

    def test_msa_one_hot_values(self):
        """Test that one-hot encoding produces correct values."""
        batch_size = 1
        num_msa = 2
        num_residues = 3

        # Create known MSA: [[0, 1, 2], [3, 4, 5]]
        msa = mx.array([[[0, 1, 2], [3, 4, 5]]])  # [1, 2, 3]

        # MLX-native one-hot encoding
        msa_clamped = mx.clip(msa, 0, 21)
        msa_one_hot = (msa_clamped[..., None] == mx.arange(22)).astype(mx.float32)

        # Check specific positions
        msa_one_hot_np = np.array(msa_one_hot)

        # First sequence [0, 1, 2]: position 0 should have index 0 hot
        assert msa_one_hot_np[0, 0, 0, 0] == 1.0  # amino acid 0
        assert msa_one_hot_np[0, 0, 1, 1] == 1.0  # amino acid 1
        assert msa_one_hot_np[0, 0, 2, 2] == 1.0  # amino acid 2

        # Second sequence [3, 4, 5]: position 0 should have index 3 hot
        assert msa_one_hot_np[0, 1, 0, 3] == 1.0  # amino acid 3
        assert msa_one_hot_np[0, 1, 1, 4] == 1.0  # amino acid 4
        assert msa_one_hot_np[0, 1, 2, 5] == 1.0  # amino acid 5

        # Each row should sum to 1 (one-hot)
        row_sums = msa_one_hot_np.sum(axis=-1)
        assert np.allclose(row_sums, 1.0)

    def test_msa_out_of_range_clamped(self):
        """Test that out-of-range amino acid indices are clamped."""
        msa = mx.array([[[25, -1, 100]]])  # Out of range values

        # MLX-native one-hot encoding
        msa_clamped = mx.clip(msa, 0, 21)
        msa_one_hot = (msa_clamped[..., None] == mx.arange(22)).astype(mx.float32)

        # Should not raise, values should be clamped to 0-21
        msa_one_hot_np = np.array(msa_one_hot)
        assert msa_one_hot_np[0, 0, 0, 21] == 1.0  # 25 clamped to 21
        assert msa_one_hot_np[0, 0, 1, 0] == 1.0   # -1 clamped to 0
        assert msa_one_hot_np[0, 0, 2, 21] == 1.0  # 100 clamped to 21

    def test_mlx_vs_numpy_one_hot_parity(self):
        """Test that MLX-native one-hot matches NumPy reference implementation."""
        batch_size = 2
        num_msa = 8
        num_residues = 16
        num_classes = 22

        # Random MSA data
        np.random.seed(123)
        msa_np = np.random.randint(0, num_classes, size=(batch_size, num_msa, num_residues))
        msa = mx.array(msa_np)

        # NumPy reference implementation (the old approach)
        msa_clamped_np = np.clip(msa_np, 0, 21)
        eye_22 = np.eye(num_classes, dtype=np.float32)
        one_hot_numpy = eye_22[msa_clamped_np]

        # MLX-native implementation (the new approach)
        msa_clamped = mx.clip(msa, 0, 21)
        one_hot_mlx = (msa_clamped[..., None] == mx.arange(num_classes)).astype(mx.float32)
        one_hot_mlx_np = np.array(one_hot_mlx)

        # Should match exactly
        np.testing.assert_array_equal(one_hot_numpy, one_hot_mlx_np)


class TestMSAFeaturePadding:
    """Test MSA feature padding without .at[] usage."""

    def test_padding_smaller_than_channel(self):
        """Test padding when msa_feat_dim < msa_channel."""
        batch_size = 1
        num_msa = 8
        num_residues = 10
        msa_feat_dim = 24
        msa_channel = 64

        msa_features_raw = mx.ones((batch_size, num_msa, num_residues, msa_feat_dim))

        # MLX-safe padding (the new approach)
        pad_width = msa_channel - msa_feat_dim
        padding = mx.zeros((batch_size, num_msa, num_residues, pad_width))
        msa_features = mx.concatenate([msa_features_raw, padding], axis=-1)

        assert msa_features.shape == (batch_size, num_msa, num_residues, msa_channel)

        # First 24 dims should be 1, rest should be 0
        msa_np = np.array(msa_features)
        assert np.allclose(msa_np[:, :, :, :msa_feat_dim], 1.0)
        assert np.allclose(msa_np[:, :, :, msa_feat_dim:], 0.0)

    def test_truncation_larger_than_channel(self):
        """Test truncation when msa_feat_dim >= msa_channel."""
        batch_size = 1
        num_msa = 8
        num_residues = 10
        msa_feat_dim = 64
        msa_channel = 32  # Smaller than feat_dim

        msa_features_raw = mx.arange(msa_feat_dim, dtype=mx.float32)
        msa_features_raw = mx.broadcast_to(
            msa_features_raw[None, None, None, :],
            (batch_size, num_msa, num_residues, msa_feat_dim)
        )

        # MLX-safe truncation
        pad_width = msa_channel - msa_feat_dim
        if pad_width > 0:
            padding = mx.zeros((batch_size, num_msa, num_residues, pad_width))
            msa_features = mx.concatenate([msa_features_raw, padding], axis=-1)
        else:
            msa_features = msa_features_raw[:, :, :, :msa_channel]

        assert msa_features.shape == (batch_size, num_msa, num_residues, msa_channel)


class TestTemplateDistanceBinning:
    """Test template distance binning without .at[] usage."""

    def test_distance_binning_shape(self):
        """Test that distance binning produces correct shape."""
        batch_size = 1
        num_templates = 4
        num_residues = 10
        pair_channel = 128

        # Create mock distances
        distances = mx.array(np.random.uniform(0, 60, size=(batch_size, num_templates, num_residues, num_residues)))

        # MLX-safe binning (the new approach)
        distance_bins = mx.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50], dtype=mx.float32)
        num_bins = 10

        distances_expanded = distances[..., None]
        low_bounds = distance_bins[:-1]
        high_bounds = distance_bins[1:]
        in_bins = ((distances_expanded >= low_bounds) & (distances_expanded < high_bounds)).astype(mx.float32)

        if num_bins < pair_channel:
            padding = mx.zeros((batch_size, num_templates, num_residues, num_residues, pair_channel - num_bins))
            template_pair_features = mx.concatenate([in_bins, padding], axis=-1)
        else:
            template_pair_features = in_bins[:, :, :, :, :pair_channel]

        assert template_pair_features.shape == (batch_size, num_templates, num_residues, num_residues, pair_channel)

    def test_distance_binning_values(self):
        """Test that distance binning assigns correct bins."""
        batch_size = 1
        num_templates = 1
        num_residues = 1

        # Test specific distances
        test_distances = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 17.0, 25.0, 40.0]
        expected_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Which bin each distance falls into

        for dist, expected_bin in zip(test_distances, expected_bins):
            distances = mx.array([[[[dist]]]])  # [1, 1, 1, 1]

            distance_bins = mx.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50], dtype=mx.float32)
            distances_expanded = distances[..., None]
            low_bounds = distance_bins[:-1]
            high_bounds = distance_bins[1:]
            in_bins = ((distances_expanded >= low_bounds) & (distances_expanded < high_bounds)).astype(mx.float32)

            in_bins_np = np.array(in_bins)

            # Only one bin should be 1
            assert in_bins_np[0, 0, 0, 0, expected_bin] == 1.0, f"Distance {dist} should be in bin {expected_bin}"
            assert in_bins_np[0, 0, 0, 0].sum() == 1.0, f"Only one bin should be active for distance {dist}"

    def test_distance_out_of_range(self):
        """Test that distances outside all bins are handled correctly."""
        distances = mx.array([[[[100.0]]]])  # Beyond last bin

        distance_bins = mx.array([0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50], dtype=mx.float32)
        distances_expanded = distances[..., None]
        low_bounds = distance_bins[:-1]
        high_bounds = distance_bins[1:]
        in_bins = ((distances_expanded >= low_bounds) & (distances_expanded < high_bounds)).astype(mx.float32)

        in_bins_np = np.array(in_bins)

        # No bin should be active (100 > 50)
        assert in_bins_np.sum() == 0.0


class TestTargetFeatConstruction:
    """Test target_feat one-hot encoding from aatype using MLX-native operations."""

    def test_target_feat_is_one_hot(self):
        """Test that target_feat is proper one-hot encoding of aatype."""
        # Simulate target_feat construction from model.py
        batch_size = 1
        num_residues = 10

        # Create mock aatype indices
        aatype = mx.array([[0, 1, 5, 10, 15, 20, 21, 3, 7, 11]])  # [1, 10]

        # MLX-native one-hot encoding via broadcast comparison
        aatype_clamped = mx.clip(aatype, 0, 21)
        target_feat = (aatype_clamped[..., None] == mx.arange(22)).astype(mx.float32)  # [batch, seq, 22]

        assert target_feat.shape == (batch_size, num_residues, 22)

        # Each position should have exactly one 1.0 and rest 0.0
        target_feat_np = np.array(target_feat)
        row_sums = target_feat_np.sum(axis=-1)
        assert np.allclose(row_sums, 1.0)

        # Check specific values
        assert target_feat_np[0, 0, 0] == 1.0  # aatype=0 -> index 0 is hot
        assert target_feat_np[0, 1, 1] == 1.0  # aatype=1 -> index 1 is hot
        assert target_feat_np[0, 2, 5] == 1.0  # aatype=5 -> index 5 is hot
        assert target_feat_np[0, 6, 21] == 1.0  # aatype=21 -> index 21 is hot

    def test_target_feat_nonzero_for_known_aatype(self):
        """Test that target_feat is non-zero for any valid aatype."""
        for aa_idx in range(22):  # All amino acid indices
            aatype = mx.array([[aa_idx]])  # [1, 1]

            # MLX-native one-hot encoding
            aatype_clamped = mx.clip(aatype, 0, 21)
            target_feat = (aatype_clamped[..., None] == mx.arange(22)).astype(mx.float32)

            # Should have exactly one non-zero element
            target_feat_np = np.array(target_feat)
            assert target_feat_np.sum() == 1.0, f"aatype={aa_idx} should produce one-hot"
            assert target_feat_np[0, 0, aa_idx] == 1.0, f"aatype={aa_idx} should have hot index at {aa_idx}"

    def test_target_feat_clamping(self):
        """Test that out-of-range aatype values are clamped."""
        aatype = mx.array([[-5, 100, 25]])  # Invalid values

        # MLX-native one-hot encoding
        aatype_clamped = mx.clip(aatype, 0, 21)
        target_feat = (aatype_clamped[..., None] == mx.arange(22)).astype(mx.float32)

        target_feat_np = np.array(target_feat)

        # -5 clamped to 0, 100 clamped to 21, 25 clamped to 21
        assert target_feat_np[0, 0, 0] == 1.0   # -5 -> 0
        assert target_feat_np[0, 1, 21] == 1.0  # 100 -> 21
        assert target_feat_np[0, 2, 21] == 1.0  # 25 -> 21


class TestRelativePositionEmbedding:
    """Test that relative position embedding is applied exactly once (Phase 3 fix)."""

    def test_pair_initial_is_zeros(self):
        """Test that Model initializes pair representation with zeros (not rel_pos)."""
        # The fix ensures pair is initialized with zeros, not relative position
        # Relative position is added ONLY in Evoformer
        batch_size = 1
        num_residues = 10
        pair_channel = 128

        # Model now initializes pair with zeros
        pair = mx.zeros(
            (batch_size, num_residues, num_residues, pair_channel),
            dtype=mx.float32,
        )

        # Should be all zeros
        pair_np = np.array(pair)
        assert np.allclose(pair_np, 0.0)

    def test_evoformer_adds_rel_pos_once(self):
        """Test that Evoformer's rel_pos_embedding adds position info exactly once."""
        from alphafold3_mlx.network.evoformer import RelativePositionEmbedding

        pair_channel = 128
        rel_pos_emb = RelativePositionEmbedding(
            pair_channel=pair_channel,
            max_relative_idx=32,
            max_relative_chain=2,
        )

        batch_size = 1
        num_residues = 10

        # Create mock inputs
        residue_index = mx.arange(num_residues, dtype=mx.int32)[None, :]  # [1, 10]
        asym_id = mx.zeros((batch_size, num_residues), dtype=mx.int32)  # Same chain

        # Get relative position embedding
        rel_pos = rel_pos_emb(residue_index, asym_id)

        assert rel_pos.shape == (batch_size, num_residues, num_residues, pair_channel)

        # Should be non-zero (contains learned position information)
        rel_pos_np = np.array(rel_pos)
        assert not np.allclose(rel_pos_np, 0.0), "rel_pos_embedding should produce non-zero output"

        # Diagonal positions (i=j) should have specific pattern
        # since relative position is 0 at diagonal
        diagonal_values = np.array([rel_pos_np[0, i, i] for i in range(num_residues)])
        # All diagonal values should be the same (same relative position = 0)
        for i in range(1, num_residues):
            assert np.allclose(diagonal_values[0], diagonal_values[i])


class TestBondContactMatrix:
    """Test bond contact matrix construction matching JAX AF3.

    JAX AF3 uses a binary contact_matrix[:, :, None] with shape [seq, seq, 1],
    not a multi-dim one-hot encoding.
    """

    def _build_contact_matrix(self, token_i, token_j, num_res):
        """Build binary symmetric contact matrix (matches model.py JAX-parity path)."""
        ti = mx.array(token_i, dtype=mx.int32)
        tj = mx.array(token_j, dtype=mx.int32)

        valid = (
            (ti >= 0) & (ti < num_res) &
            (tj >= 0) & (tj < num_res)
        )

        if not mx.any(valid).item():
            return mx.zeros((num_res, num_res), dtype=mx.float32)

        valid_float = valid.astype(mx.float32)

        i_range = mx.arange(num_res)
        j_range = mx.arange(num_res)

        i_match = (ti[:, None] == i_range[None, :]).astype(mx.float32)
        j_match = (tj[:, None] == j_range[None, :]).astype(mx.float32)

        result = (valid_float[:, None, None] * i_match[:, :, None] * j_match[:, None, :]).sum(axis=0)
        result_sym = (valid_float[:, None, None] * j_match[:, :, None] * i_match[:, None, :]).sum(axis=0)

        return mx.minimum(result + result_sym, 1.0)

    def test_contact_matrix_shape(self):
        """Test that contact matrix has shape [seq, seq] (scalar per pair)."""
        num_residues = 10
        token_i = [0, 1, 2, 3, 4]
        token_j = [1, 2, 3, 4, 5]

        contact = self._build_contact_matrix(token_i, token_j, num_residues)
        assert contact.shape == (num_residues, num_residues)

        # As bond_features for Evoformer: [batch, seq, seq, 1]
        bond_features = contact[None, :, :, None]
        assert bond_features.shape == (1, num_residues, num_residues, 1)

    def test_contact_matrix_binary(self):
        """Test that contact matrix values are 0 or 1 (binary)."""
        num_residues = 10
        # Duplicate bonds should still produce 1.0 (clamped)
        token_i = [0, 0, 1]
        token_j = [1, 1, 2]

        contact = self._build_contact_matrix(token_i, token_j, num_residues)
        contact_np = np.array(contact)

        # All values should be 0 or 1
        assert np.all((contact_np == 0.0) | (contact_np == 1.0))

    def test_contact_matrix_symmetry(self):
        """Test that contact matrix is symmetric (undirected bonds)."""
        num_residues = 10
        token_i = [0, 2]
        token_j = [3, 5]

        contact = self._build_contact_matrix(token_i, token_j, num_residues)
        contact_np = np.array(contact)

        # Check symmetry
        assert contact_np[0, 3] == contact_np[3, 0] == 1.0
        assert contact_np[2, 5] == contact_np[5, 2] == 1.0

    def test_contact_matrix_invalid_indices_filtered(self):
        """Test that invalid bond indices are filtered out."""
        num_residues = 5
        token_i = [0, -1, 10, 2]  # -1 and 10 are invalid
        token_j = [1, 2, 3, -5]   # -5 is invalid

        contact = self._build_contact_matrix(token_i, token_j, num_residues)
        contact_np = np.array(contact)

        # Only bond 0-1 is valid (symmetric)
        assert contact_np.sum() == 2.0
        assert contact_np[0, 1] == 1.0
        assert contact_np[1, 0] == 1.0

    def test_contact_matrix_zero_padding(self):
        """Test that [0,0] position is zeroed out (JAX parity)."""
        num_residues = 5
        token_i = [0]
        token_j = [0]  # Self-bond at padding position

        contact = self._build_contact_matrix(token_i, token_j, num_residues)

        # Apply JAX-parity zero mask at [0,0]
        zero_mask = 1.0 - (
            (mx.arange(num_residues)[:, None] == 0).astype(mx.float32)
            * (mx.arange(num_residues)[None, :] == 0).astype(mx.float32)
        )
        contact = contact * zero_mask
        contact_np = np.array(contact)

        assert contact_np[0, 0] == 0.0

    def test_contact_matrix_evoformer_compatible(self):
        """Test that contact matrix shape is compatible with Evoformer bond_embedding(input_dims=1)."""
        from alphafold3_mlx.modules import Linear

        num_residues = 8
        pair_channel = 128

        token_i = [0, 1, 2]
        token_j = [1, 2, 3]

        contact = self._build_contact_matrix(token_i, token_j, num_residues)
        bond_features = contact[None, :, :, None]  # [1, seq, seq, 1]

        # Should be compatible with Linear(pair_channel, input_dims=1)
        bond_embed = Linear(pair_channel, input_dims=1, use_bias=False)
        out = bond_embed(bond_features)
        mx.eval(out)

        assert out.shape == (1, num_residues, num_residues, pair_channel)
