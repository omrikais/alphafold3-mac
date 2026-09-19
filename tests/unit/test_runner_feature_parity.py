"""Regression tests for preserving canonical AF3 feature semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from alphafold3_mlx.pipeline.runner import InferenceRunner


def test_sequence_only_features_keep_query_msa_and_empty_templates() -> None:
    """The MLX runner must not delete features consumed by canonical AF3."""

    depth = 4104
    msa = np.zeros((depth, 3), dtype=np.int8)
    active_rows = {
        (1, 2, 3): 7,
        (4, 5, 6): 11,
    }
    msa[0] = np.array([1, 2, 3], dtype=np.int8)
    msa[-1] = np.array([4, 5, 6], dtype=np.int8)
    msa_mask = np.zeros_like(msa, dtype=bool)
    msa_mask[0] = True
    msa_mask[-1] = True
    deletion_matrix = np.zeros_like(msa, dtype=np.int8)
    deletion_matrix[0] = 7
    deletion_matrix[-1] = 11
    template_aatype = np.zeros((4, 3), dtype=np.int32)
    template_mask = np.zeros((4, 3, 24), dtype=bool)
    batch_dict = {
        "aatype": np.zeros((3,), dtype=np.int32),
        "msa": msa,
        "msa_mask": msa_mask,
        "deletion_matrix": deletion_matrix,
        "template_aatype": template_aatype,
        "template_all_atom_positions": np.zeros((4, 3, 24, 3), dtype=np.float32),
        "template_all_atom_mask": template_mask,
    }

    af3_input = mock.MagicMock()
    af3_input.fill_missing_fields.return_value = af3_input
    af3_input.user_ccd = None
    af3_input.chains = [object()]
    args = SimpleNamespace(
        max_tokens=None,
        max_template_date="2021-09-30",
        verbose=False,
        seed=42,
        run_data_pipeline=False,
    )
    wrapper = SimpleNamespace(input=af3_input, total_residues=3)
    runner = InferenceRunner(args=args, input_json=wrapper)

    with (
        mock.patch(
            "alphafold3.data.featurisation.featurise_input",
            return_value=[batch_dict],
        ),
        mock.patch("alphafold3.constants.chemical_components.Ccd"),
        mock.patch(
            "alphafold3_mlx.FeatureBatch.from_numpy",
            side_effect=lambda features: features,
        ),
    ):
        prepared = runner._prepare_features()

    assert prepared["msa"] is not None
    assert prepared["msa_mask"] is not None
    assert prepared["deletion_matrix"] is not None
    assert prepared["msa"].shape[0] == depth
    np.testing.assert_array_equal(prepared["msa"], msa)
    np.testing.assert_array_equal(prepared["msa_mask"], msa_mask)
    np.testing.assert_array_equal(prepared["deletion_matrix"], deletion_matrix)
    assert prepared["template_aatype"] is template_aatype
    assert prepared["template_all_atom_mask"] is template_mask
