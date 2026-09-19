"""Focused numerical parity guards for the Evoformer network core."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from types import SimpleNamespace

from alphafold3_mlx.model.model import Model
from alphafold3_mlx.core.config import (
    ConfidenceConfig,
    DiffusionConfig,
    EvoformerConfig,
    GlobalConfig,
    ModelConfig,
    TemplateConfig,
)
from alphafold3_mlx.network.evoformer import Evoformer
from alphafold3_mlx.network.evoformer import BroadcastProjection
from alphafold3_mlx.network.outer_product import OuterProductMeanMSA
from alphafold3_mlx.network.template_modules import make_backbone_rigid, pseudo_beta_fn


def _jax_style_opm_reference(
    msa: np.ndarray,
    mask: np.ndarray,
    left_weight: np.ndarray,
    right_weight: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
) -> np.ndarray:
    """Reference the JAX order: project, sum, project-with-bias, divide."""
    mean = msa.mean(axis=-1, keepdims=True)
    variance = ((msa - mean) ** 2).mean(axis=-1, keepdims=True)
    normalized = (msa - mean) / np.sqrt(variance + 1e-5)
    left = np.einsum("mni,io->mno", normalized, left_weight)
    right = np.einsum("mni,io->mno", normalized, right_weight)
    left *= mask[..., None]
    right *= mask[..., None]
    outer_sum = np.einsum("mio,mjp->ijop", left, right)
    denominator = np.einsum("mi,mj->ij", mask, mask)
    projected = np.einsum(
        "ijo,op->ijp",
        outer_sum.reshape(outer_sum.shape[0], outer_sum.shape[1], -1),
        output_weight,
    ) + output_bias
    return projected / (1e-3 + denominator[..., None])


def test_msa_outer_product_matches_jax_masked_denominator_and_bias() -> None:
    """The MSA OPM must support canonical unbatched template-style inputs."""
    msa = np.asarray(
        [
            [[1.0, -1.0], [2.0, -2.0]],
            [[3.0, -3.0], [4.0, -4.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    opm = OuterProductMeanMSA(msa_channel=2, pair_channel=1, num_outer_channel=1)
    opm.norm.scale = mx.ones((2,))
    opm.norm.offset = mx.zeros((2,))
    opm.left_proj.weight = mx.ones((2, 1))
    opm.right_proj.weight = mx.ones((2, 1))
    opm.output_proj.weight = mx.array([[2.0]])
    opm.output_proj.bias = mx.array([3.0])

    actual = opm(mx.array(msa), mx.zeros((2, 2, 1)), mx.array(mask))
    expected = _jax_style_opm_reference(
        msa,
        mask,
        np.ones((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        np.asarray([[2.0]], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    # Cross-token pairs have no valid MSA row.  The JAX epsilon denominator is
    # retained, so the projected bias remains observable instead of being
    # silently replaced by zero.
    assert float(np.asarray(actual)[0, 1, 0]) > 2_000.0

    batched = opm(mx.array(msa[None]), mx.zeros((1, 2, 2, 1)), mx.array(mask[None]))
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(actual), rtol=0, atol=0)


def test_template_pseudobeta_uses_residue_specific_dense_atom_slot() -> None:
    """Glycine must use CA while alanine uses its residue-specific CB slot."""
    aatype = mx.array([0, 7], dtype=mx.int32)  # alanine, glycine
    positions = mx.zeros((2, 24, 3), dtype=mx.float32)
    positions[0, 4] = mx.array([11.0, 0.0, 0.0])  # alanine pseudo-beta (CB)
    positions[0, 1] = mx.array([1.0, 0.0, 0.0])   # alanine CA
    positions[1, 1] = mx.array([22.0, 0.0, 0.0])  # glycine pseudo-beta (CA)
    mask_np = np.asarray([[1.0] * 24, [1.0] * 24], dtype=np.float32)
    mask_np[0, 4] = 0.0
    mask = mx.array(mask_np)

    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, positions, mask)

    np.testing.assert_allclose(
        np.asarray(pseudo_beta),
        np.asarray([[11.0, 0.0, 0.0], [22.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(pseudo_beta_mask), np.asarray([0.0, 1.0], dtype=np.float32)
    )


def test_template_backbone_mask_requires_all_three_frame_atoms() -> None:
    """The local-frame feature is masked when C, CA, or N is absent."""
    from alphafold3.model import protein_data_processing
    from alphafold3_mlx.geometry.vector import Vec3Array

    aatype = np.asarray([0, 7], dtype=np.int32)
    group_indices = np.asarray(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX)[
        aatype
    ]
    positions_np = np.zeros((2, 24, 3), dtype=np.float32)
    positions_np[:, 2] = np.asarray([0.0, 0.0, 0.0])
    positions_np[:, 1] = np.asarray([1.0, 0.0, 0.0])
    positions_np[:, 0] = np.asarray([1.0, 1.0, 0.0])
    mask_np = np.ones((2, 24), dtype=np.float32)
    mask_np[0, 2] = 0.0  # C is group-zero slot 0 for both residues.

    _, rigid_mask = make_backbone_rigid(
        Vec3Array.from_array(mx.array(positions_np)),
        mx.array(mask_np),
        mx.array(group_indices),
    )

    np.testing.assert_allclose(np.asarray(rigid_mask), np.asarray([0.0, 1.0], dtype=np.float32))


def test_template_scalar_projection_has_canonical_learned_initialization() -> None:
    """Broadcasted scalar template features use a learned Linear weight."""
    projection = BroadcastProjection(4)
    assert projection.weight.shape == (4,)
    assert not bool(mx.all(projection.weight == 0).item())


def test_template_weight_guard_requires_every_trainable_template_parameter() -> None:
    """A partial template mapping must disable random template activations."""
    required = {
        "evoformer.template_embedding.output_linear.weight",
        "evoformer.template_embedding.single_template_embedding.output_layer_norm.scale",
        "evoformer.template_embedding.single_template_embedding.pairformer_layers.0.pair_transition.norm.scale",
    }

    assert not Model._template_weights_complete(required - {next(iter(required))}, required)
    assert Model._template_weights_complete(required, required)


def test_load_weights_disables_templates_on_partial_template_mapping(tmp_path, monkeypatch) -> None:
    """The load path applies the complete-template guard, not just its helper."""
    config = ModelConfig(
        evoformer=EvoformerConfig(num_pairformer_layers=0, num_msa_layers=0, use_msa_stack=False),
        global_config=GlobalConfig(use_compile=False),
        diffusion=DiffusionConfig(num_transformer_blocks=0, atom_transformer_num_blocks=0),
        confidence=ConfidenceConfig(num_pairformer_layers=0),
    )
    model = Model(config)
    weights_path = tmp_path / "weights.bin.zst"
    weights_path.touch()

    import alphafold3_mlx.weights.loader as loader

    monkeypatch.setattr(loader, "load_mlx_params", lambda *args, **kwargs: SimpleNamespace(params={}))
    monkeypatch.setattr(model, "_convert_jax_params", lambda params: {})

    model.load_weights(weights_path)

    assert not model.evoformer.template_embedding.enabled


def test_template_config_disabled_state_is_honored() -> None:
    """The configured template toggle must apply before the first recycle."""
    config = EvoformerConfig(
        num_pairformer_layers=0,
        template=TemplateConfig(enabled=False),
    )
    evoformer = Evoformer(config=config, global_config=GlobalConfig(use_compile=False))
    assert not evoformer.template_embedding.enabled


def test_template_module_exports_canonical_embedder_classes() -> None:
    """The template module remains the public home of template embedders."""
    from alphafold3_mlx.network.evoformer import TemplateEmbedding
    from alphafold3_mlx.network.template_modules import TemplateEmbedding as Exported

    assert Exported is TemplateEmbedding


def test_jax_template_and_opm_weights_map_to_mlx_shapes() -> None:
    """Template weights and stacked OPM output_w keep canonical dimensions."""
    config = ModelConfig(
        evoformer=EvoformerConfig(
            num_pairformer_layers=1,
            num_msa_layers=1,
            use_msa_stack=True,
        ),
        global_config=GlobalConfig(use_compile=False),
        diffusion=DiffusionConfig(num_transformer_blocks=0, atom_transformer_num_blocks=0),
        confidence=ConfidenceConfig(num_pairformer_layers=0),
    )
    model = Model(config)
    params = {
        "diffuser/evoformer/template_embedding/output_linear/weights": np.zeros(
            (64, 128), dtype=np.float32
        ),
        # expand_stack unwraps the leading MSA layer dimension, then add_param
        # flattens canonical [outer, outer, pair] to MLX [outer * outer, pair].
        "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack/outer_product_mean/output_w": np.zeros(
            (1, 32, 32, 128), dtype=np.float32
        ),
    }

    converted = model._convert_jax_params(params)

    np.testing.assert_equal(
        converted["evoformer.template_embedding.output_linear.weight"].shape,
        (64, 128),
    )
    np.testing.assert_equal(
        converted["evoformer.msa_layers.0.outer_product_mean.output_proj.weight"].shape,
        (32 * 32, 128),
    )
