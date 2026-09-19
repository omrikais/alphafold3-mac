"""Template modules (distogram features) for AF3 parity."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


def __getattr__(name: str):
    """Lazily expose the template embedders implemented by Evoformer.

    The MLX port originally kept these classes in ``evoformer.py`` while the
    canonical AF3 module exposes them from ``template_modules``.  Lazy aliases
    preserve that public API without creating an import cycle when Evoformer
    imports the atom-table helpers above.
    """
    if name in {"BroadcastProjection", "SingleTemplateEmbedding", "TemplateEmbedding"}:
        from alphafold3_mlx.network.evoformer import (
            BroadcastProjection,
            SingleTemplateEmbedding,
            TemplateEmbedding,
        )

        return {
            "BroadcastProjection": BroadcastProjection,
            "SingleTemplateEmbedding": SingleTemplateEmbedding,
            "TemplateEmbedding": TemplateEmbedding,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BroadcastProjection",
    "DistogramFeaturesConfig",
    "SingleTemplateEmbedding",
    "TemplateEmbedding",
    "dgram_from_positions",
    "make_backbone_rigid",
    "pseudo_beta_fn",
]


def pseudo_beta_fn(
    aatype: mx.array,
    dense_atom_positions: mx.array,
    dense_atom_mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """Select the residue-specific pseudo-beta atom and its mask.

    AF3 stores template atoms in the dense 24-atom layout.  The pseudo-beta
    slot is residue dependent: glycine uses CA while most amino acids use CB,
    and the table also covers the non-protein residue types.  Looking up the
    canonical table avoids assuming one global atom index.
    """
    from alphafold3.model import protein_data_processing

    index_table = mx.array(protein_data_processing.RESTYPE_PSEUDOBETA_INDEX)
    atom_index = index_table[aatype.astype(mx.int32)]
    num_residues = aatype.shape[0]
    residue_index = mx.arange(num_residues)
    pseudo_beta = dense_atom_positions[
        residue_index, atom_index.astype(mx.int32)
    ]
    pseudo_beta_mask = dense_atom_mask[
        residue_index, atom_index.astype(mx.int32)
    ]
    return pseudo_beta, pseudo_beta_mask.astype(mx.float32)


def make_backbone_rigid(
    positions,
    mask: mx.array,
    group_indices: mx.array,
):
    """Construct residue backbone frames from residue-specific atom slots.

    ``RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX`` is indexed by aatype before this
    helper is called.  Group zero is ordered ``(C, CA, N)`` in the dense atom
    layout; the resulting frame is built from ``(C-CA, N-CA)`` and translated
    to CA, matching AF3's ``make_backbone_rigid``.
    """
    from alphafold3_mlx.geometry.rotation_matrix import Rot3Array
    from alphafold3_mlx.geometry.vector import Vec3Array

    backbone_indices = group_indices[:, 0]
    c_idx = backbone_indices[:, 0].astype(mx.int32)
    ca_idx = backbone_indices[:, 1].astype(mx.int32)
    n_idx = backbone_indices[:, 2].astype(mx.int32)

    num_residues = mask.shape[0]
    residue_index = mx.arange(num_residues)
    mask_n = mask[residue_index, n_idx]
    mask_ca = mask[residue_index, ca_idx]
    mask_c = mask[residue_index, c_idx]
    rigid_mask = (mask_n * mask_ca * mask_c).astype(mx.float32)

    def gather(vector, atom_index):
        return Vec3Array(
            x=vector.x[residue_index, atom_index],
            y=vector.y[residue_index, atom_index],
            z=vector.z[residue_index, atom_index],
        )

    pos_n = gather(positions, n_idx)
    pos_ca = gather(positions, ca_idx)
    pos_c = gather(positions, c_idx)
    rotation = Rot3Array.from_two_vectors(pos_c - pos_ca, pos_n - pos_ca)

    class _Rigid:
        def __init__(self, rotation, translation):
            self.rotation = rotation
            self.translation = translation

    return _Rigid(rotation, pos_ca), rigid_mask


@dataclass
class DistogramFeaturesConfig:
    min_bin: float = 3.25
    max_bin: float = 50.75
    num_bins: int = 39


def dgram_from_positions(positions: mx.array, config: DistogramFeaturesConfig) -> mx.array:
    """Compute distogram from positions (AF3 JAX parity).

    Args:
        positions: (num_res, 3) positions.
        config: Distogram bin config.
    Returns:
        Distogram [num_res, num_res, num_bins].
    """
    lower_breaks = mx.linspace(config.min_bin, config.max_bin, config.num_bins)
    lower_breaks = lower_breaks ** 2
    upper_breaks = mx.concatenate(
        [lower_breaks[1:], mx.array([1e8], dtype=mx.float32)], axis=-1
    )
    diff = positions[:, None, :] - positions[None, :, :]
    dist2 = mx.sum(diff ** 2, axis=-1, keepdims=True)
    dgram = (dist2 > lower_breaks).astype(mx.float32) * (dist2 < upper_breaks).astype(mx.float32)
    return dgram
