"""Atom constants for AlphaFold 3 MLX.

This module provides atom37 representation constants and utilities for
converting between residue-level and atom-level representations.
"""

from __future__ import annotations

import numpy as np

# Atom37 ordering (must match alphafold3.constants.atom_types)
ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)
ATOM37_ORDER: dict[str, int] = {name: i for i, name in enumerate(ATOM37_NAMES)}
NUM_ATOMS: int = 37

# Backbone atom indices in atom37
BACKBONE_ATOM_INDICES: tuple[int, ...] = (
    ATOM37_ORDER["N"],   # 0
    ATOM37_ORDER["CA"],  # 1
    ATOM37_ORDER["C"],   # 2
    ATOM37_ORDER["O"],   # 4
)

# CB index (used for most residue types, except GLY)
CB_INDEX: int = ATOM37_ORDER["CB"]  # 3

# Residue type names (20 standard amino acids + UNK)
RESTYPE_NAMES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK", "X",  # Unknown residue types
)
NUM_RESTYPES: int = len(RESTYPE_NAMES)

# Atoms present in each residue type (indices into atom37)
# Backbone (N, CA, C, O) is always present; CB present for all except GLY
RESTYPE_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "CB", "O"),
    "ARG": ("N", "CA", "C", "CB", "O", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "CB", "O", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "CB", "O", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "CB", "O", "SG"),
    "GLN": ("N", "CA", "C", "CB", "O", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "CB", "O", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),  # No CB for glycine
    "HIS": ("N", "CA", "C", "CB", "O", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "CB", "O", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "CB", "O", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "CB", "O", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "CB", "O", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "CB", "O", "CG", "CD"),
    "SER": ("N", "CA", "C", "CB", "O", "OG"),
    "THR": ("N", "CA", "C", "CB", "O", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "CB", "O", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "CB", "O", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "CB", "O", "CG1", "CG2"),
    "UNK": ("N", "CA", "C", "CB", "O"),  # Treat unknown as ALA
    "X": ("N", "CA", "C", "CB", "O"),    # Treat X as ALA
}

# Build atom37 mask for each residue type
# Shape: [NUM_RESTYPES, NUM_ATOMS]
def _build_restype_atom37_mask() -> np.ndarray:
    """Build mask indicating which atoms exist for each residue type."""
    mask = np.zeros((NUM_RESTYPES, NUM_ATOMS), dtype=np.float32)
    for i, restype in enumerate(RESTYPE_NAMES):
        atoms = RESTYPE_ATOMS.get(restype, RESTYPE_ATOMS["UNK"])
        for atom_name in atoms:
            if atom_name in ATOM37_ORDER:
                mask[i, ATOM37_ORDER[atom_name]] = 1.0
    return mask

RESTYPE_ATOM37_MASK: np.ndarray = _build_restype_atom37_mask()

# Ideal bond lengths (Angstroms) for backbone geometry
IDEAL_BOND_LENGTHS: dict[tuple[str, str], float] = {
    ("N", "CA"): 1.458,
    ("CA", "C"): 1.523,
    ("C", "O"): 1.231,
    ("C", "N"): 1.329,  # Peptide bond
    ("CA", "CB"): 1.530,
}

# Ideal bond angles (degrees) for backbone geometry
IDEAL_BOND_ANGLES: dict[tuple[str, str, str], float] = {
    ("N", "CA", "C"): 111.2,
    ("CA", "C", "O"): 120.8,
    ("CA", "C", "N"): 116.2,  # Next residue N
    ("C", "N", "CA"): 121.7,  # From prev residue
    ("N", "CA", "CB"): 110.5,
    ("C", "CA", "CB"): 110.1,
}


def get_atom37_mask(aatype):
    """Get atom37 validity mask for given residue types.

    Args:
        aatype: Residue type indices (mx.array). Shape: [..., num_residues]

    Returns:
        Atom mask (mx.array). Shape: [..., num_residues, 37]
    """
    import mlx.core as mx
    # Convert to numpy for indexing, then back to MLX
    aatype_np = np.array(aatype)
    # Clip to valid range
    aatype_clipped = np.clip(aatype_np, 0, NUM_RESTYPES - 1)
    # Gather masks
    mask = RESTYPE_ATOM37_MASK[aatype_clipped]
    return mx.array(mask)


def backbone_to_atom37(
    backbone_coords,
    aatype,
    mask=None,
):
    """Convert backbone coordinates to atom37 representation.

    Uses ideal geometry to place side-chain atoms. Backbone atoms (N, CA, C, O)
    are taken directly from input. CB is placed using ideal tetrahedral geometry.
    Other sidechain atoms are placed at CB position (simplified placeholder).

    Args:
        backbone_coords: Backbone atom positions [N, CA, C, O] (mx.array).
            Shape: [..., num_residues, 4, 3]
        aatype: Residue type indices (mx.array). Shape: [..., num_residues]
        mask: Optional residue mask (mx.array). Shape: [..., num_residues]

    Returns:
        Tuple of:
            - atom37_coords: Full atom positions. Shape: [..., num_residues, 37, 3]
            - atom37_mask: Atom validity mask. Shape: [..., num_residues, 37]
    """
    import mlx.core as mx
    # Extract backbone positions
    n_pos = backbone_coords[..., 0, :]   # N
    ca_pos = backbone_coords[..., 1, :]  # CA
    c_pos = backbone_coords[..., 2, :]   # C
    o_pos = backbone_coords[..., 3, :]   # O

    # Compute CB position using ideal tetrahedral geometry from N, CA, C
    n_to_ca = ca_pos - n_pos
    ca_to_c = c_pos - ca_pos

    # Normalize
    n_to_ca_norm = n_to_ca / (mx.sqrt(mx.sum(n_to_ca ** 2, axis=-1, keepdims=True)) + 1e-8)
    ca_to_c_norm = ca_to_c / (mx.sqrt(mx.sum(ca_to_c ** 2, axis=-1, keepdims=True)) + 1e-8)

    # Cross product for perpendicular direction
    perp = mx.stack([
        n_to_ca_norm[..., 1] * ca_to_c_norm[..., 2] - n_to_ca_norm[..., 2] * ca_to_c_norm[..., 1],
        n_to_ca_norm[..., 2] * ca_to_c_norm[..., 0] - n_to_ca_norm[..., 0] * ca_to_c_norm[..., 2],
        n_to_ca_norm[..., 0] * ca_to_c_norm[..., 1] - n_to_ca_norm[..., 1] * ca_to_c_norm[..., 0],
    ], axis=-1)
    perp_norm = perp / (mx.sqrt(mx.sum(perp ** 2, axis=-1, keepdims=True)) + 1e-8)

    # CB is approximately opposite to C direction with tetrahedral angle
    cb_dir = -0.58 * ca_to_c_norm + 0.58 * n_to_ca_norm - 0.57 * perp_norm
    cb_dir_norm = cb_dir / (mx.sqrt(mx.sum(cb_dir ** 2, axis=-1, keepdims=True)) + 1e-8)
    cb_pos = ca_pos + IDEAL_BOND_LENGTHS[("CA", "CB")] * cb_dir_norm

    # Build atom37 coordinates by concatenating all atom positions
    # atom37 order: N(0), CA(1), C(2), CB(3), O(4), rest are sidechain (5-36)
    # For simplicity, place all sidechain atoms at CB position

    # Create list of atom positions in atom37 order
    atom_positions = [
        n_pos,    # 0: N
        ca_pos,   # 1: CA
        c_pos,    # 2: C
        cb_pos,   # 3: CB
        o_pos,    # 4: O
    ]
    # Sidechain atoms (5-36): place at CB position
    for _ in range(NUM_ATOMS - 5):
        atom_positions.append(cb_pos)

    # Stack along atom dimension: [..., residues, 37, 3]
    atom37_coords = mx.stack(atom_positions, axis=-2)

    # Get shapes for mask handling
    batch_shape = backbone_coords.shape[:-2]
    num_residues = backbone_coords.shape[-3]

    # Get atom mask from residue types
    atom37_mask = get_atom37_mask(aatype)

    # Expand mask to match batch shape if needed
    target_shape = (*batch_shape, num_residues, NUM_ATOMS)
    if atom37_mask.shape != target_shape:
        atom37_mask = mx.broadcast_to(atom37_mask, target_shape)

    # Apply residue mask if provided
    if mask is not None:
        residue_mask = mask[..., None]  # [..., num_residues, 1]
        atom37_mask = atom37_mask * residue_mask

    return atom37_coords, atom37_mask


def get_backbone_atom_indices():
    """Get indices of backbone atoms in atom37.

    Returns:
        Array of backbone atom indices [N, CA, C, O] (mx.array).
    """
    import mlx.core as mx
    return mx.array([
        ATOM37_ORDER["N"],
        ATOM37_ORDER["CA"],
        ATOM37_ORDER["C"],
        ATOM37_ORDER["O"],
    ])


def get_cb_or_ca_index(aatype):
    """Get index of CB (or CA for glycine) for each residue.

    Used for computing pseudo-beta positions for distogram.

    Args:
        aatype: Residue type indices (mx.array). Shape: [..., num_residues]

    Returns:
        Atom indices for pseudo-beta (mx.array). Shape: [..., num_residues]
    """
    import mlx.core as mx
    # GLY (index 7) uses CA, others use CB
    is_gly = (aatype == 7)
    return mx.where(is_gly, ATOM37_ORDER["CA"], ATOM37_ORDER["CB"])
