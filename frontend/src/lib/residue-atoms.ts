/** Residue-type-aware atom lookup for restraint editor.
 *
 * Provides valid atom names for each standard amino acid, and a mapping
 * from one-letter to three-letter codes for sequence interpretation.
 */

/** One-letter → three-letter amino acid code mapping. */
export const AA_1TO3: Record<string, string> = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS",
  E: "GLU", Q: "GLN", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO",
  S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};

/** Heavy atoms for each standard amino acid (backbone + side chain). */
export const AMINO_ACID_ATOMS: Record<string, string[]> = {
  ALA: ["N", "CA", "C", "O", "CB"],
  ARG: ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
  ASN: ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
  ASP: ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
  CYS: ["N", "CA", "C", "O", "CB", "SG"],
  GLN: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
  GLU: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
  GLY: ["N", "CA", "C", "O"],
  HIS: ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
  ILE: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
  LEU: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
  LYS: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
  MET: ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
  PHE: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
  PRO: ["N", "CA", "C", "O", "CB", "CG", "CD"],
  SER: ["N", "CA", "C", "O", "CB", "OG"],
  THR: ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
  TRP: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
  TYR: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
  VAL: ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
};

/** Fallback backbone atoms when residue type is unknown. */
export const BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"];

/** Chain information derived from input entities. */
export interface ChainInfo {
  id: string;
  length: number;
  entityType: string;
  sequence: string;
}

/**
 * Get valid atom names for a specific residue position in a chain.
 *
 * For protein chains, looks up the amino acid at the given position and
 * returns its specific atom list. For non-protein chains or unknown
 * residue types, returns common backbone atoms.
 */
export function getAtomOptions(
  chains: ChainInfo[],
  chainId: string,
  residueNum: number,
): string[] {
  const chain = chains.find((c) => c.id === chainId);
  if (!chain || chain.entityType !== "proteinChain") return BACKBONE_ATOMS;

  const oneLetterCode = chain.sequence[residueNum - 1]; // 1-indexed
  if (!oneLetterCode) return BACKBONE_ATOMS;

  const threeLetterCode = AA_1TO3[oneLetterCode.toUpperCase()];
  if (!threeLetterCode) return BACKBONE_ATOMS;

  return AMINO_ACID_ATOMS[threeLetterCode] ?? BACKBONE_ATOMS;
}
