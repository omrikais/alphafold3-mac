/** Constants for the job submission form. */

import type { EntityType, PtmType } from "./types";

export const ENTITY_TYPE_LABELS: Record<EntityType, string> = {
  proteinChain: "Protein",
  rnaSequence: "RNA",
  dnaSequence: "DNA",
  ligand: "Ligand",
  ion: "Ion",
};

export const ENTITY_TYPE_OPTIONS: { value: EntityType; label: string }[] = [
  { value: "proteinChain", label: "Protein" },
  { value: "rnaSequence", label: "RNA" },
  { value: "dnaSequence", label: "DNA" },
  { value: "ligand", label: "Ligand" },
  { value: "ion", label: "Ion" },
];

/** Common ions available in the AlphaFold Server. */
export const COMMON_IONS = [
  { code: "MG", name: "Magnesium (Mg²⁺)" },
  { code: "ZN", name: "Zinc (Zn²⁺)" },
  { code: "FE", name: "Iron (Fe²⁺/Fe³⁺)" },
  { code: "CA", name: "Calcium (Ca²⁺)" },
  { code: "MN", name: "Manganese (Mn²⁺)" },
  { code: "CU", name: "Copper (Cu²⁺)" },
  { code: "CO", name: "Cobalt (Co²⁺)" },
  { code: "NI", name: "Nickel (Ni²⁺)" },
  { code: "NA", name: "Sodium (Na⁺)" },
  { code: "K", name: "Potassium (K⁺)" },
  { code: "CL", name: "Chloride (Cl⁻)" },
];

/** Common post-translational modification types. */
export const PTM_TYPES = [
  { code: "CCD_SEP", name: "Phosphoserine" },
  { code: "CCD_TPO", name: "Phosphothreonine" },
  { code: "CCD_PTR", name: "Phosphotyrosine" },
  { code: "CCD_HYP", name: "Hydroxyproline" },
  { code: "CCD_MLY", name: "N-dimethyl-lysine" },
  { code: "CCD_M3L", name: "N-trimethyl-lysine" },
  { code: "CCD_ALY", name: "N-acetyl-lysine" },
  { code: "CCD_AHB", name: "Beta-hydroxyasparagine" },
  { code: "CCD_P1L", name: "S-palmitoyl-cysteine" },
  { code: "CCD_CGU", name: "Gamma-carboxyglutamic acid" },
  { code: "CCD_AGM", name: "5-methyl-arginine" },
];

/** Maps each PTM CCD code to its biologically relevant target amino acid(s). */
export const PTM_RESIDUE_TARGETS: Record<string, string[]> = {
  CCD_SEP: ["S"],
  CCD_TPO: ["T"],
  CCD_PTR: ["Y"],
  CCD_HYP: ["P"],
  CCD_MLY: ["K"],
  CCD_M3L: ["K"],
  CCD_ALY: ["K"],
  CCD_AHB: ["N"],
  CCD_P1L: ["C"],
  CCD_CGU: ["E"],
  CCD_AGM: ["R"],
};

/** Returns the PTM types applicable to a given amino acid character. */
export function getPtmsForResidue(char: string): PtmType[] {
  return PTM_TYPES.filter(
    (ptm) => PTM_RESIDUE_TARGETS[ptm.code]?.includes(char),
  );
}

export const PROTEIN_CHARS = new Set("ARNDCQEGHILKMFPSTWYVXUBZJO");
export const DNA_CHARS = new Set("ACGT");
export const RNA_CHARS = new Set("ACGU");

/** Stage display names for the progress indicator. */
export const STAGE_LABELS: Record<string, string> = {
  weight_loading: "Loading model",
  data_pipeline: "MSA & template search",
  feature_preparation: "Preparing features",
  recycling: "Recycling (Evoformer)",
  diffusion: "Diffusion denoising",
  confidence: "Computing confidence",
  output_writing: "Writing outputs",
};
