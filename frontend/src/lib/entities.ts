import type { FormEntity } from "./types";

/** Convert form entities to the AlphaFold Server sequences format. */
export function entitiesToSequences(entities: FormEntity[]): Record<string, unknown>[] {
  return entities.map((entity) => {
    switch (entity.type) {
      case "proteinChain": {
        const pc: Record<string, unknown> = {
          sequence: entity.sequence,
          count: entity.copies,
        };
        if (entity.modifications?.length) {
          pc.modifications = entity.modifications.map((m) => ({
            ptmType: m.type,
            ptmPosition: m.position,
          }));
        }
        return { proteinChain: pc };
      }
      case "rnaSequence":
        return { rnaSequence: { sequence: entity.sequence, count: entity.copies } };
      case "dnaSequence":
        return { dnaSequence: { sequence: entity.sequence, count: entity.copies } };
      case "ligand":
        return {
          ligand: {
            ...(entity.smiles
              ? { smiles: entity.smiles }
              : { ligand: `CCD_${entity.ccdCode}` }),
            count: entity.copies,
          },
        };
      case "ion":
        return { ion: { ion: entity.ccdCode || "MG", count: entity.copies } };
      default:
        return {};
    }
  });
}
