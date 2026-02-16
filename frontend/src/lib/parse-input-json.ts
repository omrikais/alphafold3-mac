import type { FormEntity, Modification } from "@/lib/types";
import type { GuidanceConfig, RestraintConfig } from "@/lib/restraints";

interface ParsedInputJson {
  jobName: string;
  seed: number | null;
  entities: FormEntity[];
  runDataPipeline?: boolean;
  restraints: RestraintConfig;
  guidance: GuidanceConfig;
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function asArray(value: unknown): unknown[] | null {
  return Array.isArray(value) ? value : null;
}

function asPositiveInt(value: unknown, fallback = 1): number {
  const parsed = asNumber(value);
  if (parsed == null) return fallback;
  const intValue = Math.floor(parsed);
  return intValue > 0 ? intValue : fallback;
}

function stripCcdPrefix(value: string): string {
  return value.startsWith("CCD_") ? value.slice(4) : value;
}

function parseModifications(value: unknown): Modification[] {
  const mods = asArray(value);
  if (!mods) return [];

  const parsed: Modification[] = [];
  for (const mod of mods) {
    if (!isRecord(mod)) continue;
    const type = asString(mod.ptmType) ?? asString(mod.type);
    const position = asNumber(mod.ptmPosition) ?? asNumber(mod.position);
    if (!type || position == null) continue;
    parsed.push({ type, position: Math.max(1, Math.floor(position)) });
  }
  return parsed;
}

function parseProteinEntity(
  payload: UnknownRecord,
  nextEntityId: () => string,
): FormEntity | null {
  const sequence = asString(payload.sequence) ?? "";
  if (!sequence) return null;
  const modifications = parseModifications(payload.modifications);

  return {
    id: nextEntityId(),
    type: "proteinChain",
    sequence,
    copies: asPositiveInt(payload.count, 1),
    modifications,
  };
}

function parseNucleicEntity(
  payload: UnknownRecord,
  type: "rnaSequence" | "dnaSequence",
  nextEntityId: () => string,
): FormEntity | null {
  const sequence = asString(payload.sequence) ?? "";
  if (!sequence) return null;

  return {
    id: nextEntityId(),
    type,
    sequence,
    copies: asPositiveInt(payload.count, 1),
  };
}

function parseLigandEntity(
  payload: UnknownRecord,
  nextEntityId: () => string,
): FormEntity | null {
  const smiles = asString(payload.smiles) ?? "";
  const ligandValue =
    asString(payload.ligand) ?? asString(payload.ccdCode) ?? asString(payload.ccd) ?? "";
  const ccdCode = ligandValue ? stripCcdPrefix(ligandValue) : "";

  if (!smiles && !ccdCode) return null;

  return {
    id: nextEntityId(),
    type: "ligand",
    sequence: "",
    copies: asPositiveInt(payload.count, 1),
    ccdCode,
    smiles,
  };
}

function parseIonEntity(
  payload: UnknownRecord,
  nextEntityId: () => string,
): FormEntity | null {
  const ionCode =
    asString(payload.ion) ?? asString(payload.ccdCode) ?? asString(payload.ccd) ?? "";
  if (!ionCode) return null;

  return {
    id: nextEntityId(),
    type: "ion",
    sequence: "",
    copies: asPositiveInt(payload.count, 1),
    ccdCode: stripCcdPrefix(ionCode),
  };
}

function parseSequenceItem(
  item: unknown,
  nextEntityId: () => string,
): FormEntity | null {
  if (!isRecord(item)) return null;

  const protein = isRecord(item.proteinChain) ? item.proteinChain : null;
  if (protein) return parseProteinEntity(protein, nextEntityId);

  const rna = isRecord(item.rnaSequence) ? item.rnaSequence : null;
  if (rna) return parseNucleicEntity(rna, "rnaSequence", nextEntityId);

  const dna = isRecord(item.dnaSequence) ? item.dnaSequence : null;
  if (dna) return parseNucleicEntity(dna, "dnaSequence", nextEntityId);

  const ligand = isRecord(item.ligand) ? item.ligand : null;
  if (ligand) return parseLigandEntity(ligand, nextEntityId);

  const ion = isRecord(item.ion) ? item.ion : null;
  if (ion) return parseIonEntity(ion, nextEntityId);

  return null;
}

function parseSeed(root: UnknownRecord): number | null {
  const modelSeeds =
    asArray(root.modelSeeds) ?? asArray(root.model_seeds) ?? asArray(root.rng_seeds);
  if (!modelSeeds || modelSeeds.length === 0) return null;
  let seed = asNumber(modelSeeds[0]);
  if (seed == null && typeof modelSeeds[0] === "string") {
    const parsed = Number(modelSeeds[0]);
    if (Number.isFinite(parsed)) seed = parsed;
  }
  if (seed == null) return null;
  return Math.floor(seed);
}

function normalizeRoot(input: unknown): UnknownRecord | null {
  let root: unknown = input;

  if (Array.isArray(root)) {
    if (root.length === 0) return null;
    root = root[0];
  }

  if (!isRecord(root)) return null;

  if (isRecord(root.input_json)) return root.input_json;
  return root;
}

export function parseInputJson(
  input: unknown,
  nextEntityId: () => string,
): ParsedInputJson | null {
  const root = normalizeRoot(input);
  if (!root) return null;

  const sequences = asArray(root.sequences);
  if (!sequences || sequences.length === 0) return null;

  const entities: FormEntity[] = [];
  for (const item of sequences) {
    const parsed = parseSequenceItem(item, nextEntityId);
    if (parsed) entities.push(parsed);
  }

  if (entities.length === 0) return null;

  const restraints = isRecord(root.restraints)
    ? (root.restraints as RestraintConfig)
    : {};
  const guidance = isRecord(root.guidance) ? (root.guidance as GuidanceConfig) : {};

  return {
    jobName: asString(root.name) ?? "",
    seed: parseSeed(root),
    entities,
    runDataPipeline: asBoolean(root.runDataPipeline),
    restraints,
    guidance,
  };
}
