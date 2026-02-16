/** TypeScript interfaces matching the API schemas. */

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export type EntityType = "proteinChain" | "rnaSequence" | "dnaSequence" | "ligand" | "ion";

export interface JobSummary {
  id: string;
  name: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  num_residues: number | null;
  num_chains: number | null;
  error_message: string | null;
  progress: number;
}

export interface JobDetail extends JobSummary {
  input_json: Record<string, unknown>;
  num_samples: number;
  diffusion_steps: number;
  run_data_pipeline: boolean;
  current_stage: string | null;
}

export interface PaginatedJobs {
  jobs: JobSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface JobCreated {
  id: string;
  status: JobStatus;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  estimated_memory_gb: number | null;
  num_residues: number | null;
  num_chains: number | null;
}

export interface SystemStatus {
  model_loaded: boolean;
  model_loading: boolean;
  chip_family: string;
  memory_gb: number;
  supports_bfloat16: boolean;
  queue_size: number;
  active_job_id: string | null;
  version: string;
  run_data_pipeline: boolean;
}

export interface DistanceSatisfaction {
  chain_i: string;
  residue_i: number;
  atom_i: string;
  chain_j: string;
  residue_j: number;
  atom_j: string;
  target_distance: number;
  actual_distance: number;
  satisfied: boolean;
}

export interface ContactSatisfaction {
  chain_i: string;
  residue_i: number;
  closest_candidate_chain: string;
  closest_candidate_residue: number;
  threshold: number;
  actual_distance: number;
  satisfied: boolean;
}

export interface RepulsiveSatisfaction {
  chain_i: string;
  residue_i: number;
  chain_j: string;
  residue_j: number;
  min_distance: number;
  actual_distance: number;
  satisfied: boolean;
}

export interface RestraintSatisfaction {
  distance?: DistanceSatisfaction[];
  contact?: ContactSatisfaction[];
  repulsive?: RepulsiveSatisfaction[];
}

export interface SampleConfidence {
  sample_index: number;
  ptm: number | null;
  iptm: number | null;
  mean_plddt: number | null;
  plddt: number[];
  pae: number[][];
  num_residues: number;
  restraint_satisfaction?: RestraintSatisfaction;
}

export interface ConfidenceResult {
  ptm: number | null;
  iptm: number | null;
  mean_plddt: number | null;
  ranking_metric: string | null;
  num_samples: number;
  samples: { ptm: number | null; iptm: number | null; mean_plddt: number | null; rank: number }[];
  best_sample_index: number;
  is_complex: boolean;
}

export interface WSMessage {
  type: "stage_change" | "progress" | "completed" | "failed" | "cancelled" | "ping" | "error";
  stage?: string;
  percent_complete?: number;
  recycling_iteration?: number;
  recycling_total?: number;
  diffusion_step?: number;
  diffusion_total?: number;
  error?: string;
}

/** Entity in the job submission form. */
export interface FormEntity {
  id: string;
  type: EntityType;
  sequence: string;
  copies: number;
  /** CCD code for ligands/ions */
  ccdCode?: string;
  /** SMILES string for custom ligands */
  smiles?: string;
  /** Post-translational modifications */
  modifications?: Modification[];
}

export interface Modification {
  type: string;
  position: number;
}

export interface PtmType {
  code: string;
  name: string;
}

/** Job submission payload (AlphaFold Server format). */
export interface JobSubmission {
  name: string;
  modelSeeds: number[];
  sequences: Record<string, unknown>[];
  dialect?: string;
  version?: number;
  numSamples?: number;
  diffusionSteps?: number;
  runDataPipeline?: boolean;
  useCache?: boolean;
  restraints?: Record<string, unknown>;
  guidance?: Record<string, unknown>;
}

export interface StructureParseResult {
  name: string;
  sequences: Record<string, unknown>[];
  dialect: string;
  version: number;
  source: string;
  pdb_id: string | null;
  num_chains: number;
  num_residues: number;
  warnings: string[];
}

export interface CacheCheckResult {
  cached: boolean;
  cache_key: string | null;
  cached_at: string | null;
  size_mb: number | null;
}
