/** TypeScript type definitions for restraint-guided docking.
 *
 * Matches contracts/api-extensions.md and contracts/restraint-schema.json.
 */

// ── Restraint input types ──────────────────────────────────────────────────

export interface DistanceRestraint {
  chain_i: string;
  residue_i: number;
  atom_i?: string; // default "CA"
  chain_j: string;
  residue_j: number;
  atom_j?: string; // default "CA"
  target_distance: number;
  sigma?: number; // default 1.0
  weight?: number; // default 1.0
}

export interface CandidateResidue {
  chain_j: string;
  residue_j: number;
}

export interface ContactRestraint {
  chain_i: string;
  residue_i: number;
  candidates: CandidateResidue[];
  threshold?: number; // default 8.0
  weight?: number; // default 1.0
}

export interface RepulsiveRestraint {
  chain_i: string;
  residue_i: number;
  chain_j: string;
  residue_j: number;
  min_distance: number;
  weight?: number; // default 1.0
}

export interface GuidanceConfig {
  scale?: number; // default 1.0
  annealing?: "linear" | "cosine" | "constant"; // default "linear"
  start_step?: number; // default 0
  end_step?: number | null; // default null (= num_steps)
}

export interface RestraintConfig {
  distance?: DistanceRestraint[];
  contact?: ContactRestraint[];
  repulsive?: RepulsiveRestraint[];
}

// ── Satisfaction output types ──────────────────────────────────────────────

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

// ── Defaults for form UI ───────────────────────────────────────────────────

export const RESTRAINT_DEFAULTS = {
  distance: {
    atom: "CA",
    sigma: 1.0,
    weight: 1.0,
  },
  contact: {
    threshold: 8.0,
    weight: 1.0,
  },
  repulsive: {
    weight: 1.0,
  },
  guidance: {
    scale: 1.0,
    annealing: "linear" as const,
    start_step: 0,
  },
} as const;
