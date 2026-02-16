"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Plus, Trash2, ArrowLeftRight, Target, ShieldOff } from "lucide-react";
import type {
  RestraintConfig,
  GuidanceConfig,
  DistanceRestraint,
  ContactRestraint,
  RepulsiveRestraint,
  CandidateResidue,
} from "@/lib/restraints";
import { RESTRAINT_DEFAULTS } from "@/lib/restraints";
import type { FormEntity } from "@/lib/types";
import { type ChainInfo, getAtomOptions } from "@/lib/residue-atoms";

/** Parse float with NaN-safe fallback. Unlike `parseFloat(v) || d`, this
 *  correctly treats 0 as a valid number rather than falling back. */
const safeFloat = (v: string, d: number) => {
  const n = parseFloat(v);
  return Number.isNaN(n) ? d : n;
};

interface Props {
  restraints: RestraintConfig;
  guidance: GuidanceConfig;
  entities: FormEntity[];
  onRestraintsChange: (r: RestraintConfig) => void;
  onGuidanceChange: (g: GuidanceConfig) => void;
}

/** Encode a 1-based integer as an mmCIF chain ID (reverse spreadsheet style).
 *  1=A, 2=B, ..., 26=Z, 27=AA, 28=BA, 29=CA, ...
 *  Matches the backend's mmcif_lib.int_id_to_str_id(). */
function intIdToStrId(num: number): string {
  let n = num - 1;
  let output = "";
  while (n >= 0) {
    output += String.fromCharCode((n % 26) + 65);
    n = Math.floor(n / 26) - 1;
  }
  return output;
}

/** Extract chain information (id, length, entityType, sequence) from entities. */
function getChainInfo(entities: FormEntity[]): ChainInfo[] {
  const chains: ChainInfo[] = [];
  let chainIdx = 1;
  for (const ent of entities) {
    const count = ent.copies || 1;
    const length =
      ent.type === "ligand" || ent.type === "ion" ? 1 : ent.sequence.length;
    for (let c = 0; c < count; c++) {
      chains.push({
        id: intIdToStrId(chainIdx),
        length,
        entityType: ent.type,
        sequence: ent.sequence,
      });
      chainIdx++;
    }
  }
  return chains;
}

/** Get protein-only chain IDs (restraints are protein-only). */
function getProteinChainIds(chains: ChainInfo[]): string[] {
  return chains.filter((c) => c.entityType === "proteinChain").map((c) => c.id);
}

/** Get the length of a chain by ID. */
function getChainLength(chains: ChainInfo[], chainId: string): number {
  return chains.find((c) => c.id === chainId)?.length ?? 999;
}

// ── Residue selector ──────────────────────────────────────────────

/** Constrained residue number input with min=1, max=chainLength. */
function ResidueInput({
  value,
  chainId,
  chains,
  onChange,
  label,
}: {
  value: number;
  chainId: string;
  chains: ChainInfo[];
  onChange: (v: number) => void;
  label: string;
}) {
  const maxRes = getChainLength(chains, chainId);
  return (
    <div className="space-y-1">
      <Tooltip>
        <TooltipTrigger asChild>
          <Label className="cursor-help text-[0.65rem] uppercase text-muted-foreground">
            {label}
          </Label>
        </TooltipTrigger>
        <TooltipContent>Valid range: 1-{maxRes}</TooltipContent>
      </Tooltip>
      <Input
        type="number"
        min={1}
        max={maxRes}
        className="h-8 w-16 text-xs"
        value={value}
        onChange={(e) => {
          const v = parseInt(e.target.value) || 1;
          onChange(Math.max(1, Math.min(v, maxRes)));
        }}
      />
    </div>
  );
}

// ── Atom selector ─────────────────────────────────────────────────

/** Residue-type-aware atom dropdown for distance restraints. */
function AtomSelect({
  value,
  chainId,
  residueNum,
  chains,
  onChange,
}: {
  value: string;
  chainId: string;
  residueNum: number;
  chains: ChainInfo[];
  onChange: (v: string) => void;
}) {
  const atoms = getAtomOptions(chains, chainId, residueNum);
  // Ensure current value is included even if it's not in the standard list
  const options = atoms.includes(value) ? atoms : [value, ...atoms];

  return (
    <div className="space-y-1">
      <Label className="text-[0.65rem] uppercase text-muted-foreground">Atom</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-8 w-[4.5rem] text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((a) => (
            <SelectItem key={a} value={a}>
              {a}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

// ── Distance restraint row ────────────────────────────────────────────────

function DistanceRow({
  r,
  chainIds,
  chains,
  onChange,
  onRemove,
}: {
  r: DistanceRestraint;
  chainIds: string[];
  chains: ChainInfo[];
  onChange: (r: DistanceRestraint) => void;
  onRemove: () => void;
}) {
  const update = (partial: Partial<DistanceRestraint>) =>
    onChange({ ...r, ...partial });

  return (
    <div className="flex flex-wrap items-end gap-2 rounded-md border border-border/70 bg-background/60 p-2.5">
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Chain i</Label>
        <Select value={r.chain_i} onValueChange={(v) => update({ chain_i: v })}>
          <SelectTrigger className="h-8 w-16"><SelectValue /></SelectTrigger>
          <SelectContent>
            {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <ResidueInput
        value={r.residue_i}
        chainId={r.chain_i}
        chains={chains}
        onChange={(v) => update({ residue_i: v })}
        label="Res i"
      />
      <AtomSelect
        value={r.atom_i ?? "CA"}
        chainId={r.chain_i}
        residueNum={r.residue_i}
        chains={chains}
        onChange={(v) => update({ atom_i: v })}
      />
      <ArrowLeftRight className="mb-1.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Chain j</Label>
        <Select value={r.chain_j} onValueChange={(v) => update({ chain_j: v })}>
          <SelectTrigger className="h-8 w-16"><SelectValue /></SelectTrigger>
          <SelectContent>
            {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <ResidueInput
        value={r.residue_j}
        chainId={r.chain_j}
        chains={chains}
        onChange={(v) => update({ residue_j: v })}
        label="Res j"
      />
      <AtomSelect
        value={r.atom_j ?? "CA"}
        chainId={r.chain_j}
        residueNum={r.residue_j}
        chains={chains}
        onChange={(v) => update({ atom_j: v })}
      />
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Target (A)</Label>
        <Input type="number" step="any" min={0.1} className="h-8 w-20 text-xs" value={r.target_distance}
          onChange={(e) => update({ target_distance: safeFloat(e.target.value, 1.5) })} />
      </div>
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Sigma</Label>
        <Input type="number" step="any" min={0.1} className="h-8 w-16 text-xs" value={r.sigma ?? 1.0}
          onChange={(e) => update({ sigma: safeFloat(e.target.value, 1.0) })} />
      </div>
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Weight</Label>
        <Input type="number" step="any" min={0.1} className="h-8 w-16 text-xs" value={r.weight ?? 1.0}
          onChange={(e) => update({ weight: safeFloat(e.target.value, 1.0) })} />
      </div>
      <Button variant="ghost" size="icon-sm" className="mb-0.5 h-8 w-8 text-muted-foreground hover:text-destructive"
        onClick={onRemove} aria-label="Remove distance restraint">
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ── Contact restraint row ─────────────────────────────────────────────────

function ContactRow({
  r,
  chainIds,
  chains,
  onChange,
  onRemove,
}: {
  r: ContactRestraint;
  chainIds: string[];
  chains: ChainInfo[];
  onChange: (r: ContactRestraint) => void;
  onRemove: () => void;
}) {
  const update = (partial: Partial<ContactRestraint>) =>
    onChange({ ...r, ...partial });

  const addCandidate = () =>
    update({
      candidates: [...r.candidates, { chain_j: chainIds[0] || "A", residue_j: 1 }],
    });

  const updateCandidate = (idx: number, cand: CandidateResidue) => {
    const next = [...r.candidates];
    next[idx] = cand;
    update({ candidates: next });
  };

  const removeCandidate = (idx: number) => {
    if (r.candidates.length <= 1) return;
    update({ candidates: r.candidates.filter((_, i) => i !== idx) });
  };

  return (
    <div className="space-y-2 rounded-md border border-border/70 bg-background/60 p-2.5">
      <div className="flex flex-wrap items-end gap-2">
        <div className="space-y-1">
          <Label className="text-[0.65rem] uppercase text-muted-foreground">Source chain</Label>
          <Select value={r.chain_i} onValueChange={(v) => update({ chain_i: v })}>
            <SelectTrigger className="h-8 w-16"><SelectValue /></SelectTrigger>
            <SelectContent>
              {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <ResidueInput
          value={r.residue_i}
          chainId={r.chain_i}
          chains={chains}
          onChange={(v) => update({ residue_i: v })}
          label="Source res"
        />
        <div className="space-y-1">
          <Label className="text-[0.65rem] uppercase text-muted-foreground">Threshold (A)</Label>
          <Input type="number" step="any" min={0.1} className="h-8 w-20 text-xs" value={r.threshold ?? 8.0}
            onChange={(e) => update({ threshold: safeFloat(e.target.value, 8.0) })} />
        </div>
        <div className="space-y-1">
          <Label className="text-[0.65rem] uppercase text-muted-foreground">Weight</Label>
          <Input type="number" step="any" min={0.1} className="h-8 w-16 text-xs" value={r.weight ?? 1.0}
            onChange={(e) => update({ weight: safeFloat(e.target.value, 1.0) })} />
        </div>
        <Button variant="ghost" size="icon-sm" className="mb-0.5 h-8 w-8 text-muted-foreground hover:text-destructive"
          onClick={onRemove} aria-label="Remove contact restraint">
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <div className="space-y-1.5 pl-4">
        <div className="flex items-center gap-2">
          <span className="text-[0.65rem] font-semibold uppercase text-muted-foreground">Candidates</span>
          <Button variant="outline" size="sm" className="h-6 gap-1 px-2 text-[0.65rem]" onClick={addCandidate}>
            <Plus className="h-2.5 w-2.5" /> Add
          </Button>
        </div>
        {r.candidates.map((cand, ci) => (
          <div key={ci} className="flex items-end gap-2">
            <Select value={cand.chain_j} onValueChange={(v) => updateCandidate(ci, { ...cand, chain_j: v })}>
              <SelectTrigger className="h-7 w-14 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
              </SelectContent>
            </Select>
            <ResidueInput
              value={cand.residue_j}
              chainId={cand.chain_j}
              chains={chains}
              onChange={(v) => updateCandidate(ci, { ...cand, residue_j: v })}
              label=""
            />
            {r.candidates.length > 1 && (
              <Button variant="ghost" size="icon-sm" className="h-7 w-7 text-muted-foreground hover:text-destructive"
                onClick={() => removeCandidate(ci)}>
                <Trash2 className="h-3 w-3" />
              </Button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Repulsive restraint row ───────────────────────────────────────────────

function RepulsiveRow({
  r,
  chainIds,
  chains,
  onChange,
  onRemove,
}: {
  r: RepulsiveRestraint;
  chainIds: string[];
  chains: ChainInfo[];
  onChange: (r: RepulsiveRestraint) => void;
  onRemove: () => void;
}) {
  const update = (partial: Partial<RepulsiveRestraint>) =>
    onChange({ ...r, ...partial });

  return (
    <div className="flex flex-wrap items-end gap-2 rounded-md border border-border/70 bg-background/60 p-2.5">
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Chain i</Label>
        <Select value={r.chain_i} onValueChange={(v) => update({ chain_i: v })}>
          <SelectTrigger className="h-8 w-16"><SelectValue /></SelectTrigger>
          <SelectContent>
            {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <ResidueInput
        value={r.residue_i}
        chainId={r.chain_i}
        chains={chains}
        onChange={(v) => update({ residue_i: v })}
        label="Res i"
      />
      <ArrowLeftRight className="mb-1.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Chain j</Label>
        <Select value={r.chain_j} onValueChange={(v) => update({ chain_j: v })}>
          <SelectTrigger className="h-8 w-16"><SelectValue /></SelectTrigger>
          <SelectContent>
            {chainIds.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>
      <ResidueInput
        value={r.residue_j}
        chainId={r.chain_j}
        chains={chains}
        onChange={(v) => update({ residue_j: v })}
        label="Res j"
      />
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Min dist (A)</Label>
        <Input type="number" step="any" min={0.1} className="h-8 w-20 text-xs" value={r.min_distance}
          onChange={(e) => update({ min_distance: safeFloat(e.target.value, 15.0) })} />
      </div>
      <div className="space-y-1">
        <Label className="text-[0.65rem] uppercase text-muted-foreground">Weight</Label>
        <Input type="number" step="any" min={0.1} className="h-8 w-16 text-xs" value={r.weight ?? 1.0}
          onChange={(e) => update({ weight: safeFloat(e.target.value, 1.0) })} />
      </div>
      <Button variant="ghost" size="icon-sm" className="mb-0.5 h-8 w-8 text-muted-foreground hover:text-destructive"
        onClick={onRemove} aria-label="Remove repulsive restraint">
        <Trash2 className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ── Main restraint editor ─────────────────────────────────────────────────

export function RestraintEditor({
  restraints,
  guidance,
  entities,
  onRestraintsChange,
  onGuidanceChange,
}: Props) {
  const chains = getChainInfo(entities);
  const chainIds = getProteinChainIds(chains);
  const defaultChain = chainIds[0] || "A";
  const hasEnoughChains = chainIds.length >= 2;

  const totalCount =
    (restraints.distance?.length ?? 0) +
    (restraints.contact?.length ?? 0) +
    (restraints.repulsive?.length ?? 0);

  // ── Distance ──

  const addDistance = () => {
    const next: DistanceRestraint = {
      chain_i: defaultChain,
      residue_i: 1,
      chain_j: chainIds[1] || defaultChain,
      residue_j: 1,
      target_distance: 1.5,
      atom_i: "CA",
      atom_j: "CA",
      sigma: RESTRAINT_DEFAULTS.distance.sigma,
      weight: RESTRAINT_DEFAULTS.distance.weight,
    };
    onRestraintsChange({
      ...restraints,
      distance: [...(restraints.distance ?? []), next],
    });
  };

  const updateDistance = (idx: number, r: DistanceRestraint) => {
    const arr = [...(restraints.distance ?? [])];
    arr[idx] = r;
    onRestraintsChange({ ...restraints, distance: arr });
  };

  const removeDistance = (idx: number) => {
    onRestraintsChange({
      ...restraints,
      distance: (restraints.distance ?? []).filter((_, i) => i !== idx),
    });
  };

  // ── Contact ──

  const addContact = () => {
    const next: ContactRestraint = {
      chain_i: defaultChain,
      residue_i: 1,
      candidates: [{ chain_j: chainIds[1] || defaultChain, residue_j: 1 }],
      threshold: RESTRAINT_DEFAULTS.contact.threshold,
      weight: RESTRAINT_DEFAULTS.contact.weight,
    };
    onRestraintsChange({
      ...restraints,
      contact: [...(restraints.contact ?? []), next],
    });
  };

  const updateContact = (idx: number, r: ContactRestraint) => {
    const arr = [...(restraints.contact ?? [])];
    arr[idx] = r;
    onRestraintsChange({ ...restraints, contact: arr });
  };

  const removeContact = (idx: number) => {
    onRestraintsChange({
      ...restraints,
      contact: (restraints.contact ?? []).filter((_, i) => i !== idx),
    });
  };

  // ── Repulsive ──

  const addRepulsive = () => {
    const next: RepulsiveRestraint = {
      chain_i: defaultChain,
      residue_i: 1,
      chain_j: chainIds[1] || defaultChain,
      residue_j: 1,
      min_distance: 15.0,
      weight: RESTRAINT_DEFAULTS.repulsive.weight,
    };
    onRestraintsChange({
      ...restraints,
      repulsive: [...(restraints.repulsive ?? []), next],
    });
  };

  const updateRepulsive = (idx: number, r: RepulsiveRestraint) => {
    const arr = [...(restraints.repulsive ?? [])];
    arr[idx] = r;
    onRestraintsChange({ ...restraints, repulsive: arr });
  };

  const removeRepulsive = (idx: number) => {
    onRestraintsChange({
      ...restraints,
      repulsive: (restraints.repulsive ?? []).filter((_, i) => i !== idx),
    });
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">
              Restraint-guided docking
              {totalCount > 0 && (
                <Badge variant="secondary" className="ml-2 text-[0.65rem]">
                  {totalCount} restraint{totalCount > 1 ? "s" : ""}
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Spatial constraints to guide structure prediction.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {!hasEnoughChains && (
          <p className="text-xs text-amber-600 dark:text-amber-400">
            Restraints require at least two protein chains. Add more protein entities above.
          </p>
        )}
        {/* ── Distance restraints ── */}
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <ArrowLeftRight className="h-3.5 w-3.5 text-blue-500" />
              <span className="text-sm font-semibold">Distance</span>
            </div>
            <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={addDistance} disabled={!hasEnoughChains}>
              <Plus className="h-3 w-3" /> Add
            </Button>
          </div>
          {(restraints.distance ?? []).map((r, i) => (
            <DistanceRow key={i} r={r} chainIds={chainIds} chains={chains}
              onChange={(r) => updateDistance(i, r)} onRemove={() => removeDistance(i)} />
          ))}
          {(restraints.distance ?? []).length === 0 && (
            <p className="text-xs text-muted-foreground">No distance restraints defined.</p>
          )}
        </section>

        {/* ── Contact restraints ── */}
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Target className="h-3.5 w-3.5 text-green-500" />
              <span className="text-sm font-semibold">Contact</span>
            </div>
            <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={addContact} disabled={!hasEnoughChains}>
              <Plus className="h-3 w-3" /> Add
            </Button>
          </div>
          {(restraints.contact ?? []).map((r, i) => (
            <ContactRow key={i} r={r} chainIds={chainIds} chains={chains}
              onChange={(r) => updateContact(i, r)} onRemove={() => removeContact(i)} />
          ))}
          {(restraints.contact ?? []).length === 0 && (
            <p className="text-xs text-muted-foreground">No contact restraints defined.</p>
          )}
        </section>

        {/* ── Repulsive restraints ── */}
        <section className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <ShieldOff className="h-3.5 w-3.5 text-red-500" />
              <span className="text-sm font-semibold">Repulsive</span>
            </div>
            <Button variant="outline" size="sm" className="h-7 gap-1 text-xs" onClick={addRepulsive} disabled={!hasEnoughChains}>
              <Plus className="h-3 w-3" /> Add
            </Button>
          </div>
          {(restraints.repulsive ?? []).map((r, i) => (
            <RepulsiveRow key={i} r={r} chainIds={chainIds} chains={chains}
              onChange={(r) => updateRepulsive(i, r)} onRemove={() => removeRepulsive(i)} />
          ))}
          {(restraints.repulsive ?? []).length === 0 && (
            <p className="text-xs text-muted-foreground">No repulsive restraints defined.</p>
          )}
        </section>

        {/* ── Guidance parameters ── */}
        {totalCount > 0 && (
          <section className="af-panel bg-secondary/35 px-3.5 py-3">
            <p className="mb-2.5 text-sm font-semibold">Guidance parameters</p>
            <div className="flex flex-wrap items-end gap-3">
              <div className="space-y-1">
                <Label className="text-[0.65rem] uppercase text-muted-foreground">Scale</Label>
                <Input type="number" step="any" min={0} className="h-8 w-20 text-xs"
                  value={guidance.scale ?? 1.0}
                  onChange={(e) => onGuidanceChange({ ...guidance, scale: safeFloat(e.target.value, 1.0) })} />
              </div>
              <div className="space-y-1">
                <Label className="text-[0.65rem] uppercase text-muted-foreground">Annealing</Label>
                <Select value={guidance.annealing ?? "linear"}
                  onValueChange={(v) => onGuidanceChange({ ...guidance, annealing: v as GuidanceConfig["annealing"] })}>
                  <SelectTrigger className="h-8 w-24"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="linear">Linear</SelectItem>
                    <SelectItem value="cosine">Cosine</SelectItem>
                    <SelectItem value="constant">Constant</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="cursor-help text-[0.65rem] uppercase text-muted-foreground">Start step</Label>
                  </TooltipTrigger>
                  <TooltipContent>First diffusion step to apply guidance (0 = from start)</TooltipContent>
                </Tooltip>
                <Input type="number" min={0} className="h-8 w-20 text-xs"
                  value={guidance.start_step ?? 0}
                  onChange={(e) => onGuidanceChange({ ...guidance, start_step: parseInt(e.target.value) || 0 })} />
              </div>
              <div className="space-y-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Label className="cursor-help text-[0.65rem] uppercase text-muted-foreground">End step</Label>
                  </TooltipTrigger>
                  <TooltipContent>Last diffusion step for guidance (empty = all steps)</TooltipContent>
                </Tooltip>
                <Input type="number" min={1} className="h-8 w-20 text-xs"
                  value={guidance.end_step ?? ""}
                  placeholder="all"
                  onChange={(e) => {
                    const val = e.target.value ? parseInt(e.target.value) : null;
                    onGuidanceChange({ ...guidance, end_step: val });
                  }} />
              </div>
            </div>
          </section>
        )}
      </CardContent>
    </Card>
  );
}
