"use client";

import { useEffect, useReducer } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import type { FormEntity } from "@/lib/types";
import { ENTITY_TYPE_LABELS } from "@/lib/constants";
import { checkCache } from "@/lib/api";
import { Database, Loader2 } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  jobName: string;
  seed: number;
  runPipeline: boolean;
  entities: FormEntity[];
  onSubmit: (useCache: boolean) => void;
  isSubmitting: boolean;
}

interface CacheState {
  loading: boolean;
  cached: boolean;
  cachedAt: string | null;
  sizeMb: number | null;
  useCache: boolean;
}

type CacheAction =
  | { type: "reset" }
  | { type: "loading" }
  | { type: "success"; cached: boolean; cachedAt: string | null; sizeMb: number | null }
  | { type: "error" }
  | { type: "toggle"; value: boolean };

const INITIAL_CACHE_STATE: CacheState = {
  loading: false,
  cached: false,
  cachedAt: null,
  sizeMb: null,
  useCache: true,
};

function cacheReducer(state: CacheState, action: CacheAction): CacheState {
  switch (action.type) {
    case "reset":
      return INITIAL_CACHE_STATE;
    case "loading":
      return {
        ...state,
        loading: true,
        cached: false,
        cachedAt: null,
        sizeMb: null,
      };
    case "success":
      return {
        ...state,
        loading: false,
        cached: action.cached,
        cachedAt: action.cachedAt,
        sizeMb: action.sizeMb,
        useCache: true, // Always enable: read on hit, write on miss
      };
    case "error":
      return {
        ...state,
        loading: false,
        cached: false,
        cachedAt: null,
        sizeMb: null,
        useCache: true,
      };
    case "toggle":
      return {
        ...state,
        useCache: action.value,
      };
    default:
      return state;
  }
}

function entitiesToSequences(entities: FormEntity[]): Record<string, unknown>[] {
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

function timeAgo(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export function PreviewModal({
  open,
  onOpenChange,
  jobName,
  seed,
  runPipeline,
  entities,
  onSubmit,
  isSubmitting,
}: Props) {
  const [cacheState, dispatchCache] = useReducer(
    cacheReducer,
    INITIAL_CACHE_STATE,
  );

  // Check cache when modal opens and MSA pipeline is on
  useEffect(() => {
    if (!open || !runPipeline) {
      dispatchCache({ type: "reset" });
      return;
    }

    let cancelled = false;
    dispatchCache({ type: "loading" });

    const sequences = entitiesToSequences(entities);
    checkCache(sequences)
      .then((result) => {
        if (!cancelled) {
          dispatchCache({
            type: "success",
            cached: result.cached,
            cachedAt: result.cached_at,
            sizeMb: result.size_mb,
          });
        }
      })
      .catch(() => {
        if (!cancelled) {
          dispatchCache({ type: "error" });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [open, runPipeline, entities]);

  const totalResidues = entities.reduce((sum, e) => {
    if (
      e.type === "proteinChain" ||
      e.type === "rnaSequence" ||
      e.type === "dnaSequence"
    ) {
      return sum + e.sequence.length * e.copies;
    }
    return sum + e.copies;
  }, 0);

  const totalChains = entities.reduce((sum, e) => sum + e.copies, 0);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="tracking-tight">Submit prediction job</DialogTitle>
          <DialogDescription>
            Review key parameters, cache status, and entity composition before running.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-1">
          <section className="af-panel bg-secondary/35 p-3.5">
            <div className="grid gap-2.5 sm:grid-cols-2">
              <div className="flex min-w-0 items-center justify-between gap-3 text-sm sm:col-span-2">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="shrink-0 text-muted-foreground">Job name</span>
                  <span className="truncate font-medium">{jobName}</span>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="text-muted-foreground">Seed</span>
                  <span className="font-mono text-xs">{seed === -1 ? "Random" : seed}</span>
                </div>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">MSA search</span>
                <span>{runPipeline ? "Enabled" : "Disabled"}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Total residues</span>
                <span>{totalResidues.toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Total chains</span>
                <span>{totalChains}</span>
              </div>
            </div>
          </section>

          {runPipeline && (
            <section className="af-panel p-3.5">
              <div className="flex items-center gap-2">
                <Database className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-sm font-semibold">MSA cache</span>
                {cacheState.loading && (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                )}
                {!cacheState.loading && cacheState.cached && (
                  <Badge variant="secondary" className="text-[10px]">
                    Available
                  </Badge>
                )}
                {!cacheState.loading && !cacheState.cached && (
                  <Badge variant="outline" className="text-[10px]">
                    Not found
                  </Badge>
                )}
              </div>
              {!cacheState.loading && cacheState.cached && (
                <>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Cached {cacheState.cachedAt ? timeAgo(cacheState.cachedAt) : ""}
                    {cacheState.sizeMb != null && ` (${cacheState.sizeMb} MB)`}. Reusing
                    cache skips HMMER search.
                  </p>
                  <div className="mt-2.5 flex items-center gap-2">
                    <Switch
                      id="use-cache"
                      checked={cacheState.useCache}
                      onCheckedChange={(value) => dispatchCache({ type: "toggle", value })}
                    />
                    <Label htmlFor="use-cache" className="cursor-pointer text-xs">
                      {cacheState.useCache
                        ? "Use cached MSA data"
                        : "Run fresh HMMER search"}
                    </Label>
                  </div>
                </>
              )}
              {!cacheState.loading && !cacheState.cached && (
                <p className="mt-2 text-xs text-muted-foreground">
                  No cached MSA data found for this sequence set. HMMER search will
                  run and cache future results.
                </p>
              )}
            </section>
          )}

          <section className="space-y-2">
            <p className="text-sm font-semibold">Entities</p>
            {entities.map((entity) => (
              <div
                key={entity.id}
                className="af-panel flex items-center justify-between gap-2 px-3 py-2 text-xs"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <Badge variant="secondary" className="h-5 rounded-full px-2 text-[10px]">
                    {ENTITY_TYPE_LABELS[entity.type]}
                  </Badge>
                  {entity.modifications?.length ? (
                    <Badge variant="outline" className="h-5 rounded-full px-2 text-[10px]">
                      {entity.modifications.length} PTM
                      {entity.modifications.length > 1 ? "s" : ""}
                    </Badge>
                  ) : null}
                  {entity.type === "proteinChain" ||
                  entity.type === "rnaSequence" ||
                  entity.type === "dnaSequence" ? (
                    <span className="truncate font-mono text-muted-foreground">
                      {entity.sequence.slice(0, 34)}
                      {entity.sequence.length > 34 ? "..." : ""}
                    </span>
                  ) : (
                    <span className="truncate font-mono text-muted-foreground">
                      {entity.ccdCode || entity.smiles || "—"}
                    </span>
                  )}
                </div>
                <span className="text-muted-foreground">
                  {entity.copies > 1 ? `x${entity.copies}` : ""}
                </span>
              </div>
            ))}
          </section>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => onSubmit(cacheState.useCache)}
            disabled={isSubmitting}
            className="gap-1.5"
          >
            {isSubmitting && <Loader2 className="h-4 w-4 animate-spin" />}
            Submit job
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
