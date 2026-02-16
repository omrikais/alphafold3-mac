"use client";

import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ArrowLeftRight, Target, ShieldOff, CheckCircle2, XCircle } from "lucide-react";
import type { RestraintSatisfaction } from "@/lib/restraints";

interface Props {
  satisfaction: RestraintSatisfaction | null;
}

function SatisfiedIcon({ satisfied }: { satisfied: boolean }) {
  return satisfied ? (
    <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
  ) : (
    <XCircle className="h-3.5 w-3.5 text-red-500" />
  );
}

/**
 * Display restraint satisfaction results after job completion.
 *
 * Shows per-restraint satisfaction status (satisfied/unsatisfied) with
 * actual vs target distances. For Mol* viewer integration, the 3D
 * distance-line rendering would require direct access to the Mol* plugin
 * instance; this component provides the data overlay panel.
 */
export function RestraintViz({ satisfaction }: Props) {
  if (!satisfaction) return null;

  const distances = satisfaction.distance ?? [];
  const contacts = satisfaction.contact ?? [];
  const repulsives = satisfaction.repulsive ?? [];

  const total = distances.length + contacts.length + repulsives.length;
  if (total === 0) return null;

  const satisfied =
    distances.filter((d) => d.satisfied).length +
    contacts.filter((c) => c.satisfied).length +
    repulsives.filter((r) => r.satisfied).length;

  return (
    <section className="af-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Restraint satisfaction</h3>
        <Badge
          variant={satisfied === total ? "default" : "secondary"}
          className="text-[0.65rem]"
        >
          {satisfied}/{total} satisfied
        </Badge>
      </div>

      <div className="space-y-3">
        {/* Distance restraints */}
        {distances.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <ArrowLeftRight className="h-3 w-3 text-blue-500" />
              <span className="text-xs font-semibold text-muted-foreground">Distance</span>
            </div>
            {distances.map((d, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-md border border-border/50 bg-background/50 px-2.5 py-1.5 text-xs"
              >
                <SatisfiedIcon satisfied={d.satisfied} />
                <span className="font-mono">
                  {d.chain_i}:{d.residue_i}:{d.atom_i}
                </span>
                <ArrowLeftRight className="h-2.5 w-2.5 text-muted-foreground" />
                <span className="font-mono">
                  {d.chain_j}:{d.residue_j}:{d.atom_j}
                </span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="ml-auto tabular-nums">
                      <span className={d.satisfied ? "text-green-600" : "text-red-600"}>
                        {d.actual_distance.toFixed(1)}
                      </span>
                      <span className="text-muted-foreground">
                        {" / "}
                        {d.target_distance.toFixed(1)} A
                      </span>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Actual: {d.actual_distance.toFixed(2)} A, Target: {d.target_distance.toFixed(2)} A
                  </TooltipContent>
                </Tooltip>
              </div>
            ))}
          </div>
        )}

        {/* Contact restraints */}
        {contacts.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <Target className="h-3 w-3 text-green-500" />
              <span className="text-xs font-semibold text-muted-foreground">Contact</span>
            </div>
            {contacts.map((c, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-md border border-border/50 bg-background/50 px-2.5 py-1.5 text-xs"
              >
                <SatisfiedIcon satisfied={c.satisfied} />
                <span className="font-mono">
                  {c.chain_i}:{c.residue_i}
                </span>
                <span className="text-muted-foreground">nearest</span>
                <span className="font-mono">
                  {c.closest_candidate_chain}:{c.closest_candidate_residue}
                </span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="ml-auto tabular-nums">
                      <span className={c.satisfied ? "text-green-600" : "text-red-600"}>
                        {c.actual_distance.toFixed(1)}
                      </span>
                      <span className="text-muted-foreground">
                        {" / "}
                        {c.threshold.toFixed(1)} A
                      </span>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Actual: {c.actual_distance.toFixed(2)} A, Threshold: {c.threshold.toFixed(2)} A
                  </TooltipContent>
                </Tooltip>
              </div>
            ))}
          </div>
        )}

        {/* Repulsive restraints */}
        {repulsives.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <ShieldOff className="h-3 w-3 text-red-500" />
              <span className="text-xs font-semibold text-muted-foreground">Repulsive</span>
            </div>
            {repulsives.map((r, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-md border border-border/50 bg-background/50 px-2.5 py-1.5 text-xs"
              >
                <SatisfiedIcon satisfied={r.satisfied} />
                <span className="font-mono">
                  {r.chain_i}:{r.residue_i}
                </span>
                <ArrowLeftRight className="h-2.5 w-2.5 text-muted-foreground" />
                <span className="font-mono">
                  {r.chain_j}:{r.residue_j}
                </span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="ml-auto tabular-nums">
                      <span className={r.satisfied ? "text-green-600" : "text-red-600"}>
                        {r.actual_distance.toFixed(1)}
                      </span>
                      <span className="text-muted-foreground">
                        {" / min "}
                        {r.min_distance.toFixed(1)} A
                      </span>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>
                    Actual: {r.actual_distance.toFixed(2)} A, Min: {r.min_distance.toFixed(2)} A
                  </TooltipContent>
                </Tooltip>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
