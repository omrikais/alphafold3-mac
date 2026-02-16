"use client";

import { Check, Circle, Loader2 } from "lucide-react";
import { STAGE_LABELS } from "@/lib/constants";

const STAGES = [
  "weight_loading",
  "data_pipeline",
  "feature_preparation",
  "recycling",
  "diffusion",
  "confidence",
  "output_writing",
];

interface Props {
  currentStage: string | null;
  completed: boolean;
}

export function StageIndicator({ currentStage, completed }: Props) {
  const currentIdx = currentStage ? STAGES.indexOf(currentStage) : -1;

  return (
    <div className="af-panel bg-secondary/35 space-y-1.5 p-3.5">
      <p className="text-sm font-semibold">Pipeline stages</p>
      {STAGES.map((stage, i) => {
        const isDone = completed || i < currentIdx;
        const isCurrent = !completed && i === currentIdx;

        return (
          <div key={stage} className="flex items-center gap-2.5 py-0.5">
            {isDone ? (
              <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
            ) : isCurrent ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0 text-primary" />
            ) : (
              <Circle className="h-3.5 w-3.5 shrink-0 text-muted-foreground/40" />
            )}
            <span
              className={`text-xs ${
                isDone
                  ? "text-foreground"
                  : isCurrent
                    ? "text-foreground font-medium"
                    : "text-muted-foreground/60"
              }`}
            >
              {STAGE_LABELS[stage] ?? stage}
            </span>
          </div>
        );
      })}
    </div>
  );
}
