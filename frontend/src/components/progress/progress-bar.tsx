"use client";

import { Progress } from "@/components/ui/progress";

interface Props {
  percent: number;
  stage: string | null;
  diffusionStep: number | null;
  diffusionTotal: number | null;
  recyclingIteration: number | null;
  recyclingTotal: number | null;
}

export function ProgressBar({
  percent,
  stage,
  diffusionStep,
  diffusionTotal,
  recyclingIteration,
  recyclingTotal,
}: Props) {
  let detail = "";
  if (stage === "diffusion" && diffusionStep != null && diffusionTotal != null) {
    detail = `Step ${diffusionStep} / ${diffusionTotal}`;
  } else if (
    stage === "recycling" &&
    recyclingIteration != null &&
    recyclingTotal != null
  ) {
    detail = `Iteration ${recyclingIteration} / ${recyclingTotal}`;
  } else if (stage === "data_pipeline") {
    detail = "Running HMMER search (this may take several minutes)";
  }

  return (
    <div className="af-panel bg-secondary/35 space-y-2 p-3.5">
      <p className="text-sm font-semibold">Live progress</p>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{detail || "Running model stages"}</span>
        <span className="text-xs font-semibold tabular-nums">
          {Math.round(percent)}%
        </span>
      </div>
      <Progress value={percent} className="h-2" />
    </div>
  );
}
