"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertCircle, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { useJobProgress } from "@/hooks/use-job-progress";
import { useCancelJob } from "@/hooks/use-jobs";
import { StageIndicator } from "./stage-indicator";
import { ProgressBar } from "./progress-bar";
import type { JobDetail } from "@/lib/types";

interface Props {
  job: JobDetail;
}

export function JobProgress({ job }: Props) {
  const isActive = job.status === "pending" || job.status === "running";
  const progress = useJobProgress(isActive ? job.id : null);
  const cancelMutation = useCancelJob();

  const currentStage =
    progress.stage ?? job.current_stage ?? null;
  const percentComplete =
    progress.percentComplete > 0 ? progress.percentComplete : job.progress;
  const isCompleted =
    job.status === "completed" || progress.terminalType === "completed";
  const isFailed =
    job.status === "failed" || progress.terminalType === "failed";
  const isCancelled =
    job.status === "cancelled" || progress.terminalType === "cancelled";
  const errorMessage = progress.error ?? job.error_message;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <CardTitle className="truncate">{job.name}</CardTitle>
            <p className="text-sm text-muted-foreground">
              {job.num_residues != null && `${job.num_residues} residues`}
              {job.num_chains != null && ` / ${job.num_chains} chains`}
              {` / ${job.num_samples} samples`}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {isCompleted && (
              <Badge variant="success" className="h-6 gap-1 rounded-full px-2 text-[11px]">
                <CheckCircle2 className="h-3.5 w-3.5" />
                Completed
              </Badge>
            )}
            {isFailed && (
              <Badge variant="destructive" className="h-6 gap-1 rounded-full px-2 text-[11px]">
                <AlertCircle className="h-3.5 w-3.5" />
                Failed
              </Badge>
            )}
            {isCancelled && (
              <Badge variant="outline" className="h-6 gap-1 rounded-full px-2 text-[11px]">
                <XCircle className="h-3.5 w-3.5" />
                Cancelled
              </Badge>
            )}
            {isActive && !progress.terminal && (
              <>
                <Badge variant="default" className="h-6 gap-1 rounded-full px-2 text-[11px]">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {job.status === "pending" ? "Queued" : "Running"}
                </Badge>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 rounded-md px-3 text-xs font-semibold uppercase tracking-wide"
                  onClick={() => cancelMutation.mutate(job.id)}
                  disabled={cancelMutation.isPending}
                >
                  Cancel
                </Button>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {(isActive || percentComplete > 0) && (
          <ProgressBar
            percent={percentComplete}
            stage={currentStage}
            diffusionStep={progress.diffusionStep}
            diffusionTotal={progress.diffusionTotal}
            recyclingIteration={progress.recyclingIteration}
            recyclingTotal={progress.recyclingTotal}
          />
        )}

        <StageIndicator
          currentStage={currentStage}
          completed={isCompleted}
        />

        {errorMessage && (
          <div className="rounded-md border border-destructive/20 bg-destructive/10 px-3 py-2.5 overflow-hidden">
            <p className="text-xs text-destructive break-words whitespace-pre-wrap">
              {errorMessage.length > 500 ? errorMessage.slice(0, 500) + "..." : errorMessage}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
