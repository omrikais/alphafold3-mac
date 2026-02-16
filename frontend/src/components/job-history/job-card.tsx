"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { RotateCcw, Trash2, X } from "lucide-react";
import type { JobSummary, JobStatus } from "@/lib/types";
import { useCancelJob, useDeleteJob } from "@/hooks/use-jobs";

const STATUS_VARIANT: Record<
  JobStatus,
  "default" | "secondary" | "success" | "destructive" | "outline"
> = {
  pending: "outline",
  running: "default",
  completed: "success",
  failed: "destructive",
  cancelled: "outline",
};

const STATUS_LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

interface Props {
  job: JobSummary;
}

export function JobCard({ job }: Props) {
  const router = useRouter();
  const cancelMutation = useCancelJob();
  const deleteMutation = useDeleteJob();

  const isActive = job.status === "pending" || job.status === "running";
  const isDone =
    job.status === "completed" ||
    job.status === "failed" ||
    job.status === "cancelled";

  return (
    <Card className="border-border/75 bg-background/65 p-3 transition-colors hover:bg-secondary/60">
      <div className="flex items-center justify-between gap-3">
        <Link
          href={`/job?id=${job.id}`}
          className="min-w-0 flex-1 space-y-1.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold">{job.name}</span>
            <Badge
              variant={STATUS_VARIANT[job.status]}
              className="h-5 shrink-0 rounded-full px-2 text-[10px] font-medium"
            >
              {STATUS_LABEL[job.status]}
            </Badge>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {job.num_residues != null && (
              <span>{job.num_residues} residues</span>
            )}
            {job.num_chains != null && <span>{job.num_chains} chains</span>}
            <span>{timeAgo(job.created_at)}</span>
          </div>
          {job.status === "running" && (
            <Progress value={job.progress} className="mt-1 h-1.5" />
          )}
          {job.error_message && (
            <p className="text-xs text-destructive line-clamp-2 break-words">
              {job.error_message}
            </p>
          )}
        </Link>

        <div className="flex shrink-0 gap-1">
          {isActive && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="h-8 w-8 rounded-md"
                  onClick={() => cancelMutation.mutate(job.id)}
                  disabled={cancelMutation.isPending}
                  aria-label="Cancel job"
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Cancel job</TooltipContent>
            </Tooltip>
          )}
          {isDone && (
            <>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 rounded-md px-2 text-xs font-medium"
                    onClick={() => router.push(`/?loadJob=${job.id}`)}
                    aria-label="Reuse inputs"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    <span className="hidden sm:inline">Reuse inputs</span>
                    <span className="sm:hidden">Reuse</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Load this job&apos;s inputs</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-8 w-8 rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                    onClick={() => {
                      const confirmed = window.confirm(
                        `Delete "${job.name}"? This cannot be undone.`,
                      );
                      if (!confirmed) return;
                      deleteMutation.mutate(job.id);
                    }}
                    disabled={deleteMutation.isPending}
                    aria-label="Delete job"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Delete job</TooltipContent>
              </Tooltip>
            </>
          )}
        </div>
      </div>
    </Card>
  );
}
