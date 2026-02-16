"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft, RotateCcw } from "lucide-react";
import { useJobDetail } from "@/hooks/use-job-detail";
import { JobProgress } from "./job-progress";
import { JobResultsView } from "@/components/results/job-results-view";

interface Props {
  jobId: string;
}

export function JobDetailView({ jobId }: Props) {
  const { data: job, isLoading, error } = useJobDetail(jobId);

  if (isLoading) {
    return (
      <div className="af-panel flex items-center justify-center py-20">
        <p className="text-sm text-muted-foreground">Loading job...</p>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="space-y-4">
        <Link href="/">
          <Button variant="ghost" size="sm" className="h-8 gap-1.5 rounded-md px-2">
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </Button>
        </Link>
        <div className="af-panel flex items-center justify-center py-20">
          <p className="text-sm text-destructive">
            {error ? String(error) : "Job not found"}
          </p>
        </div>
      </div>
    );
  }

  const isCompleted = job.status === "completed";

  if (isCompleted) {
    return <JobResultsView job={job} />;
  }

  const isTerminal = job.status === "failed" || job.status === "cancelled";

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-center gap-2">
        <Link href="/">
          <Button variant="ghost" size="sm" className="h-8 gap-1.5 rounded-md px-2">
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to jobs
          </Button>
        </Link>
        {isTerminal && (
          <Link href={`/?loadJob=${jobId}`}>
            <Button variant="outline" size="sm" className="h-8 gap-1.5 rounded-md px-3">
              <RotateCcw className="h-3.5 w-3.5" />
              Reuse inputs
            </Button>
          </Link>
        )}
      </div>
      <JobProgress job={job} />
    </div>
  );
}
