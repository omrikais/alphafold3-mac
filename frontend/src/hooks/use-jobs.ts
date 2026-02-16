"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listJobs, submitJob, cancelJob, deleteJob } from "@/lib/api";
import type { JobSubmission } from "@/lib/types";

export function useJobs(params?: {
  status?: string;
  search?: string;
  page?: number;
}) {
  return useQuery({
    queryKey: ["jobs", params],
    queryFn: () => listJobs(params),
    refetchInterval: 5_000, // poll every 5s for status updates
    retry: 1,
  });
}

export function useSubmitJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (submission: JobSubmission) => submitJob(submission),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useCancelJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}
