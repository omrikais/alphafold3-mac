"use client";

import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { getJob, getResults, getSampleConfidence } from "@/lib/api";

export function useJobDetail(jobId: string) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false; // stop polling
      }
      return 3_000; // poll every 3s while running
    },
  });
}

export function useJobResults(jobId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["job-results", jobId],
    queryFn: () => getResults(jobId),
    enabled,
  });
}

export function useSampleConfidence(
  jobId: string,
  sampleIndex: number,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ["sample-confidence", jobId, sampleIndex],
    queryFn: () => getSampleConfidence(jobId, sampleIndex),
    enabled,
    // Keep showing previous sample's data while new sample loads
    placeholderData: keepPreviousData,
  });
}
