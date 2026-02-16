"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Loader2, AlertCircle } from "lucide-react";
import { useJobResults } from "@/hooks/use-job-detail";
import { getStructureUrl } from "@/lib/api";

interface Props {
  jobId: string;
}

function confidenceColor(value: number | null | undefined): string {
  if (value == null) return "text-muted-foreground";
  if (value >= 0.8) return "text-green-600 dark:text-green-400";
  if (value >= 0.5) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
}

function confidenceBg(value: number | null | undefined): string {
  if (value == null) return "bg-muted";
  if (value >= 0.8) return "bg-green-100 dark:bg-green-900/30";
  if (value >= 0.5) return "bg-yellow-100 dark:bg-yellow-900/30";
  return "bg-red-100 dark:bg-red-900/30";
}

function formatScore(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toFixed(2);
}

function formatPlddt(value: number | null | undefined): string {
  if (value == null) return "—";
  // pLDDT is 0-100 scale
  return value.toFixed(1);
}

export function JobResults({ jobId }: Props) {
  const { data: results, isLoading, isError } = useJobResults(jobId, true);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 py-4">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading results...</span>
      </div>
    );
  }

  if (isError || !results) {
    return (
      <div className="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-2">
        <AlertCircle className="h-4 w-4 text-destructive" />
        <p className="text-xs text-destructive">Failed to load results.</p>
      </div>
    );
  }

  const samples = results.samples ?? [];
  const hasSamples = samples.length > 0;

  return (
    <div className="space-y-4">
      {/* Confidence summary */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className={`rounded-lg px-3 py-2 text-center ${confidenceBg(results.ptm)}`}>
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">pTM</p>
          <p className={`text-lg font-semibold tabular-nums ${confidenceColor(results.ptm)}`}>
            {formatScore(results.ptm)}
          </p>
        </div>
        <div className={`rounded-lg px-3 py-2 text-center ${confidenceBg(results.iptm)}`}>
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">ipTM</p>
          <p className={`text-lg font-semibold tabular-nums ${confidenceColor(results.iptm)}`}>
            {formatScore(results.iptm)}
          </p>
        </div>
        <div className={`rounded-lg px-3 py-2 text-center ${confidenceBg(results.mean_plddt != null ? results.mean_plddt / 100 : null)}`}>
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">pLDDT</p>
          <p className={`text-lg font-semibold tabular-nums ${confidenceColor(results.mean_plddt != null ? results.mean_plddt / 100 : null)}`}>
            {formatPlddt(results.mean_plddt)}
          </p>
        </div>
      </div>

      {/* Per-sample table */}
      {hasSamples && samples.length > 1 && (
        <div className="af-panel overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b bg-secondary/60">
                <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Rank</th>
                <th className="px-3 py-1.5 text-right font-medium text-muted-foreground">pTM</th>
                <th className="px-3 py-1.5 text-right font-medium text-muted-foreground">ipTM</th>
                <th className="px-3 py-1.5 text-right font-medium text-muted-foreground">pLDDT</th>
                <th className="px-3 py-1.5 text-right font-medium text-muted-foreground"></th>
              </tr>
            </thead>
            <tbody>
              {samples.map((sample, i) => (
                <tr key={i} className="border-b last:border-0">
                  <td className="px-3 py-1.5">
                    <Badge variant="outline" className="text-[10px]">#{i + 1}</Badge>
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {formatScore(sample.ptm as number | null)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {formatScore(sample.iptm as number | null)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {formatPlddt(sample.mean_plddt as number | null)}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    <a href={getStructureUrl(jobId, i + 1)} download>
                      <Button variant="ghost" size="sm" className="h-6 text-[10px] gap-1">
                        <Download className="h-3 w-3" />
                        .cif
                      </Button>
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Download buttons */}
      <div className="flex flex-wrap gap-2">
        {(hasSamples ? samples : [null]).map((_, i) => (
          <a key={i} href={getStructureUrl(jobId, i + 1)} download>
            <Button variant="outline" size="sm" className="h-8 gap-1.5 rounded-md text-xs">
              <Download className="h-3.5 w-3.5" />
              {hasSamples && samples.length > 1
                ? `Download rank ${i + 1}`
                : "Download structure"}
            </Button>
          </a>
        ))}
      </div>
    </div>
  );
}
