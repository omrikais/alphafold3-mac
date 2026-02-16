"use client";

import { useState } from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { useJobResults, useSampleConfidence } from "@/hooks/use-job-detail";
import { getStructureUrl } from "@/lib/api";
import { ResultsHeader } from "./results-header";
import { ConfidenceMetrics } from "./confidence-metrics";
import { SampleTable } from "./sample-table";
import { DownloadPanel } from "./download-panel";
import { StructureViewer } from "./structure-viewer-dynamic";
import { PlddtChart } from "./plddt-chart";
import { PaeHeatmap } from "./pae-heatmap";
import { RestraintViz } from "@/components/restraint-viz";
import type { JobDetail } from "@/lib/types";
import type { RestraintSatisfaction } from "@/lib/restraints";

interface Props {
  job: JobDetail;
}

export function JobResultsView({ job }: Props) {
  const { data: results, isLoading, isError } = useJobResults(job.id, true);
  const [selectedSample, setSelectedSample] = useState<number | null>(null);

  const sampleIndex = selectedSample ?? results?.best_sample_index ?? 0;

  const {
    data: sampleData,
    isFetching: sampleFetching,
  } = useSampleConfidence(job.id, sampleIndex, !!results);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 py-16">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading results...</span>
      </div>
    );
  }

  if (isError || !results) {
    return (
      <div className="flex items-center gap-2 rounded-md bg-destructive/10 px-4 py-3">
        <AlertCircle className="h-4 w-4 text-destructive" />
        <p className="text-sm text-destructive">Failed to load results.</p>
      </div>
    );
  }

  const currentSample = results.samples[sampleIndex];
  const structureRank = currentSample?.rank ?? sampleIndex + 1;
  const cifUrl = getStructureUrl(job.id, structureRank);
  const hasSampleTable = results.samples.length > 1;

  return (
    <div className="space-y-6 xl:space-y-7">
      <ResultsHeader
        jobId={job.id}
        jobName={job.name}
        numResidues={job.num_residues}
        numChains={job.num_chains}
        numSamples={results.num_samples}
        bestSampleIndex={results.best_sample_index}
        selectedSample={sampleIndex}
        onSampleChange={setSelectedSample}
      />

      <ConfidenceMetrics
        ptm={currentSample?.ptm ?? results.ptm}
        iptm={currentSample?.iptm ?? results.iptm}
        meanPlddt={currentSample?.mean_plddt ?? results.mean_plddt}
        isComplex={results.is_complex}
      />

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
        <section className="space-y-5">
          <StructureViewer
            cifUrl={cifUrl}
            satisfaction={sampleData?.restraint_satisfaction as RestraintSatisfaction | undefined}
          />
          <div
            className={`grid grid-cols-1 gap-5 lg:grid-cols-2 transition-opacity duration-200 ${sampleFetching ? "opacity-55" : "opacity-100"}`}
          >
            <PlddtChart plddt={sampleData?.plddt ?? []} />
            <PaeHeatmap pae={sampleData?.pae ?? []} />
          </div>
        </section>

        <aside className="space-y-4 xl:sticky xl:top-20">
          <section className="af-panel p-4">
            <DownloadPanel
              jobId={job.id}
              selectedRank={structureRank}
              numSamples={results.num_samples}
            />
          </section>

          {hasSampleTable && (
            <section className="af-panel p-4">
              <SampleTable
                samples={results.samples}
                bestSampleIndex={results.best_sample_index}
                selectedSample={sampleIndex}
                isComplex={results.is_complex}
                onSelect={setSelectedSample}
              />
            </section>
          )}

          {sampleData?.restraint_satisfaction && (
            <RestraintViz
              satisfaction={sampleData.restraint_satisfaction as RestraintSatisfaction}
            />
          )}
        </aside>
      </div>
    </div>
  );
}
