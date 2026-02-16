"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, CheckCircle2, RotateCcw, Star } from "lucide-react";

interface Props {
  jobId: string;
  jobName: string;
  numResidues: number | null;
  numChains: number | null;
  numSamples: number;
  bestSampleIndex: number;
  selectedSample: number;
  onSampleChange: (index: number) => void;
}

export function ResultsHeader({
  jobId,
  jobName,
  numResidues,
  numChains,
  numSamples,
  bestSampleIndex,
  selectedSample,
  onSampleChange,
}: Props) {
  const isTopRankSelected = selectedSample === bestSampleIndex;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Link href="/">
          <Button
            variant="ghost"
            size="sm"
            className="-ml-2 h-8 gap-1.5 rounded-md px-2 text-sm font-medium text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden />
            <span>Back to jobs</span>
          </Button>
        </Link>
        <Link href={`/?loadJob=${jobId}`}>
          <Button variant="outline" size="sm" className="h-8 gap-1.5 rounded-md px-3">
            <RotateCcw className="h-3.5 w-3.5" />
            Reuse inputs
          </Button>
        </Link>
      </div>

      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="min-w-0 space-y-2.5">
          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="truncate text-[1.95rem] font-semibold tracking-tight text-foreground md:text-[2.15rem]">
              {jobName}
            </h1>
            <Badge
              variant="success"
              className="h-6 gap-1 rounded-full px-2.5 text-[11px] font-semibold"
            >
              <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
              Completed
            </Badge>
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-sm text-muted-foreground">
            <code className="rounded-md border border-border/70 bg-secondary px-1.5 py-0.5 font-mono text-xs font-medium tracking-wide">
              {jobId.slice(0, 8)}
            </code>
            {numResidues != null && (
              <>
                <span className="opacity-40">·</span>
                <span>{numResidues} residues</span>
              </>
            )}
            {numChains != null && (
              <>
                <span className="opacity-40">·</span>
                <span>
                  {numChains} {numChains === 1 ? "chain" : "chains"}
                </span>
              </>
            )}
            <span className="opacity-40">·</span>
            <span>
              {numSamples} {numSamples === 1 ? "sample" : "samples"}
            </span>
          </div>
        </div>

        {numSamples > 1 && (
          <div className="af-panel flex w-full flex-col gap-1.5 px-3 py-2 xl:w-auto">
            <span className="af-panel-subtitle">Active sample</span>
            <Select
              value={String(selectedSample)}
              onValueChange={(v) => onSampleChange(Number(v))}
            >
              <SelectTrigger
                className="h-10 min-w-[180px] bg-background text-sm font-medium"
                aria-label="Select sample"
              >
                <SelectValue>
                  <span className="flex items-center gap-2">
                    {isTopRankSelected && (
                      <Star className="h-3.5 w-3.5 shrink-0 fill-amber-500 text-amber-500" />
                    )}
                    Sample {selectedSample + 1}
                    {isTopRankSelected && (
                      <span className="text-xs text-muted-foreground">
                        (Top-ranked)
                      </span>
                    )}
                  </span>
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="end">
                {Array.from({ length: numSamples }, (_, i) => (
                  <SelectItem key={i} value={String(i)} className="text-sm">
                    <span className="flex items-center gap-2">
                      Sample {i + 1}
                      {i === bestSampleIndex && (
                        <Badge
                          variant="secondary"
                          className="h-5 rounded-full px-1.5 text-[10px] font-medium"
                        >
                          Top-ranked
                        </Badge>
                      )}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
    </div>
  );
}
