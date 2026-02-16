"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Download, FileText, Archive, FolderOpen } from "lucide-react";
import {
  getStructureUrl,
  getDownloadAllUrl,
  getConfidenceJsonUrl,
  openJobDirectory,
} from "@/lib/api";

interface Props {
  jobId: string;
  selectedRank: number;
  numSamples: number;
}

function SecondaryAction({
  icon: Icon,
  label,
  href,
}: {
  icon: typeof FileText;
  label: string;
  href: string;
}) {
  return (
    <a href={href} download>
      <Button
        variant="ghost"
        size="sm"
        className="h-9 w-full justify-start gap-2.5 rounded-md px-2.5 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground"
      >
        <Icon className="h-4 w-4 shrink-0" aria-hidden />
        <span>{label}</span>
      </Button>
    </a>
  );
}

export function DownloadPanel({ jobId, selectedRank, numSamples }: Props) {
  const [opening, setOpening] = useState(false);

  async function handleOpenInFinder() {
    setOpening(true);
    try {
      await openJobDirectory(jobId);
    } catch {
      // silently ignore
    } finally {
      setOpening(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-0.5">
        <h3 className="af-panel-header">Export</h3>
        <p className="af-panel-subtitle">Save structure and confidence files.</p>
      </div>

      <a href={getStructureUrl(jobId, selectedRank)} download className="block">
        <Button className="h-10 w-full gap-2 rounded-md text-sm font-semibold">
          <Download className="h-4 w-4" aria-hidden />
          Download selected CIF
        </Button>
      </a>

      <div className="space-y-1">
        <SecondaryAction
          icon={FileText}
          href={getConfidenceJsonUrl(jobId)}
          label="Confidence JSON"
        />
        {numSamples > 0 && (
          <SecondaryAction
            icon={Archive}
            href={getDownloadAllUrl(jobId)}
            label="All samples (ZIP)"
          />
        )}
        <Button
          variant="ghost"
          size="sm"
          className="h-9 w-full justify-start gap-2.5 rounded-md px-2.5 text-sm font-medium text-muted-foreground hover:bg-secondary hover:text-foreground"
          onClick={handleOpenInFinder}
          disabled={opening}
          aria-label="Open output directory in Finder"
        >
          <FolderOpen className="h-4 w-4 shrink-0" aria-hidden />
          Open in Finder
        </Button>
      </div>
    </div>
  );
}
