"use client";

import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import type { Modification } from "@/lib/types";

interface PTMSummaryProps {
  modifications: Modification[];
  sequence: string;
  onRemove: (position: number) => void;
}

export function PTMSummary({
  modifications,
  sequence,
  onRemove,
}: PTMSummaryProps) {
  if (modifications.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Click residues above to add modifications
      </p>
    );
  }

  return (
    <div className="flex flex-wrap gap-1">
      {modifications
        .slice()
        .sort((a, b) => a.position - b.position)
        .map((mod) => (
          <Badge
            key={mod.position}
            variant="secondary"
            className="gap-0.5 pr-0.5 text-[10px]"
          >
            {mod.type} @ {sequence[mod.position - 1]}
            {mod.position}
            <button
              type="button"
              onClick={() => onRemove(mod.position)}
              className="ml-0.5 rounded-full p-0.5 hover:bg-foreground/10"
              aria-label={`Remove ${mod.type} at position ${mod.position}`}
            >
              <X className="size-2.5" />
            </button>
          </Badge>
        ))}
    </div>
  );
}
