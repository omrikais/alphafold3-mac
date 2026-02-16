"use client";

import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ChevronRight } from "lucide-react";
import { ResidueGrid } from "./residue-grid";
import { PTMSummary } from "./ptm-summary";
import type { Modification } from "@/lib/types";

interface PTMWidgetProps {
  sequence: string;
  modifications: Modification[];
  onModificationsChange: (mods: Modification[]) => void;
}

export function PTMWidget({
  sequence,
  modifications,
  onModificationsChange,
}: PTMWidgetProps) {
  const [expanded, setExpanded] = useState(false);

  // Auto-expand when modifications exist
  useEffect(() => {
    if (modifications.length > 0) setExpanded(true);
  }, [modifications.length]);

  // Prune stale modifications when sequence shortens
  useEffect(() => {
    const pruned = modifications.filter((m) => m.position <= sequence.length);
    if (pruned.length !== modifications.length) {
      onModificationsChange(pruned);
    }
  }, [sequence.length, modifications, onModificationsChange]);

  const handleAdd = useCallback(
    (position: number, ptmType: string) => {
      // Don't allow duplicate at same position
      if (modifications.some((m) => m.position === position)) return;
      onModificationsChange([
        ...modifications,
        { type: ptmType, position },
      ]);
    },
    [modifications, onModificationsChange],
  );

  const handleRemove = useCallback(
    (position: number) => {
      onModificationsChange(
        modifications.filter((m) => m.position !== position),
      );
    },
    [modifications, onModificationsChange],
  );

  const handleChange = useCallback(
    (position: number, ptmType: string) => {
      onModificationsChange(
        modifications.map((m) =>
          m.position === position ? { ...m, type: ptmType } : m,
        ),
      );
    },
    [modifications, onModificationsChange],
  );

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        <ChevronRight
          className={`size-3.5 transition-transform ${expanded ? "rotate-90" : ""}`}
        />
        Modifications
        {modifications.length > 0 && (
          <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">
            {modifications.length}
          </Badge>
        )}
      </button>

      {expanded && (
        <div className="space-y-2 pl-5">
          <ResidueGrid
            sequence={sequence}
            modifications={modifications}
            onAdd={handleAdd}
            onRemove={handleRemove}
            onChange={handleChange}
          />
          <PTMSummary
            modifications={modifications}
            sequence={sequence}
            onRemove={handleRemove}
          />
        </div>
      )}
    </div>
  );
}
