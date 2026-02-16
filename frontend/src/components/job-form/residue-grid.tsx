"use client";

import { useMemo } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ResidueCell } from "./residue-cell";
import { getPtmsForResidue } from "@/lib/constants";
import type { Modification } from "@/lib/types";

interface ResidueGridProps {
  sequence: string;
  modifications: Modification[];
  onAdd: (position: number, ptmType: string) => void;
  onRemove: (position: number) => void;
  onChange: (position: number, ptmType: string) => void;
}

export function ResidueGrid({
  sequence,
  modifications,
  onAdd,
  onRemove,
  onChange,
}: ResidueGridProps) {
  const modMap = useMemo(
    () => new Map(modifications.map((m) => [m.position, m])),
    [modifications],
  );

  const elements: React.ReactNode[] = [];
  for (let i = 0; i < sequence.length; i++) {
    const pos = i + 1;
    const char = sequence[i];
    // Insert position marker every 10 residues
    if (pos % 10 === 0) {
      elements.push(
        <span
          key={`marker-${pos}`}
          className="flex h-[22px] w-[18px] items-center justify-center text-[9px] text-muted-foreground"
        >
          {pos}
        </span>,
      );
    }
    elements.push(
      <ResidueCell
        key={pos}
        char={char}
        position={pos}
        modification={modMap.get(pos)}
        applicablePtms={getPtmsForResidue(char)}
        onAdd={(ptmType) => onAdd(pos, ptmType)}
        onRemove={() => onRemove(pos)}
        onChange={(ptmType) => onChange(pos, ptmType)}
      />,
    );
  }

  return (
    <ScrollArea className="max-h-[240px]">
      <div className="flex flex-wrap gap-[1px]">{elements}</div>
    </ScrollArea>
  );
}
