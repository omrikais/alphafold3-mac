"use client";

import { useCallback } from "react";
import { Star } from "lucide-react";

interface SampleSummary {
  ptm: number | null;
  iptm: number | null;
  mean_plddt: number | null;
}

interface Props {
  samples: SampleSummary[];
  bestSampleIndex: number;
  selectedSample: number;
  isComplex: boolean;
  onSelect: (index: number) => void;
}

function fmt(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "\u2014";
  return value.toFixed(decimals);
}

export function SampleTable({
  samples,
  bestSampleIndex,
  selectedSample,
  isComplex,
  onSelect,
}: Props) {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, index: number) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onSelect(index);
      } else if (e.key === "ArrowDown" && index < samples.length - 1) {
        e.preventDefault();
        onSelect(index + 1);
      } else if (e.key === "ArrowUp" && index > 0) {
        e.preventDefault();
        onSelect(index - 1);
      }
    },
    [onSelect, samples.length],
  );

  if (samples.length <= 1) return null;

  return (
    <div className="space-y-3">
      <div className="space-y-0.5">
        <h3 className="af-panel-header">Sample ranking</h3>
        <p className="af-panel-subtitle">Pick the sample to inspect and export.</p>
      </div>
      <table
        className="w-full text-[13px]"
        role="grid"
        aria-label="Sample ranking"
      >
        <thead>
          <tr className="border-b border-border/80">
            <th className="px-2 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Sample
            </th>
            <th className="px-2 py-2 text-right text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              pTM
            </th>
            {isComplex && (
              <th className="px-2 py-2 text-right text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                ipTM
              </th>
            )}
            <th className="px-2 py-2 text-right text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              pLDDT
            </th>
          </tr>
        </thead>
        <tbody>
          {samples.map((s, i) => {
            const isSelected = i === selectedSample;
            const isBest = i === bestSampleIndex;
            return (
              <tr
                key={i}
                onClick={() => onSelect(i)}
                onKeyDown={(e) => handleKeyDown(e, i)}
                tabIndex={0}
                role="row"
                aria-selected={isSelected}
                className={`
                  border-b border-border/70 last:border-b-0 cursor-pointer transition-colors duration-150
                  outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset
                  ${isSelected
                    ? "bg-primary/10 dark:bg-primary/15 font-medium"
                    : "hover:bg-secondary/65"
                  }
                `}
              >
                <td className="px-2 py-2.5">
                  <span className="flex items-center gap-1.5">
                    <span className="tabular-nums">{i + 1}</span>
                    {isBest && (
                      <Star
                        className="h-3 w-3 text-amber-500 fill-amber-500 shrink-0"
                        aria-label="Best sample"
                      />
                    )}
                  </span>
                </td>
                <td className="px-2 py-2.5 text-right tabular-nums">
                  {fmt(s.ptm)}
                </td>
                {isComplex && (
                  <td className="px-2 py-2.5 text-right tabular-nums">
                    {fmt(s.iptm)}
                  </td>
                )}
                <td className="px-2 py-2.5 text-right tabular-nums">
                  {fmt(s.mean_plddt, 1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
