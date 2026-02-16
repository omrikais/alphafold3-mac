"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

const BANDS = [
  { min: 90, color: "#3B82F6", label: "Very high (>90)" },
  { min: 70, color: "#22D3EE", label: "Confident (70\u201390)" },
  { min: 50, color: "#FBBF24", label: "Low (50\u201370)" },
  { min: 0, color: "#F97316", label: "Very low (<50)" },
] as const;

function plddtColor(value: number): string {
  if (value >= 90) return BANDS[0].color;
  if (value >= 70) return BANDS[1].color;
  if (value >= 50) return BANDS[2].color;
  return BANDS[3].color;
}

interface Props {
  plddt: number[];
}

export function PlddtChart({ plddt }: Props) {
  const data = useMemo(
    () => plddt.map((v, i) => ({ residue: i + 1, plddt: v })),
    [plddt],
  );

  const tickInterval = useMemo(() => {
    const n = plddt.length;
    if (n <= 50) return 5;
    if (n <= 150) return 10;
    if (n <= 400) return 25;
    return 50;
  }, [plddt.length]);

  const xTicks = useMemo(() => {
    const ticks: number[] = [];
    for (let i = tickInterval; i <= plddt.length; i += tickInterval) {
      ticks.push(i);
    }
    if (ticks.length === 0 && plddt.length > 0) ticks.push(1);
    return ticks;
  }, [plddt.length, tickInterval]);

  if (plddt.length === 0) {
    return (
      <div className="af-panel p-5">
        <h3 className="af-panel-header mb-1.5">Per-residue confidence</h3>
        <p className="text-sm text-muted-foreground">
          No pLDDT data available.
        </p>
      </div>
    );
  }

  return (
    <article className="af-panel p-5">
      <div className="mb-3.5 flex items-start justify-between gap-4">
        <div>
          <h3 className="af-panel-header">Per-residue confidence</h3>
          <p className="af-panel-subtitle">pLDDT score distribution by residue index.</p>
        </div>
        <div className="flex flex-wrap justify-end gap-x-3 gap-y-1.5">
          {BANDS.map((band) => (
            <div key={band.label} className="flex items-center gap-1.5">
              <div
                className="h-2 w-2 rounded-full shrink-0"
                style={{ backgroundColor: band.color }}
              />
              <span className="text-xs text-muted-foreground">
                {band.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 12, left: 2, bottom: 22 }}
            barCategoryGap={0}
            barGap={0}
          >
            <CartesianGrid
              vertical={false}
              strokeDasharray="4 4"
              stroke="var(--border)"
              strokeOpacity={0.45}
            />
            <XAxis
              dataKey="residue"
              ticks={xTicks}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ strokeWidth: 1 }}
              label={{
                value: "Residue",
                position: "insideBottom",
                offset: -12,
                fontSize: 11,
              }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fontSize: 11 }}
              tickLine={false}
              axisLine={{ strokeWidth: 1 }}
              width={62}
              label={{
                value: "pLDDT",
                angle: -90,
                position: "left",
                dx: -10,
                fontSize: 11,
                fill: "var(--muted-foreground)",
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const d = payload[0].payload as {
                  residue: number;
                  plddt: number;
                };
                return (
                  <div className="rounded-md border bg-popover px-3 py-2 text-sm shadow-md">
                    <span className="text-muted-foreground">
                      Residue {d.residue}:{" "}
                    </span>
                    <span
                      className="font-semibold"
                      style={{ color: plddtColor(d.plddt) }}
                    >
                      {d.plddt.toFixed(1)}
                    </span>
                  </div>
                );
              }}
            />
            <ReferenceLine
              y={90}
              stroke={BANDS[0].color}
              strokeDasharray="4 4"
              strokeOpacity={0.3}
            />
            <ReferenceLine
              y={70}
              stroke={BANDS[1].color}
              strokeDasharray="4 4"
              strokeOpacity={0.3}
            />
            <ReferenceLine
              y={50}
              stroke={BANDS[2].color}
              strokeDasharray="4 4"
              strokeOpacity={0.3}
            />
            <Bar dataKey="plddt" maxBarSize={6} isAnimationActive={false}>
              {data.map((entry) => (
                <Cell key={entry.residue} fill={plddtColor(entry.plddt)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </article>
  );
}
