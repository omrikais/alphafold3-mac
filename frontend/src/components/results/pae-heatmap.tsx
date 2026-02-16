"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  pae: number[][];
}

const COLOR_STOPS: Array<[number, [number, number, number]]> = [
  [0, [14, 165, 233]],
  [0.45, [34, 197, 170]],
  [0.72, [234, 179, 8]],
  [1, [124, 58, 22]],
];

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function paeColor(value: number): [number, number, number] {
  const t = Math.min(value / 30, 1);
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [t0, c0] = COLOR_STOPS[i];
    const [t1, c1] = COLOR_STOPS[i + 1];
    if (t <= t1) {
      const local = (t - t0) / (t1 - t0);
      return [
        lerp(c0[0], c1[0], local),
        lerp(c0[1], c1[1], local),
        lerp(c0[2], c1[2], local),
      ];
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1][1];
}

export function PaeHeatmap({ pae }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    i: number;
    j: number;
    value: number;
    containerWidth: number;
  } | null>(null);

  const n = pae.length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || n === 0) return;

    canvas.width = n;
    canvas.height = n;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(n, n);
    const px = imageData.data;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const [r, g, b] = paeColor(pae[i]?.[j] ?? 0);
        const idx = (i * n + j) * 4;
        px[idx] = r;
        px[idx + 1] = g;
        px[idx + 2] = b;
        px[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pae, n]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas || n === 0) return;

      const rect = canvas.getBoundingClientRect();
      const j = Math.floor(((e.clientX - rect.left) / rect.width) * n);
      const i = Math.floor(((e.clientY - rect.top) / rect.height) * n);

      if (i >= 0 && i < n && j >= 0 && j < n) {
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          i: i + 1,
          j: j + 1,
          value: pae[i][j],
          containerWidth: containerRef.current?.clientWidth ?? 300,
        });
      } else {
        setTooltip(null);
      }
    },
    [pae, n],
  );

  if (n === 0) {
    return (
      <div className="af-panel p-5">
        <h3 className="af-panel-header mb-1.5">Predicted aligned error</h3>
        <p className="text-sm text-muted-foreground">
          No PAE data available.
        </p>
      </div>
    );
  }

  return (
    <article className="af-panel p-5">
      <div className="mb-3.5">
        <h3 className="af-panel-header">Predicted aligned error</h3>
        <p className="af-panel-subtitle">Expected position error in Angstrom units.</p>
      </div>

      <div className="flex gap-3">
        <div ref={containerRef} className="flex-1 min-w-0">
          <p className="text-xs text-muted-foreground text-center mb-1.5">
            Scored residue
          </p>
          <div className="flex">
            <div className="flex items-center pr-1.5">
              <span className="text-xs text-muted-foreground [writing-mode:vertical-lr] rotate-180">
                Aligned residue
              </span>
            </div>
            <div
              className="relative flex-1 aspect-square"
              style={{ maxHeight: 290 }}
            >
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full rounded"
                style={{ imageRendering: "pixelated" }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setTooltip(null)}
                role="img"
                aria-label={`PAE heatmap for ${n} residues`}
              />
              {tooltip && (
                <div
                  className="pointer-events-none absolute z-10 rounded-md border bg-popover px-3 py-2 text-sm shadow-md whitespace-nowrap"
                  style={{
                    left: Math.min(
                      tooltip.x + 12,
                      tooltip.containerWidth - 140,
                    ),
                    top: Math.max(tooltip.y - 36, 0),
                  }}
                >
                  <span className="text-muted-foreground">
                    ({tooltip.i}, {tooltip.j}):{" "}
                  </span>
                  <span className="font-semibold">
                    {tooltip.value.toFixed(1)}&thinsp;{"\u00C5"}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex w-9 flex-col items-center gap-1.5 py-6">
          <span className="text-xs text-muted-foreground tabular-nums font-medium leading-none">
            0
          </span>
          <div
            className="flex-1 w-3.5 rounded-full border border-border/60"
            style={{
              background:
                "linear-gradient(to bottom, rgb(14,165,233), rgb(34,197,170), rgb(234,179,8), rgb(124,58,22))",
            }}
          />
          <span className="text-xs text-muted-foreground tabular-nums font-medium leading-none">
            30
          </span>
          <span className="text-xs text-muted-foreground leading-none">
            {"\u00C5"}
          </span>
        </div>
      </div>
    </article>
  );
}
