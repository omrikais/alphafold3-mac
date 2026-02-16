"use client";

interface Props {
  ptm: number | null;
  iptm: number | null;
  meanPlddt: number | null;
  isComplex: boolean;
}

type Quality = "high" | "mid" | "low" | "none";

function getQuality(value: number | null | undefined): Quality {
  if (value == null) return "none";
  if (value >= 0.8) return "high";
  if (value >= 0.5) return "mid";
  return "low";
}

const QUALITY_STYLES: Record<Quality, string> = {
  high: "text-[var(--confidence-high)]",
  mid: "text-[var(--confidence-mid)]",
  low: "text-[var(--confidence-low)]",
  none: "text-muted-foreground",
};

const QUALITY_BG: Record<Quality, string> = {
  high: "bg-[var(--confidence-high-bg)]",
  mid: "bg-[var(--confidence-mid-bg)]",
  low: "bg-[var(--confidence-low-bg)]",
  none: "bg-secondary/70",
};

function Metric({
  label,
  value,
  format,
  helper,
}: {
  label: string;
  value: number | null;
  format: (v: number) => string;
  helper: string;
}) {
  const normalised = label === "pLDDT" && value != null ? value / 100 : value;
  const quality = getQuality(normalised);

  return (
    <article
      className={`af-panel af-panel-strong flex min-h-[112px] flex-col justify-between p-4 ${QUALITY_BG[quality]}`}
      title={helper}
    >
      <span className="af-stat-label">
        {label}
      </span>
      <div className="space-y-1">
        <p
          className={`af-stat-value text-4xl font-semibold leading-none ${QUALITY_STYLES[quality]}`}
        >
          {value != null ? format(value) : "\u2014"}
        </p>
        <p className="text-xs text-muted-foreground">{helper}</p>
      </div>
    </article>
  );
}

export function ConfidenceMetrics({ ptm, iptm, meanPlddt, isComplex }: Props) {
  const metricCount = isComplex ? 3 : 2;
  return (
    <section
      className={`grid gap-3 sm:gap-4 ${
        metricCount === 3 ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1 sm:grid-cols-2"
      }`}
      aria-label="Confidence summary"
    >
      <Metric
        label="pTM"
        value={ptm}
        format={(v) => v.toFixed(2)}
        helper="Global fold confidence (0-1)"
      />
      {isComplex && (
        <Metric
          label="ipTM"
          value={iptm}
          format={(v) => v.toFixed(2)}
          helper="Interface confidence for complexes (0-1)"
        />
      )}
      <Metric
        label="pLDDT"
        value={meanPlddt}
        format={(v) => v.toFixed(1)}
        helper="Mean per-residue confidence (0-100)"
      />
    </section>
  );
}
