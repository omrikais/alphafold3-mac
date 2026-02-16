"use client";

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

// Mol* CSS must be imported here (not in the dynamic component) so it's
// included in the page's CSS bundle. Dynamic imports with ssr:false don't
// include their CSS in the server-rendered HTML of static exports.
import "molstar/build/viewer/molstar.css";

const StructureViewerInner = dynamic(
  () =>
    import("./structure-viewer").then((mod) => mod.StructureViewerInner),
  {
    ssr: false,
    loading: () => (
      <section className="af-panel af-panel-strong p-2.5">
        <div className="flex items-center justify-between px-1.5 pb-2.5">
          <h2 className="af-panel-header">3D structure</h2>
          <span className="af-panel-subtitle">Interactive molecular viewer</span>
        </div>
        <div
          className="flex items-center justify-center overflow-hidden rounded-xl border border-border/80"
          style={{
            height: "clamp(380px, 52vh, 620px)",
            background: "var(--viewer-bg)",
          }}
        >
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-sm">Loading viewer...</span>
          </div>
        </div>
      </section>
    ),
  },
);

export { StructureViewerInner as StructureViewer };
export default StructureViewerInner;
