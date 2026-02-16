"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { JobDetailView } from "@/components/progress/job-detail-view";

function JobPageContent() {
  const params = useSearchParams();
  const id = params.get("id");

  if (!id) {
    return (
      <div className="af-panel flex items-center justify-center py-20">
        <p className="text-sm text-destructive">No job ID specified</p>
      </div>
    );
  }

  return <JobDetailView jobId={id} />;
}

export default function JobPage() {
  return (
    <Suspense
      fallback={
        <div className="af-panel flex items-center justify-center py-20">
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      }
    >
      <JobPageContent />
    </Suspense>
  );
}
