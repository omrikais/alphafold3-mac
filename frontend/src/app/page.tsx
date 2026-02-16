"use client";

import { Suspense } from "react";
import { JobForm } from "@/components/job-form/job-form";
import { JobHistory } from "@/components/job-history/job-history";

function HomePageContent() {
  return (
    <div className="space-y-5">
      <section className="af-panel px-5 py-4">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">
          New Prediction Session
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure entities, review model inputs, and launch structure predictions
          from one workflow.
        </p>
      </section>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.18fr)_minmax(340px,0.82fr)]">
        <JobForm />
        <JobHistory />
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={<div className="py-10 text-center text-sm text-muted-foreground">Loading...</div>}>
      <HomePageContent />
    </Suspense>
  );
}
