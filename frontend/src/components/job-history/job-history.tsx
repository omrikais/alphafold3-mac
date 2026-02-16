"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
import { useJobs } from "@/hooks/use-jobs";
import { JobCard } from "./job-card";
import type { JobStatus } from "@/lib/types";

const STATUS_FILTERS: { label: string; value: JobStatus | "" }[] = [
  { label: "All", value: "" },
  { label: "Running", value: "running" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
  { label: "Queued", value: "pending" },
];

export function JobHistory() {
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  const { data, isLoading, isError } = useJobs({
    status: statusFilter || undefined,
    search: search || undefined,
    page,
  });

  const totalPages = data ? Math.ceil(data.total / data.page_size) : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle>Recent jobs</CardTitle>
        <p className="text-sm text-muted-foreground">
          Monitor queued, running, and completed predictions.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search jobs..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="h-9 pl-8 text-sm"
          />
        </div>

        <div className="flex flex-wrap gap-1.5">
          {STATUS_FILTERS.map((f) => (
            <Badge
              key={f.value}
              variant={statusFilter === f.value ? "default" : "outline"}
              className="h-7 cursor-pointer rounded-full px-2.5 text-[11px] font-medium"
              onClick={() => {
                setStatusFilter(f.value);
                setPage(1);
              }}
            >
              {f.label}
            </Badge>
          ))}
        </div>

        <div className="space-y-2">
          {isLoading && !isError && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Loading...
            </p>
          )}
          {isError && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              Could not connect to backend. Start the API server with:{" "}
              <code className="rounded bg-muted px-1 py-0.5 text-[11px]">
                python -m alphafold3_mlx.api
              </code>
            </p>
          )}
          {data && data.jobs.length === 0 && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No jobs found. Submit a prediction to get started.
            </p>
          )}
          {data?.jobs.map((job) => <JobCard key={job.id} job={job} />)}
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-border/70 pt-3">
            <span className="text-xs text-muted-foreground">
              Page {page} of {totalPages} ({data?.total} jobs)
            </span>
            <div className="flex gap-1.5">
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-md"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                className="h-8 w-8 rounded-md"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
