import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { JobResultsView } from "./job-results-view";
import type { JobDetail } from "@/lib/types";

// ── Mock hooks ──────────────────────────────────────────────────────────────

let mockSampleData: Record<string, unknown> | undefined;

vi.mock("@/hooks/use-job-detail", () => ({
  useJobResults: () => ({
    data: {
      ptm: 0.85,
      iptm: 0.75,
      mean_plddt: 80.0,
      ranking_metric: "iptm",
      num_samples: 1,
      samples: [{ ptm: 0.85, iptm: 0.75, mean_plddt: 80.0, rank: 1 }],
      best_sample_index: 0,
      is_complex: true,
    },
    isLoading: false,
    isError: false,
  }),
  useSampleConfidence: () => ({
    data: mockSampleData,
    isFetching: false,
  }),
}));

vi.mock("@/lib/api", () => ({
  getStructureUrl: (jobId: string, rank: number) => `/api/jobs/${jobId}/structure/${rank}`,
}));

// ── Mock child components ───────────────────────────────────────────────────

vi.mock("./results-header", () => ({
  ResultsHeader: () => <div data-testid="results-header" />,
}));

vi.mock("./confidence-metrics", () => ({
  ConfidenceMetrics: () => <div data-testid="confidence-metrics" />,
}));

vi.mock("./sample-table", () => ({
  SampleTable: () => <div data-testid="sample-table" />,
}));

vi.mock("./download-panel", () => ({
  DownloadPanel: () => <div data-testid="download-panel" />,
}));

vi.mock("./structure-viewer-dynamic", () => ({
  StructureViewer: (props: { satisfaction?: unknown }) => (
    <div
      data-testid="structure-viewer"
      data-has-satisfaction={props.satisfaction != null ? "true" : "false"}
    />
  ),
}));

vi.mock("./plddt-chart", () => ({
  PlddtChart: () => <div data-testid="plddt-chart" />,
}));

vi.mock("./pae-heatmap", () => ({
  PaeHeatmap: () => <div data-testid="pae-heatmap" />,
}));

// Mock RestraintViz to capture passed props
vi.mock("@/components/restraint-viz", () => ({
  RestraintViz: (props: { satisfaction: unknown }) => (
    <div data-testid="restraint-viz" data-satisfaction={JSON.stringify(props.satisfaction)} />
  ),
}));

const JOB: JobDetail = {
  id: "job-123",
  name: "test-job",
  status: "completed",
  created_at: "2026-02-14T00:00:00Z",
  updated_at: "2026-02-14T01:00:00Z",
  num_residues: 100,
  num_chains: 2,
  error_message: null,
  progress: 100,
  input_json: {},
  num_samples: 1,
  diffusion_steps: 200,
  run_data_pipeline: false,
  current_stage: null,
};

describe("JobResultsView restraint data flow", () => {
  it("renders RestraintViz when sample data includes restraint_satisfaction", () => {
    mockSampleData = {
      sample_index: 0,
      ptm: 0.85,
      iptm: 0.75,
      mean_plddt: 80.0,
      plddt: [80, 85, 90],
      pae: [[0, 1], [1, 0]],
      num_residues: 100,
      restraint_satisfaction: {
        distance: [
          {
            chain_i: "A", residue_i: 10, atom_i: "CA",
            chain_j: "B", residue_j: 25, atom_j: "CA",
            target_distance: 8.0, actual_distance: 7.5, satisfied: true,
          },
        ],
      },
    };

    render(<JobResultsView job={JOB} />);

    const viz = screen.getByTestId("restraint-viz");
    expect(viz).toBeInTheDocument();

    // Verify the data passed through
    const passed = JSON.parse(viz.getAttribute("data-satisfaction")!);
    expect(passed.distance).toHaveLength(1);
    expect(passed.distance[0].satisfied).toBe(true);
    expect(passed.distance[0].actual_distance).toBe(7.5);
  });

  it("does not render RestraintViz when no restraint_satisfaction in sample data", () => {
    mockSampleData = {
      sample_index: 0,
      ptm: 0.85,
      iptm: 0.75,
      mean_plddt: 80.0,
      plddt: [80, 85, 90],
      pae: [[0, 1], [1, 0]],
      num_residues: 100,
    };

    render(<JobResultsView job={JOB} />);

    expect(screen.queryByTestId("restraint-viz")).not.toBeInTheDocument();
  });

  it("passes satisfaction to StructureViewer when available", () => {
    mockSampleData = {
      sample_index: 0,
      ptm: 0.85,
      iptm: 0.75,
      mean_plddt: 80.0,
      plddt: [80, 85, 90],
      pae: [[0, 1], [1, 0]],
      num_residues: 100,
      restraint_satisfaction: {
        distance: [
          {
            chain_i: "A", residue_i: 1, atom_i: "CA",
            chain_j: "B", residue_j: 1, atom_j: "CA",
            target_distance: 5.0, actual_distance: 4.8, satisfied: true,
          },
        ],
      },
    };

    render(<JobResultsView job={JOB} />);

    const viewer = screen.getByTestId("structure-viewer");
    expect(viewer.getAttribute("data-has-satisfaction")).toBe("true");
  });

  it("does not pass satisfaction to StructureViewer when absent", () => {
    mockSampleData = {
      sample_index: 0,
      ptm: 0.85,
      iptm: 0.75,
      mean_plddt: 80.0,
      plddt: [],
      pae: [],
      num_residues: 100,
    };

    render(<JobResultsView job={JOB} />);

    const viewer = screen.getByTestId("structure-viewer");
    expect(viewer.getAttribute("data-has-satisfaction")).toBe("false");
  });
});
