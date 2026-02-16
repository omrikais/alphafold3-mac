import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { JobForm } from "./job-form";

// ── Mock external hooks ────────────────────────────────────────────────────

const mutateFn = vi.fn();
const mockParseStructureFile = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/hooks/use-jobs", () => ({
  useSubmitJob: () => ({
    mutate: mutateFn,
    isPending: false,
  }),
}));

vi.mock("@/hooks/use-system-status", () => ({
  useSystemStatus: () => ({
    data: { run_data_pipeline: false },
  }),
}));

vi.mock("@/lib/api", () => ({
  getJob: vi.fn(),
  parseStructureFile: (...args: unknown[]) => mockParseStructureFile(...args),
}));

// ── Mock child components to isolate serialization logic ───────────────────

// Track restraints/guidance passed to the RestraintEditor mock
let capturedOnRestraintsChange: ((r: Record<string, unknown>) => void) | null = null;
let capturedOnGuidanceChange: ((g: Record<string, unknown>) => void) | null = null;

// Track PdbSearch onResult callback
let capturedOnPdbResult: ((r: Record<string, unknown>) => void) | null = null;

vi.mock("./pdb-search", () => ({
  PdbSearch: (props: { onResult: (r: unknown) => void; disabled?: boolean }) => {
    capturedOnPdbResult = props.onResult as (r: Record<string, unknown>) => void;
    return <div data-testid="pdb-search" />;
  },
}));

vi.mock("@/components/restraint-editor", () => ({
  RestraintEditor: (props: Record<string, unknown>) => {
    capturedOnRestraintsChange = props.onRestraintsChange as (
      r: Record<string, unknown>,
    ) => void;
    capturedOnGuidanceChange = props.onGuidanceChange as (
      g: Record<string, unknown>,
    ) => void;
    return <div data-testid="restraint-editor" />;
  },
}));

vi.mock("./entity-builder", () => ({
  EntityBuilder: ({ onChange }: { onChange: (e: unknown[]) => void }) => (
    <button
      data-testid="set-entity"
      onClick={() =>
        onChange([
          {
            id: "e1",
            type: "proteinChain",
            sequence: "ACDEF",
            copies: 1,
          },
        ])
      }
    >
      Set entity
    </button>
  ),
}));

vi.mock("./preview-modal", () => ({
  PreviewModal: ({
    open,
    onSubmit,
  }: {
    open: boolean;
    onSubmit: (useCache: boolean) => void;
  }) =>
    open ? (
      <button data-testid="submit-btn" onClick={() => onSubmit(false)}>
        Submit
      </button>
    ) : null,
}));

// Mock Radix-portal UI components
vi.mock("@/components/ui/select", () => import("@/test/ui-mocks"));
vi.mock("@/components/ui/tooltip", () => import("@/test/ui-mocks"));

describe("JobForm guidance serialization", () => {
  beforeEach(() => {
    mutateFn.mockClear();
    capturedOnRestraintsChange = null;
    capturedOnGuidanceChange = null;
  });

  it("omits restraints and guidance from payload when no restraints defined", async () => {
    const user = userEvent.setup();
    render(<JobForm />);

    // Set a valid entity so canSubmit = true
    await user.click(screen.getByTestId("set-entity"));

    // Open preview modal
    await user.click(screen.getByText("Preview & submit"));

    // Submit
    await user.click(screen.getByTestId("submit-btn"));

    expect(mutateFn).toHaveBeenCalledTimes(1);
    const submission = mutateFn.mock.calls[0][0];
    expect(submission).not.toHaveProperty("restraints");
    expect(submission).not.toHaveProperty("guidance");
  });

  it("includes restraints and guidance in payload when restraints exist", async () => {
    const user = userEvent.setup();
    render(<JobForm />);

    // Set a valid entity so canSubmit = true
    await user.click(screen.getByTestId("set-entity"));

    // Simulate the RestraintEditor calling onRestraintsChange with a distance restraint
    expect(capturedOnRestraintsChange).not.toBeNull();
    act(() => {
      capturedOnRestraintsChange!({
        distance: [
          {
            chain_i: "A",
            residue_i: 1,
            chain_j: "B",
            residue_j: 1,
            target_distance: 8.0,
          },
        ],
      });
    });

    // Open preview modal
    await user.click(screen.getByText("Preview & submit"));

    // Submit
    await user.click(screen.getByTestId("submit-btn"));

    expect(mutateFn).toHaveBeenCalledTimes(1);
    const submission = mutateFn.mock.calls[0][0];
    expect(submission).toHaveProperty("restraints");
    expect(submission.restraints).toEqual({
      distance: [
        expect.objectContaining({
          chain_i: "A",
          target_distance: 8.0,
        }),
      ],
    });
    expect(submission).toHaveProperty("guidance");
  });
});

// ── Import-reset behavior ─────────────────────────────────────────────────

const MOCK_STRUCTURE_RESULT = {
  name: "test-structure",
  sequences: [{ proteinChain: { sequence: "MVLSG", count: 1 } }],
  dialect: "alphafoldserver",
  version: 1,
  source: "upload",
  pdb_id: null,
  num_chains: 1,
  num_residues: 5,
  warnings: [],
};

describe("JobForm import-reset behavior", () => {
  beforeEach(() => {
    mutateFn.mockClear();
    mockParseStructureFile.mockReset();
    capturedOnRestraintsChange = null;
    capturedOnGuidanceChange = null;
    capturedOnPdbResult = null;
  });

  it("structure file upload clears stale restraints and guidance", async () => {
    mockParseStructureFile.mockResolvedValue(MOCK_STRUCTURE_RESULT);
    const user = userEvent.setup();
    render(<JobForm />);

    // 1. Set initial entity, restraints, and guidance
    await user.click(screen.getByTestId("set-entity"));
    expect(capturedOnRestraintsChange).not.toBeNull();
    expect(capturedOnGuidanceChange).not.toBeNull();
    act(() => {
      capturedOnRestraintsChange!({
        distance: [
          {
            chain_i: "A",
            residue_i: 1,
            chain_j: "B",
            residue_j: 1,
            target_distance: 8.0,
          },
        ],
      });
      capturedOnGuidanceChange!({
        step_range: [0, 50],
        weight: 1.5,
      });
    });

    // 2. Upload a .pdb file to trigger structure import
    const fileInput = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const file = new File(["ATOM ..."], "test.pdb", {
      type: "chemical/x-pdb",
    });
    fireEvent.change(fileInput, { target: { files: [file] } });

    // Wait for async parseStructureFile to resolve and state to update
    await waitFor(() => {
      expect(mockParseStructureFile).toHaveBeenCalledOnce();
    });

    // 3. Submit and verify restraints/guidance are cleared
    await user.click(screen.getByText("Preview & submit"));
    await user.click(screen.getByTestId("submit-btn"));

    expect(mutateFn).toHaveBeenCalledTimes(1);
    const submission = mutateFn.mock.calls[0][0];
    expect(submission).not.toHaveProperty("restraints");
    expect(submission).not.toHaveProperty("guidance");
    // Entities should come from the imported structure
    expect(submission.sequences).toHaveLength(1);
    expect(submission.sequences[0]).toHaveProperty("proteinChain");
  });

  it("PDB fetch clears stale restraints and guidance", async () => {
    const user = userEvent.setup();
    render(<JobForm />);

    // 1. Set initial entity, restraints, and guidance
    await user.click(screen.getByTestId("set-entity"));
    expect(capturedOnRestraintsChange).not.toBeNull();
    expect(capturedOnGuidanceChange).not.toBeNull();
    act(() => {
      capturedOnRestraintsChange!({
        distance: [
          {
            chain_i: "A",
            residue_i: 1,
            chain_j: "B",
            residue_j: 1,
            target_distance: 8.0,
          },
        ],
      });
      capturedOnGuidanceChange!({
        step_range: [0, 50],
        weight: 1.5,
      });
    });

    // 2. Simulate PDB fetch result via PdbSearch callback
    expect(capturedOnPdbResult).not.toBeNull();
    act(() => {
      capturedOnPdbResult!({
        ...MOCK_STRUCTURE_RESULT,
        source: "rcsb",
        pdb_id: "1UBQ",
        name: "1UBQ",
      });
    });

    // 3. Submit and verify restraints/guidance are cleared
    await user.click(screen.getByText("Preview & submit"));
    await user.click(screen.getByTestId("submit-btn"));

    expect(mutateFn).toHaveBeenCalledTimes(1);
    const submission = mutateFn.mock.calls[0][0];
    expect(submission).not.toHaveProperty("restraints");
    expect(submission).not.toHaveProperty("guidance");
    // Entities should come from the PDB fetch result
    expect(submission.sequences).toHaveLength(1);
    expect(submission.sequences[0]).toHaveProperty("proteinChain");
  });
});
