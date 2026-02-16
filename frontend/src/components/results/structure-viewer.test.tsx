import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { StructureViewerInner } from "./structure-viewer";
import type { RestraintSatisfaction } from "@/lib/restraints";

// ── Mock Mol* plugin object ─────────────────────────────────────────────────

const mockAddDistance = vi.fn().mockResolvedValue(undefined);
const mockCommit = vi.fn().mockResolvedValue(undefined);
const mockBuildDelete = vi.fn();
const mockBuild = vi.fn(() => ({ delete: mockBuildDelete, commit: mockCommit }));
const mockPluginClear = vi.fn().mockResolvedValue(undefined);

const mockCells = new Map<
  string,
  { transform: { ref: string; tags: string[] } }
>();

const mockPlugin = {
  clear: mockPluginClear,
  dispose: vi.fn(),
  canvas3d: {
    props: { renderer: { backgroundColor: 0 } },
    setProps: vi.fn(),
  },
  builders: {
    data: { rawData: vi.fn().mockResolvedValue({ ref: "data-ref" }) },
    structure: {
      parseTrajectory: vi.fn().mockResolvedValue({ ref: "traj-ref" }),
      hierarchy: { applyPreset: vi.fn().mockResolvedValue(undefined) },
    },
  },
  managers: {
    structure: {
      hierarchy: {
        current: {
          structures: [{ cell: { obj: { data: { id: "mock-struct" } } } }],
        },
      },
      measurement: { addDistance: mockAddDistance },
    },
  },
  state: {
    data: { cells: mockCells, build: mockBuild },
  },
};

// ── Mock Mol* modules (intercepted for both static and dynamic import()) ────

vi.mock("molstar/lib/mol-plugin-ui/index.js", () => ({
  createPluginUI: vi.fn().mockResolvedValue(mockPlugin),
}));

vi.mock("molstar/lib/mol-plugin-ui/react18.js", () => ({
  renderReact18: vi.fn(),
}));

vi.mock("molstar/lib/mol-plugin-ui/spec.js", () => ({
  DefaultPluginUISpec: vi.fn().mockReturnValue({}),
}));

vi.mock("molstar/lib/mol-script/language/builder.js", () => ({
  MolScriptBuilder: {
    struct: {
      generator: { atomGroups: vi.fn().mockReturnValue("mock-expr") },
    },
    core: { rel: { eq: vi.fn().mockReturnValue("mock-eq") } },
    ammp: vi.fn().mockReturnValue("mock-ammp"),
  },
}));

vi.mock("molstar/lib/mol-model/structure.js", () => ({
  StructureElement: {
    Loci: {
      fromExpression: vi.fn().mockReturnValue({ elements: ["mock"] }),
      isEmpty: vi.fn().mockReturnValue(false),
    },
  },
}));

// ── Mock fetch ──────────────────────────────────────────────────────────────

const mockFetch = vi.fn().mockResolvedValue({
  ok: true,
  text: () => Promise.resolve("data_fake\n_cell.length_a 1.0\n"),
});
global.fetch = mockFetch as unknown as typeof fetch;

// ── Helpers ─────────────────────────────────────────────────────────────────

function makeSatisfaction(
  overrides: Partial<RestraintSatisfaction> = {},
): RestraintSatisfaction {
  return {
    distance: [
      {
        chain_i: "A",
        residue_i: 10,
        atom_i: "CA",
        chain_j: "B",
        residue_j: 25,
        atom_j: "CB",
        target_distance: 8.0,
        actual_distance: 7.5,
        satisfied: true,
      },
    ],
    contact: [
      {
        chain_i: "A",
        residue_i: 5,
        closest_candidate_chain: "B",
        closest_candidate_residue: 30,
        threshold: 8.0,
        actual_distance: 6.2,
        satisfied: true,
      },
    ],
    repulsive: [
      {
        chain_i: "A",
        residue_i: 1,
        chain_j: "B",
        residue_j: 1,
        min_distance: 5.0,
        actual_distance: 12.0,
        satisfied: true,
      },
    ],
    ...overrides,
  };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("StructureViewerInner restraint visualization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCells.clear();
    // Re-stub fetch (clearAllMocks resets it)
    mockFetch.mockResolvedValue({
      ok: true,
      text: () => Promise.resolve("data_fake\n"),
    });
    global.fetch = mockFetch as unknown as typeof fetch;
  });

  it("draws restraint lines when satisfaction data is present", async () => {
    const sat = makeSatisfaction();

    render(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={sat} />,
    );

    // Wait for async init → loadStructure → drawRestraints chain to complete
    await waitFor(() => {
      // 1 distance + 1 contact + 1 repulsive = 3 addDistance calls
      expect(mockAddDistance).toHaveBeenCalledTimes(3);
    });

    // Verify restraint tag is passed for each call
    for (const call of mockAddDistance.mock.calls) {
      const opts = call[2] as { reprTags: string[]; selectionTags: string[] };
      expect(opts.reprTags).toContain("restraint-viz");
      expect(opts.selectionTags).toContain("restraint-viz");
    }
  });

  it("uses green color for satisfied and red for unsatisfied restraints", async () => {
    const sat = makeSatisfaction({
      distance: [
        {
          chain_i: "A",
          residue_i: 1,
          atom_i: "CA",
          chain_j: "B",
          residue_j: 2,
          atom_j: "CA",
          target_distance: 5.0,
          actual_distance: 4.5,
          satisfied: true,
        },
        {
          chain_i: "A",
          residue_i: 3,
          atom_i: "CA",
          chain_j: "B",
          residue_j: 4,
          atom_j: "CA",
          target_distance: 5.0,
          actual_distance: 12.0,
          satisfied: false,
        },
      ],
      contact: [],
      repulsive: [],
    });

    render(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={sat} />,
    );

    await waitFor(() => {
      expect(mockAddDistance).toHaveBeenCalledTimes(2);
    });

    // First call: satisfied → green (0x22cc22)
    const greenOpts = mockAddDistance.mock.calls[0][2] as {
      visualParams: { linesColor: number };
    };
    expect(greenOpts.visualParams.linesColor).toBe(0x22cc22);

    // Second call: unsatisfied → red (0xcc2222)
    const redOpts = mockAddDistance.mock.calls[1][2] as {
      visualParams: { linesColor: number };
    };
    expect(redOpts.visualParams.linesColor).toBe(0xcc2222);
  });

  it("does not crash and draws nothing when satisfaction is absent", async () => {
    render(<StructureViewerInner cifUrl="/api/test.cif" />);

    // Wait for init + loadStructure to complete (no satisfaction → no drawRestraints)
    await waitFor(() => {
      expect(mockPlugin.builders.structure.hierarchy.applyPreset).toHaveBeenCalled();
    });

    expect(mockAddDistance).not.toHaveBeenCalled();
  });

  it("clears old restraints and draws new ones when satisfaction changes", async () => {
    const satA = makeSatisfaction({
      distance: [
        {
          chain_i: "A",
          residue_i: 1,
          atom_i: "CA",
          chain_j: "B",
          residue_j: 2,
          atom_j: "CA",
          target_distance: 5.0,
          actual_distance: 5.1,
          satisfied: true,
        },
      ],
      contact: [],
      repulsive: [],
    });

    const { rerender } = render(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={satA} />,
    );

    // Wait for initial draw
    await waitFor(() => {
      expect(mockAddDistance).toHaveBeenCalledTimes(1);
    });

    // Simulate existing restraint cells in the state tree
    mockCells.set("restraint-1", {
      transform: { ref: "restraint-1", tags: ["restraint-viz"] },
    });
    mockCells.set("other-node", {
      transform: { ref: "other-node", tags: ["representation"] },
    });

    mockAddDistance.mockClear();
    mockBuild.mockClear();
    mockBuildDelete.mockClear();
    mockCommit.mockClear();

    // Change satisfaction data
    const satB = makeSatisfaction({
      distance: [
        {
          chain_i: "A",
          residue_i: 10,
          atom_i: "N",
          chain_j: "B",
          residue_j: 20,
          atom_j: "N",
          target_distance: 10.0,
          actual_distance: 15.0,
          satisfied: false,
        },
      ],
      contact: [],
      repulsive: [],
    });

    rerender(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={satB} />,
    );

    // Wait for clear + redraw
    await waitFor(() => {
      expect(mockAddDistance).toHaveBeenCalledTimes(1);
    });

    // Verify clear deleted only the restraint-tagged cell, not the other node
    expect(mockBuild).toHaveBeenCalled();
    expect(mockBuildDelete).toHaveBeenCalledWith("restraint-1");
    expect(mockBuildDelete).not.toHaveBeenCalledWith("other-node");
    expect(mockCommit).toHaveBeenCalled();
  });

  it("clears restraint lines when satisfaction becomes null", async () => {
    const sat = makeSatisfaction({
      distance: [
        {
          chain_i: "A",
          residue_i: 1,
          atom_i: "CA",
          chain_j: "B",
          residue_j: 2,
          atom_j: "CA",
          target_distance: 5.0,
          actual_distance: 5.0,
          satisfied: true,
        },
      ],
      contact: [],
      repulsive: [],
    });

    const { rerender } = render(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={sat} />,
    );

    await waitFor(() => {
      expect(mockAddDistance).toHaveBeenCalledTimes(1);
    });

    // Add a tagged cell so clearRestraints has something to remove
    mockCells.set("line-ref", {
      transform: { ref: "line-ref", tags: ["restraint-viz"] },
    });

    mockAddDistance.mockClear();

    // Remove satisfaction
    rerender(
      <StructureViewerInner cifUrl="/api/test.cif" satisfaction={null} />,
    );

    // Wait for clear to run
    await waitFor(() => {
      expect(mockBuild).toHaveBeenCalled();
    });

    // No new lines should be drawn
    expect(mockAddDistance).not.toHaveBeenCalled();
  });
});
