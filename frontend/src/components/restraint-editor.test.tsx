import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RestraintEditor } from "./restraint-editor";
import type { RestraintConfig, GuidanceConfig } from "@/lib/restraints";
import type { FormEntity } from "@/lib/types";

// Mock Radix-portal UI components that don't work in jsdom
vi.mock("@/components/ui/select", () => import("@/test/ui-mocks"));
vi.mock("@/components/ui/tooltip", () => import("@/test/ui-mocks"));

const TWO_CHAIN_ENTITIES: FormEntity[] = [
  { id: "e1", type: "proteinChain", sequence: "ACDEFGHIKLMNPQRSTVWY", copies: 1 },
  { id: "e2", type: "proteinChain", sequence: "ACDEFGHIKLMNPQRSTVWY", copies: 1 },
];

function setup(overrides?: {
  restraints?: RestraintConfig;
  guidance?: GuidanceConfig;
  entities?: FormEntity[];
}) {
  const onRestraintsChange = vi.fn();
  const onGuidanceChange = vi.fn();
  const restraints = overrides?.restraints ?? {};
  const guidance = overrides?.guidance ?? {};
  const entities = overrides?.entities ?? TWO_CHAIN_ENTITIES;

  const result = render(
    <RestraintEditor
      restraints={restraints}
      guidance={guidance}
      entities={entities}
      onRestraintsChange={onRestraintsChange}
      onGuidanceChange={onGuidanceChange}
    />,
  );

  return { onRestraintsChange, onGuidanceChange, ...result };
}

describe("RestraintEditor", () => {
  describe("add flow", () => {
    it("adds a distance restraint with defaults", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup();

      // There are three "Add" buttons: Distance, Contact, Repulsive
      const addButtons = screen.getAllByRole("button", { name: /add/i });
      // Distance Add is the first one
      await user.click(addButtons[0]);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.distance).toHaveLength(1);
      expect(updated.distance![0]).toMatchObject({
        chain_i: "A",
        chain_j: "B",
        residue_i: 1,
        residue_j: 1,
        target_distance: 1.5,
        atom_i: "CA",
        atom_j: "CA",
        sigma: 1.0,
        weight: 1.0,
      });
    });

    it("adds a contact restraint with defaults", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup();

      const addButtons = screen.getAllByRole("button", { name: /add/i });
      // Contact Add is the second one
      await user.click(addButtons[1]);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.contact).toHaveLength(1);
      expect(updated.contact![0]).toMatchObject({
        chain_i: "A",
        residue_i: 1,
        threshold: 8.0,
        weight: 1.0,
      });
      expect(updated.contact![0].candidates).toEqual([
        { chain_j: "B", residue_j: 1 },
      ]);
    });

    it("adds a repulsive restraint with defaults", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup();

      const addButtons = screen.getAllByRole("button", { name: /add/i });
      // Repulsive Add is the third one
      await user.click(addButtons[2]);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.repulsive).toHaveLength(1);
      expect(updated.repulsive![0]).toMatchObject({
        chain_i: "A",
        chain_j: "B",
        residue_i: 1,
        residue_j: 1,
        min_distance: 15.0,
        weight: 1.0,
      });
    });
  });

  describe("remove flow", () => {
    it("removes a distance restraint", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup({
        restraints: {
          distance: [
            {
              chain_i: "A", residue_i: 5, chain_j: "B", residue_j: 10,
              target_distance: 8.0, atom_i: "CA", atom_j: "CA",
            },
          ],
        },
      });

      const removeBtn = screen.getByRole("button", { name: /remove distance restraint/i });
      await user.click(removeBtn);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.distance).toHaveLength(0);
    });

    it("removes a contact restraint", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup({
        restraints: {
          contact: [
            {
              chain_i: "A", residue_i: 5,
              candidates: [{ chain_j: "B", residue_j: 10 }],
              threshold: 8.0,
            },
          ],
        },
      });

      const removeBtn = screen.getByRole("button", { name: /remove contact restraint/i });
      await user.click(removeBtn);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.contact).toHaveLength(0);
    });

    it("removes a repulsive restraint", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup({
        restraints: {
          repulsive: [
            {
              chain_i: "A", residue_i: 5, chain_j: "B", residue_j: 10,
              min_distance: 15.0,
            },
          ],
        },
      });

      const removeBtn = screen.getByRole("button", { name: /remove repulsive restraint/i });
      await user.click(removeBtn);

      expect(onRestraintsChange).toHaveBeenCalledTimes(1);
      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      expect(updated.repulsive).toHaveLength(0);
    });
  });

  describe("guidance section visibility", () => {
    it("shows guidance parameters when restraints exist", () => {
      setup({
        restraints: {
          distance: [
            {
              chain_i: "A", residue_i: 1, chain_j: "B", residue_j: 1,
              target_distance: 5.0,
            },
          ],
        },
      });

      expect(screen.getByText("Guidance parameters")).toBeInTheDocument();
    });

    it("hides guidance parameters when no restraints exist", () => {
      setup({ restraints: {} });

      expect(screen.queryByText("Guidance parameters")).not.toBeInTheDocument();
    });
  });

  describe("chain assignment from entities", () => {
    it("assigns sequential chain IDs respecting copies", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup({
        entities: [
          { id: "e1", type: "proteinChain", sequence: "ACDEF", copies: 2 },
          { id: "e2", type: "proteinChain", sequence: "GHIKL", copies: 1 },
        ],
      });

      // Add a distance restraint -- chain_j should default to second chain ID "B"
      const addButtons = screen.getAllByRole("button", { name: /add/i });
      await user.click(addButtons[0]);

      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      // With copies=2 for first entity: chains are A, B, C
      // chain_i defaults to first (A), chain_j defaults to second (B)
      expect(updated.distance![0].chain_i).toBe("A");
      expect(updated.distance![0].chain_j).toBe("B");
    });
  });

  describe("protein-only chain filtering", () => {
    it("disables Add buttons when fewer than two protein chains", () => {
      setup({
        entities: [
          { id: "e1", type: "proteinChain", sequence: "ACDEF", copies: 1 },
          { id: "e2", type: "ligand", sequence: "CCO", copies: 1 },
        ],
      });

      const addButtons = screen.getAllByRole("button", { name: /add/i });
      for (const btn of addButtons) {
        expect(btn).toBeDisabled();
      }
      expect(screen.getByText(/at least two protein chains/i)).toBeInTheDocument();
    });

    it("excludes non-protein chains from restraint defaults", async () => {
      const user = userEvent.setup();
      const { onRestraintsChange } = setup({
        entities: [
          { id: "e1", type: "ligand", sequence: "CCO", copies: 1 },
          { id: "e2", type: "proteinChain", sequence: "ACDEF", copies: 1 },
          { id: "e3", type: "proteinChain", sequence: "GHIKL", copies: 1 },
        ],
      });

      const addButtons = screen.getAllByRole("button", { name: /add/i });
      await user.click(addButtons[0]);

      const updated = onRestraintsChange.mock.calls[0][0] as RestraintConfig;
      // Ligand is chain A; proteins are B and C. Defaults should be B and C.
      expect(updated.distance![0].chain_i).toBe("B");
      expect(updated.distance![0].chain_j).toBe("C");
    });
  });

  describe("empty state messages", () => {
    it("shows empty state for all three types when no restraints", () => {
      setup({ restraints: {} });

      expect(screen.getByText("No distance restraints defined.")).toBeInTheDocument();
      expect(screen.getByText("No contact restraints defined.")).toBeInTheDocument();
      expect(screen.getByText("No repulsive restraints defined.")).toBeInTheDocument();
    });
  });

  describe("badge count", () => {
    it("shows count badge when restraints exist", () => {
      setup({
        restraints: {
          distance: [
            { chain_i: "A", residue_i: 1, chain_j: "B", residue_j: 1, target_distance: 5.0 },
            { chain_i: "A", residue_i: 2, chain_j: "B", residue_j: 2, target_distance: 6.0 },
          ],
          contact: [
            { chain_i: "A", residue_i: 3, candidates: [{ chain_j: "B", residue_j: 3 }] },
          ],
        },
      });

      expect(screen.getByText("3 restraints")).toBeInTheDocument();
    });
  });
});
