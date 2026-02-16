import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { RestraintViz } from "./restraint-viz";
import type { RestraintSatisfaction } from "@/lib/restraints";

// Mock Radix tooltip (uses portals)
vi.mock("@/components/ui/tooltip", () => import("@/test/ui-mocks"));

describe("RestraintViz", () => {
  it("renders nothing when satisfaction is null", () => {
    const { container } = render(<RestraintViz satisfaction={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when satisfaction has empty arrays", () => {
    const { container } = render(
      <RestraintViz satisfaction={{ distance: [], contact: [], repulsive: [] }} />,
    );
    expect(container.innerHTML).toBe("");
  });

  describe("distance satisfaction", () => {
    const satisfaction: RestraintSatisfaction = {
      distance: [
        {
          chain_i: "A", residue_i: 10, atom_i: "CA",
          chain_j: "B", residue_j: 25, atom_j: "CB",
          target_distance: 8.0, actual_distance: 7.5, satisfied: true,
        },
        {
          chain_i: "A", residue_i: 15, atom_i: "CA",
          chain_j: "B", residue_j: 30, atom_j: "CA",
          target_distance: 5.0, actual_distance: 12.3, satisfied: false,
        },
      ],
    };

    it("renders distance rows with residue labels", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("A:10:CA")).toBeInTheDocument();
      expect(screen.getByText("B:25:CB")).toBeInTheDocument();
      expect(screen.getByText("A:15:CA")).toBeInTheDocument();
      expect(screen.getByText("B:30:CA")).toBeInTheDocument();
    });

    it("shows actual and target distances with 1 decimal", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      // Satisfied row: 7.5 / 8.0
      expect(screen.getByText("7.5")).toBeInTheDocument();
      expect(screen.getByText(/8\.0 A/)).toBeInTheDocument();

      // Unsatisfied row: 12.3 / 5.0
      expect(screen.getByText("12.3")).toBeInTheDocument();
      expect(screen.getByText(/5\.0 A/)).toBeInTheDocument();
    });

    it("displays summary badge with counts", () => {
      render(<RestraintViz satisfaction={satisfaction} />);
      expect(screen.getByText("1/2 satisfied")).toBeInTheDocument();
    });
  });

  describe("contact satisfaction", () => {
    const satisfaction: RestraintSatisfaction = {
      contact: [
        {
          chain_i: "A", residue_i: 5,
          closest_candidate_chain: "B", closest_candidate_residue: 20,
          threshold: 8.0, actual_distance: 6.2, satisfied: true,
        },
      ],
    };

    it("renders contact row with nearest candidate", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("A:5")).toBeInTheDocument();
      expect(screen.getByText("nearest")).toBeInTheDocument();
      expect(screen.getByText("B:20")).toBeInTheDocument();
    });

    it("shows actual distance and threshold", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("6.2")).toBeInTheDocument();
      expect(screen.getByText(/8\.0 A/)).toBeInTheDocument();
    });

    it("displays 1/1 satisfied in badge", () => {
      render(<RestraintViz satisfaction={satisfaction} />);
      expect(screen.getByText("1/1 satisfied")).toBeInTheDocument();
    });
  });

  describe("repulsive satisfaction", () => {
    const satisfaction: RestraintSatisfaction = {
      repulsive: [
        {
          chain_i: "A", residue_i: 1, chain_j: "B", residue_j: 1,
          min_distance: 15.0, actual_distance: 10.2, satisfied: false,
        },
      ],
    };

    it("renders repulsive row with residue labels", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("A:1")).toBeInTheDocument();
      expect(screen.getByText("B:1")).toBeInTheDocument();
    });

    it("shows actual distance and min threshold", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("10.2")).toBeInTheDocument();
      expect(screen.getByText(/min/)).toBeInTheDocument();
      expect(screen.getByText(/15\.0 A/)).toBeInTheDocument();
    });

    it("displays 0/1 satisfied for unsatisfied repulsive", () => {
      render(<RestraintViz satisfaction={satisfaction} />);
      expect(screen.getByText("0/1 satisfied")).toBeInTheDocument();
    });
  });

  describe("mixed restraint types", () => {
    const satisfaction: RestraintSatisfaction = {
      distance: [
        {
          chain_i: "A", residue_i: 10, atom_i: "CA",
          chain_j: "B", residue_j: 25, atom_j: "CA",
          target_distance: 8.0, actual_distance: 7.5, satisfied: true,
        },
      ],
      contact: [
        {
          chain_i: "A", residue_i: 5,
          closest_candidate_chain: "B", closest_candidate_residue: 20,
          threshold: 8.0, actual_distance: 6.2, satisfied: true,
        },
      ],
      repulsive: [
        {
          chain_i: "A", residue_i: 1, chain_j: "B", residue_j: 1,
          min_distance: 15.0, actual_distance: 20.0, satisfied: true,
        },
      ],
    };

    it("renders all section headers", () => {
      render(<RestraintViz satisfaction={satisfaction} />);

      expect(screen.getByText("Distance")).toBeInTheDocument();
      expect(screen.getByText("Contact")).toBeInTheDocument();
      expect(screen.getByText("Repulsive")).toBeInTheDocument();
    });

    it("shows all 3 satisfied", () => {
      render(<RestraintViz satisfaction={satisfaction} />);
      expect(screen.getByText("3/3 satisfied")).toBeInTheDocument();
    });
  });
});
