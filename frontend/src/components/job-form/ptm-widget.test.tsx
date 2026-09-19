import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PTMWidget } from "./ptm-widget";

describe("PTMWidget", () => {
  it("starts expanded for an imported entity with modifications", () => {
    render(
      <PTMWidget
        sequence="AS"
        modifications={[{ type: "CCD_SEP", position: 2 }]}
        onModificationsChange={vi.fn()}
      />,
    );

    expect(screen.getByText("CCD_SEP @ S2")).toBeInTheDocument();
  });

  it("allows an imported entity with modifications to be collapsed", async () => {
    const user = userEvent.setup();
    render(
      <PTMWidget
        sequence="AS"
        modifications={[{ type: "CCD_SEP", position: 2 }]}
        onModificationsChange={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /modifications/i }));

    expect(screen.queryByText("CCD_SEP @ S2")).not.toBeInTheDocument();
  });
});
