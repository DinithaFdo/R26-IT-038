import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfirmActionDialog } from "@/components/shared/confirm-action-dialog";

describe("ConfirmActionDialog", () => {
  it("does not call onConfirm until the destructive action button is explicitly clicked", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <ConfirmActionDialog
        trigger={<button>Delete prediction</button>}
        title="Delete prediction?"
        description="This removes the owner-scoped prediction record."
        actionLabel="Delete"
        destructive
        onConfirm={onConfirm}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Delete prediction" }));
    expect(screen.getByText("Delete prediction?")).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("does not call onConfirm when cancelled", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <ConfirmActionDialog
        trigger={<button>Rerun prediction</button>}
        title="Rerun prediction?"
        description="This creates a new prediction if source audio is still available."
        actionLabel="Rerun"
        onConfirm={onConfirm}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Rerun prediction" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
