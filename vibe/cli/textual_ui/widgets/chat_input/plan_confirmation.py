from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static


class PlanConfirmationPrompt(Widget):
    """Banner that prompts the user to approve planner usage."""

    class Accepted(Message):
        """Dispatched when the user approves running the planner."""

    class Declined(Message):
        """Dispatched when the user declines running the planner."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id or "plan-confirmation-prompt")
        self._goal: str | None = None
        self._message_label: Static | None = None
        self.display = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="plan-confirmation-banner"):
            self._message_label = Static(
                "",
                classes="plan-confirmation-text",
            )
            yield self._message_label

            with Vertical(classes="plan-confirmation-buttons"):
                yield Button(
                    "Yes, plan it",
                    id="plan-confirm-yes",
                    classes=(
                        "plan-confirmation-button plan-confirmation-yes "
                        "approval-option approval-option-yes"
                    ),
                )
                yield Button(
                    "No, continue",
                    id="plan-confirm-no",
                    classes=(
                        "plan-confirmation-button plan-confirmation-no "
                        "approval-option approval-option-no"
                    ),
                )
                yield Static(
                    "Tab/Shift+Tab to move · Enter selects · Esc cancels",
                    classes="approval-help plan-confirmation-help",
                )

    def show_prompt(self, goal: str) -> None:
        """Display the confirmation banner for the provided goal."""
        self._goal = goal
        if self._message_label:
            preview = goal.strip() or "this request"
            if len(preview) > 120:
                preview = f"{preview[:117]}..."
            self._message_label.update(
                f"Plan with sub-agents before answering?\n\"{preview}\""
            )
        self.display = True

    def hide_prompt(self) -> None:
        """Hide the confirmation banner."""
        self.display = False
        self._goal = None

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plan-confirm-yes":
            self.post_message(self.Accepted())
        elif event.button.id == "plan-confirm-no":
            self.post_message(self.Declined())
