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
        self._option_buttons: list[Button] = []
        self._cursor_index: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(classes="plan-confirmation-banner approval-banner"):
            with Horizontal(classes="plan-confirmation-heading"):
                yield Static("Plan with sub-agents?", classes="plan-confirmation-title")
                yield Static("(auto)", classes="plan-confirmation-mode")

            self._message_label = Static(
                "",
                classes="plan-confirmation-text",
            )
            yield self._message_label

            with Horizontal(classes="approval-options plan-confirmation-options"):
                yes_btn = Button(
                    "Yes",
                    id="plan-confirm-yes",
                    classes="approval-option approval-option-yes plan-confirmation-option",
                )
                no_btn = Button(
                    "No",
                    id="plan-confirm-no",
                    classes="approval-option approval-option-no plan-confirmation-option",
                )
                self._option_buttons = [yes_btn, no_btn]
                yield yes_btn
                yield no_btn
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
                f"Run planning before answering?\n\"{preview}\""
            )
        self._cursor_index = 0
        self._update_option_states()
        self.display = True

    def hide_prompt(self) -> None:
        """Hide the confirmation banner."""
        self.display = False
        self._goal = None
        self._option_buttons = []
        self._cursor_index = 0

    def _update_option_states(self) -> None:
        for idx, button in enumerate(self._option_buttons):
            is_selected = idx == self._cursor_index
            button.set_class(is_selected, "approval-option-selected")
            button.set_class(is_selected, "plan-confirmation-option-selected")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plan-confirm-yes":
            self.post_message(self.Accepted())
        elif event.button.id == "plan-confirm-no":
            self.post_message(self.Declined())

    async def on_key(self, event) -> None:
        if not self.display or not self._option_buttons:
            return
        key = event.key.lower()
        if key in {"tab", "shift+tab"}:
            event.stop()
            direction = -1 if "shift" in key else 1
            self._cursor_index = (self._cursor_index + direction) % len(self._option_buttons)
            self._update_option_states()
            return
        if key in {"left", "right"}:
            event.stop()
            direction = -1 if key == "left" else 1
            self._cursor_index = (self._cursor_index + direction) % len(self._option_buttons)
            self._update_option_states()
            return
        if key in {"enter", "space"}:
            event.stop()
            button = self._option_buttons[self._cursor_index]
            await button.press()
