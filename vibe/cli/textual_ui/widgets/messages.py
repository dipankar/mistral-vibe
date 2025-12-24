from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Markdown, Static
from rich.markdown import Markdown as RichMarkdown
from textual.widgets._markdown import MarkdownStream

from vibe.core.types import (
    MemoryEntryEvent,
    PlanDecisionEvent,
    PlanStartedEvent,
    PlanStepUpdateEvent,
)
from vibe.core.planner import PlanState

class UserMessage(Static):
    def __init__(self, content: str, pending: bool = False) -> None:
        super().__init__()
        self.add_class("user-message")
        self._content = content
        self._pending = pending

    def compose(self) -> ComposeResult:
        with Horizontal(classes="user-message-container"):
            yield Static("> ", classes="user-message-prompt")
            yield Static(self._content, markup=False, classes="user-message-content")
            if self._pending:
                self.add_class("pending")

    async def set_pending(self, pending: bool) -> None:
        if pending == self._pending:
            return

        self._pending = pending

        if pending:
            self.add_class("pending")
            return

        self.remove_class("pending")


class AssistantMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("assistant-message")
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="assistant-message-container"):
            yield Static("● ", classes="assistant-message-dot")
            with Vertical(classes="assistant-message-content"):
                markdown = Markdown("")
                self._markdown = markdown
                yield markdown

    def _get_markdown(self) -> Markdown:
        if self._markdown is None:
            self._markdown = self.query_one(Markdown)
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        if self._stream is None:
            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    async def append_content(self, content: str) -> None:
        if not content:
            return

        self._content += content
        stream = self._ensure_stream()
        await stream.write(content)

    async def write_initial_content(self) -> None:
        if self._content:
            stream = self._ensure_stream()
            await stream.write(self._content)

    async def stop_stream(self) -> None:
        if self._stream is None:
            return

        await self._stream.stop()
        self._stream = None


class UserCommandMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("user-command-message")
        self._content = content

    def compose(self) -> ComposeResult:
        yield Markdown(self._content)


class InterruptMessage(Static):
    def __init__(self) -> None:
        super().__init__(
            "Interrupted · What should Vibe do instead?", classes="interrupt-message"
        )


class BashOutputMessage(Static):
    def __init__(self, command: str, cwd: str, output: str, exit_code: int) -> None:
        super().__init__()
        self.add_class("bash-output-message")
        self._command = command
        self._cwd = cwd
        self._output = output
        self._exit_code = exit_code

    def compose(self) -> ComposeResult:
        with Vertical(classes="bash-output-container"):
            with Horizontal(classes="bash-cwd-line"):
                yield Static(self._cwd, markup=False, classes="bash-cwd")
                yield Static("", classes="bash-cwd-spacer")
                if self._exit_code == 0:
                    yield Static("✓", classes="bash-exit-success")
                else:
                    yield Static("✗", classes="bash-exit-failure")
                    yield Static(f" ({self._exit_code})", classes="bash-exit-code")
            with Horizontal(classes="bash-command-line"):
                yield Static("> ", classes="bash-chevron")
                yield Static(self._command, markup=False, classes="bash-command")
                yield Static("", classes="bash-command-spacer")
            yield Static(self._output, markup=False, classes="bash-output")


class ErrorMessage(Static):
    def __init__(self, error: str, collapsed: bool = True) -> None:
        super().__init__(classes="error-message")
        self._error = error
        self.collapsed = collapsed

    def compose(self) -> ComposeResult:
        if self.collapsed:
            yield Static("Error. (ctrl+o to expand)", markup=False)
        else:
            yield Static(f"Error: {self._error}", markup=False)

    def set_collapsed(self, collapsed: bool) -> None:
        if self.collapsed == collapsed:
            return

        self.collapsed = collapsed
        self.remove_children()

        if self.collapsed:
            self.mount(Static("Error. (ctrl+o to expand)", markup=False))
        else:
            self.mount(Static(f"Error: {self._error}", markup=False))


class PlanStartedMessage(Static):
    def __init__(self, event: PlanStartedEvent) -> None:
        super().__init__()
        self.add_class("plan-message")
        self._event = event

    def compose(self) -> ComposeResult:
        steps_md = "\n".join(f"- {step}" for step in self._event.steps)
        summary = (
            f"### Plan started: {self._event.goal}\n\n"
            f"{self._event.summary}\n\n"
            f"#### Steps\n{steps_md}"
        )
        yield Markdown(summary)


class DecisionOptionCard(Static):
    """Single option card with radio/checkbox indicator, label, and description."""

    class Clicked(Message):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

    def __init__(
        self,
        label: str,
        description: str = "",
        multi_select: bool = False,
        selected: bool = False,
    ) -> None:
        super().__init__()
        self.add_class("decision-option-card")
        self._label = label
        self._description = description
        self._multi_select = multi_select
        self._selected = selected
        if selected:
            self.add_class("selected")

    def compose(self) -> ComposeResult:
        indicator = self._get_indicator()
        with Horizontal(classes="option-card-row"):
            yield Static(indicator, classes="option-indicator")
            with Vertical(classes="option-content"):
                yield Static(self._label, classes="option-label")
                if self._description:
                    yield Static(self._description, classes="option-description")

    def _get_indicator(self) -> str:
        if self._multi_select:
            return "☑" if self._selected else "☐"
        return "●" if self._selected else "○"

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if selected:
            self.add_class("selected")
        else:
            self.remove_class("selected")
        # Update indicator
        indicator = self.query_one(".option-indicator", Static)
        indicator.update(self._get_indicator())

    @property
    def selected(self) -> bool:
        return self._selected

    @property
    def label(self) -> str:
        return self._label

    async def on_click(self) -> None:
        self.post_message(self.Clicked(self._label))


class PlanDecisionMessage(Static):
    """Claude Code-style decision form with rich options, descriptions, and multi-select support."""

    class DecisionSelected(Message):
        def __init__(self, decision_id: str, selections: list[str]) -> None:
            super().__init__()
            self.decision_id = decision_id
            self.selections = selections

        @property
        def selection(self) -> str | None:
            """Backward compatibility."""
            return self.selections[0] if self.selections else None

    def __init__(self, event: PlanDecisionEvent) -> None:
        super().__init__()
        self.add_class("plan-message")
        self.add_class("plan-decision-form")
        self._event = event
        self._option_cards: dict[str, DecisionOptionCard] = {}
        self._other_input: Input | None = None
        self._other_card: DecisionOptionCard | None = None
        self._confirm_button: Button | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="decision-form-container"):
            # Header chip
            yield Static(self._event.header, classes="decision-header-chip")

            # Question
            yield Static(self._event.question, classes="decision-question")

            # Options container
            options_container = Vertical(classes="decision-options")
            yield options_container

            # Confirm button (shown when not resolved)
            if not self._event.resolved:
                yield Button("Confirm", id="decision-confirm", classes="decision-confirm-btn")

    async def on_mount(self) -> None:
        await self._render_options()

    async def update_event(self, event: PlanDecisionEvent) -> None:
        self._event = event
        await self._render_options()

    async def _render_options(self) -> None:
        options_container = self.query_one(".decision-options", Vertical)
        await options_container.remove_children()
        self._option_cards = {}
        self._other_input = None
        self._other_card = None

        # Update header and question
        header_widget = self.query_one(".decision-header-chip", Static)
        header_widget.update(self._event.header)

        question_widget = self.query_one(".decision-question", Static)
        question_widget.update(self._event.question)

        if self._event.resolved:
            # Show resolved state
            selections_text = ", ".join(self._event.selections) if self._event.selections else "None"
            resolved_msg = Static(
                f"✓ Selected: {selections_text}",
                classes="decision-resolved-status"
            )
            await options_container.mount(resolved_msg)

            # Hide confirm button
            try:
                confirm_btn = self.query_one("#decision-confirm", Button)
                confirm_btn.display = False
            except Exception:
                pass
            return

        # Render option cards
        for opt in self._event.options:
            is_selected = opt.label in self._event.selections
            card = DecisionOptionCard(
                label=opt.label,
                description=opt.description,
                multi_select=self._event.multi_select,
                selected=is_selected,
            )
            self._option_cards[opt.label] = card
            await options_container.mount(card)

        # Always add "Other" option with input
        other_container = Vertical(classes="other-option-container")
        await options_container.mount(other_container)

        self._other_card = DecisionOptionCard(
            label="Other...",
            description="Provide a custom response",
            multi_select=self._event.multi_select,
            selected=False,
        )
        await other_container.mount(self._other_card)

        self._other_input = Input(
            placeholder="Enter custom response...",
            classes="other-option-input",
        )
        self._other_input.display = False
        await other_container.mount(self._other_input)

        # Show confirm button
        try:
            confirm_btn = self.query_one("#decision-confirm", Button)
            confirm_btn.display = True
        except Exception:
            pass

    def on_decision_option_card_clicked(self, event: DecisionOptionCard.Clicked) -> None:
        """Handle option card clicks."""
        if self._event.resolved:
            return

        clicked_label = event.label

        # Handle "Other" option
        if clicked_label == "Other..." and self._other_card:
            if self._event.multi_select:
                # Toggle other card
                new_state = not self._other_card.selected
                self._other_card.set_selected(new_state)
                if self._other_input:
                    self._other_input.display = new_state
                    if new_state:
                        self._other_input.focus()
            else:
                # Single select - deselect all others
                for card in self._option_cards.values():
                    card.set_selected(False)
                self._other_card.set_selected(True)
                if self._other_input:
                    self._other_input.display = True
                    self._other_input.focus()
            return

        # Handle regular options
        if clicked_label in self._option_cards:
            card = self._option_cards[clicked_label]

            if self._event.multi_select:
                # Toggle selection
                card.set_selected(not card.selected)
            else:
                # Single select - deselect all others first
                for other_card in self._option_cards.values():
                    other_card.set_selected(False)
                if self._other_card:
                    self._other_card.set_selected(False)
                    if self._other_input:
                        self._other_input.display = False
                card.set_selected(True)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle confirm button press."""
        if event.button.id != "decision-confirm":
            return

        selections = self._get_current_selections()
        if selections:
            self.post_message(self.DecisionSelected(self._event.decision_id, selections))

    def _get_current_selections(self) -> list[str]:
        """Collect all selected options."""
        selections: list[str] = []

        # Check regular option cards
        for label, card in self._option_cards.items():
            if card.selected:
                selections.append(label)

        # Check "Other" option
        if self._other_card and self._other_card.selected and self._other_input:
            custom_value = self._other_input.value.strip()
            if custom_value:
                selections.append(custom_value)

        return selections


class SubagentStatusMessage(Static):
    def __init__(
        self,
        step_id: str,
        goal: str,
        owner: str | None,
        event: PlanStepUpdateEvent,
    ) -> None:
        super().__init__("", classes="plan-message subagent-status-message")
        self._step_id = step_id
        self._goal = goal
        self._owner = owner
        self._event = event

    def on_mount(self) -> None:
        self._update_content()

    def update_status(
        self,
        event: PlanStepUpdateEvent,
        owner: str | None,
    ) -> None:
        self._event = event
        self._owner = owner
        self._update_content()

    def _update_content(self) -> None:
        owner = self._owner or "planner"
        mode = self._event.mode or "code"
        notes = f"\n\nNotes:\n{self._event.notes}" if self._event.notes else ""
        body = (
            f"### Subagent `{self._step_id}`\n"
            f"- Goal: {self._goal}\n"
            f"- Title: {self._event.title}\n"
            f"- Owner: {owner}\n"
            f"- Mode: {mode}\n"
            f"- Status: {self._event.status}{notes}"
        )
        self.update(RichMarkdown(body))


class MemoryUpdateMessage(Static):
    def __init__(self, event: MemoryEntryEvent) -> None:
        super().__init__()
        self.add_class("memory-update-message")
        self._event = event

    def compose(self) -> ComposeResult:
        hints = (
            "\n".join(f"- {hint}" for hint in self._event.task_hints if hint.strip())
            if self._event.task_hints
            else None
        )
        hints_block = f"\n\n**Task hints**\n{hints}" if hints else ""
        token_text = (
            f"~{self._event.token_count:,} tokens"
            if self._event.token_count
            else "token snapshot unavailable"
        )
        body = (
            f"### Session memory updated (entry {self._event.entry_index})\n"
            f"- {token_text}\n\n"
            f"{self._event.summary.strip()}{hints_block}\n\n"
            "Use `/memory` to review all captured context."
        )
        yield Markdown(body)


class ThinkingPlanMessage(Static):
    def __init__(self, plan: PlanState) -> None:
        super().__init__()
        self.add_class("plan-message")
        self.add_class("thinking-plan-message")
        self._plan = plan

    def compose(self) -> ComposeResult:
        steps = "\n".join(
            f"{idx}. {step.title} — {step.status.value.replace('_', ' ').title()}"
            for idx, step in enumerate(self._plan.steps, start=1)
        )
        decisions = ""
        if self._plan.decisions:
            decision_lines = []
            for decision in self._plan.decisions:
                # Use option_labels() method for backward compat with new DecisionOption format
                opts = ", ".join(decision.option_labels()) if decision.options else "freeform"
                decision_lines.append(f"- {decision.decision_id}: {decision.question} ({opts})")
            decisions = "\n\n**Decisions to watch**\n" + "\n".join(decision_lines)
        summary = (
            f"### Thinking Outline: {self._plan.goal}\n"
            f"{self._plan.summarize()}\n\n"
            f"**Steps**\n{steps}{decisions}"
        )
        yield Markdown(summary)
