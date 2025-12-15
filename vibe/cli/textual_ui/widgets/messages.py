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


class PlanDecisionMessage(Static):
    class DecisionSelected(Message):
        def __init__(self, decision_id: str, selection: str) -> None:
            super().__init__()
            self.decision_id = decision_id
            self.selection = selection

    def __init__(self, event: PlanDecisionEvent) -> None:
        super().__init__()
        self.add_class("plan-message")
        self.add_class("plan-decision-update")
        self._event = event
        self._markdown: Markdown | None = None
        self._controls: Vertical | None = None
        self._input: Input | None = None
        self._submit_button: Button | None = None
        self._option_buttons: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            self._markdown = Markdown("")
            yield self._markdown
            controls = Vertical(classes="plan-decision-controls")
            self._controls = controls
            yield controls

    async def on_mount(self) -> None:
        await self._render()

    async def update_event(self, event: PlanDecisionEvent) -> None:
        self._event = event
        await self._render()

    async def _render(self) -> None:
        if not self._markdown or not self._controls:
            return

        options_list = (
            "\n".join(f"- `{option}`" for option in self._event.options)
            if self._event.options
            else "Freeform response"
        )
        status_line = (
            f"Selection: `{self._event.selection}`"
            if self._event.resolved and self._event.selection
            else "Awaiting selection"
        )
        body = (
            f"**Decision {self._event.decision_id}**\n\n"
            f"{self._event.question}\n\n"
            f"Options:\n{options_list}\n\n"
            f"{status_line}"
        )
        self._markdown.update(body)

        await self._controls.remove_children()
        self._input = None
        self._submit_button = None
        self._option_buttons = {}

        if self._event.resolved:
            await self._controls.mount(
                Static("Decision captured. No further action required.", classes="plan-decision-status")
            )
            return

        if self._event.options:
            container = Horizontal(classes="plan-decision-buttons")
            await self._controls.mount(container)
            for index, option in enumerate(self._event.options):
                button_id = f"decision-option-{self._event.decision_id}-{index}"
                button = Button(
                    option,
                    id=button_id,
                    classes="plan-decision-button",
                )
                self._option_buttons[button_id] = option
                await container.mount(button)
            return

        self._input = Input(placeholder="Enter your decision…", classes="plan-decision-input")
        await self._controls.mount(self._input)
        submit_id = f"decision-submit-{self._event.decision_id}"
        self._submit_button = Button(
            "Submit decision",
            id=submit_id,
            classes="plan-decision-button",
        )
        await self._controls.mount(self._submit_button)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button
        button_id = button.id or ""
        if button_id in self._option_buttons:
            selection = self._option_buttons[button_id]
            self.post_message(self.DecisionSelected(self._event.decision_id, selection))
            return

        if self._submit_button and button_id == self._submit_button.id and self._input:
            selection = self._input.value.strip()
            if selection:
                self.post_message(self.DecisionSelected(self._event.decision_id, selection))


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
                opts = ", ".join(decision.options) if decision.options else "freeform"
                decision_lines.append(f"- {decision.decision_id}: {decision.question} ({opts})")
            decisions = "\n\n**Decisions to watch**\n" + "\n".join(decision_lines)
        summary = (
            f"### Thinking Outline: {self._plan.goal}\n"
            f"{self._plan.summarize()}\n\n"
            f"**Steps**\n{steps}{decisions}"
        )
        yield Markdown(summary)
