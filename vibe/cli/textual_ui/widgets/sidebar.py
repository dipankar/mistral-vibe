"""Compact sidebar summaries for todos, plans, and agents."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from vibe.core.planner import PlanState, PlanStepStatus
from vibe.cli.textual_ui.widgets.subagent_activity import SubagentPanelEntry


class SidebarItem(Static):
    """Simple sidebar item with label on left, value on right."""

    def __init__(self, label: str, value: str = "", *, icon: str = "") -> None:
        super().__init__()
        self.add_class("sidebar-item")
        self._label = label
        self._icon = icon
        self._value = value
        self._value_widget: Static | None = None

    def compose(self) -> ComposeResult:
        icon_part = f"{self._icon} " if self._icon else ""
        yield Static(f"{icon_part}{self._label}", classes="sidebar-item-label")
        self._value_widget = Static(self._value, classes="sidebar-item-value", markup=False)
        yield self._value_widget

    def update(self, value: str) -> None:
        self._value = value
        if self._value_widget:
            self._value_widget.update(value)


class SummaryCard(Static):
    """Base card with title + simple multiline body."""

    def __init__(self, title: str, icon: str = "") -> None:
        super().__init__()
        self.add_class("sidebar-card")
        self._title = title
        self._icon = icon
        self._body: Static | None = None

    def compose(self) -> ComposeResult:
        icon_part = f"{self._icon} " if self._icon else ""
        yield Static(f"{icon_part}{self._title}", classes="sidebar-card-title")
        self._body = Static("", classes="sidebar-card-body", markup=False)
        yield self._body

    def update_lines(self, *lines: str, placeholder: str = "—") -> None:
        text = "\n".join(line for line in lines if line) or placeholder
        if self._body:
            self._body.update(text)


class SubagentSummaryCard(SummaryCard):
    """Compact summary of active subagents for sidebar."""

    def __init__(self) -> None:
        super().__init__("Agents", "🤖")
        self._active_count = 0
        self._current_step: str | None = None
        self._update_display()

    def update_subagents(self, entries: list[SubagentPanelEntry]) -> None:
        """Update the card with active subagent entries."""
        self._active_count = len(entries)
        if entries:
            for entry in entries:
                if entry.status.value in ("in_progress", "pending"):
                    self._current_step = entry.title
                    break
            else:
                self._current_step = entries[0].title
        else:
            self._current_step = None
        self._update_display()

    def _update_display(self) -> None:
        if self._active_count == 0:
            self.update_lines("Idle")
        elif self._active_count == 1:
            step = self._current_step or "Working"
            if len(step) > 20:
                step = step[:17] + "..."
            self.update_lines("1 active", step)
        else:
            step = self._current_step or "Multiple"
            if len(step) > 20:
                step = step[:17] + "..."
            self.update_lines(f"{self._active_count} active", step)


@dataclass
class _TodoCounts:
    total: int = 0
    pending: int = 0
    in_progress: int = 0
    completed: int = 0


class TodoSummaryCard(SummaryCard):
    def __init__(self) -> None:
        super().__init__("Todos", "☑")
        self.update_todos([])

    def update_todos(self, todos_data: list[dict]) -> None:
        counts = _TodoCounts(total=len(todos_data))
        first_pending = ""
        for todo in todos_data:
            status = (todo.get("status") or "").lower()
            if status in {"pending", "todo"}:
                counts.pending += 1
                if not first_pending:
                    first_pending = todo.get("content", "").strip()
            elif status in {"in_progress", "in-progress"}:
                counts.in_progress += 1
                if not first_pending:
                    first_pending = (
                        todo.get("active_form") or todo.get("content", "")
                    ).strip()
            elif status == "completed":
                counts.completed += 1

        if counts.total == 0:
            self.update_lines("Empty")
            return

        secondary = first_pending[:25] if first_pending else f"{counts.pending} pending"
        if len(secondary) > 25:
            secondary = secondary[:22] + "..."
        self.update_lines(f"{counts.completed}/{counts.total}", secondary)


class PlanSummaryCard(SummaryCard):
    def __init__(self) -> None:
        super().__init__("Plan", "📋")
        self.update_plan(None)

    def update_plan(self, plan: PlanState | None) -> None:
        if not plan:
            self.update_lines("None")
            return

        total = len(plan.steps)
        completed = 0
        active = ""
        for step in plan.steps:
            step_status = getattr(step.status, "value", step.status)
            if step_status == PlanStepStatus.COMPLETED.value:
                completed += 1
            if not active and step_status in (
                PlanStepStatus.IN_PROGRESS.value,
                PlanStepStatus.NEEDS_DECISION.value,
            ):
                active = step.title

        status_text = getattr(plan.status, "value", str(plan.status)).lower()
        goal = plan.goal or ""
        if len(goal) > 25:
            goal = goal[:22] + "..."

        secondary = active[:25] if active else goal
        self.update_lines(f"{completed}/{total} ({status_text})", secondary)


class Sidebar(Static):
    """Compact sidebar with status items and summary cards."""

    BINDINGS = [
        ("?", "toggle_help", "Help"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.add_class("sidebar")
        self._tokens_item = SidebarItem("Tokens", "0%", icon="📊")
        self._path_item = SidebarItem("Path", "", icon="📁")
        self._todo_card = TodoSummaryCard()
        self._plan_card = PlanSummaryCard()
        self._subagent_card = SubagentSummaryCard()
        self._show_help = False

    def compose(self) -> ComposeResult:
        # Status items at top
        yield self._tokens_item
        yield self._path_item

        # Summary cards
        yield self._todo_card
        yield self._plan_card
        yield self._subagent_card

        # Help section (collapsed by default)
        self._help_section = Vertical(id="sidebar-help", classes="sidebar-help")
        with self._help_section:
            yield Static("Keybindings", classes="sidebar-help-title")
            yield Static("Esc - Interrupt", classes="sidebar-help-item")
            yield Static("Ctrl+O - Tools", classes="sidebar-help-item")
            yield Static("Ctrl+T - Todos", classes="sidebar-help-item")
            yield Static("Shift+Tab - Auto-approve", classes="sidebar-help-item")
            yield Static("? - Toggle help", classes="sidebar-help-item")

    def action_toggle_help(self) -> None:
        """Toggle help section visibility."""
        self._show_help = not self._show_help
        self._help_section.display = self._show_help

    def update_tokens(self, percentage: str) -> None:
        """Update the tokens indicator."""
        self._tokens_item.update(percentage)

    def update_path(self, path: str) -> None:
        """Update the path indicator."""
        display = path.split("/")[-1] if "/" in path else path
        if len(display) > 12:
            display = display[:10] + ".."
        self._path_item.update(display)

    def update_subagents(self, entries: list[SubagentPanelEntry]) -> None:
        """Update the subagent summary card."""
        self._subagent_card.update_subagents(entries)

    def update_plan(self, plan: PlanState | None) -> None:
        self._plan_card.update_plan(plan)

    def update_todos(self, todos_data: list[dict] | None) -> None:
        self._todo_card.update_todos(todos_data or [])

    def get_todo_container(self):
        return None

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        return
