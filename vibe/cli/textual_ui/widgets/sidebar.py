"""Compact sidebar summaries for todos, plans, and memory."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from vibe.core.memory import MemoryEntry
from vibe.core.planner import PlanState, PlanStepStatus


class SummaryCard(Static):
    """Base card with title + simple multiline body."""

    def __init__(self, title: str, icon: str) -> None:
        super().__init__()
        self.add_class("summary-card")
        self._title = title
        self._icon = icon
        self._body: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            f"{self._icon} {self._title}",
            classes="summary-card-title",
        )
        self._body = Static("", classes="summary-card-body", markup=False)
        yield self._body

    def update_lines(self, *lines: str, placeholder: str = "—") -> None:
        if not self._body:
            return
        text = "\n".join(line for line in lines if line) or placeholder
        self._body.update(text)


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
            self.update_lines("No tasks yet")
            return

        summary = f"{counts.completed}/{counts.total} done · {counts.in_progress} running"
        secondary = f"Next: {first_pending[:60]}" if first_pending else ""
        self.update_lines(summary, secondary or f"Pending: {counts.pending}")


class PlanSummaryCard(SummaryCard):
    def __init__(self) -> None:
        super().__init__("Planning", "🗂")
        self.update_plan(None)

    def update_plan(self, plan: PlanState | None) -> None:
        if not plan:
            self.update_lines("No active plan")
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
        goal = plan.goal[:60] if plan.goal else ""
        decisions = sum(1 for d in plan.decisions if not d.resolved)

        status_text = getattr(plan.status, "value", str(plan.status)).lower()
        summary = f"{completed}/{total} steps · {status_text}"
        secondary = f"Focus: {active}" if active else f"Goal: {goal}"
        tertiary = f"Decisions: {decisions}" if decisions else ""
        self.update_lines(summary, secondary, tertiary)


class MemorySummaryCard(SummaryCard):
    def __init__(self) -> None:
        super().__init__("Memory", "🧠")
        self.update_entries([])

    def update_entries(self, entries: list[MemoryEntry]) -> None:
        total = len(entries)
        if total == 0:
            self.update_lines("No summaries captured yet")
            return
        last = entries[-1]
        snippet = (last.summary or "").strip()
        snippet = (snippet[:80] + "…") if len(snippet) > 80 else snippet
        tokens = getattr(last, "token_count", 0)
        self.update_lines(f"{total} entries · last {tokens} tokens", snippet or "Latest summary pending")


class Sidebar(Static):
    """Compact sidebar with summary cards."""

    def __init__(self) -> None:
        super().__init__()
        self.add_class("summary-sidebar")
        self._todo_card = TodoSummaryCard()
        self._plan_card = PlanSummaryCard()
        self._memory_card = MemorySummaryCard()

    def compose(self) -> ComposeResult:
        with Vertical(id="summary-card-stack"):
            yield self._todo_card
            yield self._plan_card
            yield self._memory_card

    # Compatibility with legacy hooks
    def get_todo_container(self):
        return None

    def update_plan(self, plan: PlanState | None) -> None:
        self._plan_card.update_plan(plan)

    def update_todos(self, todos_data: list[dict] | None) -> None:
        self._todo_card.update_todos(todos_data or [])

    def update_memory_summary(self, entries: list[MemoryEntry] | None) -> None:
        self._memory_card.update_entries(entries or [])

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        # Legacy no-op; summary card highlights overall plan only.
        return
