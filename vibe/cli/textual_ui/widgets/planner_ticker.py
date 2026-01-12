from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.reactive import reactive
from textual.widgets import Static


@dataclass(slots=True)
class PlannerTickerState:
    goal: str | None = None
    status: str = "Idle"
    total_steps: int = 0
    completed_steps: int = 0
    active_steps: int = 0
    pending_steps: int = 0
    pending_decisions: int = 0
    thinking_mode: bool = False
    rate_limited: bool = False
    context_tokens: int = 0
    max_tokens: int = 0
    context_percentage: int = 0
    memory_warning: bool = False


class PlannerTicker(Static):
    state = reactive(PlannerTickerState())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)

    def watch_state(self, new_state: PlannerTickerState) -> None:
        # Build a compact status line
        parts = []

        # Plan status and progress
        if new_state.total_steps > 0:
            status = new_state.status.lower() or "active"
            progress = f"{new_state.completed_steps}/{new_state.total_steps}"
            goal = new_state.goal
            if goal and len(goal) > 30:
                goal = goal[:27] + "..."
            goal_part = f" {goal}" if goal else ""
            parts.append(f"[{progress}]{goal_part} ({status})")
        else:
            parts.append("No plan")

        # Pending decisions
        if new_state.pending_decisions > 0:
            parts.append(f"⚡ {new_state.pending_decisions} decision(s)")

        # Context warning
        if new_state.memory_warning:
            parts.append("⚠️ memory low")

        # Rate limited
        if new_state.rate_limited:
            parts.append("⚠️ rate limited")

        self.update(" ".join(parts))
