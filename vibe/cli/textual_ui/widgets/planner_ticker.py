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
        goal = new_state.goal or "Plan idle"
        status = new_state.status.lower()
        progress = (
            f"{new_state.completed_steps}/{max(new_state.total_steps, 1)} done"
            if new_state.total_steps
            else "0/0 done"
        )
        plan_segment = f"{goal} · {status} · {progress}"

        agent_segment = (
            f"agents {new_state.active_steps} run/{new_state.pending_steps} wait"
        )
        decisions_segment = (
            f"decisions {new_state.pending_decisions} pending"
            if new_state.pending_decisions
            else "decisions clear"
        )

        context_segment = ""
        if new_state.max_tokens > 0:
            warn = " mem⚠" if new_state.memory_warning else ""
            context_segment = (
                f"ctx {new_state.context_percentage}% "
                f"({new_state.context_tokens:,}/{new_state.max_tokens:,}){warn}"
            )

        thinking_segment = (
            "thinking on" if new_state.thinking_mode else "thinking off"
        )
        rate_segment = "rate ⚠" if new_state.rate_limited else "rate ok"

        segments = [
            plan_segment,
            agent_segment,
            decisions_segment,
            context_segment,
            thinking_segment,
            rate_segment,
        ]

        filtered = [segment for segment in segments if segment]
        self.update(" · ".join(filtered))
