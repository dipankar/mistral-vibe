from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Callable, DefaultDict, Hashable

from vibe.core.planner import PlanRunStatus


class BottomPanelMode(StrEnum):
    INPUT = auto()
    CONFIG = auto()
    APPROVAL = auto()


@dataclass
class PlannerState:
    status: PlanRunStatus = PlanRunStatus.IDLE
    goal: str | None = None
    pending_confirmation: str | None = None


@dataclass
class UIStateStore:
    bottom_panel_mode: BottomPanelMode = BottomPanelMode.INPUT
    tools_collapsed: bool = True
    todos_collapsed: bool = False
    planner: PlannerState = field(default_factory=PlannerState)

    def __post_init__(self) -> None:
        self._subscribers: DefaultDict[
            Hashable, list[Callable[[UIStateStore], None]]
        ] = defaultdict(list)

    def subscribe(
        self, key: Hashable, callback: Callable[[UIStateStore], None]
    ) -> None:
        self._subscribers[key].append(callback)

    def _notify(self, key: Hashable) -> None:
        for callback in self._subscribers.get(key, []):
            callback(self)

    def set_bottom_panel(self, mode: BottomPanelMode) -> None:
        if self.bottom_panel_mode != mode:
            self.bottom_panel_mode = mode
            self._notify("bottom_panel_mode")

    def set_tools_collapsed(self, collapsed: bool) -> None:
        if self.tools_collapsed != collapsed:
            self.tools_collapsed = collapsed
            self._notify("tools_collapsed")

    def set_todos_collapsed(self, collapsed: bool) -> None:
        if self.todos_collapsed != collapsed:
            self.todos_collapsed = collapsed
            self._notify("todos_collapsed")

    def set_plan_status(self, status: PlanRunStatus, goal: str | None = None) -> None:
        if self.planner.status != status or self.planner.goal != goal:
            self.planner.status = status
            self.planner.goal = goal
            self._notify("planner_status")

    def request_plan_confirmation(self, goal: str | None) -> None:
        if self.planner.pending_confirmation != goal:
            self.planner.pending_confirmation = goal
            self._notify("planner_pending_confirmation")

    def clear_plan_confirmation(self) -> None:
        if self.planner.pending_confirmation is not None:
            self.planner.pending_confirmation = None
            self._notify("planner_pending_confirmation")
