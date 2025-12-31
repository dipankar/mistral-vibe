from __future__ import annotations

from vibe.cli.textual_ui.state_store import (
    BottomPanelMode,
    UIStateStore,
)
from vibe.core.planner import PlanRunStatus


def test_bottom_panel_updates_and_notifies() -> None:
    store = UIStateStore()
    observed: list[BottomPanelMode] = []
    store.subscribe("bottom_panel_mode", lambda s: observed.append(s.bottom_panel_mode))

    store.set_bottom_panel(BottomPanelMode.CONFIG)

    assert store.bottom_panel_mode is BottomPanelMode.CONFIG
    assert observed == [BottomPanelMode.CONFIG]


def test_plan_confirmation_flow() -> None:
    store = UIStateStore()
    observed: list[str | None] = []
    store.subscribe(
        "planner_pending_confirmation",
        lambda s: observed.append(s.planner.pending_confirmation),
    )

    store.request_plan_confirmation("build api")
    store.clear_plan_confirmation()

    assert observed == ["build api", None]
    assert store.planner.pending_confirmation is None


def test_plan_status_updates_goal() -> None:
    store = UIStateStore()
    observed: list[tuple[PlanRunStatus, str | None]] = []
    store.subscribe(
        "planner_status",
        lambda s: observed.append((s.planner.status, s.planner.goal)),
    )

    store.set_plan_status(PlanRunStatus.ACTIVE, "ship feature")

    assert observed == [(PlanRunStatus.ACTIVE, "ship feature")]
    assert store.planner.goal == "ship feature"


def test_collapsed_flags_toggle() -> None:
    store = UIStateStore()
    tool_flags: list[bool] = []
    todo_flags: list[bool] = []
    store.subscribe("tools_collapsed", lambda s: tool_flags.append(s.tools_collapsed))
    store.subscribe("todos_collapsed", lambda s: todo_flags.append(s.todos_collapsed))

    store.set_tools_collapsed(False)
    store.set_todos_collapsed(True)

    assert store.tools_collapsed is False
    assert store.todos_collapsed is True
    assert tool_flags == [False]
    assert todo_flags == [True]
