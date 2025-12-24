"""Plan Panel widget for displaying plan steps in the sidebar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.status_icons import (
    StatusIcon,
    get_plan_status_icon,
    get_step_status_icon,
    render_progress_bar,
)
from vibe.core.planner import PlanState, PlanStep, PlanStepStatus, PlanRunStatus


class PlanStepItem(Static):
    """Individual step item in the plan panel."""

    class Clicked(Message):
        def __init__(self, step_id: str) -> None:
            super().__init__()
            self.step_id = step_id

    def __init__(self, step: PlanStep, index: int) -> None:
        super().__init__()
        self.add_class("plan-step-item")
        self._step = step
        self._index = index
        self._update_classes()

    def _update_classes(self) -> None:
        # Remove all status classes first
        for status in PlanStepStatus:
            self.remove_class(f"step-{status.value}")
        # Add current status class
        self.add_class(f"step-{self._step.status.value}")

    def compose(self) -> ComposeResult:
        icon = self._get_status_icon()
        mode_badge = f" [{self._step.mode or 'code'}]" if self._step.mode else ""
        yield Static(
            f"{icon} {self._index}. {self._step.title}{mode_badge}",
            classes="step-content"
        )

    def _get_status_icon(self) -> str:
        return get_step_status_icon(self._step.status.value)

    def update_step(self, step: PlanStep) -> None:
        self._step = step
        self._update_classes()
        # Re-render content
        content = self.query_one(".step-content", Static)
        icon = self._get_status_icon()
        mode_badge = f" [{self._step.mode or 'code'}]" if self._step.mode else ""
        content.update(f"{icon} {self._index}. {self._step.title}{mode_badge}")

    async def on_click(self) -> None:
        self.post_message(self.Clicked(self._step.step_id))


class SubagentIndicator(Static):
    """Shows which subagent/step is currently active."""

    active_step: reactive[str | None] = reactive(None)
    active_mode: reactive[str | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self.add_class("subagent-indicator")

    def compose(self) -> ComposeResult:
        yield Static("", id="indicator-content")

    def watch_active_step(self, step: str | None) -> None:
        self._update_display()

    def watch_active_mode(self, mode: str | None) -> None:
        self._update_display()

    def _update_display(self) -> None:
        try:
            content = self.query_one("#indicator-content", Static)
            if self.active_step:
                mode_text = f" ({self.active_mode})" if self.active_mode else ""
                content.update(f"{StatusIcon.IN_PROGRESS} Running: {self.active_step}{mode_text}")
                self.display = True
            else:
                content.update("")
                self.display = False
        except Exception:
            pass

    def set_active(self, step_id: str | None, mode: str | None = None) -> None:
        self.active_step = step_id
        self.active_mode = mode

    def clear(self) -> None:
        self.active_step = None
        self.active_mode = None


class PlanPanel(Static):
    """Panel showing current plan with steps and progress."""

    plan: reactive[PlanState | None] = reactive(None)
    collapsed: reactive[bool] = reactive(False)

    class StepClicked(Message):
        def __init__(self, step_id: str) -> None:
            super().__init__()
            self.step_id = step_id

    class CollapseToggled(Message):
        def __init__(self, collapsed: bool) -> None:
            super().__init__()
            self.collapsed = collapsed

    def __init__(self) -> None:
        super().__init__()
        self.add_class("plan-panel")
        self._step_items: dict[str, PlanStepItem] = {}
        self._subagent_indicator: SubagentIndicator | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="plan-panel-container"):
            # Header with collapse toggle
            yield Static("Plan", id="plan-panel-header", classes="panel-header")

            # Subagent activity indicator
            self._subagent_indicator = SubagentIndicator()
            yield self._subagent_indicator

            # Plan content area
            with Vertical(id="plan-content"):
                # Goal display
                yield Static("", id="plan-goal", classes="plan-goal")

                # Progress bar
                yield Static("", id="plan-progress", classes="plan-progress")

                # Steps list
                with VerticalScroll(id="plan-steps-scroll"):
                    yield Vertical(id="plan-steps-list")

                # No plan message
                yield Static(
                    "[dim]No active plan[/dim]\n/plan <goal> to start",
                    id="no-plan-message",
                    classes="no-plan-message"
                )

    async def on_mount(self) -> None:
        await self._render_plan()

    def watch_plan(self, plan: PlanState | None) -> None:
        self.call_later(self._render_plan)

    def watch_collapsed(self, collapsed: bool) -> None:
        try:
            content = self.query_one("#plan-content", Vertical)
            content.display = not collapsed
            header = self.query_one("#plan-panel-header", Static)
            icon = "▶" if collapsed else "▼"
            header.update(f"{icon} Plan")
        except Exception:
            pass

    async def _render_plan(self) -> None:
        try:
            goal_widget = self.query_one("#plan-goal", Static)
            progress_widget = self.query_one("#plan-progress", Static)
            steps_list = self.query_one("#plan-steps-list", Vertical)
            no_plan_msg = self.query_one("#no-plan-message", Static)
        except Exception:
            return

        if not self.plan:
            goal_widget.update("")
            progress_widget.update("")
            await steps_list.remove_children()
            self._step_items = {}
            no_plan_msg.display = True
            if self._subagent_indicator:
                self._subagent_indicator.clear()
            return

        no_plan_msg.display = False

        # Update goal
        status_icon = get_plan_status_icon(self.plan.status.value)
        goal_widget.update(f"{status_icon} {self.plan.goal[:40]}...")

        # Update progress
        completed = sum(1 for s in self.plan.steps if s.status == PlanStepStatus.COMPLETED)
        total = len(self.plan.steps)
        progress_widget.update(render_progress_bar(completed, total))

        # Update steps
        await steps_list.remove_children()
        self._step_items = {}

        for idx, step in enumerate(self.plan.steps, start=1):
            item = PlanStepItem(step, idx)
            self._step_items[step.step_id] = item
            await steps_list.mount(item)

    def update_plan(self, plan: PlanState | None) -> None:
        self.plan = plan

    def update_step(self, step_id: str, step: PlanStep) -> None:
        if step_id in self._step_items:
            self._step_items[step_id].update_step(step)
        # Update progress bar
        if self.plan:
            self.call_later(self._update_progress)

    async def _update_progress(self) -> None:
        if not self.plan:
            return
        try:
            progress_widget = self.query_one("#plan-progress", Static)
            completed = sum(1 for s in self.plan.steps if s.status == PlanStepStatus.COMPLETED)
            total = len(self.plan.steps)
            progress_widget.update(render_progress_bar(completed, total))
        except Exception:
            pass

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        """Highlight the currently executing step."""
        if self._subagent_indicator:
            self._subagent_indicator.set_active(step_id, mode)

        # Update step item styling
        for sid, item in self._step_items.items():
            if sid == step_id:
                item.add_class("active-step")
            else:
                item.remove_class("active-step")

    def on_plan_step_item_clicked(self, event: PlanStepItem.Clicked) -> None:
        self.post_message(self.StepClicked(event.step_id))

    async def on_click(self, event) -> None:
        # Check if header was clicked
        try:
            header = self.query_one("#plan-panel-header", Static)
            if header in event.widget.ancestors_with_self:
                self.collapsed = not self.collapsed
                self.post_message(self.CollapseToggled(self.collapsed))
        except Exception:
            pass
