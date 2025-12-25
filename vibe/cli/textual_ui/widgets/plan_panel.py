"""Plan Panel widget for displaying plan steps in the sidebar."""

from __future__ import annotations

from time import monotonic

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import QueryError
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


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


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
        self._start_time: float | None = None
        self._duration: float | None = None
        self._update_classes()

    def _update_classes(self) -> None:
        # Remove all status classes first
        for status in PlanStepStatus:
            self.remove_class(f"step-{status.value}")
        # Add current status class
        self.add_class(f"step-{self._step.status.value}")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="step-row"):
            yield Static(self._build_content(), classes="step-content", markup=True)
            yield Static("", classes="step-duration", id=f"duration-{self._step.step_id}")

    def _get_status_icon(self) -> str:
        return get_step_status_icon(self._step.status.value)

    def _build_content(self) -> str:
        icon = self._get_status_icon()
        # Truncate title if too long
        title = self._step.title[:25] + "…" if len(self._step.title) > 25 else self._step.title
        mode_badge = f" [dim][{self._step.mode or 'code'}][/dim]" if self._step.mode else ""
        return f"{icon} {self._index}. {title}{mode_badge}"

    def _build_duration(self) -> str:
        if self._duration is not None:
            return f"[dim]{format_duration(self._duration)}[/dim]"
        if self._start_time is not None:
            elapsed = monotonic() - self._start_time
            return f"[yellow]{format_duration(elapsed)}[/yellow]"
        return ""

    def update_step(self, step: PlanStep) -> None:
        old_status = self._step.status
        self._step = step
        self._update_classes()

        # Track timing
        if step.status == PlanStepStatus.IN_PROGRESS and old_status != PlanStepStatus.IN_PROGRESS:
            self._start_time = monotonic()
            self._duration = None
        elif step.status in (PlanStepStatus.COMPLETED, PlanStepStatus.BLOCKED) and self._start_time:
            self._duration = monotonic() - self._start_time
            self._start_time = None

        # Re-render content
        try:
            content = self.query_one(".step-content", Static)
            content.update(self._build_content())
            duration = self.query_one(f"#duration-{self._step.step_id}", Static)
            duration.update(self._build_duration())
        except QueryError:
            pass

    def get_duration(self) -> float | None:
        """Get the step duration if completed."""
        return self._duration

    async def on_click(self) -> None:
        self.post_message(self.Clicked(self._step.step_id))


class SubagentIndicator(Static):
    """Shows which subagent/step is currently active with progress and timing."""

    active_step: reactive[str | None] = reactive(None)
    active_mode: reactive[str | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self.add_class("subagent-indicator")
        self._step_index: int = 0
        self._total_steps: int = 0
        self._step_title: str = ""
        self._start_time: float | None = None
        self._timer = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="indicator-container"):
            yield Static("", id="indicator-progress", classes="indicator-progress")
            yield Static("", id="indicator-content", classes="indicator-content")
            yield Static("", id="indicator-timer", classes="indicator-timer")

    def on_mount(self) -> None:
        # Start timer for live updates
        self._timer = self.set_interval(1.0, self._update_timer)

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None

    def watch_active_step(self, step: str | None) -> None:
        if step:
            self._start_time = monotonic()
        else:
            self._start_time = None
        self._update_display()

    def watch_active_mode(self, mode: str | None) -> None:
        self._update_display()

    def _update_timer(self) -> None:
        """Update the elapsed time display."""
        if not self.active_step or not self._start_time:
            return
        try:
            timer = self.query_one("#indicator-timer", Static)
            elapsed = monotonic() - self._start_time
            timer.update(f"[yellow]⏱ {format_duration(elapsed)}[/yellow]")
        except QueryError:
            pass

    def _update_display(self) -> None:
        try:
            progress = self.query_one("#indicator-progress", Static)
            content = self.query_one("#indicator-content", Static)
            timer = self.query_one("#indicator-timer", Static)

            if self.active_step:
                # Progress line
                if self._total_steps > 0:
                    progress.update(
                        f"[bold]Step {self._step_index}/{self._total_steps}[/bold]"
                    )
                else:
                    progress.update("")

                # Content line with step info
                mode_text = f" [dim]({self.active_mode})[/dim]" if self.active_mode else ""
                title = self._step_title[:30] + "…" if len(self._step_title) > 30 else self._step_title
                content.update(f"{StatusIcon.IN_PROGRESS} {title}{mode_text}")

                # Timer line
                if self._start_time:
                    elapsed = monotonic() - self._start_time
                    timer.update(f"[yellow]⏱ {format_duration(elapsed)}[/yellow]")
                else:
                    timer.update("")

                self.display = True
            else:
                progress.update("")
                content.update("")
                timer.update("")
                self.display = False
        except QueryError:
            # Widget not yet mounted
            pass

    def set_active(
        self,
        step_id: str | None,
        mode: str | None = None,
        step_index: int = 0,
        total_steps: int = 0,
        step_title: str = "",
    ) -> None:
        self._step_index = step_index
        self._total_steps = total_steps
        self._step_title = step_title
        self.active_mode = mode
        self.active_step = step_id  # Set this last to trigger watch

    def clear(self) -> None:
        self._step_index = 0
        self._total_steps = 0
        self._step_title = ""
        self.active_step = None
        self.active_mode = None


class ExecutionSummary(Static):
    """Shows execution statistics for the plan."""

    def __init__(self) -> None:
        super().__init__()
        self.add_class("execution-summary")
        self._plan_start_time: float | None = None
        self._completed_durations: list[float] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="summary-content", classes="summary-content")

    def start_plan(self) -> None:
        """Called when a plan starts executing."""
        self._plan_start_time = monotonic()
        self._completed_durations = []
        self._update_display()

    def record_step_completion(self, duration: float) -> None:
        """Record a step completion duration."""
        self._completed_durations.append(duration)
        self._update_display()

    def clear(self) -> None:
        """Clear all statistics."""
        self._plan_start_time = None
        self._completed_durations = []
        self._update_display()

    def _update_display(self) -> None:
        try:
            content = self.query_one("#summary-content", Static)
            if not self._plan_start_time:
                content.update("")
                self.display = False
                return

            elapsed = monotonic() - self._plan_start_time
            completed = len(self._completed_durations)

            parts = [f"[dim]Total:[/dim] {format_duration(elapsed)}"]
            if completed > 0:
                avg = sum(self._completed_durations) / completed
                parts.append(f"[dim]Avg:[/dim] {format_duration(avg)}")

            content.update(" · ".join(parts))
            self.display = True
        except QueryError:
            pass


class PlanPanel(Static):
    """Panel showing current plan with steps and progress."""

    plan: reactive[PlanState | None] = reactive(None)

    class StepClicked(Message):
        def __init__(self, step_id: str) -> None:
            super().__init__()
            self.step_id = step_id

    def __init__(self) -> None:
        super().__init__()
        self.add_class("plan-panel")
        self._step_items: dict[str, PlanStepItem] = {}
        self._subagent_indicator: SubagentIndicator | None = None
        self._execution_summary: ExecutionSummary | None = None
        self._current_plan_id: str | None = None
        self._render_pending: bool = False
        self._generation: int = 0  # Guards against stale updates

    def compose(self) -> ComposeResult:
        with Vertical(classes="plan-panel-container"):
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

                # Execution summary
                self._execution_summary = ExecutionSummary()
                yield self._execution_summary

                # No plan message
                yield Static(
                    "[dim]No active plan[/dim]\n/plan <goal> to start",
                    id="no-plan-message",
                    classes="no-plan-message"
                )

    async def on_mount(self) -> None:
        await self._render_plan()

    def watch_plan(self, plan: PlanState | None) -> None:
        # Increment generation on any plan change to invalidate pending renders
        self._generation += 1
        generation = self._generation

        # Debounce rapid updates
        if self._render_pending:
            return
        self._render_pending = True
        self.call_later(lambda: self._do_render(generation))

    def _do_render(self, generation: int) -> None:
        """Wrapper to call async render from call_later."""
        self._render_pending = False
        # Skip if a newer generation has been requested
        if generation != self._generation:
            return
        self.call_later(self._render_plan)

    async def _render_plan(self) -> None:
        try:
            goal_widget = self.query_one("#plan-goal", Static)
            progress_widget = self.query_one("#plan-progress", Static)
            steps_list = self.query_one("#plan-steps-list", Vertical)
            no_plan_msg = self.query_one("#no-plan-message", Static)
        except QueryError:
            # Widget not yet mounted
            return

        if not self.plan:
            goal_widget.update("")
            progress_widget.update("")
            await steps_list.remove_children()
            self._step_items = {}
            self._current_plan_id = None
            no_plan_msg.display = True
            if self._subagent_indicator:
                self._subagent_indicator.clear()
            if self._execution_summary:
                self._execution_summary.clear()
            return

        no_plan_msg.display = False

        # Update goal
        status_icon = get_plan_status_icon(self.plan.status.value)
        goal_widget.update(f"{status_icon} {self.plan.goal[:40]}...")

        # Update progress
        completed = sum(1 for s in self.plan.steps if s.status == PlanStepStatus.COMPLETED)
        total = len(self.plan.steps)
        progress_widget.update(render_progress_bar(completed, total))

        # Check if this is the same plan - update in place if so
        if self._current_plan_id == self.plan.plan_id and self._step_items:
            # Update existing step items in place
            for step in self.plan.steps:
                if step.step_id in self._step_items:
                    self._step_items[step.step_id].update_step(step)
            return

        # New plan - start execution summary
        self._current_plan_id = self.plan.plan_id
        if self._execution_summary:
            self._execution_summary.start_plan()

        await steps_list.remove_children()
        self._step_items = {}

        for idx, step in enumerate(self.plan.steps, start=1):
            item = PlanStepItem(step, idx)
            self._step_items[step.step_id] = item
            await steps_list.mount(item)

    def update_plan(self, plan: PlanState | None) -> None:
        self.plan = plan

    def clear(self) -> None:
        """Explicitly clear the panel and invalidate all pending updates."""
        self._generation += 1  # Invalidate any pending renders
        self.plan = None

    def update_step(self, step_id: str, step: PlanStep) -> None:
        # Ignore updates if no active plan (e.g., plan was cancelled)
        if not self.plan:
            return

        if step_id in self._step_items:
            item = self._step_items[step_id]
            old_status = item._step.status
            item.update_step(step)

            # Record step completion in execution summary
            if (
                step.status == PlanStepStatus.COMPLETED
                and old_status != PlanStepStatus.COMPLETED
                and self._execution_summary
            ):
                duration = item.get_duration()
                if duration is not None:
                    self._execution_summary.record_step_completion(duration)

        # Update progress bar
        self.call_later(self._update_progress)

    async def _update_progress(self) -> None:
        if not self.plan:
            return
        try:
            progress_widget = self.query_one("#plan-progress", Static)
            completed = sum(1 for s in self.plan.steps if s.status == PlanStepStatus.COMPLETED)
            total = len(self.plan.steps)
            progress_widget.update(render_progress_bar(completed, total))
        except QueryError:
            # Widget not yet mounted
            pass

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        """Highlight the currently executing step."""
        if self._subagent_indicator:
            if step_id and self.plan:
                # Find step info for the indicator
                step_index = 0
                step_title = ""
                for idx, step in enumerate(self.plan.steps, start=1):
                    if step.step_id == step_id:
                        step_index = idx
                        step_title = step.title
                        break
                self._subagent_indicator.set_active(
                    step_id,
                    mode,
                    step_index=step_index,
                    total_steps=len(self.plan.steps),
                    step_title=step_title,
                )
            else:
                self._subagent_indicator.clear()

        # Update step item styling
        for sid, item in self._step_items.items():
            if sid == step_id:
                item.add_class("active-step")
            else:
                item.remove_class("active-step")

    def on_plan_step_item_clicked(self, event: PlanStepItem.Clicked) -> None:
        self.post_message(self.StepClicked(event.step_id))

