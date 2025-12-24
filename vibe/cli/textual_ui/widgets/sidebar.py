"""Enhanced sidebar with Plan and Todo panels."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from vibe.core.planner import PlanState, PlanStep
from vibe.cli.textual_ui.widgets.plan_panel import PlanPanel
from vibe.cli.textual_ui.widgets.todo_panel import TodoPanel


class CollapsibleSection(Static):
    """A collapsible section with header and content."""

    collapsed: reactive[bool] = reactive(False)

    class Toggled(Message):
        def __init__(self, section_id: str, collapsed: bool) -> None:
            super().__init__()
            self.section_id = section_id
            self.collapsed = collapsed

    def __init__(
        self,
        title: str,
        section_id: str,
        initial_collapsed: bool = False,
    ) -> None:
        super().__init__()
        self.add_class("collapsible-section")
        self._title = title
        self._section_id = section_id
        self.collapsed = initial_collapsed

    def compose(self) -> ComposeResult:
        icon = "▶" if self.collapsed else "▼"
        yield Static(
            f"{icon} {self._title}",
            id=f"{self._section_id}-header",
            classes="section-header"
        )
        with Vertical(id=f"{self._section_id}-content", classes="section-content"):
            yield from self._compose_content()

    def _compose_content(self) -> ComposeResult:
        """Override in subclasses to provide content."""
        yield Static("")

    def watch_collapsed(self, collapsed: bool) -> None:
        try:
            header = self.query_one(f"#{self._section_id}-header", Static)
            content = self.query_one(f"#{self._section_id}-content", Vertical)
            icon = "▶" if collapsed else "▼"
            header.update(f"{icon} {self._title}")
            content.display = not collapsed
        except Exception:
            pass

    async def on_click(self, event) -> None:
        # Check if header was clicked
        try:
            header = self.query_one(f"#{self._section_id}-header", Static)
            # Simple check - if click is in the top area, toggle
            if event.y <= 1:
                self.collapsed = not self.collapsed
                self.post_message(self.Toggled(self._section_id, self.collapsed))
                event.stop()
        except Exception:
            pass


class TodoSection(CollapsibleSection):
    """Todo section in the sidebar."""

    def __init__(self, initial_collapsed: bool = False) -> None:
        super().__init__("Todo", "todo", initial_collapsed)
        self._todo_panel: TodoPanel | None = None

    def _compose_content(self) -> ComposeResult:
        yield Static("Ctrl+Shift+C copies", classes="section-tip")
        self._todo_panel = TodoPanel()
        yield self._todo_panel

    def get_todo_container(self) -> Vertical | None:
        """Legacy method for backwards compatibility."""
        if self._todo_panel:
            try:
                return self._todo_panel.query_one("#todo-list", Vertical)
            except Exception:
                return None
        return None

    def get_todo_panel(self) -> TodoPanel | None:
        return self._todo_panel

    def update_todos(self, todos_data: list[dict]) -> None:
        if self._todo_panel:
            self._todo_panel.update_todos(todos_data)

    def set_placeholder_visible(self, visible: bool) -> None:
        # Now handled by TodoPanel automatically
        pass


class PlanSection(CollapsibleSection):
    """Plan section in the sidebar."""

    def __init__(self, initial_collapsed: bool = False) -> None:
        super().__init__("Plan", "plan", initial_collapsed)
        self._plan_panel: PlanPanel | None = None

    def _compose_content(self) -> ComposeResult:
        self._plan_panel = PlanPanel()
        yield self._plan_panel

    def update_plan(self, plan: PlanState | None) -> None:
        if self._plan_panel:
            self._plan_panel.update_plan(plan)

    def update_step(self, step_id: str, step: PlanStep) -> None:
        if self._plan_panel:
            self._plan_panel.update_step(step_id, step)

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        if self._plan_panel:
            self._plan_panel.set_active_step(step_id, mode)

    def get_plan_panel(self) -> PlanPanel | None:
        return self._plan_panel


class Sidebar(Static):
    """Enhanced sidebar with Plan and Todo panels."""

    plan_collapsed: reactive[bool] = reactive(False)
    todo_collapsed: reactive[bool] = reactive(False)

    class SectionToggled(Message):
        def __init__(self, section: str, collapsed: bool) -> None:
            super().__init__()
            self.section = section
            self.collapsed = collapsed

    def __init__(self) -> None:
        super().__init__()
        self.add_class("enhanced-sidebar")
        self._plan_section: PlanSection | None = None
        self._todo_section: TodoSection | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="sidebar-container"):
            # Plan section
            self._plan_section = PlanSection(initial_collapsed=self.plan_collapsed)
            yield self._plan_section

            # Separator
            yield Static("", classes="sidebar-separator")

            # Todo section
            self._todo_section = TodoSection(initial_collapsed=self.todo_collapsed)
            yield self._todo_section

            # Help tips at bottom
            yield Static("/memory · memory overlay", classes="sidebar-help")

    def on_collapsible_section_toggled(self, event: CollapsibleSection.Toggled) -> None:
        if event.section_id == "plan":
            self.plan_collapsed = event.collapsed
        elif event.section_id == "todo":
            self.todo_collapsed = event.collapsed
        self.post_message(self.SectionToggled(event.section_id, event.collapsed))

    # Plan methods
    def update_plan(self, plan: PlanState | None) -> None:
        if self._plan_section:
            self._plan_section.update_plan(plan)

    def update_step(self, step_id: str, step: PlanStep) -> None:
        if self._plan_section:
            self._plan_section.update_step(step_id, step)

    def set_active_step(self, step_id: str | None, mode: str | None = None) -> None:
        if self._plan_section:
            self._plan_section.set_active_step(step_id, mode)

    def get_plan_panel(self) -> PlanPanel | None:
        if self._plan_section:
            return self._plan_section.get_plan_panel()
        return None

    # Todo methods
    def get_todo_container(self) -> Vertical | None:
        if self._todo_section:
            return self._todo_section.get_todo_container()
        return None

    def get_todo_panel(self) -> TodoPanel | None:
        if self._todo_section:
            return self._todo_section.get_todo_panel()
        return None

    def update_todos(self, todos_data: list[dict]) -> None:
        if self._todo_section:
            self._todo_section.update_todos(todos_data)

    def set_todo_placeholder_visible(self, visible: bool) -> None:
        if self._todo_section:
            self._todo_section.set_placeholder_visible(visible)

    # Collapse methods
    def collapse_plan(self, collapsed: bool = True) -> None:
        if self._plan_section:
            self._plan_section.collapsed = collapsed
            self.plan_collapsed = collapsed

    def collapse_todo(self, collapsed: bool = True) -> None:
        if self._todo_section:
            self._todo_section.collapsed = collapsed
            self.todo_collapsed = collapsed

    def toggle_plan(self) -> None:
        self.collapse_plan(not self.plan_collapsed)

    def toggle_todo(self) -> None:
        self.collapse_todo(not self.todo_collapsed)
