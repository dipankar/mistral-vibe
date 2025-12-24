"""Enhanced Todo Panel widget for displaying todos in the sidebar."""

from __future__ import annotations

from dataclasses import dataclass
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.status_icons import (
    get_step_status_icon,
    render_progress_bar,
    render_status_summary,
    ProgressBarStyle,
)


@dataclass
class TodoItem:
    """Represents a todo item."""
    content: str
    status: str  # pending, in_progress, completed, cancelled
    active_form: str = ""


class TodoItemWidget(Static):
    """Individual todo item in the panel."""

    class Clicked(Message):
        def __init__(self, content: str, status: str) -> None:
            super().__init__()
            self.content = content
            self.status = status

    def __init__(self, item: TodoItem, index: int) -> None:
        super().__init__()
        self.add_class("todo-item-widget")
        self._item = item
        self._index = index
        self._update_classes()

    def _update_classes(self) -> None:
        # Remove all status classes first
        for status in ["pending", "in_progress", "completed", "cancelled"]:
            self.remove_class(f"todo-{status}")
        # Add current status class
        self.add_class(f"todo-{self._item.status}")

    def compose(self) -> ComposeResult:
        icon = self._get_status_icon()
        content = self._item.active_form if self._item.status == "in_progress" else self._item.content
        yield Static(
            f"{icon} {content}",
            classes="todo-content"
        )

    def _get_status_icon(self) -> str:
        return get_step_status_icon(self._item.status)

    def update_item(self, item: TodoItem) -> None:
        self._item = item
        self._update_classes()
        # Re-render content
        content = self.query_one(".todo-content", Static)
        icon = self._get_status_icon()
        display_text = item.active_form if item.status == "in_progress" else item.content
        content.update(f"{icon} {display_text}")

    async def on_click(self) -> None:
        self.post_message(self.Clicked(self._item.content, self._item.status))


class TodoPanel(Static):
    """Panel showing current todos with progress and filtering."""

    todos: reactive[list[TodoItem]] = reactive(list, always_update=True)
    show_completed: reactive[bool] = reactive(True)

    class ItemClicked(Message):
        def __init__(self, content: str, status: str) -> None:
            super().__init__()
            self.content = content
            self.status = status

    class FilterToggled(Message):
        def __init__(self, show_completed: bool) -> None:
            super().__init__()
            self.show_completed = show_completed

    def __init__(self) -> None:
        super().__init__()
        self.add_class("todo-panel")
        self._item_widgets: list[TodoItemWidget] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="todo-panel-container"):
            # Summary bar
            yield Static("", id="todo-summary", classes="todo-summary")

            # Filter controls
            yield Static(
                "[dim]Click to toggle filter[/dim]",
                id="todo-filter-hint",
                classes="todo-filter-hint"
            )

            # Todo list
            with VerticalScroll(id="todo-list-scroll", classes="todo-list-scroll"):
                yield Vertical(id="todo-list")

            # Empty message
            yield Static(
                "[dim]No tasks yet[/dim]",
                id="todo-empty",
                classes="todo-empty"
            )

    async def on_mount(self) -> None:
        await self._render_todos()

    def watch_todos(self, todos: list[TodoItem]) -> None:
        self.call_later(self._render_todos)

    def watch_show_completed(self, show_completed: bool) -> None:
        self.call_later(self._render_todos)

    async def _render_todos(self) -> None:
        try:
            summary = self.query_one("#todo-summary", Static)
            todo_list = self.query_one("#todo-list", Vertical)
            empty_msg = self.query_one("#todo-empty", Static)
        except Exception:
            return

        if not self.todos:
            summary.update("")
            await todo_list.remove_children()
            self._item_widgets = []
            empty_msg.display = True
            return

        empty_msg.display = False

        # Calculate summary
        completed = sum(1 for t in self.todos if t.status == "completed")
        in_progress = sum(1 for t in self.todos if t.status == "in_progress")
        pending = sum(1 for t in self.todos if t.status == "pending")
        total = len(self.todos)

        # Progress bar with status summary
        progress = render_progress_bar(completed, total, width=12)
        status = render_status_summary(in_progress=in_progress, pending=pending)
        summary.update(f"{progress} {status}")

        # Filter todos
        visible_todos = [
            t for t in self.todos
            if self.show_completed or t.status != "completed"
        ]

        # Render todos
        await todo_list.remove_children()
        self._item_widgets = []

        for idx, todo in enumerate(visible_todos, start=1):
            item = TodoItemWidget(todo, idx)
            self._item_widgets.append(item)
            await todo_list.mount(item)

    def update_todos(self, todos_data: list[dict]) -> None:
        """Update todos from raw data (from todo tool result)."""
        items = []
        for data in todos_data:
            items.append(TodoItem(
                content=data.get("content", ""),
                status=data.get("status", "pending"),
                active_form=data.get("activeForm", data.get("active_form", "")),
            ))
        self.todos = items

    def toggle_filter(self) -> None:
        self.show_completed = not self.show_completed
        self.post_message(self.FilterToggled(self.show_completed))

    def on_todo_item_widget_clicked(self, event: TodoItemWidget.Clicked) -> None:
        self.post_message(self.ItemClicked(event.content, event.status))

    async def on_click(self, event) -> None:
        # Check if filter hint was clicked
        try:
            hint = self.query_one("#todo-filter-hint", Static)
            summary = self.query_one("#todo-summary", Static)
            # If click is near top, toggle filter
            if event.y <= 2:
                self.toggle_filter()
                event.stop()
        except Exception:
            pass
