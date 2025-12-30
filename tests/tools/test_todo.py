from __future__ import annotations

from vibe.core.tools.builtins.todo import Todo, TodoItem, TodoResult, TodoStatus
from vibe.core.types import ToolResultEvent


def test_todo_result_display_groups_by_status() -> None:
    result = TodoResult(
        message="Updated todos",
        todos=[
            TodoItem(id="1", content="ship feature", status=TodoStatus.IN_PROGRESS),
            TodoItem(id="2", content="write tests", status=TodoStatus.PENDING),
            TodoItem(id="3", content="cleanup", status=TodoStatus.COMPLETED),
        ],
        total_count=3,
    )
    event = ToolResultEvent(
        tool_name="todo",
        tool_class=Todo,
        result=result,
        tool_call_id="call-1",
    )

    display = Todo.get_result_display(event)

    assert display.success is True
    grouped = display.details["todos_by_status"]
    assert grouped["in_progress"][0]["content"] == "ship feature"
    assert grouped["pending"][0]["content"] == "write tests"
    assert grouped["completed"][0]["content"] == "cleanup"
    assert display.details["total_count"] == 3
