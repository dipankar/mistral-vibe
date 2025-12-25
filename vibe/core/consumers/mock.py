"""Mock event consumer for testing."""

from __future__ import annotations

from vibe.core.protocols import IEventConsumer
from vibe.core.types import (
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    MemoryEntryEvent,
    PlanCompletedEvent,
    PlanDecisionEvent,
    PlanResourceWarningEvent,
    PlanStartedEvent,
    PlanStepUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class MockEventConsumer(IEventConsumer):
    """Collects events for testing assertions.

    Records all received events for later inspection and provides
    assertion helpers for common test patterns.
    """

    def __init__(self) -> None:
        self.events: list[BaseEvent] = []
        self.assistant_events: list[AssistantEvent] = []
        self.tool_calls: list[ToolCallEvent] = []
        self.tool_results: list[ToolResultEvent] = []
        self.compact_starts: list[CompactStartEvent] = []
        self.compact_ends: list[CompactEndEvent] = []
        self.memory_entries: list[MemoryEntryEvent] = []
        self.plan_starts: list[PlanStartedEvent] = []
        self.plan_step_updates: list[PlanStepUpdateEvent] = []
        self.plan_decisions: list[PlanDecisionEvent] = []
        self.plan_completions: list[PlanCompletedEvent] = []
        self.plan_warnings: list[PlanResourceWarningEvent] = []
        self.unknown_events: list[BaseEvent] = []

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self.assistant_events.clear()
        self.tool_calls.clear()
        self.tool_results.clear()
        self.compact_starts.clear()
        self.compact_ends.clear()
        self.memory_entries.clear()
        self.plan_starts.clear()
        self.plan_step_updates.clear()
        self.plan_decisions.clear()
        self.plan_completions.clear()
        self.plan_warnings.clear()
        self.unknown_events.clear()

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Record assistant event."""
        self.events.append(event)
        self.assistant_events.append(event)

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Record tool call event."""
        self.events.append(event)
        self.tool_calls.append(event)

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        """Record tool result event."""
        self.events.append(event)
        self.tool_results.append(event)

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        """Record compact start event."""
        self.events.append(event)
        self.compact_starts.append(event)

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        """Record compact end event."""
        self.events.append(event)
        self.compact_ends.append(event)

    async def on_memory_entry(self, event: MemoryEntryEvent) -> None:
        """Record memory entry event."""
        self.events.append(event)
        self.memory_entries.append(event)

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Record plan started event."""
        self.events.append(event)
        self.plan_starts.append(event)

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Record plan step update event."""
        self.events.append(event)
        self.plan_step_updates.append(event)

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        """Record plan decision event."""
        self.events.append(event)
        self.plan_decisions.append(event)

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Record plan completed event."""
        self.events.append(event)
        self.plan_completions.append(event)

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        """Record plan resource warning event."""
        self.events.append(event)
        self.plan_warnings.append(event)

    async def on_unknown(self, event: BaseEvent) -> None:
        """Record unknown event."""
        self.events.append(event)
        self.unknown_events.append(event)

    # Assertion helpers

    def assert_tool_called(self, name: str) -> None:
        """Assert a tool was called by name."""
        assert any(
            tc.tool_name == name for tc in self.tool_calls
        ), f"Tool '{name}' was not called. Called: {[tc.tool_name for tc in self.tool_calls]}"

    def assert_tool_not_called(self, name: str) -> None:
        """Assert a tool was not called."""
        assert not any(
            tc.tool_name == name for tc in self.tool_calls
        ), f"Tool '{name}' was called but should not have been"

    def assert_tool_succeeded(self, name: str) -> None:
        """Assert a tool call succeeded (no error)."""
        results = [tr for tr in self.tool_results if tr.tool_name == name]
        assert results, f"No results for tool '{name}'"
        assert all(
            tr.error is None for tr in results
        ), f"Tool '{name}' had errors: {[tr.error for tr in results if tr.error]}"

    def assert_tool_failed(self, name: str) -> None:
        """Assert a tool call failed (has error)."""
        results = [tr for tr in self.tool_results if tr.tool_name == name]
        assert results, f"No results for tool '{name}'"
        assert any(
            tr.error is not None for tr in results
        ), f"Tool '{name}' did not fail"

    def assert_plan_started(self) -> None:
        """Assert a plan was started."""
        assert self.plan_starts, "No plan was started"

    def assert_plan_completed(self, status: str | None = None) -> None:
        """Assert a plan completed, optionally with specific status."""
        assert self.plan_completions, "No plan completed"
        if status:
            assert any(
                pc.final_status == status for pc in self.plan_completions
            ), f"No plan completed with status '{status}'"

    def get_assistant_content(self) -> str:
        """Get all assistant content concatenated."""
        return "".join(ae.content for ae in self.assistant_events)

    def get_tool_call_count(self, name: str | None = None) -> int:
        """Get count of tool calls, optionally filtered by name."""
        if name:
            return sum(1 for tc in self.tool_calls if tc.tool_name == name)
        return len(self.tool_calls)
