"""Base event consumer with no-op implementations."""

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


class BaseEventConsumer(IEventConsumer):
    """Base event consumer with no-op implementations.

    Subclasses can override only the methods they care about.
    All methods are async no-ops by default.
    """

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Handle assistant text output."""
        pass

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool invocation announcement."""
        pass

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool execution result."""
        pass

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        """Handle context compaction starting."""
        pass

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        """Handle context compaction completion."""
        pass

    async def on_memory_entry(self, event: MemoryEntryEvent) -> None:
        """Handle memory entry creation."""
        pass

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Handle plan initiation."""
        pass

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Handle plan step status change."""
        pass

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        """Handle plan decision point."""
        pass

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Handle plan completion."""
        pass

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        """Handle plan resource warning."""
        pass

    async def on_unknown(self, event: BaseEvent) -> None:
        """Handle unrecognized event types."""
        pass
