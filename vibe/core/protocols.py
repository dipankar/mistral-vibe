"""Protocol definitions for UI-Agent abstraction layer.

This module defines the core protocols that enable multi-UI support
(TUI, CLI, Web API) by abstracting the interface between agents and
their presentation layers.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vibe.core.planner import PlanState, PlanStep
    from vibe.core.types import (
        AgentStats,
        ApprovalResponse,
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


@runtime_checkable
class IEventConsumer(Protocol):
    """Consumer of agent events - implemented by each UI.

    Each UI implementation (Textual TUI, CLI, Web API) implements this
    protocol to receive and handle events from the agent layer.

    Methods are called by the EventDispatcher when corresponding events
    are produced by agents.
    """

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Handle assistant text output (streaming or complete)."""
        ...

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool invocation announcement."""
        ...

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool execution result."""
        ...

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        """Handle context compaction starting."""
        ...

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        """Handle context compaction completion."""
        ...

    async def on_memory_entry(self, event: MemoryEntryEvent) -> None:
        """Handle memory entry creation."""
        ...

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Handle plan initiation."""
        ...

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Handle plan step status change."""
        ...

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        """Handle plan decision point requiring user input."""
        ...

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Handle plan completion."""
        ...

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        """Handle plan resource warning (budget, rate limit, etc.)."""
        ...

    async def on_unknown(self, event: BaseEvent) -> None:
        """Handle unrecognized event types (extensibility)."""
        ...


@runtime_checkable
class IEventDispatcher(Protocol):
    """Routes events from agents to registered consumers.

    The dispatcher decouples event producers (agents) from consumers (UIs),
    allowing multiple consumers to receive the same event stream.
    """

    def add_consumer(self, consumer: IEventConsumer) -> None:
        """Register an event consumer."""
        ...

    def remove_consumer(self, consumer: IEventConsumer) -> None:
        """Unregister an event consumer."""
        ...

    async def dispatch(self, event: BaseEvent) -> None:
        """Dispatch event to all registered consumers."""
        ...


@runtime_checkable
class IAgentRunner(Protocol):
    """Manages agent execution lifecycle.

    Wraps an Agent and provides a clean interface for running prompts
    and managing agent state. Decouples UI from direct Agent usage.
    """

    @property
    def is_running(self) -> bool:
        """Whether agent is currently processing a prompt."""
        ...

    @property
    def stats(self) -> AgentStats:
        """Current agent statistics."""
        ...

    async def run(self, prompt: str) -> AsyncGenerator[BaseEvent, None]:
        """Execute agent with prompt, yielding events.

        Events are dispatched to consumers and yielded for optional
        caller-level handling.
        """
        ...

    async def interrupt(self) -> None:
        """Request interruption of current execution."""
        ...

    async def clear_history(self) -> None:
        """Clear conversation history."""
        ...

    async def compact(self) -> str:
        """Trigger context compaction, return summary."""
        ...


@runtime_checkable
class IApprovalHandler(Protocol):
    """Handles tool execution approval requests.

    Different UIs handle approval differently:
    - TUI: Modal dialog
    - CLI: y/n prompt
    - Web API: Async webhook
    - Testing: Auto-approve
    """

    async def request_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ApprovalResponse, str | None]:
        """Request approval for tool execution.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            tool_call_id: Unique identifier for this tool call

        Returns:
            Tuple of (ApprovalResponse, optional_feedback)
        """
        ...

    def is_auto_approved(self, tool_name: str) -> bool:
        """Check if tool is configured for auto-approval."""
        ...


@runtime_checkable
class IUIState(Protocol):
    """UI state contract.

    Provides read access to UI state that affects event handling
    (e.g., whether tool results should be collapsed).
    """

    @property
    def auto_approve(self) -> bool:
        """Whether tool executions are auto-approved."""
        ...

    @property
    def tools_collapsed(self) -> bool:
        """Whether tool results should render collapsed by default."""
        ...

    @property
    def todos_collapsed(self) -> bool:
        """Whether todo results should render collapsed by default."""
        ...

    @property
    def streaming_enabled(self) -> bool:
        """Whether streaming output is enabled."""
        ...


@runtime_checkable
class IPlanObserver(Protocol):
    """Observes planner state changes.

    Complements IEventConsumer for plan-specific state observation.
    Plans are state-based (PlanState), so this provides direct
    state access in addition to plan events.
    """

    def on_plan_created(self, plan: PlanState) -> None:
        """Called when a new plan is created."""
        ...

    def on_plan_updated(self, plan: PlanState) -> None:
        """Called when plan state changes."""
        ...

    def on_step_updated(self, step: PlanStep) -> None:
        """Called when a step status changes."""
        ...

    def on_plan_cleared(self) -> None:
        """Called when plan is cleared/cancelled."""
        ...
