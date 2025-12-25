"""Textual UI event consumer implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from textual.widgets import Static

from vibe.cli.textual_ui.widgets.compact import CompactMessage
from vibe.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    MemoryUpdateMessage,
)
from vibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from vibe.core.protocols import IEventConsumer, IUIState
from vibe.core.tools.ui import ToolUIDataAdapter
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
    ToolDisplayDestination,
    ToolResultEvent,
)
from vibe.core.utils import TaggedText

if TYPE_CHECKING:
    from textual.widget import Widget

    from vibe.cli.textual_ui.widgets.loading import LoadingWidget


class ITextualApp(Protocol):
    """Protocol for Textual app interface needed by consumer."""

    async def mount_to_chat(self, widget: Widget) -> None:
        """Mount widget to chat/messages area."""
        ...

    async def mount_to_sidebar(self, widget: Widget) -> None:
        """Mount widget to sidebar area."""
        ...

    def get_loading_widget(self) -> LoadingWidget | None:
        """Get current loading widget if any."""
        ...

    def get_ui_state(self) -> IUIState:
        """Get current UI state."""
        ...

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Handle plan started event."""
        ...

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Handle plan step update."""
        ...

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        """Handle plan decision."""
        ...

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Handle plan completion."""
        ...

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        """Handle resource warning."""
        ...

    async def on_memory_updated(self) -> None:
        """Handle memory panel refresh."""
        ...


class TextualEventConsumer(IEventConsumer):
    """Textual TUI event consumer - creates and mounts widgets.

    Implements IEventConsumer protocol to receive agent events and
    render them as Textual widgets.
    """

    def __init__(self, app: ITextualApp) -> None:
        self._app = app
        self._current_tool_call: ToolCallMessage | None = None
        self._current_compact: CompactMessage | None = None
        self._tool_results: list[ToolResultMessage] = []

    @property
    def current_tool_call(self) -> ToolCallMessage | None:
        """Currently executing tool call widget."""
        return self._current_tool_call

    @property
    def current_compact(self) -> CompactMessage | None:
        """Current compaction message widget."""
        return self._current_compact

    @current_compact.setter
    def current_compact(self, value: CompactMessage | None) -> None:
        """Set current compaction message widget."""
        self._current_compact = value

    @property
    def tool_results(self) -> list[ToolResultMessage]:
        """All tool result widgets from this session."""
        return self._tool_results

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Handle assistant text output."""
        await self._app.mount_to_chat(AssistantMessage(event.content))

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Handle tool invocation announcement."""
        tool_call = ToolCallMessage(event)

        # Update loading widget status
        loading = self._app.get_loading_widget()
        if loading and event.tool_class:
            adapter = ToolUIDataAdapter(event.tool_class)
            status_text = adapter.get_status_text()
            loading.set_status(status_text)

        # Check display destination - don't show sidebar tools in chat
        destination = self._get_display_destination(event.tool_class)
        if destination != ToolDisplayDestination.SIDEBAR:
            await self._app.mount_to_chat(tool_call)

        self._current_tool_call = tool_call

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        """Handle tool execution result."""
        # Sanitize error messages
        sanitized_event = self._sanitize_event(event)

        # Determine destination and collapsed state
        ui_state = self._app.get_ui_state()
        destination = sanitized_event.display_destination

        # Fallback: check tool class if destination not set on event
        if destination == ToolDisplayDestination.CHAT and sanitized_event.tool_class:
            destination = self._get_display_destination(sanitized_event.tool_class)

        if destination == ToolDisplayDestination.SIDEBAR:
            collapsed = ui_state.todos_collapsed
            tool_result = ToolResultMessage(
                sanitized_event, self._current_tool_call, collapsed=collapsed
            )
            await self._app.mount_to_sidebar(tool_result)
        else:
            collapsed = ui_state.tools_collapsed
            tool_result = ToolResultMessage(
                sanitized_event, self._current_tool_call, collapsed=collapsed
            )
            await self._app.mount_to_chat(tool_result)

        self._tool_results.append(tool_result)
        self._current_tool_call = None

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        """Handle context compaction starting."""
        compact_msg = CompactMessage(preemptive=event.preemptive)
        self._current_compact = compact_msg
        await self._app.mount_to_chat(compact_msg)

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        """Handle context compaction completion."""
        if self._current_compact:
            self._current_compact.set_complete(
                old_tokens=event.old_context_tokens,
                new_tokens=event.new_context_tokens,
            )
            self._current_compact = None

    async def on_memory_entry(self, event: MemoryEntryEvent) -> None:
        """Handle memory entry creation."""
        await self._app.mount_to_chat(MemoryUpdateMessage(event))
        await self._app.on_memory_updated()

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Handle plan initiation."""
        await self._app.on_plan_started(event)

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Handle plan step status change."""
        await self._app.on_plan_step_update(event)

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        """Handle plan decision point."""
        await self._app.on_plan_decision(event)

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Handle plan completion."""
        await self._app.on_plan_completed(event)

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        """Handle plan resource warning."""
        await self._app.on_plan_resource_warning(event)

    async def on_unknown(self, event: BaseEvent) -> None:
        """Handle unrecognized event types."""
        await self._app.mount_to_chat(
            Static(str(event), markup=False, classes="unknown-event")
        )

    def _sanitize_event(self, event: ToolResultEvent) -> ToolResultEvent:
        """Sanitize error messages in tool result."""
        return ToolResultEvent(
            tool_name=event.tool_name,
            tool_class=event.tool_class,
            result=event.result,
            error=TaggedText.from_string(event.error).message if event.error else None,
            skipped=event.skipped,
            skip_reason=TaggedText.from_string(event.skip_reason).message
            if event.skip_reason
            else None,
            duration=event.duration,
            tool_call_id=event.tool_call_id,
            display_destination=event.display_destination,
            subagent_id=event.subagent_id,
            subagent_step_id=event.subagent_step_id,
        )

    def _get_display_destination(
        self, tool_class: type[Any] | None
    ) -> ToolDisplayDestination:
        """Get display destination for a tool class."""
        if tool_class is None:
            return ToolDisplayDestination.CHAT

        adapter = ToolUIDataAdapter(tool_class)
        return adapter.get_display_destination()

    def stop_current_tool_call(self) -> None:
        """Stop blinking on current tool call widget."""
        if self._current_tool_call:
            self._current_tool_call.stop_blinking()
            self._current_tool_call = None

    def stop_current_compact(self) -> None:
        """Stop blinking on current compact message."""
        if self._current_compact:
            self._current_compact.stop_blinking(success=False)
            self._current_compact = None

    def get_last_tool_result(self) -> ToolResultMessage | None:
        """Get the last tool result widget."""
        return self._tool_results[-1] if self._tool_results else None

    def reset(self) -> None:
        """Clear all state and widget references."""
        self.stop_current_tool_call()
        self.stop_current_compact()
        self._tool_results.clear()

    async def handle_event(
        self,
        event: BaseEvent,
        loading_active: bool = False,
        loading_widget: Any = None,
    ) -> ToolCallMessage | None:
        """Handle an event - facade method for compatibility.

        Dispatches to the appropriate on_* method based on event type.
        Returns ToolCallMessage for ToolCallEvent (for backward compatibility).
        """
        match event:
            case ToolCallEvent():
                await self.on_tool_call(event)
                return self._current_tool_call
            case ToolResultEvent():
                await self.on_tool_result(event)
                return None
            case AssistantEvent():
                await self.on_assistant(event)
                return None
            case CompactStartEvent():
                await self.on_compact_start(event)
                return None
            case CompactEndEvent():
                await self.on_compact_end(event)
                return None
            case MemoryEntryEvent():
                await self.on_memory_entry(event)
                return None
            case PlanStartedEvent():
                await self.on_plan_started(event)
                return None
            case PlanStepUpdateEvent():
                await self.on_plan_step_update(event)
                return None
            case PlanDecisionEvent():
                await self.on_plan_decision(event)
                return None
            case PlanCompletedEvent():
                await self.on_plan_completed(event)
                return None
            case PlanResourceWarningEvent():
                await self.on_plan_resource_warning(event)
                return None
            case _:
                await self.on_unknown(event)
                return None
