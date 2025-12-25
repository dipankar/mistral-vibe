"""Event serialization for Web API."""

from __future__ import annotations

import json
from typing import Any

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


class EventSerializer:
    """Serializes agent events to JSON for Web API transmission."""

    @staticmethod
    def serialize(event: BaseEvent) -> dict[str, Any]:
        """Serialize an event to a JSON-compatible dictionary."""
        base = {
            "type": type(event).__name__,
            "subagent_id": event.subagent_id,
            "subagent_step_id": event.subagent_step_id,
        }

        match event:
            case AssistantEvent():
                return {
                    **base,
                    "content": event.content,
                    "stopped_by_middleware": event.stopped_by_middleware,
                }

            case ToolCallEvent():
                args_dict = {}
                if hasattr(event.args, "model_dump"):
                    args_dict = event.args.model_dump()
                return {
                    **base,
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "args": args_dict,
                }

            case ToolResultEvent():
                result_dict = {}
                if event.result and hasattr(event.result, "model_dump"):
                    result_dict = event.result.model_dump()
                return {
                    **base,
                    "tool_name": event.tool_name,
                    "tool_call_id": event.tool_call_id,
                    "result": result_dict,
                    "error": event.error,
                    "skipped": event.skipped,
                    "skip_reason": event.skip_reason,
                    "duration": event.duration,
                    "display_destination": event.display_destination.value,
                }

            case CompactStartEvent():
                return {
                    **base,
                    "current_context_tokens": event.current_context_tokens,
                    "threshold": event.threshold,
                    "preemptive": event.preemptive,
                }

            case CompactEndEvent():
                return {
                    **base,
                    "old_context_tokens": event.old_context_tokens,
                    "new_context_tokens": event.new_context_tokens,
                    "summary_length": event.summary_length,
                    "preemptive": event.preemptive,
                }

            case MemoryEntryEvent():
                return {
                    **base,
                    "entry_index": event.entry_index,
                    "summary": event.summary,
                    "token_count": event.token_count,
                    "task_hints": event.task_hints,
                }

            case PlanStartedEvent():
                return {
                    **base,
                    "plan_id": event.plan_id,
                    "goal": event.goal,
                    "summary": event.summary,
                    "steps": event.steps,
                }

            case PlanStepUpdateEvent():
                return {
                    **base,
                    "plan_id": event.plan_id,
                    "step_id": event.step_id,
                    "title": event.title,
                    "status": event.status,
                    "notes": event.notes,
                    "mode": event.mode,
                }

            case PlanDecisionEvent():
                return {
                    **base,
                    "plan_id": event.plan_id,
                    "decision_id": event.decision_id,
                    "header": event.header,
                    "question": event.question,
                    "options": [
                        {"label": o.label, "description": o.description}
                        for o in event.options
                    ],
                    "multi_select": event.multi_select,
                    "resolved": event.resolved,
                    "selections": event.selections,
                }

            case PlanCompletedEvent():
                return {
                    **base,
                    "plan_id": event.plan_id,
                    "final_status": event.final_status,
                    "summary": event.summary,
                    "steps_completed": event.steps_completed,
                    "steps_total": event.steps_total,
                    "resources_used": event.resources_used,
                }

            case PlanResourceWarningEvent():
                return {
                    **base,
                    "plan_id": event.plan_id,
                    "warning_type": event.warning_type,
                    "level": event.level,
                    "message": event.message,
                    "details": event.details,
                }

            case _:
                # Unknown event - serialize what we can
                return {
                    **base,
                    "data": str(event),
                }

    @staticmethod
    def to_json(event: BaseEvent) -> str:
        """Serialize an event to a JSON string."""
        return json.dumps(EventSerializer.serialize(event))

    @staticmethod
    def to_sse(event: BaseEvent, event_id: int | None = None) -> str:
        """Format an event as a Server-Sent Events message.

        Args:
            event: The event to serialize
            event_id: Optional event ID for SSE

        Returns:
            SSE-formatted string ready to send to client
        """
        event_type = type(event).__name__
        data = EventSerializer.to_json(event)

        lines = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event_type}")
        lines.append(f"data: {data}")
        lines.append("")  # Empty line to end the event

        return "\n".join(lines) + "\n"
