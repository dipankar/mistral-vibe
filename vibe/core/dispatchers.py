"""Event dispatcher implementations.

This module provides the EventDispatcher that routes agent events
to registered consumers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vibe.core.protocols import IEventConsumer, IEventDispatcher
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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EventDispatcher(IEventDispatcher):
    """Routes events to all registered consumers.

    Implements fan-out pattern: each event is dispatched to all
    registered consumers. Errors in one consumer don't affect others.
    """

    def __init__(self) -> None:
        self._consumers: list[IEventConsumer] = []

    def add_consumer(self, consumer: IEventConsumer) -> None:
        """Register an event consumer."""
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def remove_consumer(self, consumer: IEventConsumer) -> None:
        """Unregister an event consumer."""
        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def clear_consumers(self) -> None:
        """Remove all registered consumers."""
        self._consumers.clear()

    @property
    def consumer_count(self) -> int:
        """Number of registered consumers."""
        return len(self._consumers)

    async def dispatch(self, event: BaseEvent) -> None:
        """Dispatch event to all registered consumers.

        Routes event to the appropriate handler method based on event type.
        Errors in individual consumers are logged but don't propagate.
        """
        for consumer in self._consumers:
            try:
                await self._dispatch_to_consumer(consumer, event)
            except Exception as e:
                logger.error(
                    "Event handler error for %s in %s: %s",
                    type(event).__name__,
                    type(consumer).__name__,
                    e,
                )

    async def _dispatch_to_consumer(
        self, consumer: IEventConsumer, event: BaseEvent
    ) -> None:
        """Route event to appropriate consumer method."""
        match event:
            case AssistantEvent():
                await consumer.on_assistant(event)
            case ToolCallEvent():
                await consumer.on_tool_call(event)
            case ToolResultEvent():
                await consumer.on_tool_result(event)
            case CompactStartEvent():
                await consumer.on_compact_start(event)
            case CompactEndEvent():
                await consumer.on_compact_end(event)
            case MemoryEntryEvent():
                await consumer.on_memory_entry(event)
            case PlanStartedEvent():
                await consumer.on_plan_started(event)
            case PlanStepUpdateEvent():
                await consumer.on_plan_step_update(event)
            case PlanDecisionEvent():
                await consumer.on_plan_decision(event)
            case PlanCompletedEvent():
                await consumer.on_plan_completed(event)
            case PlanResourceWarningEvent():
                await consumer.on_plan_resource_warning(event)
            case _:
                await consumer.on_unknown(event)


class FilteringEventDispatcher(EventDispatcher):
    """Event dispatcher with filtering support.

    Allows consumers to subscribe to specific event types only,
    reducing unnecessary handler calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self._filters: dict[IEventConsumer, set[type[BaseEvent]] | None] = {}

    def add_consumer(
        self,
        consumer: IEventConsumer,
        event_types: set[type[BaseEvent]] | None = None,
    ) -> None:
        """Register a consumer, optionally filtered to specific event types.

        Args:
            consumer: The event consumer to register
            event_types: Set of event types to receive, or None for all events
        """
        if consumer not in self._consumers:
            self._consumers.append(consumer)
        self._filters[consumer] = event_types

    def remove_consumer(self, consumer: IEventConsumer) -> None:
        """Unregister a consumer and its filters."""
        super().remove_consumer(consumer)
        self._filters.pop(consumer, None)

    def clear_consumers(self) -> None:
        """Remove all consumers and filters."""
        super().clear_consumers()
        self._filters.clear()

    async def dispatch(self, event: BaseEvent) -> None:
        """Dispatch event to filtered consumers only."""
        event_type = type(event)
        for consumer in self._consumers:
            filters = self._filters.get(consumer)
            # None means all events, otherwise check if event type is in filter
            if filters is not None and event_type not in filters:
                continue
            try:
                await self._dispatch_to_consumer(consumer, event)
            except Exception as e:
                logger.error(
                    "Event handler error for %s in %s: %s",
                    type(event).__name__,
                    type(consumer).__name__,
                    e,
                )
