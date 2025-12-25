"""SSE Event Consumer for Web API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator

from vibe.cli.consumers.base import BaseEventConsumer
from vibe.api.events import EventSerializer
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

logger = logging.getLogger(__name__)


class SSEEventConsumer(BaseEventConsumer):
    """Event consumer that queues events for SSE streaming.

    Use this consumer to stream agent events to web clients via
    Server-Sent Events.

    Usage:
        consumer = SSEEventConsumer()
        app.add_event_consumer(consumer)

        # In your SSE endpoint:
        async def sse_endpoint():
            async for event_str in consumer.events():
                yield event_str
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        """Initialize SSE consumer.

        Args:
            max_queue_size: Maximum events to queue before dropping old ones
        """
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._event_id: int = 0
        self._closed: bool = False
        self._serializer = EventSerializer()

    async def _enqueue(self, event: BaseEvent) -> None:
        """Add event to queue, dropping oldest if full."""
        if self._closed:
            return

        self._event_id += 1
        sse_message = self._serializer.to_sse(event, self._event_id)

        try:
            # Try to put without blocking
            self._queue.put_nowait(sse_message)
        except asyncio.QueueFull:
            # Queue full - drop oldest event and retry
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(sse_message)
            except asyncio.QueueEmpty:
                pass

    async def events(self) -> AsyncGenerator[str, None]:
        """Async generator that yields SSE-formatted events.

        Use this in your SSE endpoint to stream events to the client.

        Yields:
            SSE-formatted event strings
        """
        while not self._closed:
            try:
                # Wait for next event with timeout
                event_str = await asyncio.wait_for(
                    self._queue.get(), timeout=30.0
                )
                yield event_str
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
            except asyncio.CancelledError:
                break

    def close(self) -> None:
        """Close the consumer and stop generating events."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Check if consumer is closed."""
        return self._closed

    @property
    def queue_size(self) -> int:
        """Current number of events in queue."""
        return self._queue.qsize()

    # IEventConsumer implementation - all events go through _enqueue

    async def on_assistant(self, event: AssistantEvent) -> None:
        await self._enqueue(event)

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        await self._enqueue(event)

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        await self._enqueue(event)

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        await self._enqueue(event)

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        await self._enqueue(event)

    async def on_memory_entry(self, event: MemoryEntryEvent) -> None:
        await self._enqueue(event)

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        await self._enqueue(event)

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        await self._enqueue(event)

    async def on_plan_decision(self, event: PlanDecisionEvent) -> None:
        await self._enqueue(event)

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        await self._enqueue(event)

    async def on_plan_resource_warning(self, event: PlanResourceWarningEvent) -> None:
        await self._enqueue(event)

    async def on_unknown(self, event: BaseEvent) -> None:
        await self._enqueue(event)


class BufferedSSEConsumer(SSEEventConsumer):
    """SSE consumer that buffers assistant content for smoother streaming.

    Groups rapid assistant events into larger chunks before sending,
    reducing network overhead and providing smoother output.
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        buffer_timeout: float = 0.05,  # 50ms
        min_chunk_size: int = 10,
    ) -> None:
        super().__init__(max_queue_size)
        self._buffer_timeout = buffer_timeout
        self._min_chunk_size = min_chunk_size
        self._content_buffer: str = ""
        self._buffer_task: asyncio.Task | None = None

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Buffer assistant content for smoother streaming."""
        self._content_buffer += event.content

        # If buffer is large enough, flush immediately
        if len(self._content_buffer) >= self._min_chunk_size:
            await self._flush_buffer()
        else:
            # Schedule delayed flush
            if self._buffer_task is None or self._buffer_task.done():
                self._buffer_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self) -> None:
        """Flush buffer after timeout."""
        await asyncio.sleep(self._buffer_timeout)
        await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush buffered content as a single event."""
        if self._content_buffer:
            # Create a combined assistant event
            combined_event = AssistantEvent(content=self._content_buffer)
            self._content_buffer = ""
            await self._enqueue(combined_event)

    def close(self) -> None:
        """Close and flush any remaining buffer."""
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        # Note: Can't await flush here, so buffer may be lost
        super().close()
