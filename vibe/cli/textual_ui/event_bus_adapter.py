from __future__ import annotations

import asyncio
import logging

from vibe.core.types import BaseEvent
from vibe.ipc.event_bus import EventBusPublisher, EventBusSubscriber, default_event_bus_config

logger = logging.getLogger(__name__)


class EventBusAdapter:
    """Encapsulates publisher/subscriber wiring for the Textual UI."""

    def __init__(self, workdir) -> None:
        config = default_event_bus_config(workdir)
        self.publisher = EventBusPublisher(config)
        self.subscriber = EventBusSubscriber(config)
        self._listener_task: asyncio.Task | None = None

    def start_listener(self, handler) -> None:
        if self._listener_task:
            return
        self._listener_task = asyncio.create_task(
            self.subscriber.listen(handler),
            name="event-bus-listener",
        )
        logger.info(
            "ipc.event_bus_subscribed",
            extra={"address": self.subscriber.address},
        )

    async def stop_listener(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

    def close(self) -> None:
        self.subscriber.close()
        self.publisher.close()
