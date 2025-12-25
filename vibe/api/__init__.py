"""Web API for Vibe agent interaction."""

from vibe.api.consumer import SSEEventConsumer
from vibe.api.events import EventSerializer

__all__ = ["SSEEventConsumer", "EventSerializer"]
