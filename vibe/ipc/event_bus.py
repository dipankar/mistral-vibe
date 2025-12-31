from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from vibe.core.types import BaseEvent

try:  # pragma: no cover - dependency must be present
    import pynng  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "pynng is required for Vibe's IPC event bus. "
        "Install it with `uv add pynng` or ensure the dependency is available."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class EventBusConfig:
    """Configuration for the NNG-backed event bus."""

    address: str


def _workspace_slug(workdir: Path | None) -> str:
    resolved = workdir.resolve() if workdir else None
    safe_name = "workspace"
    if resolved:
        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "-",
            resolved.name or "workspace",
        ).strip("-") or "workspace"
    digest_source = str(resolved) if resolved else "default"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"{safe_name.lower()}-{digest}"


def default_event_bus_config(workdir: Path | None = None) -> EventBusConfig:
    """Build a default event bus configuration scoped to a workspace."""

    if override := os.environ.get("VIBE_EVENT_BUS_ADDRESS"):
        return EventBusConfig(address=override)

    bus_dir = Path.home() / ".vibe" / "ipc"
    try:
        bus_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem edge case
        logger.debug("Unable to create event bus directory", exc_info=exc)

    slug = _workspace_slug(workdir)
    socket_path = bus_dir / f"{slug}.bus"
    address = f"ipc://{socket_path.as_posix()}"
    return EventBusConfig(address=address)


type EventHandler = Callable[[BaseEvent], Awaitable[None]]


def _serialize_event(event: BaseEvent) -> bytes:
    return pickle.dumps(event)


def _deserialize_event(data: bytes) -> BaseEvent:
    return pickle.loads(data)


class EventBusPublisher:
    """Publish BaseEvent instances over NNG."""

    def __init__(self, config: EventBusConfig | None = None) -> None:
        self._config = config or default_event_bus_config()
        self._socket: pynng.Pub0 | None = None
        try:
            self._socket = pynng.Pub0(listen=self._config.address)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - depends on host setup
            raise RuntimeError(
                f"Failed to start event bus publisher at {self._config.address}"
            ) from exc

    @property
    def address(self) -> str:
        return self._config.address

    async def publish(self, event: BaseEvent) -> None:
        if not self._socket:
            raise RuntimeError("Event bus publisher socket is unavailable")
        try:
            payload = _serialize_event(event)
            await asyncio.to_thread(self._socket.send, payload)
        except Exception as exc:  # pragma: no cover - transport errors
            logger.debug("Event bus publish failure", exc_info=exc)

    def close(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._socket = None


class EventBusSubscriber:
    """Subscribe to BaseEvent instances from NNG."""

    def __init__(self, config: EventBusConfig | None = None) -> None:
        self._config = config or default_event_bus_config()
        self._socket: pynng.Sub0 | None = None
        self._closed = False

        try:
            self._socket = pynng.Sub0(dial=self._config.address)  # type: ignore[arg-type]
            self._socket.subscribe(b"")
        except Exception as exc:  # pragma: no cover - depends on host setup
            raise RuntimeError(
                f"Failed to start event bus subscriber at {self._config.address}"
            ) from exc

    @property
    def address(self) -> str:
        return self._config.address

    async def listen(self, handler: EventHandler) -> None:
        if not self._socket:
            raise RuntimeError("Event bus subscriber socket is unavailable")

        try:
            while not self._closed:
                try:
                    data = await asyncio.to_thread(self._socket.recv)
                except Exception as exc:  # pragma: no cover - transport errors
                    if self._closed:
                        break
                    logger.debug("Event bus receive failure", exc_info=exc)
                    await asyncio.sleep(0.1)
                    continue

                try:
                    event = _deserialize_event(data)
                except Exception as exc:  # pragma: no cover - deserialization errors
                    logger.debug("Failed to decode event bus payload", exc_info=exc)
                    continue

                try:
                    await handler(event)
                except Exception:
                    logger.exception("Event bus handler raised an error")
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise

    def close(self) -> None:
        self._closed = True
        if self._socket:
            try:
                self._socket.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._socket = None


__all__ = [
    "EventBusConfig",
    "EventBusPublisher",
    "EventBusSubscriber",
    "default_event_bus_config",
]
