from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable, Sequence

from vibe.core.types import LLMMessage, Role

MEMORY_MESSAGE_NAME = "session_memory"


@dataclass(slots=True)
class MemoryEntry:
    summary: str
    task_hints: list[str] = field(default_factory=list)
    token_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_message(self, index: int) -> LLMMessage:
        lines = [f"[Memory #{index}] {self.summary.strip()}"]
        if self.task_hints:
            lines.append("Tasks & context:")
            for hint in self.task_hints:
                if hint.strip():
                    lines.append(f"- {hint.strip()}")
        if self.token_count:
            lines.append(f"(Captured at ~{self.token_count:,} tokens)")
        return LLMMessage(
            role=Role.user,
            name=MEMORY_MESSAGE_NAME,
            content="\n".join(lines).strip(),
        )

    @classmethod
    def from_message(cls, message: LLMMessage) -> MemoryEntry:
        return cls(summary=message.content or "")


class SessionMemory:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []
        self._synced = False

    def clear(self) -> None:
        self.entries.clear()
        self._synced = False

    def add_entry(
        self,
        summary: str,
        *,
        task_hints: Iterable[str] | None = None,
        token_count: int = 0,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            summary=summary.strip(),
            task_hints=list(task_hints or []),
            token_count=token_count,
        )
        self.entries.append(entry)
        return entry

    def sync_from_messages(self, messages: Sequence[LLMMessage]) -> None:
        if self._synced:
            return

        for message in messages:
            if self.is_memory_message(message):
                self.entries.append(MemoryEntry.from_message(message))

        self._synced = True

    def as_messages(self) -> list[LLMMessage]:
        return [entry.to_message(idx + 1) for idx, entry in enumerate(self.entries)]

    @staticmethod
    def is_memory_message(message: LLMMessage) -> bool:
        return (message.name or "").startswith(MEMORY_MESSAGE_NAME)
