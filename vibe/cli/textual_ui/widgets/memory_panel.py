from __future__ import annotations

from datetime import UTC, datetime
from typing import Sequence

from textual.widgets import Static

from vibe.core.memory import MemoryEntry


class MemoryEntryCard(Static):
    def __init__(self, entry: MemoryEntry, index: int) -> None:
        timestamp_source = entry.created_at
        if not isinstance(timestamp_source, datetime):
            timestamp_source = datetime.now(UTC)
        if timestamp_source.tzinfo is None:
            timestamp_source = timestamp_source.replace(tzinfo=UTC)
        timestamp = timestamp_source.astimezone().strftime("%H:%M:%S")
        token_text = (
            f"~{entry.token_count:,} tokens" if entry.token_count else "token snapshot n/a"
        )
        hints = (
            "\n".join(f"- {hint}" for hint in entry.task_hints if hint.strip())
            if entry.task_hints
            else ""
        )
        hints_block = f"\n{hints}" if hints else ""
        super().__init__(
            (
                f"[bold]Memory {index}[/bold] · {token_text} @ {timestamp}\n"
                f"{entry.summary.strip()}{hints_block}"
            ),
            classes="memory-entry",
        )


class MemoryPanel(Static):
    def __init__(self) -> None:
        super().__init__("", id="memory-panel")
        self.display = "none"
        self._visible = False
        self._entries: list[MemoryEntry] = []

    @property
    def is_visible(self) -> bool:
        return self._visible

    async def update_entries(self, entries: Sequence[MemoryEntry]) -> None:
        self._entries = list(entries)
        if self._visible:
            await self._render_entries()

    async def show_panel(self) -> None:
        self._visible = True
        self.display = "block"
        await self._render_entries()

    def hide_panel(self) -> None:
        if not self._visible:
            return
        self._visible = False
        self.display = "none"

    async def _render_entries(self) -> None:
        await self.remove_children()
        await self.mount(
            Static("Session Memory - /memory to close", classes="memory-panel-header")
        )
        if not self._entries:
            await self.mount(
                Static("No memory entries yet.", classes="memory-empty-message")
            )
            return

        for idx, entry in enumerate(reversed(self._entries), 1):
            await self.mount(MemoryEntryCard(entry, idx))
