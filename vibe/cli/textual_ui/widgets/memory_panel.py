from __future__ import annotations

from typing import Sequence

from textual.widgets import Static

from vibe.core.memory import MemoryEntry


class MemoryEntryCard(Static):
    def __init__(self, entry: MemoryEntry, index: int) -> None:
        super().__init__(
            f"[bold]Memory {index}[/bold]\n{entry.summary.strip()}"
            + (
                "\n" + "\n".join(f"- {hint}" for hint in entry.task_hints)
                if entry.task_hints
                else ""
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

        for idx, entry in enumerate(self._entries, 1):
            await self.mount(MemoryEntryCard(entry, idx))
