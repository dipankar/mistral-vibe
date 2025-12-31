"""Compact tabbed tree for live subagent activity."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from vibe.core.planner import PlanStepStatus


@dataclass(slots=True)
class SubagentPanelEntry:
    """State for a single active subagent."""

    step_id: str
    title: str
    goal: str
    owner: str | None
    mode: str | None
    status: PlanStepStatus
    notes: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    activity: str | None = None
    subagent_id: str | None = None


class SubagentTab(Button):
    """Button styled as a tab for selecting a subagent."""

    def __init__(self, label: str, step_id: str, *, active: bool = False) -> None:
        super().__init__(label, classes="subagent-tab", variant="primary" if active else "default")
        self.step_id = step_id
        if active:
            self.add_class("subagent-tab-active")


class SubagentActivityPanel(Static):
    """Tabbed tree view summarizing live subagent work."""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.add_class("subagent-activity-panel")
        self._tab_container: Horizontal | None = None
        self._detail: Static | None = None
        self._footer: Static | None = None
        self._selected_step_id: str | None = None
        self._ordered_ids: list[str] = []
        self._entries: dict[str, SubagentPanelEntry] = {}
        self.display = False

    def compose(self) -> ComposeResult:
        with Vertical(id="subagent-panel"):
            yield Static("Active subagents", classes="subagent-panel-title")
            self._tab_container = Horizontal(id="subagent-tabs")
            yield self._tab_container
            self._detail = Static(
                "No active steps yet.",
                id="subagent-tree",
                classes="subagent-tree",
                markup=True,
            )
            yield self._detail
            self._footer = Static("", id="subagent-footer", classes="subagent-footer")
            yield self._footer

    async def update_entries(self, entries: list[SubagentPanelEntry]) -> None:
        """Refresh panel with the provided active subagents."""
        old_entries = self._entries
        old_ids = list(self._ordered_ids)
        new_entries = {entry.step_id: entry for entry in entries}
        new_ids = [entry.step_id for entry in entries]
        if not new_entries:
            await self.clear_entries()
            return

        self._entries = new_entries
        self._ordered_ids = new_ids

        if not self._selected_step_id or self._selected_step_id not in self._entries:
            self._selected_step_id = self._ordered_ids[0]

        needs_tab_refresh = new_ids != old_ids or any(
            entry.step_id not in old_entries
            or entry.status != old_entries[entry.step_id].status
            or entry.title != old_entries[entry.step_id].title
            for entry in entries
        )

        if needs_tab_refresh:
            await self._render_tabs()
        else:
            self._update_tab_states()

        self._render_tree()
        self._update_footer()
        self.display = True

    async def clear_entries(self) -> None:
        """Remove all entries and collapse the panel."""
        self._entries.clear()
        self._ordered_ids.clear()
        self._selected_step_id = None
        if self._tab_container:
            await self._tab_container.remove_children()
            placeholder = Static(
                "No active subagents",
                classes="subagent-tab-empty",
            )
            await self._tab_container.mount(placeholder)
        if self._detail:
            self._detail.update("No active steps yet.")
        if self._footer:
            self._footer.update("")
        self.display = False

    async def _render_tabs(self) -> None:
        if not self._tab_container:
            return
        await self._tab_container.remove_children()
        for step_id in self._ordered_ids:
            entry = self._entries[step_id]
            label = self._build_tab_label(entry)
            active = step_id == self._selected_step_id
            tab = SubagentTab(label, step_id, active=active)
            await self._tab_container.mount(tab)
        self._update_tab_states()

    def _build_tab_label(self, entry: SubagentPanelEntry) -> str:
        title = entry.title.strip() or entry.step_id
        if len(title) > 18:
            title = f"{title[:17]}…"
        status = self._format_status(entry.status)
        return f"{status}: {title}"

    def _render_tree(self) -> None:
        if not self._detail:
            return
        entry = self._entries.get(self._selected_step_id) if self._selected_step_id else None

        if not entry:
            self._detail.update("No active steps yet.")
            return

        lines = [f"[bold]{entry.title}[/bold] · {self._format_status(entry.status)}"]
        rows: list[tuple[str, str]] = [
            ("Mode", entry.mode or "code"),
            ("Owner", entry.owner or "planner"),
            ("Goal", self._truncate(entry.goal)),
            (
                "Tokens",
                self._format_tokens(entry.prompt_tokens, entry.completion_tokens),
            ),
            (
                "Tools",
                self._format_tool_usage(entry.tool_calls, entry.tool_successes, entry.tool_failures),
            ),
        ]

        if entry.subagent_id:
            rows.append(("Subagent", entry.subagent_id))

        rows.append(("Activity", entry.activity or "Working"))
        if entry.notes:
            rows.append(("Notes", self._truncate(entry.notes, 120)))

        for index, (label, value) in enumerate(rows):
            connector = "└─" if index == len(rows) - 1 else "├─"
            lines.append(f"{connector} {label}: {value}")

        self._detail.update("\n".join(lines))

    def _update_footer(self) -> None:
        if not self._footer:
            return
        count = len(self._ordered_ids)
        suffix = "step" if count == 1 else "steps"
        self._footer.update(f"{count} active {suffix}")

    def focus_step(self, step_id: str | None) -> None:
        """Hint the panel to focus a specific step."""
        if step_id and step_id in self._entries:
            self._selected_step_id = step_id
        elif not step_id and self._ordered_ids:
            self._selected_step_id = self._ordered_ids[0]
        self._update_tab_states()
        self._render_tree()

    def _update_tab_states(self) -> None:
        if not self._tab_container:
            return
        for tab in self._tab_container.query(SubagentTab):
            is_active = tab.step_id == self._selected_step_id
            tab.set_class(is_active, "subagent-tab-active")
            tab.variant = "primary" if is_active else "default"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if isinstance(event.button, SubagentTab):
            self._selected_step_id = event.button.step_id
            await self._render_tabs()
            self._render_tree()

    def _format_status(self, status: PlanStepStatus) -> str:
        return status.value.replace("_", " ").title()

    def _truncate(self, text: str, limit: int = 80) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text or "—"
        return f"{text[: limit - 1]}…"

    def _format_tokens(self, prompt: int, completion: int) -> str:
        total = prompt + completion
        return (
            f"[cyan]{total:,}[/cyan] total "
            f"(prompt {prompt:,} · completion {completion:,})"
        )

    def _format_tool_usage(
        self,
        calls: int,
        successes: int,
        failures: int,
    ) -> str:
        if calls == 0:
            return "No tools invoked"
        parts = [f"{successes}/{calls} succeeded"]
        if failures:
            parts.append(f"{failures} failed")
        return ", ".join(parts)
