from __future__ import annotations

from __future__ import annotations

from vibe.cli.textual_ui.state_store import UIStateStore
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer


class ChatInputPresenter:
    """Syncs UI state store changes into the chat input widget."""

    def __init__(self, store: UIStateStore) -> None:
        self._store = store
        self._chat_input: ChatInputContainer | None = None
        store.subscribe("planner_pending_confirmation", self._handle_confirmation_update)

    def set_container(self, chat_input: ChatInputContainer | None) -> None:
        self._chat_input = chat_input
        self._handle_confirmation_update(self._store)

    def _handle_confirmation_update(self, store: UIStateStore) -> None:
        if not self._chat_input:
            return
        goal = store.planner.pending_confirmation
        if goal:
            self._chat_input.show_plan_confirmation(goal)
        else:
            self._chat_input.hide_plan_confirmation()
