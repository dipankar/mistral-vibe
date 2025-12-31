from __future__ import annotations

from typing import TYPE_CHECKING

from vibe.cli.textual_ui.state_store import BottomPanelMode, UIStateStore
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.cli.textual_ui.widgets.messages import UserCommandMessage

if TYPE_CHECKING:
    from vibe.cli.textual_ui.app import VibeApp


class BottomPanelManager:
    """Controls which widget is mounted in the bottom panel."""

    def __init__(self, app: VibeApp, store: UIStateStore) -> None:
        self._app = app
        self._store = store

    async def show_config(self) -> None:
        if self._store.bottom_panel_mode == BottomPanelMode.CONFIG:
            return

        bottom_container = self._app.query_one("#bottom-app-container")
        await self._app._mount_and_scroll(UserCommandMessage("Configuration opened..."))
        await self._remove_chat_input()

        if self._app._mode_indicator:
            self._app._mode_indicator.display = False

        config_app = ConfigApp(self._app.config)
        await bottom_container.mount(config_app)
        self._store.set_bottom_panel(BottomPanelMode.CONFIG)
        self._app.call_after_refresh(config_app.focus)

    async def show_approval(self, tool_name: str, tool_args: dict) -> None:
        bottom_container = self._app.query_one("#bottom-app-container")
        await self._remove_chat_input()

        if self._app._mode_indicator:
            self._app._mode_indicator.display = False

        approval_app = ApprovalApp(
            tool_name=tool_name,
            tool_args=tool_args,
            workdir=str(self._app.config.effective_workdir),
            config=self._app.config,
        )
        await bottom_container.mount(approval_app)
        self._store.set_bottom_panel(BottomPanelMode.APPROVAL)
        self._app.call_after_refresh(approval_app.focus)
        self._app.call_after_refresh(self._app._scroll_to_bottom)

    async def show_input(self) -> None:
        bottom_container = self._app.query_one("#bottom-app-container")

        await self._remove_if_present("#config-app")
        await self._remove_if_present("#approval-app")

        if self._app._mode_indicator:
            self._app._mode_indicator.display = True

        try:
            chat_input = self._app.query_one(ChatInputContainer)
        except Exception:
            chat_input = None

        if chat_input is None:
            chat_input = ChatInputContainer(
                history_file=self._app.history_file,
                command_registry=self._app.commands,
                id="input-container",
                show_warning=self._app.auto_approve,
            )
            await bottom_container.mount(chat_input)

        self._app._chat_input_container = chat_input
        if self._app._chat_input_presenter:
            self._app._chat_input_presenter.set_container(chat_input)
        chat_input.set_thinking_mode(self._app._thinking_mode)
        self._store.set_bottom_panel(BottomPanelMode.INPUT)
        self._app.call_after_refresh(chat_input.focus_input)

    def focus_current(self) -> None:
        try:
            mode = self._store.bottom_panel_mode
            if mode is BottomPanelMode.INPUT:
                self._app.query_one(ChatInputContainer).focus_input()
            elif mode is BottomPanelMode.CONFIG:
                self._app.query_one(ConfigApp).focus()
            elif mode is BottomPanelMode.APPROVAL:
                self._app.query_one(ApprovalApp).focus()
        except Exception:
            pass

    def handle_interrupt(self) -> bool:
        if self._store.bottom_panel_mode == BottomPanelMode.CONFIG:
            try:
                self._app.query_one(ConfigApp).action_close()
                return True
            except Exception:
                return False

        if self._store.bottom_panel_mode == BottomPanelMode.APPROVAL:
            try:
                self._app.query_one(ApprovalApp).action_reject()
                return True
            except Exception:
                return False

        return False

    async def _remove_chat_input(self) -> None:
        container = self._app._chat_input_container
        if container:
            try:
                await container.remove()
            except Exception:
                pass
            self._app._chat_input_container = None
            if self._app._chat_input_presenter:
                self._app._chat_input_presenter.set_container(None)

    async def _remove_if_present(self, selector: str) -> None:
        try:
            widget = self._app.query_one(selector)
            await widget.remove()
        except Exception:
            pass
