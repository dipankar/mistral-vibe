"""UI state management for Textual app."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe.core.protocols import IUIState

if TYPE_CHECKING:
    from vibe.cli.textual_ui.app import VibeApp


class UIState(IUIState):
    """UI state adapter that reads from VibeApp.

    Provides IUIState protocol interface for TextualEventConsumer.
    """

    def __init__(self, app: VibeApp) -> None:
        self._app = app

    @property
    def auto_approve(self) -> bool:
        """Whether tool executions are auto-approved."""
        return self._app.auto_approve

    @property
    def tools_collapsed(self) -> bool:
        """Whether tool results should render collapsed by default."""
        return self._app._tools_collapsed

    @property
    def todos_collapsed(self) -> bool:
        """Whether todo results should render collapsed by default."""
        return self._app._todos_collapsed

    @property
    def streaming_enabled(self) -> bool:
        """Whether streaming output is enabled."""
        return self._app.enable_streaming
