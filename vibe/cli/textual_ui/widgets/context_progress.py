from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class TokenState:
    max_tokens: int = 0
    current_tokens: int = 0
    soft_limit_ratio: float = 1.0


class ContextProgress(Static):
    tokens = reactive(TokenState())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def watch_tokens(self, new_state: TokenState) -> None:
        if new_state.max_tokens == 0:
            self.update("")
            return

        percentage = min(
            100, int((new_state.current_tokens / new_state.max_tokens) * 100)
        )
        bar_width = 20
        filled = min(bar_width, round(bar_width * (percentage / 100)))
        soft_ratio = max(0.0, min(new_state.soft_limit_ratio, 1.0))
        soft_cutoff = min(bar_width, max(0, round(bar_width * soft_ratio)))

        bar_chars = []
        for idx in range(bar_width):
            if idx < filled:
                bar_chars.append("█" if idx < soft_cutoff else "▓")
            else:
                bar_chars.append("░")

        soft_limit = int(new_state.max_tokens * soft_ratio)
        text = (
            f"[{''.join(bar_chars)}] "
            f"{new_state.current_tokens:,}/{new_state.max_tokens:,} tokens "
            f"(memory @ {soft_limit:,})"
        )
        self.update(text)
