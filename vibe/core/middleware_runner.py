from __future__ import annotations

from vibe.core.config import VibeConfig
from vibe.core.middleware import (
    AutoCompactMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ResetReason,
    TurnLimitMiddleware,
)


class MiddlewareRunner:
    """Configures and executes middleware pipelines for the Agent."""

    def __init__(self, config: VibeConfig) -> None:
        self._config = config
        self._pipeline = MiddlewarePipeline()

    def configure(self, *, max_turns: int | None, max_price: float | None) -> None:
        self._pipeline.clear()

        if max_turns is not None:
            self._pipeline.add(TurnLimitMiddleware(max_turns))

        if max_price is not None:
            self._pipeline.add(PriceLimitMiddleware(max_price))

        if self._config.auto_compact_threshold > 0:
            self._pipeline.add(
                AutoCompactMiddleware(
                    self._config.auto_compact_threshold,
                    self._config.memory_soft_limit_ratio,
                )
            )
            if self._config.context_warnings:
                self._pipeline.add(
                    ContextWarningMiddleware(
                        self._config.memory_soft_limit_ratio,
                        self._config.auto_compact_threshold,
                    )
                )

    async def run_before_turn(
        self, context: ConversationContext
    ) -> MiddlewareResult:
        return await self._pipeline.run_before_turn(context)

    async def run_after_turn(
        self, context: ConversationContext
    ) -> MiddlewareResult:
        return await self._pipeline.run_after_turn(context)

    def reset(self, *, reason: ResetReason | None = None) -> None:
        if reason is not None:
            self._pipeline.reset(reset_reason=reason)
        else:
            self._pipeline.reset()
