"""Agent runner implementations.

This module provides the AgentRunner that wraps an Agent and integrates
it with the event dispatcher system.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from vibe.core.protocols import IAgentRunner, IApprovalHandler, IEventDispatcher
from vibe.core.types import AgentStats, ApprovalResponse, BaseEvent

if TYPE_CHECKING:
    from vibe.core.agent import Agent

logger = logging.getLogger(__name__)


class AgentRunner(IAgentRunner):
    """Standard agent runner with event dispatching.

    Wraps an Agent and:
    - Dispatches events to registered consumers
    - Handles approval via IApprovalHandler
    - Provides lifecycle management (interrupt, clear, compact)
    """

    def __init__(
        self,
        agent: Agent,
        dispatcher: IEventDispatcher,
        approval_handler: IApprovalHandler,
    ) -> None:
        self._agent = agent
        self._dispatcher = dispatcher
        self._approval_handler = approval_handler
        self._running = False
        self._interrupt_requested = False

        # Wire up approval callback
        agent.set_approval_callback(self._handle_approval)

    @property
    def is_running(self) -> bool:
        """Whether agent is currently processing a prompt."""
        return self._running

    @property
    def stats(self) -> AgentStats:
        """Current agent statistics."""
        return self._agent.stats

    @property
    def agent(self) -> Agent:
        """Direct access to underlying agent (for advanced use cases)."""
        return self._agent

    async def run(self, prompt: str) -> AsyncGenerator[BaseEvent, None]:
        """Execute agent with prompt, yielding events.

        Events are dispatched to all registered consumers and yielded
        for optional caller-level handling.

        Args:
            prompt: User prompt to process

        Yields:
            BaseEvent instances as they are produced
        """
        self._running = True
        self._interrupt_requested = False

        try:
            async for event in self._agent.act(prompt):
                if self._interrupt_requested:
                    logger.info("Agent run interrupted")
                    break

                # Dispatch to all consumers
                await self._dispatcher.dispatch(event)

                # Also yield for caller
                yield event
        finally:
            self._running = False

    async def interrupt(self) -> None:
        """Request interruption of current execution."""
        self._interrupt_requested = True

    async def clear_history(self) -> None:
        """Clear conversation history."""
        await self._agent.clear_history()

    async def compact(self) -> str:
        """Trigger context compaction, return summary."""
        return await self._agent.compact()

    async def _handle_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ApprovalResponse, str | None]:
        """Handle approval request via the IApprovalHandler."""
        # Check auto-approval first
        if self._approval_handler.is_auto_approved(tool_name):
            return ApprovalResponse.YES, None

        # Delegate to approval handler
        return await self._approval_handler.request_approval(
            tool_name, args, tool_call_id
        )


class AutoApproveHandler(IApprovalHandler):
    """Approval handler that auto-approves all tool calls.

    Useful for testing, batch processing, or trusted environments.
    """

    def __init__(self, auto_approve_all: bool = True) -> None:
        self._auto_approve_all = auto_approve_all

    async def request_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ApprovalResponse, str | None]:
        """Always approve."""
        return ApprovalResponse.YES, None

    def is_auto_approved(self, tool_name: str) -> bool:
        """All tools are auto-approved."""
        return self._auto_approve_all


class DenyAllHandler(IApprovalHandler):
    """Approval handler that denies all tool calls.

    Useful for testing or read-only modes.
    """

    async def request_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ApprovalResponse, str | None]:
        """Always deny."""
        return ApprovalResponse.NO, "Auto-denied by policy"

    def is_auto_approved(self, tool_name: str) -> bool:
        """No tools are auto-approved."""
        return False


class CallbackApprovalHandler(IApprovalHandler):
    """Approval handler that delegates to a callback function.

    Bridges the new protocol interface to existing callback-based code.
    """

    def __init__(
        self,
        callback: Any,  # ApprovalCallback type
        auto_approved_tools: set[str] | None = None,
    ) -> None:
        self._callback = callback
        self._auto_approved_tools = auto_approved_tools or set()

    async def request_approval(
        self,
        tool_name: str,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ApprovalResponse, str | None]:
        """Delegate to callback."""
        result = self._callback(tool_name, args, tool_call_id)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def is_auto_approved(self, tool_name: str) -> bool:
        """Check if tool is in auto-approved set."""
        return tool_name in self._auto_approved_tools
