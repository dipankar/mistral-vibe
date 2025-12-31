from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from enum import StrEnum, auto
from typing import Any, Awaitable, cast

from pydantic import BaseModel

from vibe.core.llm.format import APIToolFormatHandler, ResolvedMessage
from vibe.core.tools.base import (
    BaseTool,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from vibe.core.tools.manager import ToolManager
from vibe.core.types import (
    AgentStats,
    ApprovalCallback,
    ApprovalResponse,
    AsyncApprovalCallback,
    LLMMessage,
    SyncApprovalCallback,
    ToolCallEvent,
    ToolResultEvent,
)
from vibe.core.utils import (
    TOOL_ERROR_TAG,
    CancellationReason,
    get_user_cancellation_message,
)


class ToolExecutionResponse(StrEnum):
    SKIP = auto()
    EXECUTE = auto()


class ToolDecision(BaseModel):
    verdict: ToolExecutionResponse
    feedback: str | None = None


class ToolInvocationManager:
    """Encapsulates tool execution flow for Agent."""

    def __init__(
        self,
        tool_manager_getter: Callable[[], ToolManager],
        format_handler: APIToolFormatHandler,
        stats_getter: Callable[[], AgentStats],
        append_message: Callable[[LLMMessage], None],
        auto_approve_getter: Callable[[], bool],
        approval_callback_getter: Callable[[], ApprovalCallback | None],
        persist_interaction: Callable[[], Awaitable[None]],
    ) -> None:
        self._tool_manager_getter = tool_manager_getter
        self._format_handler = format_handler
        self._stats_getter = stats_getter
        self._append_message = append_message
        self._auto_approve_getter = auto_approve_getter
        self._approval_callback_getter = approval_callback_getter
        self._persist_interaction = persist_interaction

    def _tool_manager(self) -> ToolManager:
        return self._tool_manager_getter()

    def _stats(self) -> AgentStats:
        return self._stats_getter()

    def _auto_approve(self) -> bool:
        return self._auto_approve_getter()

    def _approval_callback(self) -> ApprovalCallback | None:
        return self._approval_callback_getter()

    async def handle(
        self, resolved: ResolvedMessage
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent]:
        for failed in resolved.failed_calls:
            error_msg = f"<{TOOL_ERROR_TAG}>{failed.tool_name}: {failed.error}</{TOOL_ERROR_TAG}>"
            yield ToolResultEvent(
                tool_name=failed.tool_name,
                tool_class=None,
                error=error_msg,
                tool_call_id=failed.call_id,
            )
            self._stats().tool_calls_failed += 1
            self._append_message(
                LLMMessage.model_validate(
                    self._format_handler.create_failed_tool_response_message(
                        failed, error_msg
                    )
                )
            )

        for tool_call in resolved.tool_calls:
            tool_call_id = tool_call.call_id
            yield ToolCallEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                args=tool_call.validated_args,
                tool_call_id=tool_call_id,
            )

            try:
                tool_instance = self._tool_manager().get(tool_call.tool_name)
            except Exception as exc:
                error_msg = f"Error getting tool '{tool_call.tool_name}': {exc}"
                yield ToolResultEvent(
                    tool_name=tool_call.tool_name,
                    tool_class=tool_call.tool_class,
                    error=error_msg,
                    tool_call_id=tool_call_id,
                )
                self._append_message(
                    LLMMessage.model_validate(
                        self._format_handler.create_tool_response_message(
                            tool_call, error_msg
                        )
                    )
                )
                continue

            decision = await self._should_execute_tool(
                tool_instance, tool_call.args_dict, tool_call_id
            )
            if decision.verdict == ToolExecutionResponse.SKIP:
                self._stats().tool_calls_rejected += 1
                skip_reason = decision.feedback or str(
                    get_user_cancellation_message(
                        CancellationReason.TOOL_SKIPPED, tool_call.tool_name
                    )
                )
                yield ToolResultEvent(
                    tool_name=tool_call.tool_name,
                    tool_class=tool_call.tool_class,
                    skipped=True,
                    skip_reason=skip_reason,
                    tool_call_id=tool_call_id,
                )
                self._append_message(
                    LLMMessage.model_validate(
                        self._format_handler.create_tool_response_message(
                            tool_call, skip_reason
                        )
                    )
                )
                continue

            self._stats().tool_calls_agreed += 1
            try:
                start_time = time.perf_counter()
                result_model = await tool_instance.invoke(**tool_call.args_dict)
                duration = time.perf_counter() - start_time
                text = "\n".join(
                    f"{k}: {v}" for k, v in result_model.model_dump().items()
                )
                self._append_message(
                    LLMMessage.model_validate(
                        self._format_handler.create_tool_response_message(
                            tool_call, text
                        )
                    )
                )
                yield ToolResultEvent(
                    tool_name=tool_call.tool_name,
                    tool_class=tool_call.tool_class,
                    result=result_model,
                    duration=duration,
                    tool_call_id=tool_call_id,
                )
                self._stats().tool_calls_succeeded += 1
            except asyncio.CancelledError:
                yield await self._handle_interruption(tool_call, tool_call_id)
                raise
            except KeyboardInterrupt:
                yield await self._handle_interruption(tool_call, tool_call_id)
                raise
            except (ToolError, ToolPermissionError) as exc:
                yield self._handle_tool_error(tool_call, tool_call_id, tool_instance, exc)

    async def _handle_interruption(self, tool_call, tool_call_id: str) -> ToolResultEvent:
        cancel = str(
            get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
        )
        event = ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=cancel,
            tool_call_id=tool_call_id,
        )
        self._append_message(
            LLMMessage.model_validate(
                self._format_handler.create_tool_response_message(tool_call, cancel)
            )
        )
        await self._persist_interaction()
        return event

    def _handle_tool_error(
        self,
        tool_call,
        tool_call_id: str,
        tool_instance: BaseTool,
        exc: Exception,
    ) -> ToolResultEvent:
        error_msg = (
            f"<{TOOL_ERROR_TAG}>{tool_instance.get_name()} failed: {exc}</{TOOL_ERROR_TAG}>"
        )
        event = ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=error_msg,
            tool_call_id=tool_call_id,
        )
        if isinstance(exc, ToolPermissionError):
            stats = self._stats()
            stats.tool_calls_agreed -= 1
            stats.tool_calls_rejected += 1
        else:
            self._stats().tool_calls_failed += 1
        self._append_message(
            LLMMessage.model_validate(
                self._format_handler.create_tool_response_message(
                    tool_call, error_msg
                )
            )
        )
        return event

    async def _should_execute_tool(
        self, tool: BaseTool, args: dict[str, Any], tool_call_id: str
    ) -> ToolDecision:
        if self._auto_approve():
            return ToolDecision(verdict=ToolExecutionResponse.EXECUTE)

        args_model, _ = tool._get_tool_args_results()
        validated_args = args_model.model_validate(args)

        allowlist_denylist_result = tool.check_allowlist_denylist(validated_args)
        if allowlist_denylist_result == ToolPermission.ALWAYS:
            return ToolDecision(verdict=ToolExecutionResponse.EXECUTE)
        if allowlist_denylist_result == ToolPermission.NEVER:
            denylist_patterns = tool.config.denylist
            denylist_str = ", ".join(repr(pattern) for pattern in denylist_patterns)
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                feedback=f"Tool '{tool.get_name()}' blocked by denylist: [{denylist_str}]",
            )

        tool_name = tool.get_name()
        perm = self._tool_manager().get_tool_config(tool_name).permission

        if perm is ToolPermission.ALWAYS:
            return ToolDecision(verdict=ToolExecutionResponse.EXECUTE)
        if perm is ToolPermission.NEVER:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                feedback=f"Tool '{tool_name}' is permanently disabled",
            )

        return await self._ask_approval(tool_name, args, tool_call_id)

    async def _ask_approval(
        self, tool_name: str, args: dict[str, Any], tool_call_id: str
    ) -> ToolDecision:
        callback = self._approval_callback()
        if not callback:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                feedback="Tool execution not permitted.",
            )
        if asyncio.iscoroutinefunction(callback):
            async_callback = cast(AsyncApprovalCallback, callback)
            response, feedback = await async_callback(tool_name, args, tool_call_id)
        else:
            sync_callback = cast(SyncApprovalCallback, callback)
            response, feedback = sync_callback(tool_name, args, tool_call_id)

        if response is ApprovalResponse.YES:
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE, feedback=feedback
            )
        return ToolDecision(verdict=ToolExecutionResponse.SKIP, feedback=feedback)


__all__ = [
    "ToolInvocationManager",
    "ToolDecision",
    "ToolExecutionResponse",
]
