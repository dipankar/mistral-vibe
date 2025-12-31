from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable
import time
from typing import Any
from uuid import uuid4

from vibe.core.config import VibeConfig
from vibe.core.interaction_logger import InteractionLogger
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.format import APIToolFormatHandler, ResolvedMessage
from vibe.core.llm.types import BackendLike
from vibe.core.memory import MemoryEntry, SessionMemory
from vibe.core.conversation_state import ConversationState
from vibe.core.middleware import (
    ConversationContext,
    MiddlewareAction,
    MiddlewareResult,
    ResetReason,
)
from vibe.core.prompts import UtilityPrompt
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.manager import ToolManager
from vibe.core.types import (
    AgentStats,
    ApprovalCallback,
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    LLMChunk,
    LLMMessage,
    MemoryEntryEvent,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)
from vibe.core.utils import (
    VIBE_STOP_EVENT_TAG,
    get_user_agent,
    is_user_cancellation_event,
)
from vibe.core.middleware_runner import MiddlewareRunner
from vibe.core.tool_invocation_manager import ToolInvocationManager


class AgentError(Exception):
    """Base exception for Agent errors."""


class AgentStateError(AgentError):
    """Raised when agent is in an invalid state."""


class LLMResponseError(AgentError):
    """Raised when LLM response is malformed or missing expected data."""


class Agent:
    def __init__(
        self,
        config: VibeConfig,
        auto_approve: bool = False,
        message_observer: Callable[[LLMMessage], None] | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
        backend: BackendLike | None = None,
        enable_streaming: bool = False,
    ) -> None:
        self.config = config

        self.tool_manager = ToolManager(config)
        self.format_handler = APIToolFormatHandler()

        self.backend_factory = lambda: backend or self._select_backend()
        self.backend = self.backend_factory()

        self._middleware = MiddlewareRunner(config)
        self.enable_streaming = enable_streaming
        self._setup_middleware(max_turns, max_price)

        system_prompt = get_universal_system_prompt(self.tool_manager, config)

        self._conversation = ConversationState(
            system_prompt,
            session_memory=SessionMemory(),
            message_observer=message_observer,
        )

        self.stats = AgentStats()
        try:
            active_model = config.get_active_model()
            self.stats.input_price_per_million = active_model.input_price
            self.stats.output_price_per_million = active_model.output_price
        except ValueError:
            pass

        self.auto_approve = auto_approve
        self.approval_callback: ApprovalCallback | None = None

        self.session_id = str(uuid4())

        self.interaction_logger = InteractionLogger(
            config.session_logging,
            self.session_id,
            auto_approve,
            config.effective_workdir,
        )

        self._last_chunk: LLMChunk | None = None
        self._tool_invocation_manager = ToolInvocationManager(
            tool_manager_getter=lambda: self.tool_manager,
            format_handler=self.format_handler,
            stats_getter=lambda: self.stats,
            append_message=self.add_message,
            auto_approve_getter=lambda: self.auto_approve,
            approval_callback_getter=lambda: self.approval_callback,
            persist_interaction=self._persist_interaction,
        )

    @property
    def messages(self) -> list[LLMMessage]:
        return self._conversation.messages

    @messages.setter
    def messages(self, value: list[LLMMessage]) -> None:
        self._conversation.messages = value

    @property
    def session_memory(self) -> SessionMemory:
        return self._conversation.session_memory

    def _select_backend(self) -> BackendLike:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        timeout = self.config.api_timeout
        return BACKEND_FACTORY[provider.backend](provider=provider, timeout=timeout)

    def add_message(self, message: LLMMessage) -> None:
        self._conversation.add_message(message)

    def _ensure_memory_initialized(self) -> None:
        self._conversation.ensure_memory_initialized()

    def _flush_new_messages(self) -> None:
        self._conversation.flush_new_messages()

    async def act(self, msg: str) -> AsyncGenerator[BaseEvent]:
        self._ensure_memory_initialized()
        self._clean_message_history()
        async for event in self._conversation_loop(msg):
            yield event

    def _setup_middleware(self, max_turns: int | None, max_price: float | None) -> None:
        self._middleware.configure(max_turns=max_turns, max_price=max_price)

    async def _handle_middleware_result(
        self, result: MiddlewareResult
    ) -> AsyncGenerator[BaseEvent]:
        match result.action:
            case MiddlewareAction.STOP:
                yield AssistantEvent(
                    content=f"<{VIBE_STOP_EVENT_TAG}>{result.reason}</{VIBE_STOP_EVENT_TAG}>",
                    stopped_by_middleware=True,
                )
                await self.interaction_logger.save_interaction(
                    self.messages, self.stats, self.config, self.tool_manager
                )

            case MiddlewareAction.INJECT_MESSAGE:
                if result.message and len(self.messages) > 0:
                    last_msg = self.messages[-1]
                    if last_msg.content:
                        last_msg.content += f"\n\n{result.message}"
                    else:
                        last_msg.content = result.message

            case MiddlewareAction.COMPACT:
                if not self._has_compressible_history(force_full=True):
                    return
                old_tokens = result.metadata.get(
                    "old_tokens", self.stats.context_tokens
                )
                threshold = result.metadata.get(
                    "threshold", self.config.auto_compact_threshold
                )

                yield CompactStartEvent(
                    current_context_tokens=old_tokens, threshold=threshold
                )

                summary, entry = await self._compress_history_into_memory(
                    force_full=True, token_snapshot=old_tokens
                )
                summary_text = summary or ""

                yield CompactEndEvent(
                    old_context_tokens=old_tokens,
                    new_context_tokens=self.stats.context_tokens,
                    summary_length=len(summary_text),
                )
                if entry:
                    yield MemoryEntryEvent(
                        entry_index=len(self.session_memory.entries),
                        summary=entry.summary,
                        token_count=entry.token_count,
                        task_hints=list(entry.task_hints),
                    )

            case MiddlewareAction.MEMORY_COMPACT:
                if not self._has_compressible_history():
                    return
                old_tokens = result.metadata.get(
                    "old_tokens", self.stats.context_tokens
                )
                threshold = result.metadata.get(
                    "threshold", self.config.auto_compact_threshold
                )

                yield CompactStartEvent(
                    current_context_tokens=old_tokens,
                    threshold=threshold,
                    preemptive=True,
                )

                summary, entry = await self._compress_history_into_memory(
                    token_snapshot=old_tokens
                )
                summary_text = summary or ""

                yield CompactEndEvent(
                    old_context_tokens=old_tokens,
                    new_context_tokens=self.stats.context_tokens,
                    summary_length=len(summary_text),
                    preemptive=True,
                )
                if entry:
                    yield MemoryEntryEvent(
                        entry_index=len(self.session_memory.entries),
                        summary=entry.summary,
                        token_count=entry.token_count,
                        task_hints=list(entry.task_hints),
                    )

            case MiddlewareAction.CONTINUE:
                pass

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages, stats=self.stats, config=self.config
        )

    async def _conversation_loop(self, user_msg: str) -> AsyncGenerator[BaseEvent]:
        self.messages.append(LLMMessage(role=Role.user, content=user_msg))
        self.stats.steps += 1

        try:
            should_break_loop = False
            while not should_break_loop:
                result = await self._middleware.run_before_turn(
                    self._get_context()
                )

                async for event in self._handle_middleware_result(result):
                    yield event

                if result.action == MiddlewareAction.STOP:
                    self._flush_new_messages()
                    return

                self.stats.steps += 1
                user_cancelled = False
                async for event in self._perform_llm_turn():
                    if is_user_cancellation_event(event):
                        user_cancelled = True
                    yield event

                last_message = self.messages[-1]
                should_break_loop = (
                    last_message.role != Role.tool
                    and self._last_chunk is not None
                    and self._last_chunk.finish_reason is not None
                )

                self._flush_new_messages()
                await self.interaction_logger.save_interaction(
                    self.messages, self.stats, self.config, self.tool_manager
                )

                if user_cancelled:
                    self._flush_new_messages()
                    await self.interaction_logger.save_interaction(
                        self.messages, self.stats, self.config, self.tool_manager
                    )
                    return

                after_result = await self._middleware.run_after_turn(
                    self._get_context()
                )

                async for event in self._handle_middleware_result(after_result):
                    yield event

                if after_result.action == MiddlewareAction.STOP:
                    self._flush_new_messages()
                    return

                self._flush_new_messages()
                await self.interaction_logger.save_interaction(
                    self.messages, self.stats, self.config, self.tool_manager
                )

        except Exception:
            self._flush_new_messages()
            await self.interaction_logger.save_interaction(
                self.messages, self.stats, self.config, self.tool_manager
            )
            raise

    async def _perform_llm_turn(
        self,
    ) -> AsyncGenerator[AssistantEvent | ToolCallEvent | ToolResultEvent]:
        if self.enable_streaming:
            async for event in self._stream_assistant_events():
                yield event
        else:
            assistant_event = await self._get_assistant_event()
            if assistant_event.content:
                yield assistant_event

        last_message = self.messages[-1]
        last_chunk = self._last_chunk
        if last_chunk is None or last_chunk.usage is None:
            raise LLMResponseError("LLM response missing chunk or usage data")

        parsed = self.format_handler.parse_message(last_message)
        resolved = self.format_handler.resolve_tool_calls(
            parsed, self.tool_manager, self.config
        )

        if last_chunk.usage.completion_tokens > 0 and self.stats.last_turn_duration > 0:
            self.stats.tokens_per_second = (
                last_chunk.usage.completion_tokens / self.stats.last_turn_duration
            )

        if not resolved.tool_calls and not resolved.failed_calls:
            return

        async for event in self._tool_invocation_manager.handle(resolved):
            yield event

    def _create_assistant_event(
        self, content: str, chunk: LLMChunk | None
    ) -> AssistantEvent:
        return AssistantEvent(content=content)

    async def _stream_assistant_events(self) -> AsyncGenerator[AssistantEvent]:
        chunks: list[LLMChunk] = []
        content_buffer = ""
        chunks_with_content = 0
        BATCH_SIZE = 5

        async for chunk in self._chat_streaming():
            chunks.append(chunk)

            if chunk.message.tool_calls and chunk.finish_reason is None:
                if chunk.message.content:
                    content_buffer += chunk.message.content
                    chunks_with_content += 1

                if content_buffer:
                    yield self._create_assistant_event(content_buffer, chunk)
                    content_buffer = ""
                    chunks_with_content = 0
                continue

            if chunk.message.content:
                content_buffer += chunk.message.content
                chunks_with_content += 1

                if chunks_with_content >= BATCH_SIZE:
                    yield self._create_assistant_event(content_buffer, chunk)
                    content_buffer = ""
                    chunks_with_content = 0

        if content_buffer:
            last_chunk = chunks[-1] if chunks else None
            yield self._create_assistant_event(content_buffer, last_chunk)

        full_content = ""
        full_tool_calls_map = OrderedDict[int, ToolCall]()
        for chunk in chunks:
            full_content += chunk.message.content or ""
            if not chunk.message.tool_calls:
                continue

            for tc in chunk.message.tool_calls:
                if tc.index is None:
                    raise LLMResponseError("Tool call chunk missing index")
                if tc.index not in full_tool_calls_map:
                    full_tool_calls_map[tc.index] = tc
                else:
                    new_args_str = (
                        full_tool_calls_map[tc.index].function.arguments or ""
                    ) + (tc.function.arguments or "")
                    full_tool_calls_map[tc.index].function.arguments = new_args_str

        full_tool_calls = list(full_tool_calls_map.values()) or None
        last_message = LLMMessage(
            role=Role.assistant, content=full_content, tool_calls=full_tool_calls
        )
        self.messages.append(last_message)
        finish_reason = next(
            (c.finish_reason for c in chunks if c.finish_reason is not None), None
        )
        self._last_chunk = LLMChunk(
            message=last_message, usage=chunks[-1].usage, finish_reason=finish_reason
        )

    async def _get_assistant_event(self) -> AssistantEvent:
        llm_result = await self._chat()
        if llm_result.usage is None:
            raise LLMResponseError(
                "Usage data missing in non-streaming completion response"
            )
        self._last_chunk = llm_result
        assistant_msg = llm_result.message
        self.messages.append(assistant_msg)

        return AssistantEvent(content=assistant_msg.content or "")

    async def _chat(self, max_tokens: int | None = None) -> LLMChunk:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)

        available_tools = self.format_handler.get_available_tools(
            self.tool_manager, self.config
        )
        tool_choice = self.format_handler.get_tool_choice()

        try:
            start_time = time.perf_counter()

            async with self.backend as backend:
                result = await backend.complete(
                    model=active_model,
                    messages=self.messages,
                    temperature=active_model.temperature,
                    tools=available_tools,
                    tool_choice=tool_choice,
                    extra_headers={
                        "user-agent": get_user_agent(provider.backend),
                        "x-affinity": self.session_id,
                    },
                    max_tokens=max_tokens,
                )

            end_time = time.perf_counter()
            if result.usage is None:
                raise LLMResponseError(
                    "Usage data missing in non-streaming completion response"
                )

            self.stats.last_turn_duration = end_time - start_time
            self.stats.last_turn_prompt_tokens = result.usage.prompt_tokens
            self.stats.last_turn_completion_tokens = result.usage.completion_tokens
            self.stats.session_prompt_tokens += result.usage.prompt_tokens
            self.stats.session_completion_tokens += result.usage.completion_tokens
            self.stats.context_tokens = (
                result.usage.prompt_tokens + result.usage.completion_tokens
            )

            processed_message = self.format_handler.process_api_response_message(
                result.message
            )

            return LLMChunk(
                message=processed_message,
                usage=result.usage,
                finish_reason=result.finish_reason,
            )

        except Exception as e:
            raise RuntimeError(
                f"API error from {provider.name} (model: {active_model.name}): {e}"
            ) from e

    async def _chat_streaming(
        self, max_tokens: int | None = None
    ) -> AsyncGenerator[LLMChunk]:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)

        available_tools = self.format_handler.get_available_tools(
            self.tool_manager, self.config
        )
        tool_choice = self.format_handler.get_tool_choice()
        try:
            start_time = time.perf_counter()
            last_chunk = None
            async with self.backend as backend:
                async for chunk in backend.complete_streaming(
                    model=active_model,
                    messages=self.messages,
                    temperature=active_model.temperature,
                    tools=available_tools,
                    tool_choice=tool_choice,
                    extra_headers={
                        "user-agent": get_user_agent(provider.backend),
                        "x-affinity": self.session_id,
                    },
                    max_tokens=max_tokens,
                ):
                    last_chunk = chunk
                    processed_message = (
                        self.format_handler.process_api_response_message(chunk.message)
                    )
                    yield LLMChunk(
                        message=processed_message,
                        usage=chunk.usage,
                        finish_reason=chunk.finish_reason,
                    )

            end_time = time.perf_counter()
            if last_chunk is None:
                raise LLMResponseError("Streamed completion returned no chunks")
            if last_chunk.usage is None:
                raise LLMResponseError(
                    "Usage data missing in final chunk of streamed completion"
                )

            self.stats.last_turn_duration = end_time - start_time
            self.stats.last_turn_prompt_tokens = last_chunk.usage.prompt_tokens
            self.stats.last_turn_completion_tokens = last_chunk.usage.completion_tokens
            self.stats.session_prompt_tokens += last_chunk.usage.prompt_tokens
            self.stats.session_completion_tokens += last_chunk.usage.completion_tokens
            self.stats.context_tokens = (
                last_chunk.usage.prompt_tokens + last_chunk.usage.completion_tokens
            )

        except Exception as e:
            raise RuntimeError(
                f"API error from {provider.name} (model: {active_model.name}): {e}"
            ) from e

    def _clean_message_history(self) -> None:
        self._conversation.clean_history()

    def _conversation_without_memory(self) -> list[LLMMessage]:
        return self._conversation.conversation_without_memory()

    def _split_history_for_memory(
        self, *, force_full: bool = False
    ) -> tuple[list[LLMMessage], list[LLMMessage]]:
        return self._conversation.split_history_for_memory(force_full=force_full)

    def _has_compressible_history(self, *, force_full: bool = False) -> bool:
        return self._conversation.has_compressible_history(force_full=force_full)

    async def _compress_history_into_memory(
        self, force_full: bool = False, token_snapshot: int | None = None
    ) -> tuple[str | None, MemoryEntry | None]:
        try:
            self._ensure_memory_initialized()
            self._clean_message_history()
            await self.interaction_logger.save_interaction(
                self.messages, self.stats, self.config, self.tool_manager
            )

            memory_candidates, recent_messages = self._split_history_for_memory(
                force_full=force_full
            )
            if not memory_candidates:
                return None, None

            summary = await self._summarize_messages(memory_candidates)
            if not summary.strip():
                return None, None

            last_request = next(
                (
                    msg.content or ""
                    for msg in reversed(memory_candidates)
                    if msg.role is Role.user and (msg.content or "").strip()
                ),
                None,
            )
            hints = [last_request] if last_request else []
            snapshot = (
                token_snapshot
                if token_snapshot is not None
                else self.stats.context_tokens
            )
            entry = self.session_memory.add_entry(
                summary, task_hints=hints, token_count=snapshot
            )

            self._rebuild_messages_with_memory(recent_messages)
            await self._refresh_context_tokens()

            self._reset_session()
            await self.interaction_logger.save_interaction(
                self.messages, self.stats, self.config, self.tool_manager
            )
            self._middleware.reset(reason=ResetReason.COMPACT)
            return summary, entry

        except Exception:
            await self.interaction_logger.save_interaction(
                self.messages, self.stats, self.config, self.tool_manager
            )
            raise

    async def _summarize_messages(self, targets: list[LLMMessage]) -> str:
        summary_request = UtilityPrompt.COMPACT.read()
        preserved_messages = self.messages
        system_prompt = preserved_messages[0].content if preserved_messages else ""
        self.messages = [LLMMessage(role=Role.system, content=system_prompt or "")]
        self.messages.extend(targets)
        self.messages.append(LLMMessage(role=Role.user, content=summary_request))
        self.stats.steps += 1

        try:
            summary_result = await self._chat()
            if summary_result.usage is None:
                raise LLMResponseError(
                    "Usage data missing in compaction summary response"
                )
            return summary_result.message.content or ""
        finally:
            self.messages = preserved_messages

    def _rebuild_messages_with_memory(self, recent_messages: list[LLMMessage]) -> None:
        self._conversation.rebuild_messages_with_memory(recent_messages)

    async def _refresh_context_tokens(self) -> None:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        async with self.backend as backend:
            actual_context_tokens = await backend.count_tokens(
                model=active_model,
                messages=self.messages,
                tools=self.format_handler.get_available_tools(
                    self.tool_manager, self.config
                ),
                extra_headers={"user-agent": get_user_agent(provider.backend)},
            )
        self.stats.context_tokens = actual_context_tokens

    def _reset_session(self) -> None:
        self.session_id = str(uuid4())
        self.interaction_logger.reset_session(self.session_id)

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        self.approval_callback = callback

    async def _persist_interaction(self) -> None:
        await self.interaction_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager
        )

    async def clear_history(self) -> None:
        await self.interaction_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager
        )
        self.messages = self.messages[:1]
        self.session_memory.clear()
        self._conversation.mark_memory_synced(False)

        self.stats = AgentStats()

        try:
            active_model = self.config.get_active_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
        except ValueError:
            pass

        self._middleware.reset()
        self.tool_manager.reset_all()
        self._reset_session()

    async def compact(self) -> str:
        summary, _ = await self._compress_history_into_memory(
            force_full=True, token_snapshot=self.stats.context_tokens
        )
        return summary or ""

    async def reload_with_initial_messages(
        self,
        config: VibeConfig | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
    ) -> None:
        await self.interaction_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager
        )

        preserved_messages = self.messages[1:] if len(self.messages) > 1 else []
        old_system_prompt = self.messages[0].content if len(self.messages) > 0 else ""

        if config is not None:
            self.config = config
            self.backend = self.backend_factory()

        self.tool_manager = ToolManager(self.config)

        new_system_prompt = get_universal_system_prompt(self.tool_manager, self.config)
        self.messages = [LLMMessage(role=Role.system, content=new_system_prompt)]
        did_system_prompt_change = old_system_prompt != new_system_prompt

        if preserved_messages:
            self.messages.extend(preserved_messages)

        self.session_memory.clear()
        self._conversation.mark_memory_synced(False)
        self._ensure_memory_initialized()
        if self.session_memory.entries:
            self._rebuild_messages_with_memory(self._conversation_without_memory())

        if len(self.messages) == 1 or did_system_prompt_change:
            self.stats.reset_context_state()

        try:
            active_model = self.config.get_active_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
        except ValueError:
            pass

        self._setup_middleware(max_turns, max_price)

        self._conversation.reset_observer_view()

        self.tool_manager.reset_all()

        await self.interaction_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager
        )
