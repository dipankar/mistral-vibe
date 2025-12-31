"""SubAgent - Isolated agent instance for parallel/specialized task execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from vibe.core.config import VibeConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.modes import get_subagent_instructions
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.llm.types import BackendLike
from vibe.core.tools.base import BaseTool, ToolError, ToolPermission, ToolPermissionError
from vibe.core.tools.manager import NoSuchToolError, ToolManager
from vibe.core.types import (
    ApprovalCallback,
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    LLMChunk,
    LLMMessage,
    Role,
    SubagentProgressEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)
from vibe.core.utils import (
    TOOL_ERROR_TAG,
    get_user_agent,
    logger,
)


@dataclass
class SubAgentStats:
    """Statistics for a single subagent execution."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class SubAgentConfig:
    """Configuration for a subagent."""
    step_id: str
    title: str
    mode: str = "code"
    allowed_tools: list[str] | None = None  # None = all tools
    denied_tools: list[str] | None = None
    max_tokens: int | None = None  # Token budget for this subagent
    max_turns: int = 10  # Max conversation turns
    system_prompt_suffix: str = ""  # Additional context for system prompt


@dataclass
class SubAgentResult:
    """Result from a subagent execution."""
    step_id: str
    success: bool
    output: str = ""
    error: str | None = None
    stats: SubAgentStats = field(default_factory=SubAgentStats)
    artifacts: list[str] = field(default_factory=list)  # Files modified, etc.


class SubAgent:
    """
    Isolated agent instance for executing plan steps.

    Unlike the main Agent, SubAgent:
    - Has its own isolated message history
    - Can have restricted tool access
    - Tracks its own token budget
    - Is designed for single-task execution
    - Reports results back to the parent
    """

    def __init__(
        self,
        config: VibeConfig,
        subagent_config: SubAgentConfig,
        system_prompt: str,
        parent_approval_callback: ApprovalCallback | None = None,
        backend: BackendLike | None = None,
        enable_streaming: bool = True,
    ) -> None:
        self.config = config
        self.subagent_config = subagent_config
        self.subagent_id = f"subagent-{subagent_config.step_id}-{uuid4().hex[:8]}"

        # Create tool manager with filtering
        self.tool_manager = self._create_filtered_tool_manager(config, subagent_config)
        self.format_handler = APIToolFormatHandler()

        # Backend setup
        self.backend_factory = lambda: backend or self._select_backend()
        self.backend = self.backend_factory()

        # Initialize message history with system prompt
        full_system_prompt = self._build_system_prompt(system_prompt, subagent_config)
        self.messages: list[LLMMessage] = [
            LLMMessage(role=Role.system, content=full_system_prompt)
        ]

        # Stats tracking
        self.stats = SubAgentStats()
        self._turn_count = 0

        # Approval callback (inherited from parent)
        self.approval_callback = parent_approval_callback

        # Streaming
        self.enable_streaming = enable_streaming
        self._last_chunk: LLMChunk | None = None

        logger.debug(
            "SubAgent %s created for step %s with %d tools",
            self.subagent_id,
            subagent_config.step_id,
            len(self.tool_manager.available_tools()),
        )

    def _tag_event(self, event: BaseEvent) -> BaseEvent:
        """Tag an event with subagent context."""
        event.subagent_id = self.subagent_id
        event.subagent_step_id = self.subagent_config.step_id
        return event

    def _create_filtered_tool_manager(
        self, config: VibeConfig, subagent_config: SubAgentConfig
    ) -> ToolManager:
        """Create a tool manager with filtered tools based on subagent config."""
        manager = ToolManager(config)
        manager.filter_tools(
            allowed=subagent_config.allowed_tools,
            denied=subagent_config.denied_tools,
        )
        return manager

    def _build_system_prompt(
        self, base_prompt: str, subagent_config: SubAgentConfig
    ) -> str:
        """Build the system prompt with subagent-specific context."""
        parts = [base_prompt]

        # Add mode-specific guidance
        mode_guidance = get_subagent_instructions(subagent_config.mode)
        if mode_guidance:
            parts.append(f"\n\n## Mode: {subagent_config.mode.title()}\n{mode_guidance}")

        # Add token budget warning if set
        if subagent_config.max_tokens:
            parts.append(
                f"\n\n## Token Budget\n"
                f"You have a budget of approximately {subagent_config.max_tokens:,} tokens. "
                f"Be concise and efficient. Prioritize completing the task over explanations."
            )

        # Add custom suffix
        if subagent_config.system_prompt_suffix:
            parts.append(f"\n\n{subagent_config.system_prompt_suffix}")

        return "".join(parts)

    def _select_backend(self) -> BackendLike:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        timeout = self.config.api_timeout
        return BACKEND_FACTORY[provider.backend](provider=provider, timeout=timeout)

    def _build_progress_event(self, activity: str | None = None) -> SubagentProgressEvent:
        """Create a progress event snapshot for the current stats."""
        return SubagentProgressEvent(
            step_id=self.subagent_config.step_id,
            subagent_id=self.subagent_id,
            prompt_tokens=self.stats.prompt_tokens,
            completion_tokens=self.stats.completion_tokens,
            tool_calls=self.stats.tool_calls,
            tool_successes=self.stats.tool_successes,
            tool_failures=self.stats.tool_failures,
            activity=activity,
        )

    async def execute(self, task_prompt: str) -> AsyncGenerator[BaseEvent, None]:
        """
        Execute the subagent task.

        Yields events during execution and returns the final result.
        """
        import time
        start_time = time.time()

        try:
            async for event in self._conversation_loop(task_prompt):
                yield self._tag_event(event)
        finally:
            self.stats.duration_seconds = time.time() - start_time

    async def _conversation_loop(self, user_msg: str) -> AsyncGenerator[BaseEvent, None]:
        """Main conversation loop for the subagent."""
        self.messages.append(LLMMessage(role=Role.user, content=user_msg))

        while self._turn_count < self.subagent_config.max_turns:
            self._turn_count += 1

            # Check token budget
            if self._is_over_budget():
                yield AssistantEvent(
                    content="Token budget exceeded. Stopping execution.",
                    stopped_by_middleware=True,
                )
                return

            # Get LLM response
            try:
                async for event in self._get_llm_response():
                    yield event
            except Exception as exc:
                logger.error("SubAgent LLM error: %s", exc)
                yield AssistantEvent(
                    content=f"Error: {exc}",
                    stopped_by_middleware=True,
                )
                return

            # Check token budget after LLM response
            if self._is_over_budget():
                yield AssistantEvent(
                    content="Token budget exceeded after response. Stopping execution.",
                    stopped_by_middleware=True,
                )
                return

            # Check if we have tool calls to execute
            last_message = self.messages[-1] if self.messages else None
            if not last_message or not last_message.tool_calls:
                # No tool calls - conversation complete
                return

            # Execute tool calls
            async for event in self._execute_tool_calls(last_message.tool_calls):
                yield event

    async def _get_llm_response(self) -> AsyncGenerator[BaseEvent, None]:
        """Get response from LLM."""
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)

        tools = self.format_handler.get_available_tools(
            self.tool_manager, self.config
        )

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        async with self.backend as client:
            async for chunk in client.complete_streaming(
                model=active_model,
                messages=self.messages,
                temperature=active_model.temperature,
                tools=tools if tools else None,
                max_tokens=None,
                tool_choice="auto" if tools else None,
                extra_headers={
                    "user-agent": get_user_agent(provider.backend),
                    "x-affinity": f"subagent-{self.subagent_id}",
                },
            ):
                self._last_chunk = chunk

                # Accumulate content
                if chunk.message.content:
                    content_parts.append(chunk.message.content)
                    if self.enable_streaming:
                        yield AssistantEvent(content=chunk.message.content)

                # Accumulate tool calls
                if chunk.message.tool_calls:
                    for tc in chunk.message.tool_calls:
                        self._merge_tool_call(tool_calls, tc)

                # Update stats from usage
                if chunk.usage:
                    self.stats.prompt_tokens += chunk.usage.prompt_tokens
                    self.stats.completion_tokens += chunk.usage.completion_tokens
                    yield self._tag_event(
                        self._build_progress_event(activity="Generating response")
                    )

        # Build final message
        full_content = "".join(content_parts)
        final_message = LLMMessage(
            role=Role.assistant,
            content=full_content or None,
            tool_calls=tool_calls if tool_calls else None,
        )
        self.messages.append(final_message)

        # Emit final content if not streaming
        if not self.enable_streaming and full_content:
            yield AssistantEvent(content=full_content)

    def _merge_tool_call(self, tool_calls: list[ToolCall], new_tc: ToolCall) -> None:
        """Merge streaming tool call chunks."""
        if new_tc.index is not None:
            while len(tool_calls) <= new_tc.index:
                tool_calls.append(ToolCall())
            existing = tool_calls[new_tc.index]
            if new_tc.id:
                existing.id = new_tc.id
            if new_tc.function.name:
                existing.function.name = new_tc.function.name
            if new_tc.function.arguments:
                existing.function.arguments = (
                    existing.function.arguments or ""
                ) + new_tc.function.arguments
        elif new_tc.id:
            tool_calls.append(new_tc)

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute tool calls and yield events."""
        for tc in tool_calls:
            if not tc.function.name or not tc.id:
                continue

            tool_name = tc.function.name
            tool_call_id = tc.id
            self.stats.tool_calls += 1
            yield self._tag_event(
                self._build_progress_event(activity=f"Running tool `{tool_name}`")
            )

            # Get tool
            try:
                tool = self.tool_manager.get(tool_name)
            except NoSuchToolError:
                error_msg = f"Unknown tool: {tool_name}"
                self._add_tool_result(tool_call_id, tool_name, error=error_msg)
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=None,
                    error=error_msg,
                    tool_call_id=tool_call_id,
                )
                self.stats.tool_failures += 1
                yield self._tag_event(
                    self._build_progress_event(activity=f"Tool `{tool_name}` unavailable")
                )
                continue

            # Parse arguments
            try:
                raw_args = json.loads(tc.function.arguments or "{}")
                args_model, _ = tool._get_tool_args_results()
                args = args_model.model_validate(raw_args)
            except Exception as exc:
                error_msg = f"Invalid arguments: {exc}"
                self._add_tool_result(tool_call_id, tool_name, error=error_msg)
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=type(tool),
                    error=error_msg,
                    tool_call_id=tool_call_id,
                )
                self.stats.tool_failures += 1
                yield self._tag_event(
                    self._build_progress_event(activity=f"Tool `{tool_name}` argument error")
                )
                continue

            # Emit tool call event
            yield ToolCallEvent(
                tool_name=tool_name,
                tool_class=type(tool),
                args=args,
                tool_call_id=tool_call_id,
            )

            # Check approval
            approved = await self._check_approval(tool, args, tool_call_id)
            if not approved:
                self._add_tool_result(
                    tool_call_id, tool_name, error="Tool execution denied by user"
                )
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=type(tool),
                    skipped=True,
                    skip_reason="User denied",
                    tool_call_id=tool_call_id,
                )
                continue

            # Execute tool
            import time
            start = time.time()
            try:
                result = await tool.run(args)
                duration = time.time() - start
                self._add_tool_result(tool_call_id, tool_name, result=result)
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=type(tool),
                    result=result,
                    duration=duration,
                    tool_call_id=tool_call_id,
                )
                self.stats.tool_successes += 1
                yield self._tag_event(
                    self._build_progress_event(activity=f"Tool `{tool_name}` completed")
                )
            except (ToolError, ToolPermissionError) as exc:
                duration = time.time() - start
                error_msg = str(exc)
                self._add_tool_result(tool_call_id, tool_name, error=error_msg)
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=type(tool),
                    error=error_msg,
                    duration=duration,
                    tool_call_id=tool_call_id,
                )
                self.stats.tool_failures += 1
                yield self._tag_event(
                    self._build_progress_event(activity=f"Tool `{tool_name}` failed")
                )
            except Exception as exc:
                duration = time.time() - start
                error_msg = f"Unexpected error: {exc}"
                self._add_tool_result(tool_call_id, tool_name, error=error_msg)
                yield ToolResultEvent(
                    tool_name=tool_name,
                    tool_class=type(tool),
                    error=error_msg,
                    duration=duration,
                    tool_call_id=tool_call_id,
                )
                self.stats.tool_failures += 1
                yield self._tag_event(
                    self._build_progress_event(activity=f"Tool `{tool_name}` crashed")
                )

    def _add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: BaseModel | None = None,
        error: str | None = None,
    ) -> None:
        """Add tool result to message history."""
        if error:
            content = f"<{TOOL_ERROR_TAG}>{error}</{TOOL_ERROR_TAG}>"
        elif result:
            # Format result like Agent does
            content = "\n".join(
                f"{k}: {v}" for k, v in result.model_dump().items()
            )
        else:
            content = "Tool executed successfully"

        self.messages.append(
            LLMMessage(
                role=Role.tool,
                content=content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
        )

    async def _check_approval(
        self, tool: BaseTool, args: BaseModel, tool_call_id: str
    ) -> bool:
        """Check if tool execution is approved."""
        tool_name = tool.get_name()

        # Check allowlist/denylist first
        allowlist_result = tool.check_allowlist_denylist(args)
        if allowlist_result == ToolPermission.ALWAYS:
            return True
        if allowlist_result == ToolPermission.NEVER:
            return False

        # Check tool config permission
        permission = self.tool_manager.get_tool_config(tool_name).permission
        if permission == ToolPermission.ALWAYS:
            return True
        if permission == ToolPermission.NEVER:
            return False

        # Need to ask for approval
        if not self.approval_callback:
            # No callback = auto-approve
            return True

        try:
            args_dict = args.model_dump()
            # Check if callback is async or sync
            if asyncio.iscoroutinefunction(self.approval_callback):
                response, _ = await self.approval_callback(
                    tool_name, args_dict, tool_call_id
                )
            else:
                response, _ = self.approval_callback(
                    tool_name, args_dict, tool_call_id
                )
            return response == ApprovalResponse.YES
        except (TypeError, ValueError, AttributeError):
            return False

    def _is_over_budget(self) -> bool:
        """Check if subagent is over its token budget."""
        if not self.subagent_config.max_tokens:
            return False
        return self.stats.total_tokens >= self.subagent_config.max_tokens

    def get_result(self) -> SubAgentResult:
        """Get the final result of the subagent execution."""
        # Extract final output from last assistant message
        output = ""
        for msg in reversed(self.messages):
            if msg.role == Role.assistant and msg.content:
                output = msg.content
                break

        # Check for errors
        has_errors = self.stats.tool_failures > 0
        error = None
        if has_errors:
            error = f"{self.stats.tool_failures} tool(s) failed during execution"

        return SubAgentResult(
            step_id=self.subagent_config.step_id,
            success=not has_errors,
            output=output,
            error=error,
            stats=self.stats,
        )
