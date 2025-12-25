"""CLI event consumer for headless/programmatic use."""

from __future__ import annotations

import sys
from typing import TextIO

from vibe.cli.consumers.base import BaseEventConsumer
from vibe.core.types import (
    AssistantEvent,
    CompactEndEvent,
    CompactStartEvent,
    PlanCompletedEvent,
    PlanStartedEvent,
    PlanStepUpdateEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class CLIEventConsumer(BaseEventConsumer):
    """Simple CLI consumer - prints events to stdout.

    Useful for headless operation, scripting, or piping output.
    Provides minimal formatting suitable for terminal output.
    """

    def __init__(
        self,
        stream: TextIO = sys.stdout,
        verbose: bool = False,
        show_tools: bool = True,
        show_plans: bool = True,
    ) -> None:
        """Initialize CLI consumer.

        Args:
            stream: Output stream (default: stdout)
            verbose: Show detailed output
            show_tools: Show tool call/result messages
            show_plans: Show plan-related messages
        """
        self._stream = stream
        self._verbose = verbose
        self._show_tools = show_tools
        self._show_plans = show_plans

    def _write(self, text: str, end: str = "\n") -> None:
        """Write text to stream."""
        print(text, end=end, file=self._stream, flush=True)

    async def on_assistant(self, event: AssistantEvent) -> None:
        """Print assistant output."""
        self._write(event.content, end="")

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Print tool call announcement."""
        if not self._show_tools:
            return

        if self._verbose:
            args_str = str(event.args.model_dump()) if hasattr(event.args, "model_dump") else ""
            self._write(f"\n[Tool: {event.tool_name}] {args_str}")
        else:
            self._write(f"\n[{event.tool_name}]", end=" ")

    async def on_tool_result(self, event: ToolResultEvent) -> None:
        """Print tool result."""
        if not self._show_tools:
            return

        if event.error:
            self._write(f"ERROR: {event.error}")
        elif event.skipped:
            self._write(f"SKIPPED: {event.skip_reason or 'unknown'}")
        elif self._verbose:
            result_str = str(event.result.model_dump()) if event.result and hasattr(event.result, "model_dump") else "OK"
            self._write(f"OK: {result_str}")
        else:
            self._write("OK")

    async def on_compact_start(self, event: CompactStartEvent) -> None:
        """Print compaction start."""
        if self._verbose:
            self._write(f"\n[Compacting context: {event.current_context_tokens} tokens]")

    async def on_compact_end(self, event: CompactEndEvent) -> None:
        """Print compaction end."""
        if self._verbose:
            self._write(
                f"[Compacted: {event.old_context_tokens} -> {event.new_context_tokens} tokens]"
            )

    async def on_plan_started(self, event: PlanStartedEvent) -> None:
        """Print plan start."""
        if not self._show_plans:
            return

        self._write(f"\n[Plan: {event.goal}]")
        if self._verbose:
            for i, step in enumerate(event.steps, 1):
                self._write(f"  {i}. {step}")

    async def on_plan_step_update(self, event: PlanStepUpdateEvent) -> None:
        """Print plan step update."""
        if not self._show_plans:
            return

        if self._verbose:
            self._write(f"[Step {event.step_id}: {event.title} -> {event.status}]")

    async def on_plan_completed(self, event: PlanCompletedEvent) -> None:
        """Print plan completion."""
        if not self._show_plans:
            return

        self._write(
            f"\n[Plan {event.final_status}: {event.steps_completed}/{event.steps_total} steps]"
        )
