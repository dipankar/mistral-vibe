from __future__ import annotations

import pytest

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent import Agent
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.types import (
    AssistantEvent,
    CompactEndEvent,
    CompactStartEvent,
    LLMMessage,
    Role,
)


@pytest.mark.asyncio
async def test_auto_compact_triggers_and_batches_observer() -> None:
    observed: list[tuple[Role, str | None]] = []

    def observer(msg: LLMMessage) -> None:
        observed.append((msg.role, msg.content))

    backend = FakeBackend([
        mock_llm_chunk(content="<summary>"),
        mock_llm_chunk(content="<final>"),
    ])
    cfg = VibeConfig(
        session_logging=SessionLoggingConfig(enabled=False), auto_compact_threshold=1
    )
    agent = Agent(cfg, message_observer=observer, backend=backend)
    agent.stats.context_tokens = 2
    agent.messages.extend(
        [
            LLMMessage(role=Role.user, content="Earlier task"),
            LLMMessage(role=Role.assistant, content="Resolved"),
        ]
    )

    events = [ev async for ev in agent.act("Hello")]

    assert len(events) == 3
    assert isinstance(events[0], CompactStartEvent)
    assert isinstance(events[1], CompactEndEvent)
    assert isinstance(events[2], AssistantEvent)
    start: CompactStartEvent = events[0]
    end: CompactEndEvent = events[1]
    final: AssistantEvent = events[2]
    assert start.current_context_tokens == 2
    assert start.threshold == 1
    assert start.preemptive is False
    assert end.old_context_tokens == 2
    assert end.new_context_tokens >= 1
    assert end.preemptive is False
    assert final.content == "<final>"

    roles = [r for r, _ in observed]
    assert roles == [Role.system, Role.user, Role.assistant]
    assert observed[1][1] is not None and "[Memory #1]" in observed[1][1]
    assert "- Hello" in (observed[1][1] or "")
    assert observed[2][1] == "<final>"


@pytest.mark.asyncio
async def test_preemptive_memory_compaction_before_limit() -> None:
    backend = FakeBackend([
        mock_llm_chunk(content="<summary>"),
        mock_llm_chunk(content="<final>"),
    ])
    cfg = VibeConfig(
        session_logging=SessionLoggingConfig(enabled=False),
        auto_compact_threshold=100,
    )
    agent = Agent(cfg, backend=backend)
    agent.stats.context_tokens = 96
    agent.messages.extend(
        [
            LLMMessage(role=Role.user, content="Legacy"),
            LLMMessage(role=Role.assistant, content="Done"),
            LLMMessage(role=Role.user, content="Legacy 2"),
            LLMMessage(role=Role.assistant, content="Done 2"),
            LLMMessage(role=Role.user, content="Legacy 3"),
            LLMMessage(role=Role.assistant, content="Done 3"),
        ]
    )

    events = [ev async for ev in agent.act("Ping")]

    assert len(events) == 3
    start = events[0]
    end = events[1]
    final = events[2]
    assert isinstance(start, CompactStartEvent)
    assert isinstance(end, CompactEndEvent)
    assert isinstance(final, AssistantEvent)
    assert start.preemptive is True
    assert end.preemptive is True
    assert agent.session_memory.entries
    assert agent.session_memory.entries[0].task_hints == ["Legacy 2"]
    assert final.content == "<final>"
