from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

from vibe.core.memory import SessionMemory
from vibe.core.types import LLMMessage, Role
from vibe.core.utils import CancellationReason, get_user_cancellation_message


class ConversationState:
    """Tracks conversation messages, session memory, and observer fan-out."""

    def __init__(
        self,
        system_prompt: str,
        *,
        session_memory: SessionMemory | None = None,
        message_observer: Callable[[LLMMessage], None] | None = None,
    ) -> None:
        self.session_memory = session_memory or SessionMemory()
        self._messages: list[LLMMessage] = [
            LLMMessage(role=Role.system, content=system_prompt)
        ]
        self._memory_synced = False
        self._observer = message_observer
        self._last_observed_index = 0
        if self._observer:
            self._observer(self._messages[0])
            self._last_observed_index = 1

    @property
    def messages(self) -> list[LLMMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[LLMMessage]) -> None:
        self._messages = value

    def add_message(self, message: LLMMessage) -> None:
        self._messages.append(message)

    @property
    def memory_synced(self) -> bool:
        return self._memory_synced

    def mark_memory_synced(self, synced: bool) -> None:
        self._memory_synced = synced

    def ensure_memory_initialized(self) -> None:
        if self._memory_synced:
            return
        self.session_memory.sync_from_messages(self._messages)
        self._memory_synced = True

    def flush_new_messages(self) -> None:
        if not self._observer:
            return
        if self._last_observed_index >= len(self._messages):
            return
        for msg in self._messages[self._last_observed_index :]:
            self._observer(msg)
        self._last_observed_index = len(self._messages)

    def clean_history(self) -> None:
        ACCEPTABLE_HISTORY_SIZE = 2
        if len(self._messages) < ACCEPTABLE_HISTORY_SIZE:
            return
        self._fill_missing_tool_responses()
        self._ensure_assistant_after_tools()

    def conversation_without_memory(self) -> list[LLMMessage]:
        return [
            msg
            for msg in self._messages[1:]
            if not SessionMemory.is_memory_message(msg)
        ]

    def split_history_for_memory(
        self, *, force_full: bool = False
    ) -> tuple[list[LLMMessage], list[LLMMessage]]:
        conversation = self.conversation_without_memory()
        min_recent = 0 if force_full else 4
        if len(conversation) <= min_recent:
            return [], conversation

        cutoff = len(conversation) - min_recent
        return conversation[:cutoff], conversation[cutoff:]

    def has_compressible_history(self, *, force_full: bool = False) -> bool:
        memory_candidates, _ = self.split_history_for_memory(force_full=force_full)
        return bool(memory_candidates)

    def rebuild_messages_with_memory(self, recent_messages: Sequence[LLMMessage]) -> None:
        system_message = self._messages[0]
        filtered_recent = [
            msg for msg in recent_messages if not SessionMemory.is_memory_message(msg)
        ]
        rebuilt = [system_message, *self.session_memory.as_messages(), *filtered_recent]
        self._messages = rebuilt
        self._memory_synced = True

    def reset_observer_view(self) -> None:
        self._last_observed_index = 0
        if self._observer:
            for msg in self._messages:
                self._observer(msg)
            self._last_observed_index = len(self._messages)

    def _fill_missing_tool_responses(self) -> None:
        i = 1
        while i < len(self._messages):  # noqa: PLR1702
            msg = self._messages[i]

            if msg.role == Role.assistant and msg.tool_calls:
                expected_responses = len(msg.tool_calls)

                if expected_responses > 0:
                    actual_responses = 0
                    j = i + 1
                    while j < len(self._messages) and self._messages[j].role == Role.tool:
                        actual_responses += 1
                        j += 1

                    if actual_responses < expected_responses:
                        insertion_point = i + 1 + actual_responses

                        for call_idx in range(actual_responses, expected_responses):
                            tool_call_data = msg.tool_calls[call_idx]

                            empty_response = LLMMessage(
                                role=Role.tool,
                                tool_call_id=tool_call_data.id or "",
                                name=(tool_call_data.function.name or "")
                                if tool_call_data.function
                                else "",
                                content=str(
                                    get_user_cancellation_message(
                                        CancellationReason.TOOL_NO_RESPONSE
                                    )
                                ),
                            )

                            self._messages.insert(insertion_point, empty_response)
                            insertion_point += 1

                    i = i + 1 + expected_responses
                    continue

            i += 1

    def _ensure_assistant_after_tools(self) -> None:
        MIN_MESSAGE_SIZE = 2
        if len(self._messages) < MIN_MESSAGE_SIZE:
            return

        last_msg = self._messages[-1]
        if last_msg.role is Role.tool:
            empty_assistant_msg = LLMMessage(role=Role.assistant, content="Understood.")
            self._messages.append(empty_assistant_msg)
