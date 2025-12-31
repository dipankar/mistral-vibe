from __future__ import annotations

from abc import ABC
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from vibe.core.tools.base import BaseTool


@dataclass
class ResumeSessionInfo:
    type: Literal["continue", "resume"]
    session_id: str
    session_time: str

    def message(self) -> str:
        action = None
        match self.type:
            case "continue":
                action = "Continuing"
            case "resume":
                action = "Resuming"
        return f"{action} session `{self.session_id}` from {self.session_time}"


class AgentStats(BaseModel):
    steps: int = 0
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    tool_calls_agreed: int = 0
    tool_calls_rejected: int = 0
    tool_calls_failed: int = 0
    tool_calls_succeeded: int = 0

    context_tokens: int = 0

    last_turn_prompt_tokens: int = 0
    last_turn_completion_tokens: int = 0
    last_turn_duration: float = 0.0
    tokens_per_second: float = 0.0

    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0

    @computed_field
    @property
    def session_total_llm_tokens(self) -> int:
        return self.session_prompt_tokens + self.session_completion_tokens

    @computed_field
    @property
    def last_turn_total_tokens(self) -> int:
        return self.last_turn_prompt_tokens + self.last_turn_completion_tokens

    @computed_field
    @property
    def session_cost(self) -> float:
        """Calculate the total session cost in dollars based on token usage and pricing.

        NOTE: This is a rough estimate and is worst-case scenario.
        The actual cost may be lower due to prompt caching.
        If the model changes mid-session, this uses current pricing for all tokens.
        """
        input_cost = (
            self.session_prompt_tokens / 1_000_000
        ) * self.input_price_per_million
        output_cost = (
            self.session_completion_tokens / 1_000_000
        ) * self.output_price_per_million
        return input_cost + output_cost

    def update_pricing(self, input_price: float, output_price: float) -> None:
        """Update pricing info when model changes.

        NOTE: session_cost will be recalculated using new pricing for all
        accumulated tokens. This is a known approximation when models change.
        This should not be a big issue, pricing is only used for max_price which is in
        programmatic mode, so user should not update models there.
        """
        self.input_price_per_million = input_price
        self.output_price_per_million = output_price

    def reset_context_state(self) -> None:
        """Reset context-related fields while preserving cumulative session stats.

        Used after config reload or similar operations where the context
        changes but we want to preserve session totals.
        """
        self.context_tokens = 0
        self.last_turn_prompt_tokens = 0
        self.last_turn_completion_tokens = 0
        self.last_turn_duration = 0.0
        self.tokens_per_second = 0.0


class SessionInfo(BaseModel):
    session_id: str
    start_time: str
    message_count: int
    stats: AgentStats
    save_dir: str


class SessionMetadata(BaseModel):
    session_id: str
    start_time: str
    end_time: str | None
    git_commit: str | None
    git_branch: str | None
    environment: dict[str, str | None]
    auto_approve: bool = False
    username: str


StrToolChoice = Literal["auto", "none", "any", "required"]


class AvailableFunction(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class AvailableTool(BaseModel):
    type: Literal["function"] = "function"
    function: AvailableFunction


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    id: str | None = None
    index: int | None = None
    function: FunctionCall = Field(default_factory=FunctionCall)
    type: str = "function"


def _content_before(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts: list[str] = []
        for p in v:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(v)


Content = Annotated[str, BeforeValidator(_content_before)]


class Role(StrEnum):
    system = auto()
    user = auto()
    assistant = auto()
    tool = auto()


class ApprovalResponse(StrEnum):
    YES = "y"
    NO = "n"


class ToolDisplayDestination(StrEnum):
    """Where tool results should be displayed in the UI."""

    CHAT = auto()  # Main chat/messages area
    SIDEBAR = auto()  # Sidebar panel (e.g., todos)


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: Content | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_any(cls, v: Any) -> dict[str, Any] | Any:
        if isinstance(v, dict):
            v.setdefault("content", "")
            v.setdefault("role", "assistant")
            return v
        return {
            "role": str(getattr(v, "role", "assistant")),
            "content": getattr(v, "content", ""),
            "tool_calls": getattr(v, "tool_calls", None),
            "name": getattr(v, "name", None),
            "tool_call_id": getattr(v, "tool_call_id", None),
        }


class LLMUsage(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMChunk(BaseModel):
    model_config = ConfigDict(frozen=True)
    message: LLMMessage
    finish_reason: str | None = None
    usage: LLMUsage | None = None


class BaseEvent(BaseModel, ABC):
    """Abstract base class for all agent events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    # Optional subagent context for events from parallel step execution
    subagent_id: str | None = None
    subagent_step_id: str | None = None


class AssistantEvent(BaseEvent):
    content: str
    stopped_by_middleware: bool = False


class ToolCallEvent(BaseEvent):
    tool_name: str
    tool_class: type[BaseTool]
    args: BaseModel
    tool_call_id: str


class ToolResultEvent(BaseEvent):
    tool_name: str
    tool_class: type[BaseTool] | None
    result: BaseModel | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    duration: float | None = None
    tool_call_id: str
    display_destination: ToolDisplayDestination = ToolDisplayDestination.CHAT


class CompactStartEvent(BaseEvent):
    current_context_tokens: int
    threshold: int
    preemptive: bool = False


class CompactEndEvent(BaseEvent):
    old_context_tokens: int
    new_context_tokens: int
    summary_length: int
    preemptive: bool = False


class PlanEvent(BaseEvent):
    plan_id: str


class PlanStartedEvent(PlanEvent):
    goal: str
    summary: str
    steps: list[str]


class PlanStepUpdateEvent(PlanEvent):
    step_id: str
    title: str
    status: str
    notes: str | None = None
    mode: str | None = None


class DecisionOptionData(BaseModel):
    """Rich option with label and description for Claude Code-style forms."""
    label: str
    description: str = ""


class PlanDecisionEvent(PlanEvent):
    decision_id: str
    header: str = "Decision"  # Short chip label like "Database", "Auth Type"
    question: str
    options: list[DecisionOptionData] = Field(default_factory=list)
    multi_select: bool = False
    resolved: bool = False
    selections: list[str] = Field(default_factory=list)

    @property
    def selection(self) -> str | None:
        """Backward compatibility: return first selection."""
        return self.selections[0] if self.selections else None


class SubagentProgressEvent(BaseEvent):
    """Event emitted while a subagent is running to report token usage."""

    step_id: str
    subagent_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    activity: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class PlanResourceWarningEvent(PlanEvent):
    """Event emitted when planner resources are constrained."""

    warning_type: str  # "budget", "rate_limit", "decision_timeout"
    level: str  # "warning", "critical", "exceeded"
    message: str
    details: dict[str, str] = Field(default_factory=dict)


class PlanCompletedEvent(PlanEvent):
    """Event emitted when a plan completes."""

    final_status: str  # "completed", "cancelled", "failed"
    summary: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    resources_used: dict[str, str] = Field(default_factory=dict)


class MemoryEntryEvent(BaseEvent):
    entry_index: int
    summary: str
    token_count: int = 0
    task_hints: list[str] = Field(default_factory=list)


class OutputFormat(StrEnum):
    TEXT = auto()
    JSON = auto()
    STREAMING = auto()


type AsyncApprovalCallback = Callable[
    [str, dict[str, Any], str], Awaitable[tuple[ApprovalResponse, str | None]]
]

type SyncApprovalCallback = Callable[
    [str, dict[str, Any], str], tuple[ApprovalResponse, str | None]
]

type ApprovalCallback = AsyncApprovalCallback | SyncApprovalCallback
