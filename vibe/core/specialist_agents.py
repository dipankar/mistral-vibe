"""Specialized subagent types for different task categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from vibe.core.modes import ExecutionMode, get_subagent_instructions
from vibe.core.subagent import SubAgent, SubAgentConfig, SubAgentResult

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.llm.types import BackendLike
    from vibe.core.types import ApprovalCallback


class SpecialistType(StrEnum):
    """Types of specialist agents."""

    CODE = auto()  # Implementation specialist
    TEST = auto()  # Validation specialist
    RESEARCH = auto()  # Research strategist
    DESIGN = auto()  # Design architect
    DOCS = auto()  # Documentation specialist
    RUN = auto()  # Execution specialist


@dataclass
class SpecialistConfig:
    """Configuration for a specialist agent."""

    specialist_type: SpecialistType
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    max_tokens: int = 50000
    max_turns: int = 10
    extra_instructions: str = ""

    @property
    def mode(self) -> str:
        """Get the execution mode for this specialist."""
        return self.specialist_type.value

    def to_subagent_config(self, step_id: str, title: str) -> SubAgentConfig:
        """Convert to SubAgentConfig."""
        return SubAgentConfig(
            step_id=step_id,
            title=title,
            mode=self.mode,
            allowed_tools=self.allowed_tools,
            denied_tools=self.denied_tools,
            max_tokens=self.max_tokens,
            max_turns=self.max_turns,
            system_prompt_suffix=self.extra_instructions,
        )


# Predefined specialist configurations
SPECIALIST_CONFIGS: dict[SpecialistType, SpecialistConfig] = {
    SpecialistType.CODE: SpecialistConfig(
        specialist_type=SpecialistType.CODE,
        allowed_tools=["read_file", "write_file", "search_replace", "grep", "bash"],
        max_tokens=60000,
        max_turns=15,
        extra_instructions=(
            "Focus on writing clean, maintainable code. "
            "Follow existing patterns and conventions in the codebase. "
            "Run tests if available to verify your changes."
        ),
    ),
    SpecialistType.TEST: SpecialistConfig(
        specialist_type=SpecialistType.TEST,
        allowed_tools=["read_file", "write_file", "search_replace", "grep", "bash"],
        max_tokens=50000,
        max_turns=12,
        extra_instructions=(
            "Focus on comprehensive test coverage. "
            "Include edge cases and error scenarios. "
            "Ensure tests are deterministic and fast."
        ),
    ),
    SpecialistType.RESEARCH: SpecialistConfig(
        specialist_type=SpecialistType.RESEARCH,
        allowed_tools=["read_file", "grep"],
        denied_tools=["write_file", "search_replace", "bash"],
        max_tokens=40000,
        max_turns=8,
        extra_instructions=(
            "Gather information without modifying files. "
            "Summarize findings clearly and concisely. "
            "Identify patterns, dependencies, and potential issues."
        ),
    ),
    SpecialistType.DESIGN: SpecialistConfig(
        specialist_type=SpecialistType.DESIGN,
        allowed_tools=["read_file", "grep"],
        denied_tools=["write_file", "bash"],
        max_tokens=45000,
        max_turns=10,
        extra_instructions=(
            "Analyze trade-offs between different approaches. "
            "Consider scalability, maintainability, and performance. "
            "Provide actionable recommendations with clear rationale."
        ),
    ),
    SpecialistType.DOCS: SpecialistConfig(
        specialist_type=SpecialistType.DOCS,
        allowed_tools=["read_file", "write_file", "search_replace", "grep"],
        denied_tools=["bash"],
        max_tokens=40000,
        max_turns=8,
        extra_instructions=(
            "Update documentation to be accurate and user-friendly. "
            "Include examples where helpful. "
            "Keep formatting consistent with existing docs."
        ),
    ),
    SpecialistType.RUN: SpecialistConfig(
        specialist_type=SpecialistType.RUN,
        allowed_tools=["read_file", "bash", "grep"],
        denied_tools=["write_file", "search_replace"],
        max_tokens=30000,
        max_turns=6,
        extra_instructions=(
            "Execute commands carefully and capture output. "
            "Validate that expected outcomes are achieved. "
            "Report any errors or warnings clearly."
        ),
    ),
}


def get_specialist_config(specialist_type: SpecialistType | str) -> SpecialistConfig:
    """Get the configuration for a specialist type.

    Args:
        specialist_type: The specialist type (enum or string).

    Returns:
        The specialist configuration, defaulting to CODE if not found.
    """
    if isinstance(specialist_type, str):
        try:
            specialist_type = SpecialistType(specialist_type.lower())
        except ValueError:
            specialist_type = SpecialistType.CODE

    return SPECIALIST_CONFIGS.get(specialist_type, SPECIALIST_CONFIGS[SpecialistType.CODE])


def create_specialist_agent(
    config: VibeConfig,
    specialist_type: SpecialistType | str,
    step_id: str,
    title: str,
    system_prompt: str,
    parent_approval_callback: ApprovalCallback | None = None,
    backend: BackendLike | None = None,
    enable_streaming: bool = True,
    extra_instructions: str = "",
) -> SubAgent:
    """Create a specialist subagent.

    Args:
        config: Vibe configuration.
        specialist_type: Type of specialist to create.
        step_id: Unique step identifier.
        title: Human-readable step title.
        system_prompt: Base system prompt.
        parent_approval_callback: Approval callback from parent.
        backend: LLM backend to use.
        enable_streaming: Whether to enable streaming.
        extra_instructions: Additional instructions for this specific task.

    Returns:
        Configured SubAgent instance.
    """
    specialist_config = get_specialist_config(specialist_type)

    # Merge extra instructions
    full_instructions = specialist_config.extra_instructions
    if extra_instructions:
        full_instructions = f"{full_instructions}\n\n{extra_instructions}"

    subagent_config = SubAgentConfig(
        step_id=step_id,
        title=title,
        mode=specialist_config.mode,
        allowed_tools=specialist_config.allowed_tools,
        denied_tools=specialist_config.denied_tools,
        max_tokens=specialist_config.max_tokens,
        max_turns=specialist_config.max_turns,
        system_prompt_suffix=full_instructions,
    )

    return SubAgent(
        config=config,
        subagent_config=subagent_config,
        system_prompt=system_prompt,
        parent_approval_callback=parent_approval_callback,
        backend=backend,
        enable_streaming=enable_streaming,
    )


# Convenience classes for type-specific creation
class CodeSpecialist:
    """Factory for code implementation specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.CODE,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )


class TestSpecialist:
    """Factory for test/validation specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.TEST,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )


class ResearchSpecialist:
    """Factory for research/analysis specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.RESEARCH,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )


class DesignSpecialist:
    """Factory for design/architecture specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.DESIGN,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )


class DocsSpecialist:
    """Factory for documentation specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.DOCS,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )


class RunSpecialist:
    """Factory for execution/run specialists."""

    @staticmethod
    def create(
        config: VibeConfig,
        step_id: str,
        title: str,
        system_prompt: str,
        **kwargs,
    ) -> SubAgent:
        return create_specialist_agent(
            config=config,
            specialist_type=SpecialistType.RUN,
            step_id=step_id,
            title=title,
            system_prompt=system_prompt,
            **kwargs,
        )
