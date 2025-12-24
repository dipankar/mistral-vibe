"""Unified mode guidance catalog for planner and subagents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class ExecutionMode(StrEnum):
    """Execution modes for plan steps and subagents."""

    CODE = "code"
    TEST = "test"
    RESEARCH = "research"
    DESIGN = "design"
    DOCS = "docs"
    RUN = "run"


@dataclass(frozen=True)
class ModeGuidance:
    """Guidance configuration for an execution mode."""

    title: str  # Human-readable specialist title
    planner_guidance: str  # Instructions for planner step descriptions
    subagent_guidance: str  # Detailed instructions for subagent execution
    icon: str = ""  # Optional icon for UI


# Centralized mode catalog - single source of truth
MODE_CATALOG: dict[ExecutionMode, ModeGuidance] = {
    ExecutionMode.CODE: ModeGuidance(
        title="Implementation Specialist",
        planner_guidance=(
            "- Focus on writing or editing code with best practices.\n"
            "- Update files and explain key changes."
        ),
        subagent_guidance=(
            "Focus on writing and editing code with best practices.\n"
            "- Make precise, targeted changes\n"
            "- Update files and explain key changes\n"
            "- Run tests if available to verify changes\n"
            "- Follow existing code style and patterns"
        ),
        icon="[blue]</>[/blue]",
    ),
    ExecutionMode.TEST: ModeGuidance(
        title="Validation Specialist",
        planner_guidance=(
            "- Emphasize testing: create or update tests, run suites, and report outcomes.\n"
            "- Highlight coverage gaps."
        ),
        subagent_guidance=(
            "Emphasize testing and validation.\n"
            "- Create or update tests for the changes\n"
            "- Run test suites and report outcomes\n"
            "- Highlight coverage gaps\n"
            "- Ensure edge cases are covered"
        ),
        icon="[green]T[/green]",
    ),
    ExecutionMode.RESEARCH: ModeGuidance(
        title="Research Strategist",
        planner_guidance=(
            "- Gather information from the repo (read files/search) and summarize findings.\n"
            "- Do not modify files unless necessary."
        ),
        subagent_guidance=(
            "Gather information and summarize findings.\n"
            "- Read files and search the codebase\n"
            "- Summarize relevant findings\n"
            "- Do not modify files unless explicitly needed\n"
            "- Identify patterns and dependencies"
        ),
        icon="[cyan]?[/cyan]",
    ),
    ExecutionMode.DESIGN: ModeGuidance(
        title="Design Architect",
        planner_guidance=(
            "- Outline implementation approaches, trade-offs, and decisions before coding.\n"
            "- Produce actionable guidance."
        ),
        subagent_guidance=(
            "Outline implementation approaches and decisions.\n"
            "- Analyze trade-offs between options\n"
            "- Produce actionable guidance\n"
            "- Consider scalability and maintainability\n"
            "- Document architectural decisions"
        ),
        icon="[magenta]D[/magenta]",
    ),
    ExecutionMode.DOCS: ModeGuidance(
        title="Documentation Specialist",
        planner_guidance=(
            "- Update documentation, READMEs, or comments to reflect changes.\n"
            "- Keep explanations user-friendly."
        ),
        subagent_guidance=(
            "Update documentation to reflect changes.\n"
            "- Update READMEs and doc files\n"
            "- Add or update code comments\n"
            "- Keep explanations user-friendly\n"
            "- Ensure examples are accurate"
        ),
        icon="[yellow]#[/yellow]",
    ),
    ExecutionMode.RUN: ModeGuidance(
        title="Execution Specialist",
        planner_guidance=(
            "- Execute commands or scripts, capture output, and interpret results.\n"
            "- Validate that goals are met."
        ),
        subagent_guidance=(
            "Execute commands and validate results.\n"
            "- Run commands or scripts\n"
            "- Capture and interpret output\n"
            "- Validate that goals are met\n"
            "- Report any errors or warnings"
        ),
        icon="[red]>[/red]",
    ),
}


def get_mode_guidance(mode: str | ExecutionMode | None) -> ModeGuidance:
    """Get guidance for a mode, defaulting to CODE if not found."""
    if mode is None:
        return MODE_CATALOG[ExecutionMode.CODE]

    # Normalize string to enum
    if isinstance(mode, str):
        mode_lower = mode.lower().strip()
        # Handle "tests" -> "test" alias
        if mode_lower == "tests":
            mode_lower = "test"
        try:
            mode = ExecutionMode(mode_lower)
        except ValueError:
            return MODE_CATALOG[ExecutionMode.CODE]

    return MODE_CATALOG.get(mode, MODE_CATALOG[ExecutionMode.CODE])


def get_specialist_title(mode: str | ExecutionMode | None) -> str:
    """Get the specialist title for a mode."""
    return get_mode_guidance(mode).title


def get_planner_instructions(mode: str | ExecutionMode | None) -> str:
    """Get planner-level instructions for a mode."""
    return get_mode_guidance(mode).planner_guidance


def get_subagent_instructions(mode: str | ExecutionMode | None) -> str:
    """Get detailed subagent instructions for a mode."""
    return get_mode_guidance(mode).subagent_guidance


def get_mode_icon(mode: str | ExecutionMode | None) -> str:
    """Get the icon for a mode."""
    return get_mode_guidance(mode).icon


# Thinking mode configuration
class ThinkingDepth(StrEnum):
    """Depth levels for thinking mode."""

    MINIMAL = "minimal"  # 1-2 bullet points, very concise
    STANDARD = "standard"  # 3 bullet points, balanced
    DEEP = "deep"  # 5+ bullet points, thorough analysis


@dataclass(frozen=True)
class ThinkingModeConfig:
    """Configuration for thinking mode behavior."""

    enabled: bool = False
    depth: ThinkingDepth = ThinkingDepth.STANDARD
    max_words: int = 100

    @property
    def instructions(self) -> str:
        """Generate thinking mode instructions based on depth."""
        match self.depth:
            case ThinkingDepth.MINIMAL:
                return (
                    "You are in thinking mode. Begin with a brief 'Thoughts' section "
                    "containing 1-2 bullet points with key reasoning. Follow with an "
                    "'Answer' section. Keep Thoughts under 50 words."
                )
            case ThinkingDepth.DEEP:
                return (
                    "You are in thinking mode. Begin with a detailed 'Thoughts' section "
                    "containing 5 or more bullet points that thoroughly analyze reasoning, "
                    "risks, alternatives, and implications. Follow with a comprehensive "
                    "'Answer' section. Keep Thoughts under 200 words."
                )
            case ThinkingDepth.STANDARD | _:
                return (
                    "You are in thinking mode. Begin with a concise 'Thoughts' section "
                    "containing no more than three bullet points that outline reasoning, "
                    "risks, or next checks. After that, provide an 'Answer' section with "
                    f"actionable guidance. Keep Thoughts under {self.max_words} words."
                )


# Default instance for backward compatibility
DEFAULT_THINKING_INSTRUCTIONS = ThinkingModeConfig().instructions
