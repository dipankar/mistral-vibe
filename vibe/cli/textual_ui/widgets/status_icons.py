"""Shared status icon provider for UI widgets."""

from __future__ import annotations

from enum import StrEnum


class StatusIcon(StrEnum):
    """Standard status icons used across UI components."""

    COMPLETED = "[green]\u2713[/green]"  # ✓
    IN_PROGRESS = "[yellow]\u25b6[/yellow]"  # ▶
    BLOCKED = "[red]\u2717[/red]"  # ✗
    PENDING = "[dim]\u25cb[/dim]"  # ○
    NEEDS_DECISION = "[cyan]?[/cyan]"
    ACTIVE = "[green]\u25cf[/green]"  # ●
    PAUSED = "[yellow]\u23f8[/yellow]"  # ⏸


class ProgressBarStyle:
    """Progress bar styling constants."""

    FILLED = "\u2588"  # █
    EMPTY = "\u2591"  # ░
    DEFAULT_WIDTH = 15


def get_step_status_icon(status: str) -> str:
    """Get icon for a plan/todo step status.

    Args:
        status: Status string (completed, in_progress, blocked, pending, needs_decision, cancelled)

    Returns:
        Rich-formatted icon string
    """
    status_lower = status.lower()
    match status_lower:
        case "completed":
            return StatusIcon.COMPLETED
        case "in_progress":
            return StatusIcon.IN_PROGRESS
        case "blocked" | "failed" | "cancelled":
            return StatusIcon.BLOCKED
        case "needs_decision":
            return StatusIcon.NEEDS_DECISION
        case "pending" | _:
            return StatusIcon.PENDING


def get_plan_status_icon(status: str) -> str:
    """Get icon for overall plan status.

    Args:
        status: Plan status string (active, paused, completed, cancelled)

    Returns:
        Rich-formatted icon string
    """
    status_lower = status.lower()
    match status_lower:
        case "active":
            return StatusIcon.ACTIVE
        case "paused":
            return StatusIcon.PAUSED
        case "completed":
            return StatusIcon.COMPLETED
        case "cancelled":
            return StatusIcon.BLOCKED
        case _:
            return StatusIcon.PENDING


def render_progress_bar(
    completed: int,
    total: int,
    width: int = ProgressBarStyle.DEFAULT_WIDTH,
) -> str:
    """Render a text-based progress bar.

    Args:
        completed: Number of completed items
        total: Total number of items
        width: Width of the progress bar in characters

    Returns:
        Progress bar string like "[███░░░░░░░░░░░░] 3/10"
    """
    if total <= 0:
        return f"[{ProgressBarStyle.EMPTY * width}] 0/0"

    filled = int(width * completed / total)
    bar = ProgressBarStyle.FILLED * filled + ProgressBarStyle.EMPTY * (width - filled)
    return f"[{bar}] {completed}/{total}"


def render_status_summary(
    completed: int = 0,
    in_progress: int = 0,
    pending: int = 0,
) -> str:
    """Render a compact status summary.

    Args:
        completed: Number of completed items
        in_progress: Number of in-progress items
        pending: Number of pending items

    Returns:
        Formatted status string like "✓3 ▶1 ○5"
    """
    parts = []
    if completed:
        parts.append(f"{StatusIcon.COMPLETED}{completed}")
    if in_progress:
        parts.append(f"{StatusIcon.IN_PROGRESS}{in_progress}")
    if pending:
        parts.append(f"{StatusIcon.PENDING}{pending}")
    return " ".join(parts) if parts else StatusIcon.PENDING
