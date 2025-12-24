"""Plan scheduler for managing complex multi-plan workflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vibe.core.utils import logger

if TYPE_CHECKING:
    from vibe.core.planner import PlanState


class ScheduledPlanStatus(StrEnum):
    """Status of a scheduled plan."""

    QUEUED = auto()  # Waiting to be executed
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()  # Failed during execution
    CANCELLED = auto()  # Cancelled before completion
    BLOCKED = auto()  # Blocked by dependencies


@dataclass
class ScheduledPlan:
    """A plan in the scheduler queue."""

    schedule_id: str
    goal: str
    priority: int = 0  # Higher priority = executed first
    status: ScheduledPlanStatus = ScheduledPlanStatus.QUEUED
    depends_on: list[str] = field(default_factory=list)  # List of schedule_ids
    plan_state: PlanState | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_runnable(self) -> bool:
        """Check if this plan can be executed."""
        return self.status == ScheduledPlanStatus.QUEUED

    @property
    def is_terminal(self) -> bool:
        """Check if this plan is in a terminal state."""
        return self.status in (
            ScheduledPlanStatus.COMPLETED,
            ScheduledPlanStatus.FAILED,
            ScheduledPlanStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Get the execution duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


@dataclass
class SchedulerStats:
    """Statistics for the scheduler."""

    total_queued: int = 0
    total_running: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0


class PlanScheduler:
    """Manages a queue of plans for sequential or prioritized execution.

    Features:
    - Priority-based scheduling (higher priority plans run first)
    - Dependency management (plans can depend on other plans)
    - Pause/resume support
    - Plan cancellation

    Usage:
        scheduler = PlanScheduler()
        plan_id = scheduler.enqueue("Build the feature", priority=1)
        scheduler.enqueue("Write tests", depends_on=[plan_id])

        # In execution loop
        while True:
            plan = scheduler.get_next_runnable()
            if plan:
                scheduler.mark_running(plan.schedule_id)
                # Execute plan...
                scheduler.mark_completed(plan.schedule_id)
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        """Initialize the scheduler.

        Args:
            max_concurrent: Maximum number of plans that can run concurrently.
        """
        self._queue: dict[str, ScheduledPlan] = {}
        self._max_concurrent = max_concurrent
        self._paused = False
        self._lock = asyncio.Lock()

    @property
    def is_paused(self) -> bool:
        """Check if the scheduler is paused."""
        return self._paused

    @property
    def queue_size(self) -> int:
        """Get the number of plans in the queue."""
        return len(self._queue)

    def enqueue(
        self,
        goal: str,
        priority: int = 0,
        depends_on: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a plan to the queue.

        Args:
            goal: The goal for the plan.
            priority: Priority (higher = runs first).
            depends_on: List of schedule_ids this plan depends on.
            metadata: Optional metadata for the plan.

        Returns:
            The schedule_id of the queued plan.
        """
        schedule_id = f"sched-{uuid4().hex[:8]}"

        # Validate dependencies exist
        if depends_on:
            for dep_id in depends_on:
                if dep_id not in self._queue:
                    logger.warning(
                        "Dependency %s not found for plan %s",
                        dep_id,
                        schedule_id,
                    )

        plan = ScheduledPlan(
            schedule_id=schedule_id,
            goal=goal,
            priority=priority,
            depends_on=depends_on or [],
            metadata=metadata or {},
        )

        self._queue[schedule_id] = plan
        logger.info(
            "Queued plan %s: %s (priority=%d, deps=%s)",
            schedule_id,
            goal[:50],
            priority,
            depends_on or [],
        )

        return schedule_id

    def get_next_runnable(self) -> ScheduledPlan | None:
        """Get the next runnable plan from the queue.

        Returns:
            The next plan to execute, or None if none are runnable.
        """
        if self._paused:
            return None

        # Check concurrent limit
        running_count = sum(
            1 for p in self._queue.values()
            if p.status == ScheduledPlanStatus.RUNNING
        )
        if running_count >= self._max_concurrent:
            return None

        # Find runnable plans (queued with all dependencies satisfied)
        runnable: list[ScheduledPlan] = []
        for plan in self._queue.values():
            if not plan.is_runnable:
                continue

            # Check dependencies
            deps_satisfied = all(
                self._queue.get(dep_id, ScheduledPlan(schedule_id="", goal="")).status
                == ScheduledPlanStatus.COMPLETED
                for dep_id in plan.depends_on
            )

            if deps_satisfied:
                runnable.append(plan)
            elif any(
                self._queue.get(dep_id, ScheduledPlan(schedule_id="", goal="")).status
                in (ScheduledPlanStatus.FAILED, ScheduledPlanStatus.CANCELLED)
                for dep_id in plan.depends_on
            ):
                # Dependency failed - block this plan
                plan.status = ScheduledPlanStatus.BLOCKED
                plan.error = "Blocked due to failed dependency"

        if not runnable:
            return None

        # Sort by priority (descending) then by creation time (ascending)
        runnable.sort(key=lambda p: (-p.priority, p.created_at))
        return runnable[0]

    def mark_running(self, schedule_id: str) -> bool:
        """Mark a plan as running.

        Args:
            schedule_id: The schedule ID.

        Returns:
            True if the plan was marked running, False otherwise.
        """
        plan = self._queue.get(schedule_id)
        if not plan or plan.status != ScheduledPlanStatus.QUEUED:
            return False

        plan.status = ScheduledPlanStatus.RUNNING
        plan.started_at = datetime.utcnow()
        logger.info("Plan %s started", schedule_id)
        return True

    def mark_completed(
        self,
        schedule_id: str,
        plan_state: PlanState | None = None,
    ) -> bool:
        """Mark a plan as completed.

        Args:
            schedule_id: The schedule ID.
            plan_state: The final plan state.

        Returns:
            True if the plan was marked completed, False otherwise.
        """
        plan = self._queue.get(schedule_id)
        if not plan or plan.status != ScheduledPlanStatus.RUNNING:
            return False

        plan.status = ScheduledPlanStatus.COMPLETED
        plan.completed_at = datetime.utcnow()
        plan.plan_state = plan_state
        logger.info(
            "Plan %s completed in %.1fs",
            schedule_id,
            plan.duration_seconds or 0,
        )
        return True

    def mark_failed(self, schedule_id: str, error: str) -> bool:
        """Mark a plan as failed.

        Args:
            schedule_id: The schedule ID.
            error: The error message.

        Returns:
            True if the plan was marked failed, False otherwise.
        """
        plan = self._queue.get(schedule_id)
        if not plan or plan.status != ScheduledPlanStatus.RUNNING:
            return False

        plan.status = ScheduledPlanStatus.FAILED
        plan.completed_at = datetime.utcnow()
        plan.error = error
        logger.error("Plan %s failed: %s", schedule_id, error)
        return True

    def cancel(self, schedule_id: str) -> bool:
        """Cancel a queued or running plan.

        Args:
            schedule_id: The schedule ID.

        Returns:
            True if the plan was cancelled, False otherwise.
        """
        plan = self._queue.get(schedule_id)
        if not plan or plan.is_terminal:
            return False

        plan.status = ScheduledPlanStatus.CANCELLED
        plan.completed_at = datetime.utcnow()
        logger.info("Plan %s cancelled", schedule_id)
        return True

    def cancel_all(self) -> int:
        """Cancel all queued and running plans.

        Returns:
            The number of plans cancelled.
        """
        count = 0
        for plan in self._queue.values():
            if not plan.is_terminal:
                plan.status = ScheduledPlanStatus.CANCELLED
                plan.completed_at = datetime.utcnow()
                count += 1

        logger.info("Cancelled %d plans", count)
        return count

    def pause(self) -> None:
        """Pause the scheduler (no new plans will start)."""
        self._paused = True
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume the scheduler."""
        self._paused = False
        logger.info("Scheduler resumed")

    def get_plan(self, schedule_id: str) -> ScheduledPlan | None:
        """Get a plan by its schedule ID."""
        return self._queue.get(schedule_id)

    def get_all_plans(self) -> list[ScheduledPlan]:
        """Get all plans in the queue."""
        return list(self._queue.values())

    def get_queued_plans(self) -> list[ScheduledPlan]:
        """Get all queued plans."""
        return [p for p in self._queue.values() if p.status == ScheduledPlanStatus.QUEUED]

    def get_running_plans(self) -> list[ScheduledPlan]:
        """Get all running plans."""
        return [p for p in self._queue.values() if p.status == ScheduledPlanStatus.RUNNING]

    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        stats = SchedulerStats()
        for plan in self._queue.values():
            match plan.status:
                case ScheduledPlanStatus.QUEUED:
                    stats.total_queued += 1
                case ScheduledPlanStatus.RUNNING:
                    stats.total_running += 1
                case ScheduledPlanStatus.COMPLETED:
                    stats.total_completed += 1
                case ScheduledPlanStatus.FAILED:
                    stats.total_failed += 1
                case ScheduledPlanStatus.CANCELLED:
                    stats.total_cancelled += 1
        return stats

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled plans from the queue.

        Returns:
            The number of plans removed.
        """
        to_remove = [
            sid for sid, plan in self._queue.items()
            if plan.is_terminal
        ]
        for sid in to_remove:
            del self._queue[sid]
        return len(to_remove)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the scheduler state to a dict."""
        return {
            "paused": self._paused,
            "max_concurrent": self._max_concurrent,
            "plans": [
                {
                    "schedule_id": p.schedule_id,
                    "goal": p.goal,
                    "priority": p.priority,
                    "status": p.status.value,
                    "depends_on": p.depends_on,
                    "created_at": p.created_at.isoformat(),
                    "started_at": p.started_at.isoformat() if p.started_at else None,
                    "completed_at": p.completed_at.isoformat() if p.completed_at else None,
                    "error": p.error,
                    "metadata": p.metadata,
                }
                for p in self._queue.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanScheduler:
        """Deserialize scheduler state from a dict."""
        scheduler = cls(max_concurrent=data.get("max_concurrent", 1))
        scheduler._paused = data.get("paused", False)

        for plan_data in data.get("plans", []):
            plan = ScheduledPlan(
                schedule_id=plan_data["schedule_id"],
                goal=plan_data["goal"],
                priority=plan_data.get("priority", 0),
                status=ScheduledPlanStatus(plan_data.get("status", "queued")),
                depends_on=plan_data.get("depends_on", []),
                created_at=datetime.fromisoformat(plan_data["created_at"]),
                started_at=(
                    datetime.fromisoformat(plan_data["started_at"])
                    if plan_data.get("started_at")
                    else None
                ),
                completed_at=(
                    datetime.fromisoformat(plan_data["completed_at"])
                    if plan_data.get("completed_at")
                    else None
                ),
                error=plan_data.get("error"),
                metadata=plan_data.get("metadata", {}),
            )
            scheduler._queue[plan.schedule_id] = plan

        return scheduler
