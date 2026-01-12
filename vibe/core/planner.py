from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from vibe.core.config import VibeConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.types import BackendLike
from vibe.core.modes import get_planner_instructions, get_specialist_title
from vibe.core.planner_resources import PlannerResourceManager, ResourceWarningLevel
from vibe.core.prompts import UtilityPrompt
from vibe.core.types import LLMMessage, Role
from vibe.core.utils import get_user_agent, logger

if TYPE_CHECKING:
    from vibe.core.subagent import SubAgentStats


class PlanStepStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    NEEDS_DECISION = auto()
    COMPLETED = auto()


class PlanRunStatus(StrEnum):
    IDLE = auto()
    ACTIVE = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass(slots=True)
class DecisionOption:
    """Rich option with label and description for Claude Code-style forms."""
    label: str
    description: str = ""


@dataclass(slots=True)
class PlanDecision:
    decision_id: str
    header: str  # Short chip label like "Database", "Auth Type"
    question: str
    options: list[DecisionOption] = field(default_factory=list)
    multi_select: bool = False
    selections: list[str] = field(default_factory=list)
    resolved: bool = False

    @property
    def selection(self) -> str | None:
        """Backward compatibility: return first selection."""
        return self.selections[0] if self.selections else None

    @selection.setter
    def selection(self, value: str | None) -> None:
        """Backward compatibility: set single selection."""
        if value:
            self.selections = [value]
        else:
            self.selections = []

    def option_labels(self) -> list[str]:
        """Return list of option labels for backward compat."""
        return [opt.label for opt in self.options]


@dataclass(slots=True)
class PlanStep:
    step_id: str
    title: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    owner: str | None = None
    notes: str | None = None
    mode: str | None = None
    decision_id: str | None = None
    depends_on: list[str] = field(default_factory=list)  # List of step IDs this step depends on


@dataclass(slots=True)
class PlanState:
    plan_id: str
    goal: str
    status: PlanRunStatus
    steps: list[PlanStep]
    decisions: list[PlanDecision] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def summarize(self) -> str:
        completed = sum(1 for step in self.steps if step.status == PlanStepStatus.COMPLETED)
        total = len(self.steps)
        status_label = self.status.value.replace("_", " ").lower()
        return f"Plan '{self.goal}' - {completed}/{total} steps complete ({status_label})"

    def validate_dependencies(self) -> list[str]:
        """Validate plan dependencies for circular references and missing steps.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        step_ids = {step.step_id for step in self.steps}

        # Check for missing dependencies and self-loops
        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    errors.append(
                        f"Step '{step.step_id}' depends on non-existent step '{dep_id}'"
                    )
                if dep_id == step.step_id:
                    errors.append(
                        f"Step '{step.step_id}' has circular dependency on itself"
                    )

        # Check for cycles using DFS
        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Find the step and check its dependencies
            for step in self.steps:
                if step.step_id == node:
                    for dep_id in step.depends_on:
                        if dep_id in step_ids:  # Only check valid deps
                            if dep_id not in visited:
                                if has_cycle(dep_id, visited, rec_stack):
                                    return True
                            elif dep_id in rec_stack:
                                return True
                    break

            rec_stack.remove(node)
            return False

        visited: set[str] = set()
        for step in self.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id, visited, set()):
                    errors.append(
                        f"Circular dependency detected involving step '{step.step_id}'"
                    )

        return errors


class PlannerAgent:
    """Planner orchestrator that drafts plans using the active LLM backend."""

    def __init__(
        self,
        config: VibeConfig,
        backend: BackendLike | None = None,
        prompt: str | None = None,
    ) -> None:
        self.config = config
        self._current_plan: PlanState | None = None
        self._planner_prompt = prompt or UtilityPrompt.PLANNER.read()
        self._backend_factory = (
            (lambda: backend)
            if backend
            else self._select_backend
        )
        # Resource management
        self._resource_manager: PlannerResourceManager | None = None

    def get_plan(self) -> PlanState | None:
        return self._current_plan

    @property
    def resource_manager(self) -> PlannerResourceManager:
        """Get or create the resource manager for this planner."""
        if self._resource_manager is None:
            self._resource_manager = PlannerResourceManager(config=self.config.planner)
        return self._resource_manager

    def reset_resources(self) -> None:
        """Reset resource tracking (call when starting a new plan)."""
        self._resource_manager = PlannerResourceManager(config=self.config.planner)

    def record_step_usage(self, stats: SubAgentStats) -> ResourceWarningLevel:
        """Record token usage from a completed step.

        Args:
            stats: Statistics from a SubAgent execution.

        Returns:
            The current warning level after recording usage.
        """
        return self.resource_manager.add_token_usage(
            prompt=stats.prompt_tokens,
            completion=stats.completion_tokens,
        )

    def can_start_step(self, estimated_tokens: int = 10000) -> bool:
        """Check if we have budget to start a new step."""
        return self.resource_manager.can_start_step(estimated_tokens)

    def get_resource_summary(self) -> dict[str, str]:
        """Get a summary of resource usage."""
        return self.resource_manager.get_status_summary()

    def should_pause_for_resources(self) -> tuple[bool, str]:
        """Check if execution should pause due to resource constraints."""
        return self.resource_manager.should_pause_for_resources()

    def get_runnable_steps(self) -> list[PlanStep]:
        """Get steps that are runnable (pending/in_progress with all dependencies satisfied)."""
        plan = self._current_plan
        if not plan:
            return []

        # Build set of completed step IDs
        completed_ids = {
            step.step_id
            for step in plan.steps
            if step.status == PlanStepStatus.COMPLETED
        }

        runnable = []
        for step in plan.steps:
            # Skip already completed or blocked steps
            if step.status not in (
                PlanStepStatus.PENDING,
                PlanStepStatus.IN_PROGRESS,
                PlanStepStatus.NEEDS_DECISION,
            ):
                continue

            # Check if all dependencies are satisfied
            dependencies_satisfied = all(
                dep_id in completed_ids for dep_id in step.depends_on
            )

            if dependencies_satisfied:
                runnable.append(step)

        return runnable

    def get_parallel_runnable_steps(self) -> list[PlanStep]:
        """Get steps that can be executed in parallel (no mutual dependencies)."""
        runnable = self.get_runnable_steps()

        # Filter to only pending steps (not already in_progress)
        # Multiple steps can run in parallel if they have no dependencies on each other
        pending_runnable = [
            step for step in runnable
            if step.status == PlanStepStatus.PENDING
        ]

        return pending_runnable

    def build_step_prompt(self, step_id: str) -> str | None:
        plan = self._current_plan
        if not plan:
            return None
        step = self._get_step(step_id)
        if not step:
            return None
        mode = step.mode or "code"
        specialist = get_specialist_title(mode)
        completed_steps = [
            f"{idx}. {s.title} · done ({s.notes or 'no notes'})"
            for idx, s in enumerate(plan.steps, start=1)
            if s.status == PlanStepStatus.COMPLETED
        ]
        pending_steps = [
            f"{idx}. {s.title} · {s.status.value.replace('_', ' ').title()}"
            for idx, s in enumerate(plan.steps, start=1)
            if s.step_id != step.step_id and s.status != PlanStepStatus.COMPLETED
        ]
        segments = [
            f"You are {specialist}, a specialist subagent executing a planner-defined step.",
            f"Overall Goal: {plan.goal}",
            f"Plan Summary: {plan.summarize()}",
            f"Step ID: {step.step_id}",
            f"Step Title: {step.title}",
        ]
        if step.notes:
            segments.append(f"Notes: {step.notes}")
        owner = step.owner or "planner"
        segments.append(f"Owner: {owner}")
        if completed_steps:
            segments.append("Completed steps:\n" + "\n".join(f"- {line}" for line in completed_steps))
        if pending_steps:
            segments.append("Other pending steps:\n" + "\n".join(f"- {line}" for line in pending_steps))
        if step.decision_id:
            segments.append(
                f"Decision Reference: {step.decision_id}. Ensure your work honors the decision outcome."
            )
        segments.append(
            "Operational constraints:\n"
            "- Stay ahead of context limits; compact or summarize once utilization crosses ~95% of the budget.\n"
            "- If rate limits occur, retry with exponential backoff (start at 2 seconds) before asking the user for help."
        )
        segments.append(
            "Execution guidance:\n"
            "- Use Vibe tools (read/write/search/bash) to complete this step.\n"
            "- Keep changes scoped to this step unless you discover critical blockers.\n"
            "- When finished, summarize the changes, note follow-ups, and mention any files that were touched."
        )
        mode_guidance = get_planner_instructions(mode)
        segments.append(f"Mode-specific guidance:\n{mode_guidance}")
        return "\n".join(segments)

    def update_step_status(
        self, step_id: str, status: PlanStepStatus, notes: str | None = None
    ) -> PlanStep | None:
        step = self._get_step(step_id)
        if not step:
            return None
        step.status = status
        if notes:
            step.notes = notes
        if self._current_plan:
            self._current_plan.updated_at = datetime.utcnow()
        return step

    def complete_if_possible(self) -> PlanState | None:
        plan = self._current_plan
        if not plan:
            return None
        if all(step.status == PlanStepStatus.COMPLETED for step in plan.steps):
            plan.status = PlanRunStatus.COMPLETED
            plan.updated_at = datetime.utcnow()
        return plan

    async def start_plan(self, goal: str) -> PlanState:
        plan_id = str(uuid4())

        # Reset resource tracking for the new plan
        self.reset_resources()

        try:
            response_text = await self._request_plan(goal.strip())
            plan = self._parse_plan_response(plan_id, goal, response_text)

            # Validate dependencies
            validation_errors = plan.validate_dependencies()
            if validation_errors:
                logger.warning(
                    "Plan has dependency issues: %s",
                    "; ".join(validation_errors)
                )
                # Fix invalid dependencies by removing them
                valid_step_ids = {step.step_id for step in plan.steps}
                for step in plan.steps:
                    step.depends_on = [
                        dep for dep in step.depends_on
                        if dep in valid_step_ids and dep != step.step_id
                    ]

            # Start timeout tracking for any initial decisions
            for decision in plan.decisions:
                if not decision.resolved:
                    self.resource_manager.start_decision_timeout(decision.decision_id)

        except json.JSONDecodeError as exc:
            logger.warning(
                "Planner JSON parse error (fallback triggered): %s at position %d",
                exc.msg,
                exc.pos,
            )
            plan = self._fallback_plan(plan_id, goal, f"Invalid JSON response: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Planner fallback triggered: %s", exc)
            plan = self._fallback_plan(plan_id, goal, str(exc))
        self._current_plan = plan
        return self._current_plan

    def pause(self) -> PlanState | None:
        if not self._current_plan:
            return None
        if self._current_plan.status == PlanRunStatus.ACTIVE:
            self._current_plan.status = PlanRunStatus.PAUSED
            self._current_plan.updated_at = datetime.utcnow()
        return self._current_plan

    def resume(self) -> PlanState | None:
        if not self._current_plan:
            return None
        if self._current_plan.status == PlanRunStatus.PAUSED:
            self._current_plan.status = PlanRunStatus.ACTIVE
            self._current_plan.updated_at = datetime.utcnow()
        return self._current_plan

    def cancel(self) -> None:
        if self._current_plan:
            self._current_plan.status = PlanRunStatus.CANCELLED
            self._current_plan.updated_at = datetime.utcnow()
        self._current_plan = None

    def decide(self, decision_id: str, selections: str | list[str]) -> PlanState | None:
        """Record user decision(s) for a decision checkpoint.

        Args:
            decision_id: The decision identifier
            selections: Single selection string or list of selections for multi-select
        """
        plan = self._current_plan
        if not plan:
            return None
        # Normalize to list
        selection_list = [selections] if isinstance(selections, str) else list(selections)
        for decision in plan.decisions:
            if decision.decision_id == decision_id:
                decision.selections = selection_list
                decision.resolved = True
                plan.updated_at = datetime.utcnow()

                # Resolve decision timeout tracking
                self.resource_manager.resolve_decision(decision_id)

                for step in plan.steps:
                    if step.decision_id == decision_id and step.status == PlanStepStatus.NEEDS_DECISION:
                        step.status = PlanStepStatus.PENDING
                break
        return plan

    def export_for_logging(self) -> dict[str, Any] | None:
        """Export plan state with resource info for session logging.

        Returns:
            Dict suitable for inclusion in session logs, or None if no plan.
        """
        plan_dict = self.to_dict()
        if not plan_dict:
            return None

        # Add resource summary
        plan_dict["resources"] = self.get_resource_summary()

        # Add step completion statistics
        if self._current_plan:
            completed = sum(
                1 for s in self._current_plan.steps
                if s.status == PlanStepStatus.COMPLETED
            )
            total = len(self._current_plan.steps)
            plan_dict["progress"] = {
                "completed": completed,
                "total": total,
                "percent": round(completed / total * 100, 1) if total > 0 else 0,
            }

        return plan_dict

    def to_dict(self) -> dict[str, Any] | None:
        if not self._current_plan:
            return None
        return {
            "plan_id": self._current_plan.plan_id,
            "goal": self._current_plan.goal,
            "status": self._current_plan.status.value,
            "steps": [
                {
                    "id": step.step_id,
                    "title": step.title,
                    "status": step.status.value,
                    "owner": step.owner,
                    "notes": step.notes,
                    "mode": step.mode,
                    "decision_id": step.decision_id,
                    "depends_on": step.depends_on,
                }
                for step in self._current_plan.steps
            ],
            "decisions": [
                {
                    "id": decision.decision_id,
                    "header": decision.header,
                    "question": decision.question,
                    "options": [
                        {"label": opt.label, "description": opt.description}
                        for opt in decision.options
                    ],
                    "multi_select": decision.multi_select,
                    "selections": decision.selections,
                    "resolved": decision.resolved,
                }
                for decision in self._current_plan.decisions
            ],
            "created_at": self._current_plan.created_at.isoformat(),
            "updated_at": self._current_plan.updated_at.isoformat(),
        }

    def load_from_dict(self, data: dict[str, Any]) -> PlanState:
        plan = self._deserialize_plan(data)
        self._current_plan = plan
        return plan

    async def _request_plan(self, goal: str) -> str:
        messages = [
            LLMMessage(role=Role.system, content=self._planner_prompt),
            LLMMessage(
                role=Role.user,
                content=self._format_user_prompt(goal),
            ),
        ]
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        backend = self._backend_factory()
        async with backend as client:
            chunk = await client.complete(
                model=active_model,
                messages=messages,
                temperature=min(active_model.temperature, 0.3),
                tools=None,
                tool_choice="none",
                extra_headers={
                    "user-agent": get_user_agent(provider.backend),
                    "x-affinity": f"planner-{provider.backend.value}",
                },
                max_tokens=800,
            )
        return chunk.message.content or ""

    def _format_user_prompt(self, goal: str) -> str:
        instructions = (
            "Return JSON only. Do not wrap in Markdown. "
            "Provide realistic next steps with precise filenames when relevant."
        )
        return f"GOAL:\n{goal}\n\n{instructions}"

    def _parse_plan_response(self, plan_id: str, goal: str, content: str) -> PlanState:
        data = json.loads(content)
        steps_data = data.get("steps")
        if not isinstance(steps_data, list) or not steps_data:
            raise ValueError("Planner response missing steps array")

        steps: list[PlanStep] = []
        decisions: dict[str, PlanDecision] = {}

        for index, raw_step in enumerate(steps_data, start=1):
            if not isinstance(raw_step, dict):
                continue
            step_id = str(raw_step.get("id") or f"{plan_id}-step-{index}")
            status = self._normalize_step_status(raw_step.get("status"))
            owner = self._safe_str(raw_step.get("owner"))
            notes = self._safe_str(raw_step.get("notes"))
            mode = self._normalize_mode(raw_step.get("mode"))
            # Parse dependencies
            raw_depends = raw_step.get("depends_on") or []
            depends_on = [str(d) for d in raw_depends if d]
            step = PlanStep(
                step_id=step_id,
                title=self._safe_str(raw_step.get("title")) or f"Step {index}",
                status=status,
                owner=owner,
                notes=notes,
                mode=mode,
                depends_on=depends_on,
            )
            steps.append(step)
            step_decision = raw_step.get("decision")
            if isinstance(step_decision, dict):
                decision = self._build_decision(
                    step_decision, fallback_id=f"{step_id}-decision"
                )
                decisions[decision.decision_id] = decision
                step.decision_id = decision.decision_id
                if not decision.resolved:
                    step.status = PlanStepStatus.NEEDS_DECISION

        extra_decisions = data.get("decisions")
        if isinstance(extra_decisions, list):
            for raw_decision in extra_decisions:
                if not isinstance(raw_decision, dict):
                    continue
                decision = self._build_decision(raw_decision)
                decisions.setdefault(decision.decision_id, decision)

        plan_status = self._normalize_plan_status(data.get("status"))
        plan = PlanState(
            plan_id=plan_id,
            goal=self._safe_str(data.get("goal")) or goal,
            status=plan_status,
            steps=steps,
            decisions=list(decisions.values()),
        )
        return plan

    def _build_decision(self, payload: dict[str, Any], fallback_id: str | None = None) -> PlanDecision:
        decision_id = self._safe_str(payload.get("id")) or fallback_id or str(uuid4())
        header = self._safe_str(payload.get("header")) or "Decision"
        raw_options = payload.get("options") or []
        options: list[DecisionOption] = []
        for opt in raw_options:
            if isinstance(opt, dict):
                # Rich option with label and description
                label = self._safe_str(opt.get("label")) or ""
                description = self._safe_str(opt.get("description")) or ""
                if label:
                    options.append(DecisionOption(label=label, description=description))
            elif isinstance(opt, str) and opt.strip():
                # Backward compat: simple string option
                options.append(DecisionOption(label=opt.strip(), description=""))

        # Handle selections (multi-select support)
        raw_selections = payload.get("selections") or []
        if not raw_selections:
            # Backward compat: single selection field
            single_selection = self._safe_str(payload.get("selection"))
            raw_selections = [single_selection] if single_selection else []
        selections = [s for s in raw_selections if isinstance(s, str) and s.strip()]

        return PlanDecision(
            decision_id=decision_id,
            header=header,
            question=self._safe_str(payload.get("question")) or "Need user input",
            options=options,
            multi_select=bool(payload.get("multi_select", False)),
            selections=selections,
            resolved=bool(payload.get("resolved", False)),
        )

    def _fallback_plan(self, plan_id: str, goal: str, error: str | None = None) -> PlanState:
        steps = [
            PlanStep(
                step_id=f"{plan_id}-1",
                title="Review repository context and goals",
                status=PlanStepStatus.PENDING,
                owner="planner",
                mode="research",
                notes="Summarize files, tests, and constraints relevant to the request.",
            ),
            PlanStep(
                step_id=f"{plan_id}-2",
                title="Draft detailed implementation steps",
                status=PlanStepStatus.PENDING,
                owner="planner",
                mode="design",
                notes="Break the goal into concrete edits, commands, or tests.",
            ),
            PlanStep(
                step_id=f"{plan_id}-3",
                title="Execute, verify, and summarize changes",
                status=PlanStepStatus.PENDING,
                owner="planner",
                mode="code",
                notes="Run tools/tests, validate results, and report back.",
            ),
        ]
        if error:
            steps[0].notes = f"{steps[0].notes} (planner fallback due to: {error})"
        return PlanState(
            plan_id=plan_id,
            goal=goal,
            status=PlanRunStatus.ACTIVE,
            steps=steps,
            decisions=[],
        )

    def _normalize_step_status(self, value: str | None) -> PlanStepStatus:
        if not value:
            return PlanStepStatus.PENDING
        normalized = value.replace("-", "_").replace(" ", "_").upper()
        match normalized:
            case "IN_PROGRESS":
                return PlanStepStatus.IN_PROGRESS
            case "BLOCKED":
                return PlanStepStatus.BLOCKED
            case "NEEDS_DECISION":
                return PlanStepStatus.NEEDS_DECISION
            case "COMPLETED" | "DONE":
                return PlanStepStatus.COMPLETED
            case _:
                return PlanStepStatus.PENDING

    def _normalize_plan_status(self, value: str | None) -> PlanRunStatus:
        if not value:
            return PlanRunStatus.ACTIVE
        normalized = value.replace("-", "_").replace(" ", "_").upper()
        match normalized:
            case "PAUSED":
                return PlanRunStatus.PAUSED
            case "COMPLETED":
                return PlanRunStatus.COMPLETED
            case "CANCELLED" | "CANCELED":
                return PlanRunStatus.CANCELLED
            case _:
                return PlanRunStatus.ACTIVE

    def _normalize_mode(self, value: Any) -> str | None:
        if value is None:
            return "code"
        text = str(value).strip().lower()
        if not text:
            return "code"
        allowed = {"code", "test", "tests", "research", "design", "docs", "run"}
        if text in allowed:
            return text if text != "tests" else "test"
        return "code"

    def _safe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_step(self, step_id: str) -> PlanStep | None:
        plan = self._current_plan
        if not plan:
            return None
        for step in plan.steps:
            if step.step_id == step_id:
                return step
        return None

    def _select_backend(self) -> BackendLike:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        timeout = self.config.api_timeout
        backend_cls = BACKEND_FACTORY[provider.backend]
        return backend_cls(provider=provider, timeout=timeout)

    def _deserialize_plan(self, data: dict[str, Any]) -> PlanState:
        plan_id = str(data.get("plan_id") or uuid4())
        goal = self._safe_str(data.get("goal")) or "Plan"
        status = self._normalize_plan_status(data.get("status"))
        steps: list[PlanStep] = []
        for raw in data.get("steps") or []:
            if not isinstance(raw, dict):
                continue
            step_id = str(raw.get("id") or f"{plan_id}-step-{len(steps) + 1}")
            # Parse dependencies
            raw_depends = raw.get("depends_on") or []
            depends_on = [str(d) for d in raw_depends if d]
            step = PlanStep(
                step_id=step_id,
                title=self._safe_str(raw.get("title")) or "Untitled Step",
                status=self._normalize_step_status(raw.get("status")),
                owner=self._safe_str(raw.get("owner")),
                notes=self._safe_str(raw.get("notes")),
                mode=self._normalize_mode(raw.get("mode")),
                decision_id=self._safe_str(raw.get("decision_id")),
                depends_on=depends_on,
            )
            steps.append(step)
        decisions: list[PlanDecision] = []
        for raw in data.get("decisions") or []:
            if not isinstance(raw, dict):
                continue
            # Parse options - handle both old format (list of strings) and new format (list of dicts)
            raw_options = raw.get("options") or []
            options: list[DecisionOption] = []
            for opt in raw_options:
                if isinstance(opt, dict):
                    label = self._safe_str(opt.get("label")) or ""
                    description = self._safe_str(opt.get("description")) or ""
                    if label:
                        options.append(DecisionOption(label=label, description=description))
                elif isinstance(opt, str) and opt.strip():
                    options.append(DecisionOption(label=opt.strip(), description=""))

            # Handle selections - backward compat with single selection
            raw_selections = raw.get("selections") or []
            if not raw_selections:
                single_selection = self._safe_str(raw.get("selection"))
                raw_selections = [single_selection] if single_selection else []
            selections = [s for s in raw_selections if isinstance(s, str) and s.strip()]

            decisions.append(
                PlanDecision(
                    decision_id=self._safe_str(raw.get("id")) or str(uuid4()),
                    header=self._safe_str(raw.get("header")) or "Decision",
                    question=self._safe_str(raw.get("question")) or "Need input",
                    options=options,
                    multi_select=bool(raw.get("multi_select", False)),
                    selections=selections,
                    resolved=bool(raw.get("resolved", False)),
                )
            )
        for step in steps:
            if not step.decision_id:
                continue
            match = next(
                (decision for decision in decisions if decision.decision_id == step.decision_id),
                None,
            )
            if match and not match.resolved:
                step.status = PlanStepStatus.NEEDS_DECISION
        created_at = self._parse_datetime(data.get("created_at")) or datetime.utcnow()
        updated_at = self._parse_datetime(data.get("updated_at")) or created_at
        return PlanState(
            plan_id=plan_id,
            goal=goal,
            status=status,
            steps=steps,
            decisions=decisions,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None
