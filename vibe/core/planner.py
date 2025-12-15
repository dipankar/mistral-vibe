from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
import json
from typing import Any
from uuid import uuid4

from vibe.core.config import VibeConfig
from vibe.core.llm.backend.factory import BACKEND_FACTORY
from vibe.core.llm.types import BackendLike
from vibe.core.prompts import UtilityPrompt
from vibe.core.types import LLMMessage, Role
from vibe.core.utils import get_user_agent, logger


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
class PlanDecision:
    decision_id: str
    question: str
    options: list[str] = field(default_factory=list)
    selection: str | None = None
    resolved: bool = False


@dataclass(slots=True)
class PlanStep:
    step_id: str
    title: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    owner: str | None = None
    notes: str | None = None
    mode: str | None = None
    decision_id: str | None = None


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

    def get_plan(self) -> PlanState | None:
        return self._current_plan

    def get_runnable_steps(self) -> list[PlanStep]:
        plan = self._current_plan
        if not plan:
            return []
        return [
            step
            for step in plan.steps
            if step.status
            in (
                PlanStepStatus.PENDING,
                PlanStepStatus.IN_PROGRESS,
                PlanStepStatus.NEEDS_DECISION,
            )
        ]

    def build_step_prompt(self, step_id: str) -> str | None:
        plan = self._current_plan
        if not plan:
            return None
        step = self._get_step(step_id)
        if not step:
            return None
        mode = step.mode or "code"
        specialist = SPECIALIST_TITLES.get(mode, "the Execution Specialist")
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
        mode_guidance = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["code"])
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
        try:
            response_text = await self._request_plan(goal.strip())
            plan = self._parse_plan_response(plan_id, goal, response_text)
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

    def decide(self, decision_id: str, selection: str) -> PlanState | None:
        plan = self._current_plan
        if not plan:
            return None
        for decision in plan.decisions:
            if decision.decision_id == decision_id:
                decision.selection = selection
                decision.resolved = True
                plan.updated_at = datetime.utcnow()
                for step in plan.steps:
                    if step.decision_id == decision_id and step.status == PlanStepStatus.NEEDS_DECISION:
                        step.status = PlanStepStatus.PENDING
                break
        return plan

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
                }
                for step in self._current_plan.steps
            ],
            "decisions": [
                {
                    "id": decision.decision_id,
                    "question": decision.question,
                    "options": decision.options,
                    "selection": decision.selection,
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
            step = PlanStep(
                step_id=step_id,
                title=self._safe_str(raw_step.get("title")) or f"Step {index}",
                status=status,
                owner=owner,
                notes=notes,
                mode=mode,
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
        options = payload.get("options") or []
        option_list = [self._safe_str(opt) for opt in options if self._safe_str(opt)]
        return PlanDecision(
            decision_id=decision_id,
            question=self._safe_str(payload.get("question")) or "Need user input",
            options=option_list,
            selection=self._safe_str(payload.get("selection")),
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
            step = PlanStep(
                step_id=step_id,
                title=self._safe_str(raw.get("title")) or "Untitled Step",
                status=self._normalize_step_status(raw.get("status")),
                owner=self._safe_str(raw.get("owner")),
                notes=self._safe_str(raw.get("notes")),
                mode=self._normalize_mode(raw.get("mode")),
                decision_id=self._safe_str(raw.get("decision_id")),
            )
            steps.append(step)
        decisions: list[PlanDecision] = []
        for raw in data.get("decisions") or []:
            if not isinstance(raw, dict):
                continue
            decisions.append(
                PlanDecision(
                    decision_id=self._safe_str(raw.get("id")) or str(uuid4()),
                    question=self._safe_str(raw.get("question")) or "Need input",
                    options=[
                        self._safe_str(option) or ""
                        for option in raw.get("options") or []
                        if self._safe_str(option)
                    ],
                    selection=self._safe_str(raw.get("selection")),
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
        except Exception:
            return None
SPECIALIST_TITLES = {
    "code": "the Implementation Specialist",
    "test": "the Validation Specialist",
    "research": "the Research Strategist",
    "design": "the Design Architect",
    "docs": "the Documentation Specialist",
    "run": "the Execution Specialist",
}

MODE_INSTRUCTIONS = {
    "code": "- Focus on writing or editing code with best practices.\n- Update files and explain key changes.",
    "test": "- Emphasize testing: create or update tests, run suites, and report outcomes.\n- Highlight coverage gaps.",
    "research": "- Gather information from the repo (read files/search) and summarize findings.\n- Do not modify files unless necessary.",
    "design": "- Outline implementation approaches, trade-offs, and decisions before coding.\n- Produce actionable guidance.",
    "docs": "- Update documentation, READMEs, or comments to reflect changes.\n- Keep explanations user-friendly.",
    "run": "- Execute commands or scripts, capture output, and interpret results.\n- Validate that goals are met.",
}
