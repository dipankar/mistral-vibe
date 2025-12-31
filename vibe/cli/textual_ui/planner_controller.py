from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

from textual.worker import Worker

from vibe.cli.textual_ui.state_store import UIStateStore
from vibe.core.config import VibeConfig
from vibe.core.planner import (
    PlanRunStatus,
    PlanState,
    PlanStep,
    PlanStepStatus,
    PlannerAgent,
)
from vibe.core.subagent import SubAgentResult
from vibe.core.types import ResumeSessionInfo
from vibe.core.utils import logger


@dataclass
class PlannerCallbacks:
    refresh_plan: Callable[[PlanState | None], Awaitable[None]]
    clear_plan_cards: Callable[[], Awaitable[None]]
    append_message: Callable[[str], Awaitable[None]]
    update_step_card: Callable[[PlanState, PlanStep], Awaitable[None]]
    notify_decision_required: Callable[[PlanState, PlanStep], Awaitable[None]]
    set_active_step: Callable[[str | None, str | None], None]
    run_subagent: Callable[[PlanStep, str], Awaitable[SubAgentResult]]
    wait_for_agent_idle: Callable[[], Awaitable[None]]


class PlannerController:
    """Coordinates PlannerAgent lifecycle, persistence, and step execution."""

    def __init__(
        self,
        config: VibeConfig,
        ui_store: UIStateStore,
        callbacks: PlannerCallbacks,
        *,
        run_worker: Callable[[Awaitable, bool, str | None], Worker],
        call_later: Callable[[Callable[[], None]], None],
    ) -> None:
        self._config = config
        self._ui_store = ui_store
        self._callbacks = callbacks
        self._planner_agent: PlannerAgent | None = None
        self._plan_executor_worker: Worker | None = None
        self._plan_executor_plan_id: str | None = None
        self._planner_instance_id = uuid4().hex
        self._plan_state_file = self._build_plan_state_path()
        self._run_worker = run_worker
        self._call_later = call_later
        self._step_update_condition: asyncio.Condition | None = None
        self._step_update_pending = False

    def _ensure_condition(self) -> asyncio.Condition:
        if self._step_update_condition is None:
            self._step_update_condition = asyncio.Condition()
        return self._step_update_condition

    def get_plan(self) -> PlanState | None:
        return self._planner_agent.get_plan() if self._planner_agent else None

    async def preview_plan(self, goal: str) -> PlanState | None:
        """Generate a temporary plan without mutating controller state."""
        planner = PlannerAgent(self._config)
        try:
            return await planner.start_plan(goal)
        except Exception as exc:
            logger.debug("Planner preview failed", exc_info=exc)
            return None

    async def start_plan(self, goal: str) -> PlanState:
        planner = self._ensure_planner()
        plan = await planner.start_plan(goal)
        await self._callbacks.refresh_plan(plan)
        await self._persist_plan_state_async()
        return plan

    async def start_plan_executor(self, plan_id: str) -> None:
        self.cancel_plan_executor()
        worker = self._run_worker(
            self._execute_plan_steps(plan_id),
            exclusive=False,
            name=f"plan-executor-{plan_id}",
        )
        self._plan_executor_worker = worker
        self._plan_executor_plan_id = plan_id

    def cancel_plan_executor(self) -> None:
        worker = self._plan_executor_worker
        if worker:
            worker.cancel()
            self._plan_executor_worker = None
            self._plan_executor_plan_id = None
            if not worker.is_cancelled and not worker.is_finished:
                logger.warning("Plan executor worker %s still running", worker.name)
        else:
            self._plan_executor_worker = None
            self._plan_executor_plan_id = None
        self._call_later(lambda: asyncio.create_task(self._persist_plan_state_async()))

    async def pause_plan(self) -> PlanState | None:
        planner = self._planner_agent
        if not planner:
            return None
        plan = planner.pause()
        if plan:
            await self._callbacks.refresh_plan(plan)
            await self._persist_plan_state_async()
        return plan

    async def resume_plan(self) -> PlanState | None:
        planner = self._planner_agent
        if not planner:
            return None
        plan = planner.resume()
        if plan:
            await self._callbacks.refresh_plan(plan)
            await self._persist_plan_state_async()
        return plan

    async def cancel_plan(self) -> None:
        if self._planner_agent:
            self._planner_agent.cancel()
        self.cancel_plan_executor()
        await self._callbacks.refresh_plan(None)
        await self._callbacks.clear_plan_cards()
        await self._persist_plan_state_async()

    async def decide(
        self, decision_id: str, selections: str | list[str]
    ) -> PlanState | None:
        planner = self._planner_agent
        if not planner:
            return None
        plan = planner.decide(decision_id, selections)
        if plan:
            await self._callbacks.refresh_plan(plan)
            await self._persist_plan_state_async()
        return plan

    async def load_plan_state(self, session_info: ResumeSessionInfo | None) -> None:
        if session_info is None:
            if self._plan_state_file.exists():
                try:
                    self._plan_state_file.unlink()
                except Exception:
                    pass
            return

        try:
            self._plan_state_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("Failed to ensure planner state directory", exc_info=exc)
            return

        if not self._plan_state_file.exists():
            return

        try:
            payload = json.loads(self._plan_state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to read planner state", exc_info=exc)
            return

        if not isinstance(payload, dict):
            return

        plan_data = payload.get("plan")
        if not isinstance(plan_data, dict):
            return

        owner_id = payload.get("owner_instance_id")
        owner_pid = payload.get("owner_pid")
        if owner_id and owner_id != self._planner_instance_id:
            if self._is_pid_active(owner_pid):
                logger.debug(
                    "Planner state owned by another active instance (pid=%s)",
                    owner_pid,
                )
                return

        planner = self._ensure_planner()
        try:
            plan = planner.load_from_dict(plan_data)
        except Exception as exc:
            logger.debug("Failed to deserialize planner state", exc_info=exc)
            return

        await self._callbacks.refresh_plan(plan)
        await self._callbacks.clear_plan_cards()
        await self._callbacks.append_message(
            f"Restored planning session '{plan.goal}' "
            f"({plan.status.value.replace('_', ' ').title()})."
        )
        if not self._plan_payload_owned_by_self(payload):
            await self._persist_plan_state_async()
        if plan.status == PlanRunStatus.ACTIVE:
            await self.start_plan_executor(plan.plan_id)

    async def _execute_plan_steps(self, plan_id: str) -> None:
        while True:
            planner = self._planner_agent
            if not planner:
                return
            plan = planner.get_plan()
            if not plan or plan.plan_id != plan_id:
                return
            if plan.status in (PlanRunStatus.PAUSED, PlanRunStatus.CANCELLED):
                return

            parallel_steps = planner.get_parallel_runnable_steps()
            executable_steps = [
                step
                for step in parallel_steps
                if step.status not in (PlanStepStatus.NEEDS_DECISION, PlanStepStatus.BLOCKED)
            ]

            for step in parallel_steps:
                if step.status == PlanStepStatus.NEEDS_DECISION:
                    await self._callbacks.notify_decision_required(plan, step)

            if not executable_steps:
                runnable = planner.get_runnable_steps()
                in_progress = [
                    s for s in runnable if s.status == PlanStepStatus.IN_PROGRESS
                ]
                if not in_progress and not parallel_steps:
                    await self._maybe_mark_plan_complete(plan_id)
                    return
                condition = self._ensure_condition()
                async with condition:
                    if not self._step_update_pending:
                        try:
                            await asyncio.wait_for(condition.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            pass
                    self._step_update_pending = False
                continue

            if len(executable_steps) > 1:
                await self._callbacks.append_message(
                    "Executing "
                    + ", ".join(f"`{s.step_id}`" for s in executable_steps)
                    + " in parallel."
                )
                await self._callbacks.wait_for_agent_idle()
                tasks = [
                    self._execute_single_step(step, planner, plan_id)
                    for step in executable_steps
                ]
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    return
            else:
                step = executable_steps[0]
                await self._execute_single_step(step, planner, plan_id)

            await asyncio.sleep(0)

    async def _execute_single_step(
        self, step: PlanStep, planner: PlannerAgent, plan_id: str
    ) -> None:
        plan = planner.get_plan()
        if not plan or plan.status in (PlanRunStatus.PAUSED, PlanRunStatus.CANCELLED):
            return

        await self._update_step_status(step.step_id, PlanStepStatus.IN_PROGRESS)
        self._callbacks.set_active_step(step.step_id, step.mode)

        prompt = planner.build_step_prompt(step.step_id)
        if not prompt:
            await self._update_step_status(
                step.step_id,
                PlanStepStatus.BLOCKED,
                notes="Planner missing instructions for this step.",
            )
            self._callbacks.set_active_step(None, None)
            return

        await self._callbacks.append_message(
            f"Planner executing step `{step.step_id}` · {step.title} "
            f"(SubAgent: {step.mode or 'code'})"
        )

        try:
            result = await self._callbacks.run_subagent(step, prompt)
            if result.success:
                await self._update_step_status(
                    step.step_id,
                    PlanStepStatus.COMPLETED,
                    notes=f"Completed in {result.stats.duration_seconds:.1f}s, "
                    f"{result.stats.total_tokens} tokens",
                )
            else:
                await self._update_step_status(
                    step.step_id,
                    PlanStepStatus.BLOCKED,
                    notes=result.error or "SubAgent execution failed",
                )
        except Exception as exc:
            logger.warning("SubAgent execution failed", exc_info=exc)
            await self._update_step_status(
                step.step_id,
                PlanStepStatus.BLOCKED,
                notes=f"Error: {exc}",
            )
        finally:
            self._callbacks.set_active_step(None, None)

        await self._maybe_mark_plan_complete(plan_id)

    async def _update_step_status(
        self, step_id: str, status: PlanStepStatus, notes: str | None = None
    ) -> None:
        planner = self._planner_agent
        if not planner:
            return
        step = planner.update_step_status(step_id, status, notes)
        if not step:
            return
        plan = planner.get_plan()
        await self._callbacks.refresh_plan(plan)
        if plan:
            await self._callbacks.update_step_card(plan, step)
        condition = self._ensure_condition()
        async with condition:
            self._step_update_pending = True
            condition.notify_all()

    async def _maybe_mark_plan_complete(self, plan_id: str) -> None:
        planner = self._planner_agent
        if not planner:
            return
        plan = planner.complete_if_possible()
        if not plan or plan.plan_id != plan_id:
            return
        if plan.status == PlanRunStatus.COMPLETED:
            await self._callbacks.refresh_plan(plan)
            await self._callbacks.append_message(
                f"Planner goal '{plan.goal}' completed."
            )
            self.cancel_plan_executor()
            await self._callbacks.clear_plan_cards()
            await self._persist_plan_state_async()

    async def _persist_plan_state_async(self) -> None:
        await asyncio.to_thread(self._persist_plan_state_sync)

    def _persist_plan_state_sync(self) -> None:
        plan_data = None
        if self._planner_agent:
            plan_data = self._planner_agent.to_dict()
        if plan_data:
            payload = {
                "owner_instance_id": self._planner_instance_id,
                "owner_pid": os.getpid(),
                "plan": plan_data,
            }
            try:
                self._plan_state_file.parent.mkdir(parents=True, exist_ok=True)
                self._plan_state_file.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug("Failed to persist plan state", exc_info=exc)
            return

        if not self._plan_state_file.exists():
            return

        try:
            payload = json.loads(self._plan_state_file.read_text(encoding="utf-8"))
        except Exception:
            payload = None

        if payload and not self._plan_payload_owned_by_self(payload):
            return

        try:
            self._plan_state_file.unlink()
        except Exception:
            pass

    def _ensure_planner(self) -> PlannerAgent:
        if not self._planner_agent:
            self._planner_agent = PlannerAgent(self._config)
        return self._planner_agent

    def _planner_workspace_slug(self) -> str:
        workdir = str(self._config.effective_workdir.resolve())
        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "-",
            self._config.effective_workdir.name or "workspace",
        ).strip("-")
        if not safe_name:
            safe_name = "workspace"
        digest = hashlib.sha256(workdir.encode("utf-8")).hexdigest()[:10]
        return f"{safe_name.lower()}-{digest}"

    def _build_plan_state_path(self) -> Path:
        return Path.home() / ".vibe" / "plans" / f"{self._planner_workspace_slug()}.json"

    def _plan_payload_owned_by_self(self, payload: dict[str, object]) -> bool:
        owner_id = payload.get("owner_instance_id")
        return not owner_id or owner_id == self._planner_instance_id

    def _is_pid_active(self, pid: int | None) -> bool:
        if not pid or pid < 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
