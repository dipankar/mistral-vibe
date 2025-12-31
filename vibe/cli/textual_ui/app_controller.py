from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from vibe.cli.textual_ui.planner_ui_controller import (
    PendingPlanRequest,
    PlannerUIController,
    PlanTriggerDecision,
)
from vibe.cli.textual_ui.widgets.messages import UserMessage


@dataclass
class AgentHooks:
    ensure_agent_init_task: Callable[[], asyncio.Task | None]
    mount_message: Callable[[UserMessage], Awaitable[None]]
    start_processing: Callable[[PendingPlanRequest], Awaitable[None]]


class AppController:
    """Coordinates planner checks and agent request routing."""

    def __init__(
        self,
        planner_ui: PlannerUIController,
        agent_hooks: AgentHooks,
        *,
        planner_auto_start_enabled: Callable[[], bool],
        thinking_mode_enabled: Callable[[], bool],
        should_autostart_plan: Callable[[], bool],
    ) -> None:
        self._planner_ui = planner_ui
        self._agent_hooks = agent_hooks
        self._planner_auto_start_enabled = planner_auto_start_enabled
        self._thinking_mode_enabled = thinking_mode_enabled
        self._should_autostart_plan = should_autostart_plan

    async def handle_user_message(self, text: str) -> None:
        init_task = self._agent_hooks.ensure_agent_init_task()
        pending_init = bool(init_task and not init_task.done())
        user_message = UserMessage(text, pending=pending_init)

        await self._agent_hooks.mount_message(user_message)

        if self._planner_ui.has_pending_confirmation():
            handled = await self._planner_ui.handle_confirmation_response(text)
            if handled:
                return

        thinking_mode = self._thinking_mode_enabled()
        planner_auto_start = self._planner_auto_start_enabled()
        should_prompt_for_plan = (
            (thinking_mode or planner_auto_start) and self._should_autostart_plan()
        )

        if should_prompt_for_plan:
            decision = (
                PlanTriggerDecision.PLAN
                if thinking_mode
                else self._planner_ui.assess_plan_need(text)
            )
            if decision in (PlanTriggerDecision.PLAN, PlanTriggerDecision.ASK):
                request = PendingPlanRequest(
                    goal=text,
                    user_message=user_message,
                    init_task=init_task,
                    pending_init=pending_init,
                )
                await self._planner_ui.request_confirmation(text, request)
                return

        request = PendingPlanRequest(
            goal=text,
            user_message=user_message,
            init_task=init_task,
            pending_init=pending_init,
        )
        await self._agent_hooks.start_processing(request)
