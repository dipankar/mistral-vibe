from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Awaitable, Callable

from vibe.cli.textual_ui.widgets.messages import UserCommandMessage
from vibe.core.types import LLMMessage

logger = logging.getLogger(__name__)


class PlanTriggerDecision(StrEnum):
    PLAN = "plan"
    SKIP = "skip"
    ASK = "ask"


@dataclass
class PendingPlanRequest:
    goal: str
    user_message: LLMMessage
    init_task: asyncio.Task | None
    pending_init: bool


@dataclass
class PlanConfirmationHooks:
    send_user_message: Callable[[UserCommandMessage], Awaitable[None]]
    start_planning: Callable[[str], Awaitable[None]]
    continue_request: Callable[[PendingPlanRequest], Awaitable[None]]
    get_thinking_mode: Callable[[], bool]


class PlannerUIController:
    """Coordinates planner confirmation UX."""

    def __init__(
        self,
        ui_store,
        hooks: PlanConfirmationHooks,
    ) -> None:
        self._ui_store = ui_store
        self._hooks = hooks
        self._pending_request: PendingPlanRequest | None = None

    def assess_plan_need(self, message: str) -> PlanTriggerDecision:
        normalized = message.strip().lower()
        if not normalized:
            return PlanTriggerDecision.SKIP

        action_keywords = [
            "implement",
            "build",
            "refactor",
            "migrate",
            "create",
            "develop",
            "rewrite",
            "optimize",
            "fix",
            "support",
            "scaffold",
            "automate",
            "generate",
            "test ",
            "add ",
            "remove ",
        ]
        question_prefixes = [
            "describe",
            "what",
            "why",
            "who",
            "where",
            "when",
            "which",
            "explain",
            "show",
            "list",
            "tell me",
            "summarize",
        ]

        contains_action_kw = any(keyword in normalized for keyword in action_keywords)
        strong_plan_terms = [
            "multistep",
            "multi-step",
            "roadmap",
            "strategy",
            "outline steps",
            "execution plan",
            "timeline",
        ]
        if any(term in normalized for term in strong_plan_terms):
            return PlanTriggerDecision.PLAN

        if normalized.startswith("/plan"):
            return PlanTriggerDecision.PLAN

        if any(normalized.startswith(prefix + " ") for prefix in question_prefixes):
            if not contains_action_kw:
                return PlanTriggerDecision.SKIP

        if normalized.endswith("?") and not (
            "how to" in normalized or "how do" in normalized or contains_action_kw
        ):
            return PlanTriggerDecision.SKIP

        token_count = len(normalized.split())
        bullet_points = message.count("\n-") + message.count("\n*")
        numbered_items = bool(re.search(r"\d+\.", message))
        conjunction_rich = normalized.count(" and ") + normalized.count(" then ")
        plan_score = 0

        if contains_action_kw:
            plan_score += 1
        if token_count >= 80:
            plan_score += 2
        elif token_count >= 40:
            plan_score += 1
        if bullet_points >= 2 or numbered_items:
            plan_score += 1
        if conjunction_rich >= 2:
            plan_score += 1
        if "\n" in message and len(message.splitlines()) >= 3:
            plan_score += 1

        if plan_score >= 2:
            return PlanTriggerDecision.PLAN
        if plan_score == 0:
            return PlanTriggerDecision.SKIP
        return PlanTriggerDecision.ASK

    def has_pending_confirmation(self) -> bool:
        return bool(self._ui_store.planner.pending_confirmation)

    async def request_confirmation(
        self, goal: str, request: PendingPlanRequest
    ) -> None:
        self._pending_request = request
        self._ui_store.request_plan_confirmation(goal)
        logger.info(
            "planner.confirmation_requested",
            extra={
                "goal": goal,
                "thinking_mode": self._hooks.get_thinking_mode(),
            },
        )
        await self._hooks.send_user_message(
            UserCommandMessage(
                "This request might benefit from a planning session. "
                "Reply `yes` to generate a plan with sub-agents or `no` to continue directly."
            )
        )

    async def handle_confirmation_response(self, message: str) -> bool:
        if not self._ui_store.planner.pending_confirmation:
            return False

        response = message.strip().lower()
        affirmative = {"y", "yes", "sure", "please", "do it", "go ahead"}
        negative = {"n", "no", "skip", "not now", "nope"}

        if response in affirmative or response.startswith("yes"):
            await self.apply_choice(accept=True)
            return True

        if response in negative or response.startswith("no "):
            await self.apply_choice(accept=False)
            return True

        self._pending_request = None
        self._ui_store.clear_plan_confirmation()
        return False

    async def apply_choice(self, *, accept: bool) -> None:
        goal = self._ui_store.planner.pending_confirmation
        if not goal:
            return

        logger.info(
            "planner.confirmation_choice",
            extra={
                "goal": goal,
                "accepted": accept,
            },
        )
        request = self._pending_request
        self._pending_request = None
        self._ui_store.clear_plan_confirmation()

        if accept:
            await self._hooks.start_planning(goal)
            return

        await self._hooks.send_user_message(
            UserCommandMessage("Okay, continuing without a planning session.")
        )

        if request:
            await self._hooks.continue_request(request)
