from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
from textual.containers import VerticalScroll

from vibe.cli.textual_ui.widgets.compact import CompactMessage
from vibe.cli.textual_ui.widgets.context_progress import TokenState
from vibe.cli.textual_ui.widgets.messages import (
    ErrorMessage,
    PlanDecisionMessage,
    UserCommandMessage,
)
from vibe.core.config import VibeConfig
from vibe.core.planner import PlanState

if TYPE_CHECKING:
    from vibe.cli.textual_ui.app import VibeApp
    from vibe.cli.textual_ui.planner_controller import PlannerController


class CommandController:
    """Holds slash-command behavior so VibeApp stays presentation-focused."""

    def __init__(self, app: VibeApp) -> None:
        self._app = app

    @property
    def _planner(self) -> PlannerController | None:
        return self._app._planner_controller

    async def show_help(self) -> None:
        help_text = self._app.commands.get_help_text()
        await self._app._mount_and_scroll(UserCommandMessage(help_text))

    async def start_planning(self, argument: str | None) -> None:
        goal = (argument or "").strip()
        if not goal:
            await self._app._mount_and_scroll(
                UserCommandMessage(
                    "Usage: `/plan <goal>`\n\nExample:\n`/plan add a planning sidebar`"
                )
            )
            return

        await self._app._mount_and_scroll(
            UserCommandMessage("Working on getting the subagents ready…")
        )

        controller = self._planner
        if controller is None:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "Planner controller is not available.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            return

        await self._app._clear_plan_cards()
        plan_state = await controller.start_plan(goal)
        await self._app._emit_plan_events(plan_state)
        await self._app._mount_and_scroll(
            UserCommandMessage(self._format_plan_summary(plan_state))
        )
        await controller.start_plan_executor(plan_state.plan_id)

    async def toggle_plan_auto(self, argument: str | None) -> None:
        mode = (argument or "").strip().lower()
        if mode in {"on", "true", "enable", "enabled"}:
            self._app._set_planner_auto_start(True)
            await self._app._mount_and_scroll(
                UserCommandMessage(
                    "Planner auto-start enabled. Future prompts will create a plan automatically."
                )
            )
            return

        if mode in {"off", "false", "disable", "disabled"}:
            self._app._set_planner_auto_start(False)
            await self._app._mount_and_scroll(
                UserCommandMessage("Planner auto-start disabled.")
            )
            return

        status = "enabled" if self._app._planner_auto_start else "disabled"
        await self._app._mount_and_scroll(
            UserCommandMessage(
                f"Planner auto-start is currently {status}. "
                "Use `/plan auto on` or `/plan auto off` to change."
            )
        )

    async def show_plan_status(self) -> None:
        controller = self._planner
        if not controller:
            await self._app._mount_and_scroll(
                UserCommandMessage("Planner controller is not available.")
            )
            return
        plan = controller.get_plan()
        if not plan:
            await self._app._mount_and_scroll(
                UserCommandMessage("No active plan. Run `/plan <goal>` to start one.")
            )
            return
        await self._app._mount_and_scroll(
            UserCommandMessage(self._format_plan_summary(plan))
        )

    async def pause_plan(self) -> None:
        controller = self._planner
        if not controller:
            await self._app._mount_and_scroll(
                UserCommandMessage("Planner is not active yet. Use `/plan <goal>` to begin.")
            )
            return
        plan = await controller.pause_plan()
        if not plan:
            await self._app._mount_and_scroll(
                UserCommandMessage("No active plan to pause.")
            )
            return
        controller.cancel_plan_executor()
        await self._app._mount_and_scroll(UserCommandMessage("Planning session paused."))

    async def resume_plan(self) -> None:
        controller = self._planner
        if not controller:
            await self._app._mount_and_scroll(
                UserCommandMessage("Planner is not active yet. Use `/plan <goal>` to begin.")
            )
            return
        plan = await controller.resume_plan()
        if not plan:
            await self._app._mount_and_scroll(
                UserCommandMessage("No paused plan to resume.")
            )
            return
        await self._app._mount_and_scroll(UserCommandMessage("Planning session resumed."))
        await controller.start_plan_executor(plan.plan_id)

    async def cancel_plan(self) -> None:
        controller = self._planner
        if not controller or not controller.get_plan():
            await self._app._mount_and_scroll(
                UserCommandMessage("No active plan to cancel.")
            )
            return
        await controller.cancel_plan()
        if self._app._sidebar:
            self._app._sidebar.clear_plan()
        await self._app._mount_and_scroll(UserCommandMessage("Planning session cancelled."))

    async def handle_plan_decision(self, argument: str | None) -> None:
        controller = self._planner
        if not controller or not controller.get_plan():
            await self._app._mount_and_scroll(
                UserCommandMessage("No active plan. Start one with `/plan <goal>`.")
            )
            return
        if not argument:
            await self._app._mount_and_scroll(
                UserCommandMessage("Usage: `/plan decide <decision-id> <choice>`")
            )
            return
        parts = argument.split()
        if len(parts) < 2:
            await self._app._mount_and_scroll(
                UserCommandMessage("Usage: `/plan decide <decision-id> <choice>`")
            )
            return
        decision_id, selection = parts[0], " ".join(parts[1:])
        await self._record_decision(controller, decision_id, [selection])

    async def handle_plan_decision_selections(
        self, message: PlanDecisionMessage.DecisionSelected
    ) -> None:
        selections = [s.strip() for s in message.selections if s.strip()]
        if not selections:
            return
        controller = self._planner
        if not controller or not controller.get_plan():
            await self._app._mount_and_scroll(
                UserCommandMessage("No active plan. Start one with `/plan <goal>`.")
            )
            return
        await self._record_decision(controller, message.decision_id, selections)

    async def _record_decision(
        self,
        controller: PlannerController,
        decision_id: str,
        selections: list[str],
    ) -> None:
        plan = await controller.decide(decision_id, selections)
        if not plan:
            await self._app._mount_and_scroll(
                UserCommandMessage(f"Unknown decision `{decision_id}`.")
            )
            return

        selections_text = ", ".join(selections)
        await self._app._mount_and_scroll(
            UserCommandMessage(f"Decision `{decision_id}` recorded: {selections_text}")
        )
        decision = next(
            (item for item in plan.decisions if item.decision_id == decision_id),
            None,
        )
        if decision:
            await self._app._emit_decision_event(plan, decision)
        await controller.start_plan_executor(plan.plan_id)

    def _format_plan_summary(self, plan: PlanState) -> str:
        status = plan.status.value.replace("_", " ").title()
        lines = [f"### Plan: {plan.goal}", f"- Status: {status}"]
        for step in plan.steps:
            step_status = step.status.value.replace("_", " ").title()
            owner = f" (owner: {step.owner})" if step.owner else ""
            mode = f" · mode: {step.mode}" if step.mode else ""
            lines.append(f"- [{step_status}]{owner}{mode} {step.title}")
        if plan.decisions:
            lines.append("")
            lines.append("Pending decisions:")
            for decision in plan.decisions:
                options = ", ".join(decision.option_labels()) if decision.options else "freeform"
                decision_status = (
                    f"resolved: {decision.selection}" if decision.resolved else "awaiting input"
                )
                lines.append(
                    f"- `{decision.decision_id}` {decision.question} ({options}) → {decision_status}"
                )
        return "\n".join(lines)

    async def show_status(self) -> None:
        if self._app.agent is None:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "Agent not initialized yet. Send a message first.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            return

        stats = self._app.agent.stats
        status_text = f"""## Agent Statistics

- **Steps**: {stats.steps:,}
- **Session Prompt Tokens**: {stats.session_prompt_tokens:,}
- **Session Completion Tokens**: {stats.session_completion_tokens:,}
- **Session Total LLM Tokens**: {stats.session_total_llm_tokens:,}
- **Last Turn Tokens**: {stats.last_turn_total_tokens:,}
- **Cost**: ${stats.session_cost:.4f}
"""
        await self._app._mount_and_scroll(UserCommandMessage(status_text))

    async def show_config(self) -> None:
        await self._app._bottom_panel_manager.show_config()

    async def reload_config(self) -> None:
        try:
            new_config = await asyncio.to_thread(VibeConfig.load)
            if self._app.agent:
                await self._app.agent.reload_with_initial_messages(config=new_config)
            self._app.config = new_config
            self._app._planner_auto_start = new_config.planner_auto_start
            if self._app.config.auto_compact_threshold > 0:
                current_tokens = (
                    self._app.agent.stats.context_tokens if self._app.agent else 0
                )
                self._app._set_context_tokens(
                    TokenState(
                        max_tokens=self._app.config.auto_compact_threshold,
                        current_tokens=current_tokens,
                        soft_limit_ratio=self._app.config.memory_soft_limit_ratio,
                    )
                )
            else:
                self._app._set_context_tokens(TokenState())
            await self._app._refresh_memory_panel()
            await self._app._mount_and_scroll(
                UserCommandMessage("Configuration reloaded.")
            )
        except Exception as e:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to reload config: {e}",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )

    async def clear_history(self) -> None:
        agent = self._app.agent
        if agent is None:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to clear yet.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            return

        try:
            await agent.clear_history()
            await self._app._refresh_memory_panel()
            await self._app._finalize_current_streaming_message()
            if self._app.event_handler:
                self._app.event_handler.reset()
            messages_area = self._app.query_one("#messages")
            await messages_area.remove_children()
            if self._app._sidebar:
                self._app._sidebar.update_todos([])

            current_state = self._app._current_token_state
            self._app._set_context_tokens(
                TokenState(
                    max_tokens=current_state.max_tokens,
                    current_tokens=agent.stats.context_tokens,
                    soft_limit_ratio=current_state.soft_limit_ratio
                    or self._app.config.memory_soft_limit_ratio,
                )
            )
            await self._app._mount_and_scroll(
                UserCommandMessage("Conversation history cleared!")
            )
            chat = self._app.query_one("#chat", VerticalScroll)
            chat.scroll_home(animate=False)
        except Exception as e:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to clear history: {e}",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )

    async def show_log_path(self) -> None:
        agent = self._app.agent
        if agent is None:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "No log file created yet. Send a message first.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            return

        if not agent.interaction_logger.enabled:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    "Session logging is disabled in configuration.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            return

        try:
            log_path = str(agent.interaction_logger.filepath)
            await self._app._mount_and_scroll(
                UserCommandMessage(
                    f"## Current Log File Path\n\n`{log_path}`\n\nYou can send this file to share your interaction."
                )
            )
        except Exception as e:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Failed to get log path: {e}",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )

    async def compact_history(self) -> None:
        app = self._app
        if app._agent_running:
            await app._mount_and_scroll(
                ErrorMessage(
                    "Cannot compact while agent is processing. Please wait.",
                    collapsed=app._ui_store.tools_collapsed,
                )
            )
            return

        if app.agent is None or len(app.agent.messages) <= 1:
            await app._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=app._ui_store.tools_collapsed,
                )
            )
            return

        if not app.event_handler:
            return

        old_tokens = app.agent.stats.context_tokens
        compact_msg = CompactMessage()
        app.event_handler.current_compact = compact_msg
        await app._mount_and_scroll(compact_msg)

        try:
            await app.agent.compact()
            await app._refresh_memory_panel()
            new_tokens = app.agent.stats.context_tokens
            compact_msg.set_complete(old_tokens=old_tokens, new_tokens=new_tokens)
            app.event_handler.current_compact = None

            current_state = app._current_token_state
            app._set_context_tokens(
                TokenState(
                    max_tokens=current_state.max_tokens,
                    current_tokens=new_tokens,
                    soft_limit_ratio=current_state.soft_limit_ratio
                    or app.config.memory_soft_limit_ratio,
                )
            )
        except Exception as e:
            compact_msg.set_error(str(e))
            app.event_handler.current_compact = None

    async def memory_command(self, argument: str | None = None) -> None:
        action = (argument or "").strip().lower()
        if action == "clear":
            if not self._app.agent:
                await self._app._mount_and_scroll(
                    ErrorMessage(
                        "Cannot clear memory until the agent starts.",
                        collapsed=self._app._ui_store.tools_collapsed,
                    )
                )
                return
            self._app.agent.session_memory.clear()
            await self._app._refresh_memory_panel()
            await self._app._mount_and_scroll(
                UserCommandMessage("Session memory cleared for this run.")
            )
            return

        await self._app._show_memory_panel()

    async def exit_app(self) -> None:
        self._app.exit()

    async def toggle_thinking_mode(self, argument: str | None = None) -> None:
        mode = (argument or "").strip().lower()
        if mode in {"on", "true", "enable", "enabled"}:
            self._app._set_thinking_mode(True)
            await self._app._mount_and_scroll(
                UserCommandMessage("Thinking mode enabled.")
            )
            return
        if mode in {"off", "false", "disable", "disabled"}:
            self._app._set_thinking_mode(False)
            await self._app._mount_and_scroll(
                UserCommandMessage("Thinking mode disabled.")
            )
            return

        self._app._set_thinking_mode(not self._app._thinking_mode)
        status = "enabled" if self._app._thinking_mode else "disabled"
        await self._app._mount_and_scroll(
            UserCommandMessage(f"Thinking mode {status}.")
        )

    async def retry_last_prompt(self) -> None:
        if not self._app._last_failed_prompt:
            await self._app._mount_and_scroll(
                UserCommandMessage("No failed operation to retry.")
            )
            return

        if self._app._retry_count >= self._app._max_retries:
            await self._app._mount_and_scroll(
                ErrorMessage(
                    f"Max retries ({self._app._max_retries}) exceeded. Please try again later.",
                    collapsed=self._app._ui_store.tools_collapsed,
                )
            )
            self._app._reset_retry_state()
            return

        delay = self._app._get_retry_delay()
        self._app._retry_count += 1

        await self._app._mount_and_scroll(
            UserCommandMessage(
                f"Retrying (attempt {self._app._retry_count}/{self._app._max_retries}) "
                f"after {delay:.1f}s delay..."
            )
        )

        await asyncio.sleep(delay)

        prompt = self._app._last_failed_prompt
        self._app._last_failed_prompt = None

        if self._app.agent and not self._app._agent_running:
            self._app._agent_task = self._app._create_safe_task(
                self._app._handle_agent_turn(prompt),
                name="agent-turn-retry",
            )
