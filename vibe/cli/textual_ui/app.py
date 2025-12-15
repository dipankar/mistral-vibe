from __future__ import annotations

import asyncio
from enum import StrEnum, auto
import hashlib
import os
import re
import subprocess
from typing import Any, ClassVar, assert_never

import json
from pathlib import Path
from uuid import uuid4

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker

from vibe.cli.clipboard import copy_selection_to_clipboard
from vibe.cli.commands import CommandRegistry
from vibe.cli.textual_ui.handlers.event_handler import EventHandler
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from vibe.cli.textual_ui.widgets.compact import CompactMessage
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.cli.textual_ui.widgets.context_progress import ContextProgress, TokenState
from vibe.cli.textual_ui.widgets.loading import LoadingWidget
from vibe.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    BashOutputMessage,
    ErrorMessage,
    InterruptMessage,
    PlanDecisionMessage,
    PlanStartedMessage,
    PlanStepUpdateMessage,
    UserCommandMessage,
    UserMessage,
)
from vibe.cli.textual_ui.widgets.mode_indicator import ModeIndicator
from vibe.cli.textual_ui.widgets.memory_panel import MemoryPanel
from vibe.cli.textual_ui.widgets.planner_panel import PlannerPanel
from vibe.cli.textual_ui.widgets.path_display import PathDisplay
from vibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from vibe.cli.textual_ui.widgets.welcome import WelcomeBanner
from vibe.cli.update_notifier import (
    FileSystemUpdateCacheRepository,
    PyPIVersionUpdateGateway,
    UpdateCacheRepository,
    VersionUpdateAvailability,
    VersionUpdateError,
    VersionUpdateGateway,
    get_update_if_available,
)
from vibe.core import __version__ as CORE_VERSION
from vibe.core.agent import Agent
from vibe.core.autocompletion.path_prompt_adapter import render_path_prompt
from vibe.core.config import VibeConfig
from vibe.core.config_path import HISTORY_FILE
from vibe.core.planner import (
    PlanDecision,
    PlanRunStatus,
    PlanState,
    PlanStep,
    PlanStepStatus,
    PlannerAgent,
)
from vibe.core.tools.base import BaseToolConfig, ToolPermission
from vibe.core.types import (
    ApprovalResponse,
    CompactEndEvent,
    LLMMessage,
    MemoryEntryEvent,
    PlanStartedEvent,
    PlanDecisionEvent,
    PlanStepUpdateEvent,
    ResumeSessionInfo,
    Role,
)
from vibe.core.utils import (
    CancellationReason,
    get_user_cancellation_message,
    is_dangerous_directory,
    logger,
)


class BottomApp(StrEnum):
    Approval = auto()
    Config = auto()
    Input = auto()


class VibeApp(App):
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "app.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "force_quit", "Quit", show=False),
        Binding("ctrl+shift+c", "copy_selection", "Copy Selection", show=False),
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+o", "toggle_tool", "Toggle Tool", show=False),
        Binding("ctrl+t", "toggle_todo", "Toggle Todo", show=False),
        Binding("shift+tab", "cycle_mode", "Cycle Mode", show=False, priority=True),
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False, priority=True),
        Binding(
            "shift+down", "scroll_chat_down", "Scroll Down", show=False, priority=True
        ),
    ]

    def __init__(
        self,
        config: VibeConfig,
        auto_approve: bool = False,
        enable_streaming: bool = False,
        initial_prompt: str | None = None,
        loaded_messages: list[LLMMessage] | None = None,
        session_info: ResumeSessionInfo | None = None,
        version_update_notifier: VersionUpdateGateway | None = None,
        update_cache_repository: UpdateCacheRepository | None = None,
        current_version: str = CORE_VERSION,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.auto_approve = auto_approve
        self.enable_streaming = enable_streaming
        self.agent: Agent | None = None
        self._agent_running = False
        self._agent_initializing = False
        self._interrupt_requested = False
        self._agent_task: asyncio.Task | None = None

        self._loading_widget: LoadingWidget | None = None
        self._pending_approval: asyncio.Future | None = None

        self.event_handler: EventHandler | None = None
        self.commands = CommandRegistry()

        self._chat_input_container: ChatInputContainer | None = None
        self._mode_indicator: ModeIndicator | None = None
        self._context_progress: ContextProgress | None = None
        self._memory_panel: MemoryPanel | None = None
        self._planner_panel: PlannerPanel | None = None
        self._planner_agent: PlannerAgent | None = None
        self._plan_executor_worker: Worker | None = None
        self._plan_executor_plan_id: str | None = None
        self._planner_instance_id = uuid4().hex
        self._plan_state_file = self._build_plan_state_path()
        self._current_bottom_app: BottomApp = BottomApp.Input
        self.theme = config.textual_theme

        self.history_file = HISTORY_FILE.path

        self._tools_collapsed = True
        self._todos_collapsed = False
        self._current_streaming_message: AssistantMessage | None = None
        self._version_update_notifier = version_update_notifier
        self._update_cache_repository = update_cache_repository
        self._is_update_check_enabled = config.enable_update_checks
        self._current_version = current_version
        self._update_notification_task: asyncio.Task | None = None
        self._update_notification_shown = False

        self._initial_prompt = initial_prompt
        self._loaded_messages = loaded_messages
        self._session_info = session_info
        self._agent_init_task: asyncio.Task | None = None
        # prevent a race condition where the agent initialization
        # completes exactly at the moment the user interrupts
        self._agent_init_interrupted = False
        self._auto_scroll = True

    def compose(self) -> ComposeResult:
        with Horizontal(id="app-layout"):
            with Vertical(id="main-root"):
                with VerticalScroll(id="chat"):
                    yield WelcomeBanner(self.config)
                    yield Static(id="messages")

                with Horizontal(id="loading-area"):
                    yield Static(id="loading-area-content")
                    yield ModeIndicator(auto_approve=self.auto_approve)

                with Static(id="bottom-app-container"):
                    yield ChatInputContainer(
                        history_file=self.history_file,
                        command_registry=self.commands,
                        id="input-container",
                        show_warning=self.auto_approve,
                    )

                with Horizontal(id="bottom-bar"):
                    yield PathDisplay(
                        self.config.displayed_workdir or self.config.effective_workdir
                    )
                    yield Static(id="spacer")
                    yield ContextProgress()

                yield MemoryPanel()

            with Vertical(id="sidebar"):
                with Vertical(id="todo-section"):
                    yield Static("Todo", id="todo-title")
                    yield Static("Ctrl+Shift+C copies selection", id="copy-tip")
                    yield Static("/memory or /mem · memory overlay", id="memory-tip")
                    with VerticalScroll(id="todo-scroll"):
                        yield Static(id="todo-area")
                yield Static(id="sidebar-divider")
                yield PlannerPanel()


    async def on_mount(self) -> None:
        self.event_handler = EventHandler(
            mount_callback=self._mount_and_scroll,
            scroll_callback=self._scroll_to_bottom_deferred,
            todo_area_callback=lambda: self.query_one("#todo-area"),
            get_tools_collapsed=lambda: self._tools_collapsed,
            get_todos_collapsed=lambda: self._todos_collapsed,
        )

        self._chat_input_container = self.query_one(ChatInputContainer)
        self._mode_indicator = self.query_one(ModeIndicator)
        self._context_progress = self.query_one(ContextProgress)
        self._memory_panel = self.query_one(MemoryPanel)
        self._planner_panel = self.query_one(PlannerPanel)
        await self._ensure_todo_placeholder()

        if self.config.auto_compact_threshold > 0:
            self._context_progress.tokens = TokenState(
                max_tokens=self.config.auto_compact_threshold,
                current_tokens=0,
                soft_limit_ratio=self.config.memory_soft_limit_ratio,
            )

        chat_input_container = self.query_one(ChatInputContainer)
        chat_input_container.focus_input()
        await self._show_dangerous_directory_warning()
        self._schedule_update_notification()

        if self._session_info:
            await self._mount_and_scroll(AssistantMessage(self._session_info.message()))

        if self._initial_prompt:
            self.call_after_refresh(self._process_initial_prompt)
        else:
            self._ensure_agent_init_task()

        if self._planner_panel:
            await self._planner_panel.update_plan(None)
        await self._load_plan_state()

    def _process_initial_prompt(self) -> None:
        if self._initial_prompt:
            self.run_worker(
                self._handle_user_message(self._initial_prompt), exclusive=False
            )

    async def _refresh_memory_panel(self) -> None:
        if self._memory_panel and self.agent:
            await self._memory_panel.update_entries(
                self.agent.session_memory.entries
            )

    async def _memory_command(self, argument: str | None = None) -> None:
        action = (argument or "").strip().lower()
        if action == "clear":
            if not self.agent:
                await self._mount_and_scroll(
                    ErrorMessage(
                        "Cannot clear memory until the agent starts.",
                        collapsed=self._tools_collapsed,
                    )
                )
                return
            self.agent.session_memory.clear()
            await self._refresh_memory_panel()
            await self._mount_and_scroll(
                UserCommandMessage("Session memory cleared for this run.")
            )
            return

        await self._show_memory_panel()

    async def _ensure_todo_placeholder(self) -> None:
        todo_area = self.query_one("#todo-area", Static)
        if todo_area.children:
            return
        await todo_area.mount(
            Static(
                "No todo items yet. Use the todo tool to add tasks.",
                classes="todo-placeholder",
            )
        )

    async def on_chat_input_container_submitted(
        self, event: ChatInputContainer.Submitted
    ) -> None:
        value = event.value.strip()
        if not value:
            return

        input_widget = self.query_one(ChatInputContainer)
        input_widget.value = ""

        if self._agent_running:
            await self._interrupt_agent()

        if value.startswith("!"):
            await self._handle_bash_command(value[1:])
            return

        if await self._handle_command(value):
            return

        await self._handle_user_message(value)

    async def on_approval_app_approval_granted(
        self, message: ApprovalApp.ApprovalGranted
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

        await self._switch_to_input_app()

    async def on_approval_app_approval_granted_always_tool(
        self, message: ApprovalApp.ApprovalGrantedAlwaysTool
    ) -> None:
        self._set_tool_permission_always(
            message.tool_name, save_permanently=message.save_permanently
        )

        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

        await self._switch_to_input_app()

    async def on_approval_app_approval_rejected(
        self, message: ApprovalApp.ApprovalRejected
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.OPERATION_CANCELLED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback))

        await self._switch_to_input_app()

        if self._loading_widget and self._loading_widget.parent:
            await self._remove_loading_widget()

    async def _remove_loading_widget(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
            self._loading_widget = None

    def on_config_app_setting_changed(self, message: ConfigApp.SettingChanged) -> None:
        if message.key == "textual_theme":
            self.theme = message.value

    async def on_config_app_config_closed(
        self, message: ConfigApp.ConfigClosed
    ) -> None:
        if message.changes:
            self._save_config_changes(message.changes)
            await self._reload_config()
        else:
            await self._mount_and_scroll(
                UserCommandMessage("Configuration closed (no changes saved).")
            )

        await self._switch_to_input_app()

    def _set_tool_permission_always(
        self, tool_name: str, save_permanently: bool = False
    ) -> None:
        if save_permanently:
            VibeConfig.save_updates({"tools": {tool_name: {"permission": "always"}}})

        if tool_name not in self.config.tools:
            self.config.tools[tool_name] = BaseToolConfig()

        self.config.tools[tool_name].permission = ToolPermission.ALWAYS

    def _save_config_changes(self, changes: dict[str, str]) -> None:
        if not changes:
            return

        updates: dict = {}

        for key, value in changes.items():
            match key:
                case "active_model":
                    if value != self.config.active_model:
                        updates["active_model"] = value
                case "textual_theme":
                    if value != self.config.textual_theme:
                        updates["textual_theme"] = value

        if updates:
            VibeConfig.save_updates(updates)

    async def _handle_command(self, user_input: str) -> bool:
        if match := self.commands.find_command(user_input):
            command, argument = match
            handler = getattr(self, command.handler)
            if asyncio.iscoroutinefunction(handler):
                if command.accepts_argument:
                    await handler(argument)
                else:
                    await handler()
            else:
                if command.accepts_argument:
                    handler(argument)
                else:
                    handler()
            return True
        return False

    async def _handle_bash_command(self, command: str) -> None:
        if not command:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No command provided after '!'", collapsed=self._tools_collapsed
                )
            )
            return

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=False,
                timeout=30,
                cwd=self.config.effective_workdir,
            )
            stdout = (
                result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            )
            stderr = (
                result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            )
            output = stdout or stderr or "(no output)"
            exit_code = result.returncode
            await self._mount_and_scroll(
                BashOutputMessage(
                    command, str(self.config.effective_workdir), output, exit_code
                )
            )
        except subprocess.TimeoutExpired:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Command timed out after 30 seconds",
                    collapsed=self._tools_collapsed,
                )
            )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(f"Command failed: {e}", collapsed=self._tools_collapsed)
            )

    async def _handle_user_message(self, message: str) -> None:
        init_task = self._ensure_agent_init_task()
        pending_init = bool(init_task and not init_task.done())
        user_message = UserMessage(message, pending=pending_init)

        await self._mount_and_scroll(user_message)

        self.run_worker(
            self._process_user_message_after_mount(
                message=message,
                user_message=user_message,
                init_task=init_task,
                pending_init=pending_init,
            ),
            exclusive=False,
        )

    async def _process_user_message_after_mount(
        self,
        message: str,
        user_message: UserMessage,
        init_task: asyncio.Task | None,
        pending_init: bool,
    ) -> None:
        try:
            if init_task and not init_task.done():
                loading = LoadingWidget()
                self._loading_widget = loading
                await self.query_one("#loading-area-content").mount(loading)

                try:
                    await init_task
                finally:
                    if self._loading_widget and self._loading_widget.parent:
                        await self._loading_widget.remove()
                        self._loading_widget = None
                    if pending_init:
                        await user_message.set_pending(False)
            elif pending_init:
                await user_message.set_pending(False)

            if pending_init and self._agent_init_interrupted:
                self._agent_init_interrupted = False
                return

            if self.agent and not self._agent_running:
                self._agent_task = asyncio.create_task(self._handle_agent_turn(message))
        except asyncio.CancelledError:
            self._agent_init_interrupted = False
            if pending_init:
                await user_message.set_pending(False)
            return

    async def _initialize_agent(self) -> None:
        if self.agent or self._agent_initializing:
            return

        self._agent_initializing = True
        try:
            agent = Agent(
                self.config,
                auto_approve=self.auto_approve,
                enable_streaming=self.enable_streaming,
            )

            if not self.auto_approve:
                agent.approval_callback = self._approval_callback

            if self._loaded_messages:
                non_system_messages = [
                    msg
                    for msg in self._loaded_messages
                    if not (msg.role == Role.system)
                ]
                agent.messages.extend(non_system_messages)
                logger.info(
                    "Loaded %d messages from previous session", len(non_system_messages)
                )

            self.agent = agent
            await self._refresh_memory_panel()
        except asyncio.CancelledError:
            self.agent = None
            return
        except Exception as e:
            self.agent = None
            await self._mount_and_scroll(
                ErrorMessage(str(e), collapsed=self._tools_collapsed)
            )
        finally:
            self._agent_initializing = False
            self._agent_init_task = None

    def _ensure_agent_init_task(self) -> asyncio.Task | None:
        if self.agent:
            self._agent_init_task = None
            self._agent_init_interrupted = False
            return None

        if self._agent_init_task and self._agent_init_task.done():
            if self._agent_init_task.cancelled():
                self._agent_init_task = None

        if not self._agent_init_task or self._agent_init_task.done():
            self._agent_init_interrupted = False
            self._agent_init_task = asyncio.create_task(self._initialize_agent())

        return self._agent_init_task

    async def _approval_callback(
        self, tool: str, args: dict, tool_call_id: str
    ) -> tuple[ApprovalResponse, str | None]:
        self._pending_approval = asyncio.Future()
        await self._switch_to_approval_app(tool, args)
        result = await self._pending_approval
        self._pending_approval = None
        return result

    async def _handle_agent_turn(self, prompt: str) -> None:
        if not self.agent:
            return

        self._agent_running = True

        loading_area = self.query_one("#loading-area-content")

        loading = LoadingWidget()
        self._loading_widget = loading
        await loading_area.mount(loading)

        try:
            rendered_prompt = render_path_prompt(
                prompt, base_dir=self.config.effective_workdir
            )
            async for event in self.agent.act(rendered_prompt):
                if self._context_progress and self.agent:
                    current_state = self._context_progress.tokens
                    self._context_progress.tokens = TokenState(
                        max_tokens=current_state.max_tokens,
                        current_tokens=self.agent.stats.context_tokens,
                        soft_limit_ratio=current_state.soft_limit_ratio
                        or self.config.memory_soft_limit_ratio,
                    )

                if self.event_handler:
                    await self.event_handler.handle_event(
                        event,
                        loading_active=self._loading_widget is not None,
                        loading_widget=self._loading_widget,
                    )
                if isinstance(event, MemoryEntryEvent):
                    await self._refresh_memory_panel()
                elif isinstance(event, CompactEndEvent):
                    await self._refresh_memory_panel()

        except asyncio.CancelledError:
            if self._loading_widget and self._loading_widget.parent:
                await self._loading_widget.remove()
            if self.event_handler:
                self.event_handler.stop_current_tool_call()
            raise
        except Exception as e:
            if self._loading_widget and self._loading_widget.parent:
                await self._loading_widget.remove()
            if self.event_handler:
                self.event_handler.stop_current_tool_call()
            await self._mount_and_scroll(
                ErrorMessage(str(e), collapsed=self._tools_collapsed)
            )
        finally:
            self._agent_running = False
            self._interrupt_requested = False
            self._agent_task = None
            if self._loading_widget:
                await self._loading_widget.remove()
            self._loading_widget = None
            await self._finalize_current_streaming_message()

    async def _interrupt_agent(self) -> None:
        interrupting_agent_init = bool(
            self._agent_init_task and not self._agent_init_task.done()
        )

        if (
            not self._agent_running and not interrupting_agent_init
        ) or self._interrupt_requested:
            return

        self._interrupt_requested = True

        if interrupting_agent_init and self._agent_init_task:
            self._agent_init_interrupted = True
            self._agent_init_task.cancel()
            try:
                await self._agent_init_task
            except asyncio.CancelledError:
                pass

        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

        if self.event_handler:
            self.event_handler.stop_current_tool_call()
            self.event_handler.stop_current_compact()

        self._agent_running = False
        loading_area = self.query_one("#loading-area-content")
        await loading_area.remove_children()

        await self._finalize_current_streaming_message()
        await self._mount_and_scroll(InterruptMessage())

        self._interrupt_requested = False

    async def _show_help(self) -> None:
        help_text = self.commands.get_help_text()
        await self._mount_and_scroll(UserCommandMessage(help_text))

    async def _start_planning(self, argument: str | None = None) -> None:
        goal = (argument or "").strip()
        if not goal:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Usage: `/plan <goal>`\n\nExample:\n`/plan add a planning sidebar`"
                )
            )
            return

        planner = self._ensure_planner()
        plan_state = await planner.start_plan(goal)
        await self._refresh_planner_panel(plan_state)
        await self._emit_plan_events(plan_state)
        await self._mount_and_scroll(
            UserCommandMessage(self._format_plan_summary(plan_state))
        )
        self._start_plan_executor(plan_state.plan_id)

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
                options = ", ".join(decision.options) if decision.options else "freeform"
                decision_status = (
                    f"resolved: {decision.selection}" if decision.resolved else "awaiting input"
                )
                lines.append(
                    f"- `{decision.decision_id}` {decision.question} ({options}) → {decision_status}"
                )
        return "\n".join(lines)

    async def _show_plan_status(self, _: str | None = None) -> None:
        if not self._planner_agent or not self._planner_agent.get_plan():
            await self._mount_and_scroll(
                UserCommandMessage("No active plan. Run `/plan <goal>` to start one.")
            )
            return
        plan = self._planner_agent.get_plan()
        if plan:
            await self._mount_and_scroll(UserCommandMessage(self._format_plan_summary(plan)))

    async def _pause_plan(self, _: str | None = None) -> None:
        if not self._planner_agent:
            await self._mount_and_scroll(
                UserCommandMessage("Planner is not active yet. Use `/plan <goal>` to begin.")
            )
            return
        plan = self._planner_agent.pause()
        if not plan:
            await self._mount_and_scroll(
                UserCommandMessage("No active plan to pause.")
            )
            return
        self._cancel_plan_executor()
        await self._refresh_planner_panel(plan)
        await self._mount_and_scroll(UserCommandMessage("Planning session paused."))

    async def _resume_plan(self, _: str | None = None) -> None:
        if not self._planner_agent:
            await self._mount_and_scroll(
                UserCommandMessage("Planner is not active yet. Use `/plan <goal>` to begin.")
            )
            return
        plan = self._planner_agent.resume()
        if not plan:
            await self._mount_and_scroll(
                UserCommandMessage("No paused plan to resume.")
            )
            return
        await self._refresh_planner_panel(plan)
        await self._mount_and_scroll(UserCommandMessage("Planning session resumed."))
        self._start_plan_executor(plan.plan_id)

    async def _cancel_plan(self, _: str | None = None) -> None:
        if not self._planner_agent or not self._planner_agent.get_plan():
            await self._mount_and_scroll(
                UserCommandMessage("No active plan to cancel.")
            )
            return
        self._planner_agent.cancel()
        self._cancel_plan_executor()
        await self._refresh_planner_panel(None)
        await self._mount_and_scroll(UserCommandMessage("Planning session cancelled."))

    async def _handle_plan_decision(self, argument: str | None = None) -> None:
        if not self._planner_agent or not self._planner_agent.get_plan():
            await self._mount_and_scroll(
                UserCommandMessage("No active plan. Start one with `/plan <goal>`.")
            )
            return
        if not argument:
            await self._mount_and_scroll(
                UserCommandMessage("Usage: `/plan decide <decision-id> <choice>`")
            )
            return
        parts = argument.split()
        if len(parts) < 2:
            await self._mount_and_scroll(
                UserCommandMessage("Usage: `/plan decide <decision-id> <choice>`")
            )
            return
        decision_id, selection = parts[0], " ".join(parts[1:])
        plan = self._planner_agent.decide(decision_id, selection)
        if not plan:
            await self._mount_and_scroll(
                UserCommandMessage(f"Unknown decision `{decision_id}`.")
            )
            return
        await self._refresh_planner_panel(plan)
        await self._mount_and_scroll(
            UserCommandMessage(f"Decision `{decision_id}` recorded: {selection}")
        )
        decision = next(
            (item for item in plan.decisions if item.decision_id == decision_id),
            None,
        )
        if decision:
            await self._emit_decision_event(plan, decision)
        self._start_plan_executor(plan.plan_id)

    async def on_planner_panel_decision_selected(
        self, message: PlannerPanel.DecisionSelected
    ) -> None:
        await self._handle_plan_decision(f"{message.decision_id} {message.selection}")

    async def _emit_plan_events(self, plan: PlanState) -> None:
        steps_summary = [
            f"{index}. {step.title} · {step.status.value.replace('_', ' ').title()} · mode={step.mode or 'code'}"
            for index, step in enumerate(plan.steps, start=1)
        ]
        plan_started = PlanStartedEvent(
            plan_id=plan.plan_id,
            goal=plan.goal,
            summary=plan.summarize(),
            steps=steps_summary,
        )
        await self._mount_and_scroll(PlanStartedMessage(plan_started))

        for step in plan.steps:
            event = PlanStepUpdateEvent(
                plan_id=plan.plan_id,
                step_id=step.step_id,
                title=step.title,
                status=step.status.value.replace("_", " ").title(),
                notes=step.notes,
            )
            await self._mount_and_scroll(PlanStepUpdateMessage(event))
        await self._emit_decision_events(plan)

    async def _emit_decision_events(self, plan: PlanState) -> None:
        for decision in plan.decisions:
            await self._emit_decision_event(plan, decision)

    async def _emit_decision_event(
        self, plan: PlanState, decision: PlanDecision
    ) -> None:
        event = PlanDecisionEvent(
            plan_id=plan.plan_id,
            decision_id=decision.decision_id,
            question=decision.question,
            options=decision.options,
            resolved=decision.resolved,
            selection=decision.selection,
        )
        await self._mount_and_scroll(PlanDecisionMessage(event))

    async def _show_status(self) -> None:
        if self.agent is None:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Agent not initialized yet. Send a message first.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        stats = self.agent.stats
        status_text = f"""## Agent Statistics

- **Steps**: {stats.steps:,}
- **Session Prompt Tokens**: {stats.session_prompt_tokens:,}
- **Session Completion Tokens**: {stats.session_completion_tokens:,}
- **Session Total LLM Tokens**: {stats.session_total_llm_tokens:,}
- **Last Turn Tokens**: {stats.last_turn_total_tokens:,}
- **Cost**: ${stats.session_cost:.4f}
"""
        await self._mount_and_scroll(UserCommandMessage(status_text))

    async def _show_config(self) -> None:
        """Switch to the configuration app in the bottom panel."""
        if self._current_bottom_app == BottomApp.Config:
            return
        await self._switch_to_config_app()

    async def _reload_config(self) -> None:
        try:
            new_config = VibeConfig.load()

            if self.agent:
                await self.agent.reload_with_initial_messages(config=new_config)

            self.config = new_config
            if self._context_progress:
                if self.config.auto_compact_threshold > 0:
                    current_tokens = (
                        self.agent.stats.context_tokens if self.agent else 0
                    )
                    self._context_progress.tokens = TokenState(
                        max_tokens=self.config.auto_compact_threshold,
                        current_tokens=current_tokens,
                        soft_limit_ratio=self.config.memory_soft_limit_ratio,
                    )
                else:
                    self._context_progress.tokens = TokenState()

            await self._refresh_memory_panel()
            await self._mount_and_scroll(UserCommandMessage("Configuration reloaded."))
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to reload config: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _clear_history(self) -> None:
        if self.agent is None:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to clear yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.agent:
            return

        try:
            await self.agent.clear_history()
            await self._refresh_memory_panel()
            await self._finalize_current_streaming_message()
            messages_area = self.query_one("#messages")
            await messages_area.remove_children()
            todo_area = self.query_one("#todo-area")
            await todo_area.remove_children()
            await self._ensure_todo_placeholder()

            if self._context_progress and self.agent:
                current_state = self._context_progress.tokens
                self._context_progress.tokens = TokenState(
                    max_tokens=current_state.max_tokens,
                    current_tokens=self.agent.stats.context_tokens,
                    soft_limit_ratio=current_state.soft_limit_ratio
                    or self.config.memory_soft_limit_ratio,
                )
            await self._mount_and_scroll(
                UserCommandMessage("Conversation history cleared!")
            )
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_home(animate=False)

        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to clear history: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _show_log_path(self) -> None:
        if self.agent is None:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No log file created yet. Send a message first.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.agent.interaction_logger.enabled:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Session logging is disabled in configuration.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        try:
            log_path = str(self.agent.interaction_logger.filepath)
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"## Current Log File Path\n\n`{log_path}`\n\nYou can send this file to share your interaction."
                )
            )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to get log path: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _compact_history(self) -> None:
        if self._agent_running:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Cannot compact while agent is processing. Please wait.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if self.agent is None:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if len(self.agent.messages) <= 1:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.agent or not self.event_handler:
            return

        old_tokens = self.agent.stats.context_tokens
        compact_msg = CompactMessage()
        self.event_handler.current_compact = compact_msg
        await self._mount_and_scroll(compact_msg)

        try:
            await self.agent.compact()
            await self._refresh_memory_panel()
            new_tokens = self.agent.stats.context_tokens
            compact_msg.set_complete(old_tokens=old_tokens, new_tokens=new_tokens)
            self.event_handler.current_compact = None

            if self._context_progress:
                current_state = self._context_progress.tokens
                self._context_progress.tokens = TokenState(
                    max_tokens=current_state.max_tokens,
                    current_tokens=new_tokens,
                    soft_limit_ratio=current_state.soft_limit_ratio
                    or self.config.memory_soft_limit_ratio,
                )
        except Exception as e:
            compact_msg.set_error(str(e))
            self.event_handler.current_compact = None

    async def _exit_app(self) -> None:
        self.exit()

    async def _switch_to_config_app(self) -> None:
        if self._current_bottom_app == BottomApp.Config:
            return

        bottom_container = self.query_one("#bottom-app-container")
        await self._mount_and_scroll(UserCommandMessage("Configuration opened..."))

        try:
            chat_input_container = self.query_one(ChatInputContainer)
            await chat_input_container.remove()
        except Exception:
            pass

        if self._mode_indicator:
            self._mode_indicator.display = False

        config_app = ConfigApp(self.config)
        await bottom_container.mount(config_app)
        self._current_bottom_app = BottomApp.Config

        self.call_after_refresh(config_app.focus)

    async def _switch_to_approval_app(self, tool_name: str, tool_args: dict) -> None:
        bottom_container = self.query_one("#bottom-app-container")

        try:
            chat_input_container = self.query_one(ChatInputContainer)
            await chat_input_container.remove()
        except Exception:
            pass

        if self._mode_indicator:
            self._mode_indicator.display = False

        approval_app = ApprovalApp(
            tool_name=tool_name,
            tool_args=tool_args,
            workdir=str(self.config.effective_workdir),
            config=self.config,
        )
        await bottom_container.mount(approval_app)
        self._current_bottom_app = BottomApp.Approval

        self.call_after_refresh(approval_app.focus)
        self.call_after_refresh(self._scroll_to_bottom)

    async def _switch_to_input_app(self) -> None:
        bottom_container = self.query_one("#bottom-app-container")

        try:
            config_app = self.query_one("#config-app")
            await config_app.remove()
        except Exception:
            pass

        try:
            approval_app = self.query_one("#approval-app")
            await approval_app.remove()
        except Exception:
            pass

        if self._mode_indicator:
            self._mode_indicator.display = True

        try:
            chat_input_container = self.query_one(ChatInputContainer)
            self._chat_input_container = chat_input_container
            self._current_bottom_app = BottomApp.Input
            self.call_after_refresh(chat_input_container.focus_input)
            return
        except Exception:
            pass

        chat_input_container = ChatInputContainer(
            history_file=self.history_file,
            command_registry=self.commands,
            id="input-container",
            show_warning=self.auto_approve,
        )
        await bottom_container.mount(chat_input_container)
        self._chat_input_container = chat_input_container

        self._current_bottom_app = BottomApp.Input

        self.call_after_refresh(chat_input_container.focus_input)

    def _focus_current_bottom_app(self) -> None:
        try:
            match self._current_bottom_app:
                case BottomApp.Input:
                    self.query_one(ChatInputContainer).focus_input()
                case BottomApp.Config:
                    self.query_one(ConfigApp).focus()
                case BottomApp.Approval:
                    self.query_one(ApprovalApp).focus()
                case app:
                    assert_never(app)
        except Exception:
            pass

    def action_interrupt(self) -> None:
        if self._memory_panel and self._memory_panel.is_visible:
            self._memory_panel.hide_panel()
            self._focus_current_bottom_app()
            return

        if self._current_bottom_app == BottomApp.Config:
            try:
                config_app = self.query_one(ConfigApp)
                config_app.action_close()
            except Exception:
                pass
            return

        if self._current_bottom_app == BottomApp.Approval:
            try:
                approval_app = self.query_one(ApprovalApp)
                approval_app.action_reject()
            except Exception:
                pass
            return

        has_pending_user_message = any(
            msg.has_class("pending") for msg in self.query(UserMessage)
        )

        interrupt_needed = self._agent_running or (
            self._agent_init_task
            and not self._agent_init_task.done()
            and has_pending_user_message
        )

        if interrupt_needed:
            self.run_worker(self._interrupt_agent(), exclusive=False)

        self._scroll_to_bottom()
        self._focus_current_bottom_app()

    async def action_toggle_tool(self) -> None:
        if not self.event_handler:
            return

        self._tools_collapsed = not self._tools_collapsed

        non_todo_results = [
            result
            for result in self.event_handler.tool_results
            if result.event.tool_name != "todo"
        ]

        for result in non_todo_results:
            result.collapsed = self._tools_collapsed
            await result.render_result()

        try:
            error_messages = self.query(ErrorMessage)
            for error_msg in error_messages:
                error_msg.set_collapsed(self._tools_collapsed)
        except Exception:
            pass

    async def action_toggle_todo(self) -> None:
        if not self.event_handler:
            return

        self._todos_collapsed = not self._todos_collapsed

        todo_results = [
            result
            for result in self.event_handler.tool_results
            if result.event.tool_name == "todo"
        ]

        for result in todo_results:
            result.collapsed = self._todos_collapsed
            await result.render_result()

    def action_cycle_mode(self) -> None:
        if self._current_bottom_app != BottomApp.Input:
            return

        self.auto_approve = not self.auto_approve

        if self._mode_indicator:
            self._mode_indicator.set_auto_approve(self.auto_approve)

        if self._chat_input_container:
            self._chat_input_container.set_show_warning(self.auto_approve)

        if self.agent:
            self.agent.auto_approve = self.auto_approve

            if self.auto_approve:
                self.agent.approval_callback = None
            else:
                self.agent.approval_callback = self._approval_callback

        self._focus_current_bottom_app()

    def action_force_quit(self) -> None:
        if copy_selection_to_clipboard(self):
            return

        input_widgets = self.query(ChatInputContainer)
        if input_widgets:
            input_widget = input_widgets.first()
            if input_widget.value:
                input_widget.value = ""
                return

        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()

        self.exit()

    def action_scroll_chat_up(self) -> None:
        try:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_relative(y=-5, animate=False)
            self._auto_scroll = False
        except Exception:
            pass

    def action_scroll_chat_down(self) -> None:
        try:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_relative(y=5, animate=False)
            if self._is_scrolled_to_bottom(chat):
                self._auto_scroll = True
        except Exception:
            pass

    async def _show_dangerous_directory_warning(self) -> None:
        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            warning = (
                f"⚠️ WARNING: {reason}\n\nRunning in this location is not recommended."
            )
            await self._mount_and_scroll(UserCommandMessage(warning))

    async def _finalize_current_streaming_message(self) -> None:
        if self._current_streaming_message is None:
            return

        await self._current_streaming_message.stop_stream()
        self._current_streaming_message = None

    async def _mount_and_scroll(self, widget: Widget) -> None:
        messages_area = self.query_one("#messages")
        chat = self.query_one("#chat", VerticalScroll)
        was_at_bottom = self._is_scrolled_to_bottom(chat)

        if was_at_bottom:
            self._auto_scroll = True

        if isinstance(widget, AssistantMessage):
            if self._current_streaming_message is not None:
                content = widget._content or ""
                if content:
                    await self._current_streaming_message.append_content(content)
            else:
                self._current_streaming_message = widget
                await messages_area.mount(widget)
                await widget.write_initial_content()
        else:
            await self._finalize_current_streaming_message()
            await messages_area.mount(widget)

            is_tool_message = isinstance(widget, (ToolCallMessage, ToolResultMessage))

            if not is_tool_message:
                self.call_after_refresh(self._scroll_to_bottom)

        if was_at_bottom:
            self.call_after_refresh(self._anchor_if_scrollable)

    def _is_scrolled_to_bottom(self, scroll_view: VerticalScroll) -> bool:
        try:
            threshold = 3
            return scroll_view.scroll_y >= (scroll_view.max_scroll_y - threshold)
        except Exception:
            return True

    def _scroll_to_bottom(self) -> None:
        try:
            chat = self.query_one("#chat")
            chat.scroll_end(animate=False)
        except Exception:
            pass

    def _scroll_to_bottom_deferred(self) -> None:
        self.call_after_refresh(self._scroll_to_bottom)

    def _anchor_if_scrollable(self) -> None:
        if not self._auto_scroll:
            return
        try:
            chat = self.query_one("#chat", VerticalScroll)
            if chat.max_scroll_y == 0:
                return
            chat.anchor()
        except Exception:
            pass

    def _schedule_update_notification(self) -> None:
        if (
            self._version_update_notifier is None
            or self._update_notification_task
            or not self._is_update_check_enabled
        ):
            return

        self._update_notification_task = asyncio.create_task(
            self._check_version_update(), name="version-update-check"
        )

    async def _check_version_update(self) -> None:
        try:
            if (
                self._version_update_notifier is None
                or self._update_cache_repository is None
            ):
                return

            update = await get_update_if_available(
                version_update_notifier=self._version_update_notifier,
                current_version=self._current_version,
                update_cache_repository=self._update_cache_repository,
            )
        except VersionUpdateError as error:
            self.notify(
                error.message,
                title="Update check failed",
                severity="warning",
                timeout=10,
            )
            return
        except Exception as exc:
            logger.debug("Version update check failed", exc_info=exc)
            return
        finally:
            self._update_notification_task = None

        if update is None or not update.should_notify:
            return

        self._display_update_notification(update)

    def _display_update_notification(self, update: VersionUpdateAvailability) -> None:
        if self._update_notification_shown:
            return

        message = f'{self._current_version} => {update.latest_version}\nRun "uv tool upgrade mistral-vibe" to update'

        self.notify(
            message, title="Update available", severity="information", timeout=10
        )
        self._update_notification_shown = True

    async def _show_memory_panel(self) -> None:
        if not self.agent:
            init_task = self._ensure_agent_init_task()
            if init_task and not init_task.done():
                await self._mount_and_scroll(
                    UserCommandMessage(
                        "Agent is starting… session memory will appear once ready."
                    )
                )
            else:
                await self._mount_and_scroll(
                    ErrorMessage(
                        "Session memory is unavailable until the agent starts.",
                        collapsed=self._tools_collapsed,
                    )
                )
            return

        if not self._memory_panel:
            return

        await self._refresh_memory_panel()

        if self._memory_panel.is_visible:
            self._memory_panel.hide_panel()
            return

        await self._memory_panel.show_panel()

    def action_copy_selection(self) -> None:
        copy_selection_to_clipboard(self)

    def _ensure_planner(self) -> PlannerAgent:
        if not self._planner_agent:
            self._planner_agent = PlannerAgent(self.config)
        return self._planner_agent

    def _planner_workspace_slug(self) -> str:
        workdir = str(self.config.effective_workdir.resolve())
        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]", "-", self.config.effective_workdir.name or "workspace"
        ).strip("-")
        if not safe_name:
            safe_name = "workspace"
        digest = hashlib.sha256(workdir.encode("utf-8")).hexdigest()[:10]
        return f"{safe_name.lower()}-{digest}"

    def _build_plan_state_path(self) -> Path:
        return Path.home() / ".vibe" / "plans" / f"{self._planner_workspace_slug()}.json"

    def _plan_payload_owned_by_self(self, payload: dict[str, Any]) -> bool:
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

    async def _refresh_planner_panel(self, plan: PlanState | None = None) -> None:
        if not self._planner_panel:
            return
        plan_to_use = plan
        if plan_to_use is None and self._planner_agent:
            plan_to_use = self._planner_agent.get_plan()
        await self._planner_panel.update_plan(plan_to_use)
        self._persist_plan_state()

    def _start_plan_executor(self, plan_id: str) -> None:
        self._cancel_plan_executor()
        worker = self.run_worker(self._execute_plan_steps(plan_id), exclusive=False)
        self._plan_executor_worker = worker
        self._plan_executor_plan_id = plan_id

    def _cancel_plan_executor(self) -> None:
        if self._plan_executor_worker:
            self._plan_executor_worker.cancel()
        self._plan_executor_worker = None
        self._plan_executor_plan_id = None
        self._persist_plan_state()

    def _persist_plan_state(self) -> None:
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
                    json.dumps(payload, indent=2), encoding="utf-8"
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
            runnable_steps = planner.get_runnable_steps()
            if not runnable_steps:
                await self._maybe_mark_plan_complete(plan_id)
                return
            for step in runnable_steps:
                if plan.status in (PlanRunStatus.PAUSED, PlanRunStatus.CANCELLED):
                    return
                if step.status == PlanStepStatus.NEEDS_DECISION:
                    await self._notify_step_decision_needed(step)
                    continue
                if step.status == PlanStepStatus.BLOCKED:
                    continue
                await self._update_step_status(step.step_id, PlanStepStatus.IN_PROGRESS)
                prompt = planner.build_step_prompt(step.step_id)
                if not prompt:
                    await self._update_step_status(
                        step.step_id,
                        PlanStepStatus.BLOCKED,
                        notes="Planner missing instructions for this step.",
                    )
                    continue
                await self._mount_and_scroll(
                    UserCommandMessage(
                        f"Planner executing step `{step.step_id}` · {step.title}"
                    )
                )
                await self._wait_for_agent_idle()
                await self._handle_user_message(prompt)
                agent_task = self._agent_task
                if agent_task:
                    try:
                        await agent_task
                    except asyncio.CancelledError:
                        return
                await self._update_step_status(
                    step.step_id,
                    PlanStepStatus.COMPLETED,
                )
                await self._maybe_mark_plan_complete(plan_id)
            await asyncio.sleep(0)

    async def _wait_for_agent_idle(self) -> None:
        while True:
            busy = self._agent_running or (
                self._agent_init_task and not self._agent_init_task.done()
            )
            if not busy:
                return
            await asyncio.sleep(0.05)

    async def _update_step_status(
        self, step_id: str, status: PlanStepStatus, notes: str | None = None
    ) -> None:
        if not self._planner_agent:
            return
        step = self._planner_agent.update_step_status(step_id, status, notes)
        if not step:
            return
        plan = self._planner_agent.get_plan()
        await self._refresh_planner_panel(plan)
        if plan:
            await self._emit_step_update_message(plan, step)

    async def _notify_step_decision_needed(self, step: PlanStep) -> None:
        decision_id = step.decision_id
        plan = self._planner_agent.get_plan() if self._planner_agent else None
        if plan and decision_id:
            decision = next(
                (item for item in plan.decisions if item.decision_id == decision_id),
                None,
            )
            if decision:
                await self._emit_decision_event(plan, decision)
                return
        await self._mount_and_scroll(
            UserCommandMessage(
                f"Planner step `{step.step_id}` requires decision "
                f"`{decision_id or 'unknown'}`. Use `/plan decide <id> <choice>` "
                "to continue."
            )
        )

    async def _emit_step_update_message(self, plan: PlanState, step: PlanStep) -> None:
        event = PlanStepUpdateEvent(
            plan_id=plan.plan_id,
            step_id=step.step_id,
            title=step.title,
            status=step.status.value.replace("_", " ").title(),
            notes=step.notes,
            mode=step.mode,
        )
        await self._mount_and_scroll(PlanStepUpdateMessage(event))

    async def _maybe_mark_plan_complete(self, plan_id: str) -> None:
        if not self._planner_agent:
            return
        plan = self._planner_agent.complete_if_possible()
        if not plan or plan.plan_id != plan_id:
            return
        if plan.status == PlanRunStatus.COMPLETED:
            await self._refresh_planner_panel(plan)
            await self._mount_and_scroll(
                UserCommandMessage(f"Planner goal '{plan.goal}' completed.")
            )
            self._cancel_plan_executor()
            self._persist_plan_state()

    async def _load_plan_state(self) -> None:
        try:
            self._plan_state_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("Failed to ensure planner state directory", exc_info=exc)
            return

        if not self._plan_state_file.exists():
            return

        try:
            raw_payload = json.loads(self._plan_state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to read planner state", exc_info=exc)
            return

        payload = raw_payload if isinstance(raw_payload, dict) else {}
        plan_data = payload.get("plan")
        if plan_data is None:
            plan_data = raw_payload if isinstance(raw_payload, dict) else None
        if not isinstance(plan_data, dict):
            return

        owner_id = payload.get("owner_instance_id")
        owner_pid = payload.get("owner_pid")
        if owner_id and owner_id != self._planner_instance_id:
            if self._is_pid_active(owner_pid):
                logger.debug(
                    "Planner state is owned by another active instance (pid=%s)",
                    owner_pid,
                )
                return

        planner = self._ensure_planner()
        try:
            plan = planner.load_from_dict(plan_data)
        except Exception as exc:
            logger.debug("Failed to deserialize planner state", exc_info=exc)
            return

        await self._refresh_planner_panel(plan)
        if not self._plan_payload_owned_by_self(payload):
            self._persist_plan_state()
        if plan.status == PlanRunStatus.ACTIVE:
            self._start_plan_executor(plan.plan_id)


def run_textual_ui(
    config: VibeConfig,
    auto_approve: bool = False,
    enable_streaming: bool = False,
    initial_prompt: str | None = None,
    loaded_messages: list[LLMMessage] | None = None,
    session_info: ResumeSessionInfo | None = None,
) -> None:
    update_notifier = PyPIVersionUpdateGateway(project_name="mistral-vibe")
    update_cache_repository = FileSystemUpdateCacheRepository()
    app = VibeApp(
        config=config,
        auto_approve=auto_approve,
        enable_streaming=enable_streaming,
        initial_prompt=initial_prompt,
        loaded_messages=loaded_messages,
        session_info=session_info,
        version_update_notifier=update_notifier,
        update_cache_repository=update_cache_repository,
    )
    app.run()
