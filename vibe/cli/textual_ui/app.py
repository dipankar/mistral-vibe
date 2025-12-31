from __future__ import annotations

import asyncio
import inspect
import subprocess
from collections import deque
from enum import StrEnum, auto
from typing import Any, Callable, ClassVar, Coroutine, assert_never

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker, WorkerState

from vibe.cli.clipboard import copy_selection_to_clipboard
from vibe.cli.commands import CommandRegistry
from vibe.cli.textual_ui.consumers.textual_consumer import TextualEventConsumer
from vibe.cli.textual_ui.ui_state import UIState
from vibe.cli.textual_ui.planner_controller import PlannerCallbacks, PlannerController
from vibe.cli.textual_ui.planner_ui_controller import (
    PendingPlanRequest,
    PlanConfirmationHooks,
    PlannerUIController,
)
from vibe.cli.textual_ui.app_controller import AgentHooks, AppController
from vibe.cli.textual_ui.bottom_panel_manager import BottomPanelManager
from vibe.cli.textual_ui.command_controller import CommandController
from vibe.core.dispatchers import EventDispatcher
from vibe.cli.textual_ui.event_bus_adapter import EventBusAdapter
from vibe.cli.textual_ui.presenters import ChatInputPresenter
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.cli.textual_ui.widgets.context_progress import ContextProgress, TokenState
from vibe.cli.textual_ui.widgets.loading import LoadingWidget
from vibe.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    BashOutputMessage,
    ErrorMessage,
    InterruptMessage,
    MemoryUpdateMessage,
    PlanDecisionMessage,
    PlanStartedMessage,
    ThinkingPlanMessage,
    UserCommandMessage,
    UserMessage,
)
from vibe.cli.textual_ui.widgets.mode_indicator import ModeIndicator
from vibe.cli.textual_ui.widgets.memory_panel import MemoryPanel
from vibe.cli.textual_ui.widgets.planner_ticker import PlannerTicker, PlannerTickerState
from vibe.cli.textual_ui.widgets.path_display import PathDisplay
from vibe.cli.textual_ui.widgets.subagent_activity import (
    SubagentActivityPanel,
    SubagentPanelEntry,
)
from vibe.cli.textual_ui.widgets.sidebar import Sidebar
from vibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from vibe.cli.textual_ui.widgets.welcome import WelcomeBanner
from vibe.cli.textual_ui.state_store import BottomPanelMode, UIStateStore
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
)
from vibe.core.subagent import SubAgent, SubAgentConfig, SubAgentResult
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.base import BaseToolConfig, ToolPermission
from vibe.core.types import (
    ApprovalResponse,
    BaseEvent,
    CompactEndEvent,
    LLMMessage,
    MemoryEntryEvent,
    PlanDecisionEvent,
    PlanStartedEvent,
    ResumeSessionInfo,
    Role,
    SubagentProgressEvent,
    ToolResultEvent,
)
from vibe.core.utils import (
    CancellationReason,
    get_user_cancellation_message,
    is_dangerous_directory,
    logger,
)


DEFAULT_THINKING_MODE_INSTRUCTIONS = (
    "You are in thinking mode. Begin with a concise 'Thoughts' section containing no more than "
    "three bullet points that outline reasoning, risks, or next checks. After that, provide an "
    "'Answer' section with actionable guidance. Keep Thoughts under 100 words."
)

ACTIVE_SUBAGENT_STATUSES: set[PlanStepStatus] = {
    PlanStepStatus.IN_PROGRESS,
    PlanStepStatus.NEEDS_DECISION,
    PlanStepStatus.BLOCKED,
}

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

        self.event_handler: TextualEventConsumer | None = None
        self._event_bus_adapter = EventBusAdapter(self.config.effective_workdir)
        self._event_dispatcher = EventDispatcher(event_bus=self._event_bus_adapter.publisher)
        self._ui_event_dispatcher: EventDispatcher | None = None
        self._ui_dispatcher_registered = False
        self._ui_state: UIState | None = None
        self.commands = CommandRegistry()

        self._chat_input_container: ChatInputContainer | None = None
        self._chat_input_presenter: ChatInputPresenter | None = None
        self._mode_indicator: ModeIndicator | None = None
        self._context_progress: ContextProgress | None = None
        self._planner_ticker: PlannerTicker | None = None
        self._memory_panel: MemoryPanel | None = None
        self._ui_store = UIStateStore()
        self._chat_input_presenter = ChatInputPresenter(self._ui_store)
        self._bottom_panel_manager = BottomPanelManager(self, self._ui_store)
        self._planner_controller: PlannerController | None = None
        callbacks = PlannerCallbacks(
            refresh_plan=self._refresh_planner_panel,
            clear_plan_cards=self._clear_plan_cards,
            append_message=self._planner_append_message,
            update_step_card=self._mount_or_update_step_card,
            notify_decision_required=self._notify_step_decision_needed,
            set_active_step=self._set_active_plan_step,
            run_subagent=self._execute_step_with_subagent,
            wait_for_agent_idle=self._wait_for_agent_idle,
        )
        self._planner_controller = PlannerController(
            config=self.config,
            ui_store=self._ui_store,
            callbacks=callbacks,
            run_worker=self.run_worker,
            call_later=self.call_later,
        )
        confirmation_hooks = PlanConfirmationHooks(
            send_user_message=self._mount_and_scroll,
            start_planning=self._start_planning,
            continue_request=self._continue_pending_request,
            get_thinking_mode=lambda: self._thinking_mode,
        )
        self._planner_ui = PlannerUIController(self._ui_store, confirmation_hooks)
        self._planner_auto_start = config.planner_auto_start
        self._thinking_mode = config.thinking_mode_enabled
        self._app_controller: AppController | None = None
        agent_hooks = AgentHooks(
            ensure_agent_init_task=self._ensure_agent_init_task,
            mount_message=self._mount_and_scroll,
            start_processing=self._continue_pending_request,
        )
        self._app_controller = AppController(
            planner_ui=self._planner_ui,
            agent_hooks=agent_hooks,
            planner_auto_start_enabled=lambda: self._planner_auto_start,
            thinking_mode_enabled=lambda: self._thinking_mode,
            should_autostart_plan=self._should_autostart_plan,
        )
        self._command_controller = CommandController(self)
        self.theme = config.textual_theme

        self.history_file = HISTORY_FILE.path

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
        self._subagent_entries: dict[str, SubagentPanelEntry] = {}
        self._plan_decision_cards: dict[str, PlanDecisionMessage] = {}
        self._current_token_state = TokenState()
        self._rate_limit_warning = False
        self._rate_limit_reset_task: asyncio.Task | None = None
        self._sidebar: Sidebar | None = None
        self._subagent_panel: SubagentActivityPanel | None = None

        # Event-based signaling for responsiveness (initialized lazily)
        self._agent_idle_event: asyncio.Event | None = None

        # Lock for protecting agent state transitions
        self._agent_state_lock: asyncio.Lock | None = None

        # Retry state for agent failures
        self._last_failed_prompt: str | None = None
        self._retry_count: int = 0
        self._max_retries: int = 3
        self._retry_delay_base: float = 1.0  # seconds, exponential backoff
        self._local_event_ids: deque[str] = deque()
        self._local_event_id_set: set[str] = set()
        self._local_event_window = 512

    @property
    def ui_store(self) -> UIStateStore:
        return self._ui_store

    def _ensure_events(self) -> None:
        """Ensure event objects and locks are created (must be called from async context)."""
        if self._agent_idle_event is None:
            self._agent_idle_event = asyncio.Event()
            self._agent_idle_event.set()  # Start in idle state
        if self._agent_state_lock is None:
            self._agent_state_lock = asyncio.Lock()

    def _setup_event_routing(self) -> None:
        if not self.event_handler or not self._event_dispatcher:
            return

        if not self._ui_dispatcher_registered:
            self._event_dispatcher.add_consumer(self.event_handler)
            self._ui_dispatcher_registered = True

        if self._ui_event_dispatcher is None:
            self._ui_event_dispatcher = EventDispatcher()
            self._ui_event_dispatcher.add_consumer(self.event_handler)
        self._event_bus_adapter.start_listener(self._handle_bus_event)

    def _record_local_event(self, event: BaseEvent) -> None:
        event_id = getattr(event, "event_id", None)
        if not event_id:
            return
        self._local_event_ids.append(event_id)
        self._local_event_id_set.add(event_id)
        if len(self._local_event_ids) > self._local_event_window:
            old_id = self._local_event_ids.popleft()
            self._local_event_id_set.discard(old_id)

    def _should_ignore_bus_event(self, event: BaseEvent) -> bool:
        event_id = getattr(event, "event_id", None)
        if not event_id:
            return False
        if event_id not in self._local_event_id_set:
            return False
        self._local_event_id_set.discard(event_id)
        try:
            self._local_event_ids.remove(event_id)
        except ValueError:
            pass
        return True

    async def _dispatch_ui_event(self, event: BaseEvent, *, force: bool = False) -> None:
        if not self._ui_event_dispatcher:
            return
        if not force and self._ui_dispatcher_registered:
            return
        await self._ui_event_dispatcher.dispatch(event)

    async def _fan_out_event(self, event: BaseEvent) -> None:
        if not self._event_dispatcher:
            return
        await self._event_dispatcher.dispatch(event)
        self._record_local_event(event)
        self._maybe_update_sidebar_from_event(event)
        await self._maybe_update_subagent_panel_from_event(event)
        await self._dispatch_ui_event(event)

    async def _handle_bus_event(self, event: BaseEvent) -> None:
        if self._should_ignore_bus_event(event):
            return
        await self._dispatch_ui_event(event, force=True)

    def _maybe_update_sidebar_from_event(self, event: BaseEvent) -> None:
        if not self._sidebar:
            return
        if isinstance(event, ToolResultEvent) and event.tool_name == "todo":
            todos = self._extract_todo_entries(event)
            if todos is not None:
                self._sidebar.update_todos(todos)

    async def _maybe_update_subagent_panel_from_event(self, event: BaseEvent) -> None:
        if not isinstance(event, SubagentProgressEvent):
            return
        entry = self._subagent_entries.get(event.step_id)
        if not entry:
            return
        entry.prompt_tokens = event.prompt_tokens
        entry.completion_tokens = event.completion_tokens
        entry.tool_calls = event.tool_calls
        entry.tool_successes = event.tool_successes
        entry.tool_failures = event.tool_failures
        if event.activity:
            entry.activity = event.activity
        entry.subagent_id = event.subagent_id
        await self._refresh_subagent_panel()

    def _extract_todo_entries(self, event: ToolResultEvent) -> list[dict] | None:
        result = event.result
        if result is None:
            return None

        todos = getattr(result, "todos", None)
        if todos is None and hasattr(result, "model_dump"):
            try:
                todos = result.model_dump().get("todos")
            except Exception:
                todos = None
        if not todos:
            return None

        normalized: list[dict] = []
        for item in todos:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "content": item.get("content", ""),
                        "status": str(item.get("status", "")),
                        "active_form": item.get("active_form") or item.get("activeForm", ""),
                    }
                )
                continue
            status = getattr(item, "status", "")
            normalized.append(
                {
                    "content": getattr(item, "content", ""),
                    "status": getattr(status, "value", status),
                    "active_form": getattr(item, "active_form", ""),
                }
            )
        return normalized

    async def _shutdown_event_bus(self) -> None:
        await self._event_bus_adapter.stop_listener()
        self._event_bus_adapter.close()

    def _signal_agent_busy(self) -> None:
        """Signal that the agent has started working."""
        if self._agent_idle_event:
            self._agent_idle_event.clear()

    def _signal_agent_idle(self) -> None:
        """Signal that the agent has become idle."""
        if self._agent_idle_event:
            self._agent_idle_event.set()

    def _is_retryable_error(self, error: Exception | str) -> bool:
        """Check if an error is transient and worth retrying."""
        error_str = str(error).lower()
        retryable_patterns = [
            "rate limit",
            "too many requests",
            "429",
            "503",
            "502",
            "504",
            "timeout",
            "connection",
            "network",
            "temporarily unavailable",
            "service unavailable",
            "server error",
            "internal error",
        ]
        return any(pattern in error_str for pattern in retryable_patterns)

    def _get_retry_delay(self) -> float:
        """Calculate exponential backoff delay for retry."""
        # Exponential backoff with jitter: base * 2^retry_count + random jitter
        import random
        delay = self._retry_delay_base * (2 ** self._retry_count)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        return min(delay + jitter, 30.0)  # Cap at 30 seconds

    def _reset_retry_state(self) -> None:
        """Reset retry state after successful operation."""
        self._last_failed_prompt = None
        self._retry_count = 0

    async def _retry_last_prompt(self) -> None:
        await self._command_controller.retry_last_prompt()

    def _create_safe_task(
        self,
        coro: Coroutine,
        *,
        name: str | None = None,
        error_callback: Callable[[Exception], None] | None = None,
    ) -> asyncio.Task:
        """Create an asyncio task with error boundary.

        Unhandled exceptions are logged and optionally passed to a callback.
        This prevents "Task exception was never retrieved" warnings.
        """
        task = asyncio.create_task(coro, name=name)

        def _handle_task_exception(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error(
                    "Unhandled exception in task %s: %s",
                    name or "unnamed",
                    exc,
                    exc_info=exc,
                )
                if error_callback:
                    try:
                        error_callback(exc)
                    except Exception:
                        pass

        task.add_done_callback(_handle_task_exception)
        return task

    def compose(self) -> ComposeResult:
        with Vertical(id="app-root"):
            with Horizontal(id="app-layout"):
                with Vertical(id="main-root"):
                    with VerticalScroll(id="chat"):
                        yield WelcomeBanner(self.config)
                        yield Static(id="messages")

                    with Horizontal(id="loading-area"):
                        yield Static(id="loading-area-content")
                        yield ModeIndicator(auto_approve=self.auto_approve)

                with Vertical(id="sidebar"):
                    yield Sidebar()

            with Static(id="bottom-app-container"):
                yield ChatInputContainer(
                    history_file=self.history_file,
                    command_registry=self.commands,
                    id="input-container",
                    show_warning=self.auto_approve,
                )

            with Horizontal(id="bottom-bar"):
                path_widget = PathDisplay(
                    self.config.displayed_workdir or self.config.effective_workdir
                )
                path_widget.add_class("path-display")
                yield path_widget
                yield PlannerTicker(id="planner-ticker")
                yield ContextProgress()

            yield MemoryPanel()


    async def on_mount(self) -> None:
        # Initialize event-based signaling
        self._ensure_events()

        # Get reference to enhanced sidebar
        self._sidebar = self.query_one(Sidebar)
        # Initialize UI state and consumer
        self._ui_state = UIState(self)
        self.event_handler = TextualEventConsumer(self)
        self._setup_event_routing()

        self._chat_input_container = self.query_one(ChatInputContainer)
        self._mode_indicator = self.query_one(ModeIndicator)
        self._context_progress = self.query_one(ContextProgress)
        self._planner_ticker = self.query_one(PlannerTicker)
        self._memory_panel = self.query_one(MemoryPanel)
        if self._chat_input_container:
            self._chat_input_presenter.set_container(self._chat_input_container)
            self._chat_input_container.set_thinking_mode(self._thinking_mode)
        await self._ensure_todo_placeholder()

        if self.config.auto_compact_threshold > 0:
            self._set_context_tokens(
                TokenState(
                    max_tokens=self.config.auto_compact_threshold,
                    current_tokens=0,
                    soft_limit_ratio=self.config.memory_soft_limit_ratio,
                )
            )
        else:
            self._set_context_tokens(TokenState())

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

        await self._load_plan_state()

    async def on_unmount(self) -> None:
        await self._shutdown_event_bus()
        parent_on_unmount = getattr(super(), "on_unmount", None)
        if parent_on_unmount:
            maybe_coro = parent_on_unmount()
            if inspect.isawaitable(maybe_coro):
                await maybe_coro

    def _process_initial_prompt(self) -> None:
        if self._initial_prompt:
            self.run_worker(
                self._handle_user_message(self._initial_prompt), exclusive=False
            )

    def _prepare_agent_prompt(
        self, user_text: str, thinking_outline: str | None = None
    ) -> str:
        if not self._thinking_mode:
            return user_text
        # Use configured instructions or fall back to default
        instructions = (
            self.config.thinking_mode_instructions
            or DEFAULT_THINKING_MODE_INSTRUCTIONS
        )
        outline_block = (
            f"\n\nPlan Outline:\n{thinking_outline}" if thinking_outline else ""
        )
        return f"{instructions}{outline_block}\n\nRequest:\n{user_text}"

    def _set_context_tokens(self, state: TokenState) -> None:
        self._current_token_state = state
        if self._context_progress:
            self._context_progress.tokens = state
        self._update_planner_ticker()

    def _update_planner_ticker(self, plan: PlanState | None = None) -> None:
        if not self._planner_ticker:
            return

        plan_state = plan
        if plan_state is None and self._planner_controller:
            plan_state = self._planner_controller.get_plan()

        max_tokens = self._current_token_state.max_tokens
        current_tokens = self._current_token_state.current_tokens
        context_percentage = (
            int((current_tokens / max_tokens) * 100) if max_tokens else 0
        )
        soft_limit = (
            int(max_tokens * self._current_token_state.soft_limit_ratio)
            if max_tokens
            else 0
        )
        memory_warning = bool(
            soft_limit and current_tokens >= soft_limit
        )

        goal = plan_state.goal if plan_state else None
        status = (
            plan_state.status.value.replace("_", " ").title()
            if plan_state
            else "Idle"
        )
        total_steps = len(plan_state.steps) if plan_state else 0
        completed_steps = (
            sum(1 for step in plan_state.steps if step.status == PlanStepStatus.COMPLETED)
            if plan_state
            else 0
        )
        active_steps = (
            sum(1 for step in plan_state.steps if step.status == PlanStepStatus.IN_PROGRESS)
            if plan_state
            else 0
        )
        pending_steps = (
            sum(
                1
                for step in plan_state.steps
                if step.status
                in (PlanStepStatus.PENDING, PlanStepStatus.NEEDS_DECISION)
            )
            if plan_state
            else 0
        )
        pending_decisions = (
            sum(1 for decision in plan_state.decisions if not decision.resolved)
            if plan_state
            else 0
        )

        self._planner_ticker.state = PlannerTickerState(
            goal=goal,
            status=status,
            total_steps=total_steps,
            completed_steps=completed_steps,
            active_steps=active_steps,
            pending_steps=pending_steps,
            pending_decisions=pending_decisions,
            thinking_mode=self._thinking_mode,
            rate_limited=self._rate_limit_warning,
            context_tokens=current_tokens,
            max_tokens=max_tokens,
            context_percentage=context_percentage,
            memory_warning=memory_warning,
        )

    def _trigger_rate_limit_alert(self) -> None:
        self._rate_limit_warning = True
        self._update_planner_ticker()
        if self._rate_limit_reset_task:
            self._rate_limit_reset_task.cancel()
        self._rate_limit_reset_task = self._create_safe_task(
            self._clear_rate_limit_alert(),
            name="rate-limit-reset",
        )

    async def _clear_rate_limit_alert(self, delay: float = 12.0) -> None:
        try:
            await asyncio.sleep(delay)
            self._rate_limit_warning = False
            self._update_planner_ticker()
        except asyncio.CancelledError:
            pass
        finally:
            self._rate_limit_reset_task = None

    @staticmethod
    def _looks_like_rate_limit_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "rate limit" in lowered
            or "too many requests" in lowered
            or "429" in lowered
        )

    async def _generate_thinking_outline(self, user_text: str) -> str | None:
        prompt_text = user_text.strip()
        if not prompt_text:
            return None

        truncated_prompt = prompt_text if len(prompt_text) <= 2000 else f"{prompt_text[:2000]}…"
        if not self._planner_controller:
            return None

        plan = await self._planner_controller.preview_plan(
            f"Thinking mode outline for the following request:\n{truncated_prompt}"
        )
        if not plan:
            return None

        await self._mount_and_scroll(ThinkingPlanMessage(plan))

        outline_lines = [
            f"{idx}. {step.title} [{step.mode or 'code'}]"
            for idx, step in enumerate(plan.steps, start=1)
        ]
        if plan.decisions:
            outline_lines.append("Decisions:")
            for decision in plan.decisions:
                options = ", ".join(decision.option_labels()) if decision.options else "freeform"
                selection = f" → {decision.selection}" if decision.selection else ""
                outline_lines.append(
                    f"- {decision.decision_id}: {decision.question} ({options}){selection}"
                )
        return "\n".join(outline_lines) or None

    async def _refresh_memory_panel(self) -> None:
        if not self.agent:
            if self._sidebar:
                self._sidebar.update_memory_summary([])
            return

        entries = self.agent.session_memory.entries
        if self._memory_panel:
            await self._memory_panel.update_entries(entries)
        if self._sidebar:
            self._sidebar.update_memory_summary(entries)

    async def _refresh_subagent_panel(self) -> None:
        if not self._subagent_entries:
            if self._subagent_panel:
                try:
                    await self._subagent_panel.clear_entries()
                except Exception:
                    pass
                try:
                    await self._subagent_panel.remove()
                except Exception:
                    pass
                self._subagent_panel = None
            return

        if not self._subagent_panel:
            panel = SubagentActivityPanel(id="subagent-activity-panel")
            self._subagent_panel = panel
            await self._mount_and_scroll(panel)

        entries = list(self._subagent_entries.values())
        await self._subagent_panel.update_entries(entries)

    async def _memory_command(self, argument: str | None = None) -> None:
        await self._command_controller.memory_command(argument)

    async def _ensure_todo_placeholder(self) -> None:
        # TodoPanel now handles empty state automatically
        pass

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

    async def on_chat_input_container_thinking_mode_toggle_requested(
        self, _: ChatInputContainer.ThinkingModeToggleRequested
    ) -> None:
        self._thinking_mode = not self._thinking_mode
        if self._chat_input_container:
            self._chat_input_container.set_thinking_mode(self._thinking_mode)
        self._update_planner_ticker()
        status = "enabled" if self._thinking_mode else "disabled"
        await self._mount_and_scroll(
            UserCommandMessage(f"Thinking mode {status}. Tab toggles this mode.")
        )

    async def on_chat_input_container_plan_confirmation_accepted(
        self, _: ChatInputContainer.PlanConfirmationAccepted
    ) -> None:
        await self._planner_ui.apply_choice(accept=True)

    async def on_chat_input_container_plan_confirmation_declined(
        self, _: ChatInputContainer.PlanConfirmationDeclined
    ) -> None:
        await self._planner_ui.apply_choice(accept=False)

    async def _toggle_thinking_mode(self, argument: str | None = None) -> None:
        await self._command_controller.toggle_thinking_mode(argument)

    def _set_thinking_mode(self, enabled: bool, *, persist: bool = False) -> None:
        """Set thinking mode state and optionally persist to config."""
        self._thinking_mode = enabled
        if self._chat_input_container:
            self._chat_input_container.set_thinking_mode(enabled)
        self._update_planner_ticker()
        if persist:
            VibeConfig.save_updates({"thinking_mode_enabled": enabled})

    async def on_approval_app_approval_granted(
        self, message: ApprovalApp.ApprovalGranted
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

        await self._bottom_panel_manager.show_input()

    async def on_approval_app_approval_granted_always_tool(
        self, message: ApprovalApp.ApprovalGrantedAlwaysTool
    ) -> None:
        self._set_tool_permission_always(
            message.tool_name, save_permanently=message.save_permanently
        )

        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

        await self._bottom_panel_manager.show_input()

    async def on_approval_app_approval_rejected(
        self, message: ApprovalApp.ApprovalRejected
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.OPERATION_CANCELLED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback))

        await self._bottom_panel_manager.show_input()

        if self._loading_widget and self._loading_widget.parent:
            await self._remove_loading_widget()

    async def _remove_loading_widget(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
            self._loading_widget = None

    async def _detach_loading_widget(self, loading: LoadingWidget) -> None:
        """Remove a specific loading widget instance without touching others."""
        if loading.parent:
            await loading.remove()
        if self._loading_widget is loading:
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

        await self._bottom_panel_manager.show_input()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes to propagate errors to UI."""
        if event.state == WorkerState.ERROR:
            worker = event.worker
            error = worker.error
            worker_name = worker.name or "background task"

            # Log the error
            logger.error(
                "Worker '%s' failed with error: %s",
                worker_name,
                error,
                exc_info=error,
            )

            # Show error in UI (schedule to avoid blocking)
            async def _show_worker_error() -> None:
                error_msg = str(error) if error else "Unknown error"
                await self._mount_and_scroll(
                    ErrorMessage(
                        f"Background task '{worker_name}' failed: {error_msg}",
                        collapsed=self._ui_store.tools_collapsed,
                    )
                )
                # If it's a rate limit error, trigger the alert
                if error and self._looks_like_rate_limit_error(str(error)):
                    self._trigger_rate_limit_alert()

            self.call_later(_show_worker_error)

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
                    "No command provided after '!'", collapsed=self._ui_store.tools_collapsed
                )
            )
            return

        def _run_subprocess() -> subprocess.CompletedProcess:
            """Run subprocess in a thread to avoid blocking the event loop."""
            return subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=False,
                timeout=30,
                cwd=self.config.effective_workdir,
            )

        try:
            # Run subprocess in a thread pool to keep UI responsive
            result = await asyncio.to_thread(_run_subprocess)
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
                    collapsed=self._ui_store.tools_collapsed,
                )
            )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(f"Command failed: {e}", collapsed=self._ui_store.tools_collapsed)
            )

    async def _handle_user_message(self, message: str) -> None:
        if self._app_controller:
            await self._app_controller.handle_user_message(message)
            return

        logger.warning("App controller unavailable; falling back to direct agent call")
        init_task = self._ensure_agent_init_task()
        pending_init = bool(init_task and not init_task.done())
        user_message = UserMessage(message, pending=pending_init)
        await self._mount_and_scroll(user_message)
        await self._continue_user_message(
            message=message,
            user_message=user_message,
            init_task=init_task,
            pending_init=pending_init,
        )

    async def _continue_user_message(
        self,
        *,
        message: str,
        user_message: UserMessage,
        init_task: asyncio.Task | None,
        pending_init: bool,
    ) -> None:
        thinking_outline = None
        if self._thinking_mode:
            thinking_outline = await self._generate_thinking_outline(message)

        agent_prompt = self._prepare_agent_prompt(message, thinking_outline)

        self.run_worker(
            self._process_user_message_after_mount(
                message=message,
                agent_prompt=agent_prompt,
                user_message=user_message,
                init_task=init_task,
                pending_init=pending_init,
            ),
            exclusive=False,
        )

    async def _continue_pending_request(
        self, request: PendingPlanRequest
    ) -> None:
        await self._continue_user_message(
            message=request.goal,
            user_message=request.user_message,
            init_task=request.init_task,
            pending_init=request.pending_init,
        )

    async def _process_user_message_after_mount(
        self,
        message: str,
        agent_prompt: str,
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
                    await self._detach_loading_widget(loading)
                    if pending_init:
                        await user_message.set_pending(False)
            elif pending_init:
                await user_message.set_pending(False)

            if pending_init and self._agent_init_interrupted:
                self._agent_init_interrupted = False
                return

            if self.agent and not self._agent_running:
                self._agent_task = self._create_safe_task(
                    self._handle_agent_turn(agent_prompt),
                    name="agent-turn",
                )
        except asyncio.CancelledError:
            self._agent_init_interrupted = False
            if pending_init:
                await user_message.set_pending(False)
            return

    async def _initialize_agent(self) -> None:
        if self.agent or self._agent_initializing:
            return

        self._agent_initializing = True
        # Timeout for agent initialization (30 seconds should be plenty)
        AGENT_INIT_TIMEOUT = 30.0

        try:
            async with asyncio.timeout(AGENT_INIT_TIMEOUT):
                # Run potentially blocking Agent creation in a thread
                agent = await asyncio.to_thread(
                    Agent,
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
        except asyncio.TimeoutError:
            self.agent = None
            error_msg = (
                f"Agent initialization timed out after {AGENT_INIT_TIMEOUT}s. "
                "Please check your network connection and API configuration."
            )
            await self._mount_and_scroll(
                ErrorMessage(error_msg, collapsed=self._ui_store.tools_collapsed)
            )
            # Timeout is retryable
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Use `/retry` to retry initialization, or the agent will "
                    "retry automatically on your next message."
                )
            )
        except asyncio.CancelledError:
            self.agent = None
            return
        except Exception as e:
            self.agent = None
            error_str = str(e)
            await self._mount_and_scroll(
                ErrorMessage(error_str, collapsed=self._ui_store.tools_collapsed)
            )
            # Offer retry for retryable errors
            if self._is_retryable_error(e):
                await self._mount_and_scroll(
                    UserCommandMessage(
                        "This error may be temporary. The agent will retry "
                        "automatically on your next message, or use `/retry`."
                    )
                )
                if self._looks_like_rate_limit_error(error_str):
                    self._trigger_rate_limit_alert()
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
            self._agent_init_task = self._create_safe_task(
                self._initialize_agent(),
                name="agent-init",
            )

        return self._agent_init_task

    async def _approval_callback(
        self, tool: str, args: dict, tool_call_id: str
    ) -> tuple[ApprovalResponse, str | None]:
        self._pending_approval = asyncio.Future()
        await self._bottom_panel_manager.show_approval(tool, args)
        result = await self._pending_approval
        self._pending_approval = None
        return result

    async def _handle_agent_turn(self, prompt: str) -> None:
        # Capture agent reference to prevent race conditions during iteration
        agent = self.agent
        if not agent:
            return

        self._agent_running = True
        self._signal_agent_busy()

        loading_area = self.query_one("#loading-area-content")

        loading = LoadingWidget()
        self._loading_widget = loading
        await loading_area.mount(loading)

        try:
            rendered_prompt = render_path_prompt(
                prompt, base_dir=self.config.effective_workdir
            )
            # Use captured agent reference for iteration
            async for event in agent.act(rendered_prompt):
                # Check if agent is still valid (could be cleared during cancellation)
                if self.agent is not None:
                    current_state = self._current_token_state
                    self._set_context_tokens(
                        TokenState(
                            max_tokens=current_state.max_tokens,
                            current_tokens=agent.stats.context_tokens,
                            soft_limit_ratio=current_state.soft_limit_ratio
                            or self.config.memory_soft_limit_ratio,
                        )
                    )

                # Dispatch event to all registered consumers and UI
                await self._fan_out_event(event)

                # Additional app-level event handling
                if isinstance(event, MemoryEntryEvent):
                    await self._refresh_memory_panel()
                elif isinstance(event, CompactEndEvent):
                    await self._refresh_memory_panel()

        except asyncio.CancelledError:
            await self._detach_loading_widget(loading)
            if self.event_handler:
                self.event_handler.stop_current_tool_call()
            raise
        except Exception as e:
            await self._detach_loading_widget(loading)
            if self.event_handler:
                self.event_handler.stop_current_tool_call()

            error_str = str(e)
            is_retryable = self._is_retryable_error(e)

            # Store prompt for potential retry
            if is_retryable:
                self._last_failed_prompt = prompt

            await self._mount_and_scroll(
                ErrorMessage(error_str, collapsed=self._ui_store.tools_collapsed)
            )

            if self._looks_like_rate_limit_error(error_str):
                self._trigger_rate_limit_alert()

            # Offer retry for retryable errors
            if is_retryable and self._retry_count < self._max_retries:
                await self._mount_and_scroll(
                    UserCommandMessage(
                        f"This error may be temporary. Use `/retry` to retry, "
                        f"or wait for auto-retry in {self._get_retry_delay():.0f}s. "
                        f"({self._retry_count + 1}/{self._max_retries} attempts remaining)"
                    )
                )
                # Schedule auto-retry for rate limit errors
                if self._looks_like_rate_limit_error(error_str):
                    self.call_later(self._retry_last_prompt)
        else:
            # Success - reset retry state
            self._reset_retry_state()
        finally:
            self._agent_running = False
            self._interrupt_requested = False
            self._agent_task = None
            await self._detach_loading_widget(loading)
            await self._finalize_current_streaming_message()
            self._signal_agent_idle()

    async def _interrupt_agent(self) -> None:
        # Use lock to prevent race conditions during state transitions
        if self._agent_state_lock:
            async with self._agent_state_lock:
                await self._interrupt_agent_impl()
        else:
            await self._interrupt_agent_impl()

    async def _interrupt_agent_impl(self) -> None:
        """Implementation of agent interrupt - called within state lock."""
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
        await self._command_controller.show_help()

    async def _start_planning(self, argument: str | None = None) -> None:
        await self._command_controller.start_planning(argument)

    def _should_autostart_plan(self) -> bool:
        controller = self._planner_controller
        if not controller:
            return False
        plan = controller.get_plan()
        if not plan:
            return True
        return plan.status in (PlanRunStatus.COMPLETED, PlanRunStatus.CANCELLED)

    async def _toggle_plan_auto(self, argument: str | None = None) -> None:
        await self._command_controller.toggle_plan_auto(argument)

    async def _show_plan_status(self, _: str | None = None) -> None:
        await self._command_controller.show_plan_status()

    async def _pause_plan(self, _: str | None = None) -> None:
        await self._command_controller.pause_plan()

    async def _resume_plan(self, _: str | None = None) -> None:
        await self._command_controller.resume_plan()

    async def _cancel_plan(self, _: str | None = None) -> None:
        await self._command_controller.cancel_plan()

    def _set_planner_auto_start(self, enabled: bool, *, persist: bool = True) -> None:
        self._planner_auto_start = enabled
        if persist:
            VibeConfig.save_updates({"planner_auto_start": enabled})

    async def _handle_plan_decision(self, argument: str | None = None) -> None:
        await self._command_controller.handle_plan_decision(argument)

    async def on_plan_decision_message_decision_selected(
        self, message: PlanDecisionMessage.DecisionSelected
    ) -> None:
        await self._command_controller.handle_plan_decision_selections(message)

    async def _emit_plan_events(self, plan: PlanState, include_summary: bool = True) -> None:
        if include_summary:
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
            await self._mount_or_update_step_card(plan, step)

        await self._emit_decision_events(plan)

    async def _mount_or_update_step_card(
        self, plan: PlanState, step: PlanStep
    ) -> None:
        if step.status not in ACTIVE_SUBAGENT_STATUSES:
            removed = self._subagent_entries.pop(step.step_id, None)
            if removed:
                await self._refresh_subagent_panel()
            self._update_planner_ticker(plan)
            return

        entry = self._subagent_entries.get(step.step_id)
        if entry:
            entry.title = step.title
            entry.owner = step.owner or entry.owner
            entry.mode = step.mode or entry.mode
            entry.notes = step.notes or entry.notes
            entry.status = step.status
        else:
            self._subagent_entries[step.step_id] = SubagentPanelEntry(
                step_id=step.step_id,
                title=step.title,
                goal=plan.goal,
                owner=step.owner,
                mode=step.mode,
                status=step.status,
                notes=step.notes,
            )
        await self._refresh_subagent_panel()
        self._update_planner_ticker(plan)

    async def _emit_decision_events(self, plan: PlanState) -> None:
        for decision in plan.decisions:
            await self._emit_decision_event(plan, decision)

    async def _emit_decision_event(
        self, plan: PlanState, decision: PlanDecision
    ) -> None:
        from vibe.core.types import DecisionOptionData
        event = PlanDecisionEvent(
            plan_id=plan.plan_id,
            decision_id=decision.decision_id,
            header=decision.header,
            question=decision.question,
            options=[
                DecisionOptionData(label=opt.label, description=opt.description)
                for opt in decision.options
            ],
            multi_select=decision.multi_select,
            resolved=decision.resolved,
            selections=decision.selections,
        )
        await self._mount_or_update_decision_card(event)

    async def _mount_or_update_decision_card(
        self, event: PlanDecisionEvent
    ) -> None:
        existing = self._plan_decision_cards.get(event.decision_id)
        if existing:
            await existing.update_event(event)
            self._update_planner_ticker()
            return
        message = PlanDecisionMessage(event)
        self._plan_decision_cards[event.decision_id] = message
        await self._mount_and_scroll(message)
        self._update_planner_ticker()

    async def _notify_step_decision_needed(
        self, plan: PlanState, step: PlanStep
    ) -> None:
        decision_id = step.decision_id
        if decision_id:
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

    async def _planner_append_message(self, text: str) -> None:
        await self._mount_and_scroll(UserCommandMessage(text))

    def _set_active_plan_step(self, step_id: str | None, mode: str | None) -> None:
        if self._sidebar:
            self._sidebar.set_active_step(step_id, mode)
        if self._subagent_panel:
            self._subagent_panel.focus_step(step_id)

    async def _show_status(self) -> None:
        await self._command_controller.show_status()

    async def _show_config(self) -> None:
        await self._command_controller.show_config()

    async def _reload_config(self) -> None:
        await self._command_controller.reload_config()

    async def _clear_history(self) -> None:
        await self._command_controller.clear_history()

    async def _show_log_path(self) -> None:
        await self._command_controller.show_log_path()

    async def _compact_history(self) -> None:
        await self._command_controller.compact_history()

    async def _exit_app(self) -> None:
        await self._command_controller.exit_app()

    def action_interrupt(self) -> None:
        if self._memory_panel and self._memory_panel.is_visible:
            self._memory_panel.hide_panel()
            self._bottom_panel_manager.focus_current()
            return
        if self._bottom_panel_manager.handle_interrupt():
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
        self._bottom_panel_manager.focus_current()

    async def action_toggle_tool(self) -> None:
        if not self.event_handler:
            return

        new_value = not self._ui_store.tools_collapsed
        self._ui_store.set_tools_collapsed(new_value)

        non_todo_results = [
            result
            for result in self.event_handler.tool_results
            if result.event.tool_name != "todo"
        ]

        for result in non_todo_results:
            result.collapsed = self._ui_store.tools_collapsed
            await result.render_result()

        try:
            error_messages = self.query(ErrorMessage)
            for error_msg in error_messages:
                error_msg.set_collapsed(self._ui_store.tools_collapsed)
        except Exception:
            pass

    async def action_toggle_todo(self) -> None:
        if not self.event_handler:
            return

        new_value = not self._ui_store.todos_collapsed
        self._ui_store.set_todos_collapsed(new_value)

        todo_results = [
            result
            for result in self.event_handler.tool_results
            if result.event.tool_name == "todo"
        ]

        for result in todo_results:
            result.collapsed = self._ui_store.todos_collapsed
            await result.render_result()

    def action_cycle_mode(self) -> None:
        if self._ui_store.bottom_panel_mode != BottomPanelMode.INPUT:
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

        self._bottom_panel_manager.focus_current()

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

    # ITextualApp interface methods for TextualEventConsumer

    async def mount_to_chat(self, widget: Widget) -> None:
        """Mount widget to chat/messages area (ITextualApp interface)."""
        await self._mount_and_scroll(widget)

    async def mount_to_sidebar(self, widget: Widget) -> None:
        """Sidebar now shows summaries only; ignore direct mounts."""
        return

    def get_loading_widget(self) -> LoadingWidget | None:
        """Get current loading widget (ITextualApp interface)."""
        return self._loading_widget

    def get_ui_state(self) -> UIState:
        """Get current UI state (ITextualApp interface)."""
        if self._ui_state is None:
            self._ui_state = UIState(self)
        return self._ui_state

    async def on_plan_started(self, event: Any) -> None:
        """Handle plan started event (ITextualApp interface)."""
        # Plan events are handled separately by the planner integration
        pass

    async def on_plan_step_update(self, event: Any) -> None:
        """Handle plan step update (ITextualApp interface)."""
        pass

    async def on_plan_decision(self, event: Any) -> None:
        """Handle plan decision (ITextualApp interface)."""
        pass

    async def on_plan_completed(self, event: Any) -> None:
        """Handle plan completion (ITextualApp interface)."""
        pass

    async def on_plan_resource_warning(self, event: Any) -> None:
        """Handle resource warning (ITextualApp interface)."""
        pass

    async def on_memory_updated(self) -> None:
        """Handle memory panel refresh (ITextualApp interface)."""
        await self._refresh_memory_panel()

    # Event dispatcher consumer management

    def add_event_consumer(self, consumer: Any) -> None:
        """Add an external event consumer to receive agent events.

        Enables Web API, logging, or other external consumers to
        receive events from the agent.
        """
        if self._event_dispatcher:
            self._event_dispatcher.add_consumer(consumer)

    def remove_event_consumer(self, consumer: Any) -> None:
        """Remove an event consumer."""
        if self._event_dispatcher:
            self._event_dispatcher.remove_consumer(consumer)

    @property
    def event_dispatcher(self) -> EventDispatcher | None:
        """Access to the event dispatcher for advanced use cases."""
        return self._event_dispatcher

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

        self._update_notification_task = self._create_safe_task(
            self._check_version_update(),
            name="version-update-check",
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
                        collapsed=self._ui_store.tools_collapsed,
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

    def action_copy_text(self) -> None:
        """Override Textual's default copy action to use our safe implementation.

        Textual's built-in action_copy_text has a bug where Selection coordinates
        are screen-relative but extraction expects widget-relative coordinates,
        causing IndexError on multi-line layouts.
        """
        copy_selection_to_clipboard(self)

    async def _clear_plan_cards(self) -> None:
        self._subagent_entries.clear()
        if self._subagent_panel:
            try:
                await self._subagent_panel.clear_entries()
            except Exception:
                pass
            try:
                await self._subagent_panel.remove()
            except Exception:
                pass
            self._subagent_panel = None

        for message in list(self._plan_decision_cards.values()):
            try:
                if message.parent:
                    await message.remove()
            except Exception:
                pass
        self._plan_decision_cards.clear()
        self._update_planner_ticker(None)

    async def _refresh_planner_panel(self, plan: PlanState | None = None) -> None:
        # Update sidebar plan panel
        if self._sidebar:
            self._sidebar.update_plan(plan)
        status = plan.status if plan else PlanRunStatus.IDLE
        goal = plan.goal if plan else None
        self._ui_store.set_plan_status(status, goal)
        self._update_planner_ticker(plan)

    async def _wait_for_agent_idle(self, timeout: float = 60.0) -> None:
        """Wait for the agent to become idle using event-based signaling.

        Args:
            timeout: Maximum time to wait in seconds (default 60s).
        """
        # Quick check first
        busy = self._agent_running or (
            self._agent_init_task and not self._agent_init_task.done()
        )
        if not busy:
            return

        # Wait on event with timeout
        if self._agent_idle_event:
            try:
                await asyncio.wait_for(self._agent_idle_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for agent to become idle")
        else:
            # Fallback to polling if event not initialized
            while True:
                busy = self._agent_running or (
                    self._agent_init_task and not self._agent_init_task.done()
                )
                if not busy:
                    return
                await asyncio.sleep(0.05)

    async def _execute_step_with_subagent(
        self, step: PlanStep, prompt: str
    ) -> SubAgentResult:
        """Execute a plan step using an isolated SubAgent."""
        from vibe.core.tools.manager import ToolManager

        # Create SubAgent config from step
        subagent_config = SubAgentConfig(
            step_id=step.step_id,
            title=step.title,
            mode=step.mode or "code",
            max_turns=20,  # Allow enough turns for complex tasks
            system_prompt_suffix=(
                f"\n\n## Current Task\n"
                f"You are executing step '{step.title}' as part of a larger plan.\n"
                f"Focus only on completing this specific task."
            ),
        )

        # Get base system prompt from a fresh tool manager
        tool_manager = ToolManager(self.config)
        base_system_prompt = get_universal_system_prompt(tool_manager, self.config)

        # Create SubAgent
        logger.info(
            "planner.subagent_start",
            extra={
                "step_id": step.step_id,
                "title": step.title,
                "mode": step.mode or "code",
            },
        )
        subagent = SubAgent(
            config=self.config,
            subagent_config=subagent_config,
            system_prompt=base_system_prompt,
            parent_approval_callback=self._approval_callback if not self.auto_approve else None,
            enable_streaming=self.enable_streaming,
        )

        # Execute and handle events
        loading = LoadingWidget()
        self._loading_widget = loading
        loading_area = self.query_one("#loading-area-content")
        await loading_area.mount(loading)

        try:
            async for event in subagent.execute(prompt):
                # Dispatch events to all registered consumers and UI
                await self._fan_out_event(event)
        finally:
            await self._detach_loading_widget(loading)

        result = subagent.get_result()
        stats = result.stats
        logger.info(
            "planner.subagent_complete",
            extra={
                "step_id": result.step_id,
                "success": result.success,
                "tokens": stats.total_tokens,
                "duration": stats.duration_seconds,
                "artifacts": len(result.artifacts),
            },
        )
        return result

    async def _load_plan_state(self) -> None:
        if not self._planner_controller:
            return
        await self._planner_controller.load_plan_state(self._session_info)


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
