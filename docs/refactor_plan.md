# Core Refactor Plan

This document tracks the staged refactor needed to make Mistral Vibe maintainable, testable, and consistently IPC-driven. Each phase lists concrete steps plus completion criteria so we can tick them off as work lands.

## Phase 1 – Agent Decomposition

**Goal:** Break `vibe/core/agent.py` into testable collaborators without changing external behaviour or the event bus contract.

1. **ConversationState component**
   - Encapsulate message history, memory syncing, observer flushing, and stats snapshots.
   - `Agent` should delegate `_ensure_memory_initialized`, `_flush_new_messages`, `_clean_message_history`, and related helpers to this class.
   - Unit tests should cover history trimming and observer fan-out without spinning up the full agent.
2. **MiddlewareRunner wrapper**
   - Move `_setup_middleware` logic and middleware execution into a dedicated helper (e.g., `MiddlewareCoordinator`).
   - Provide a narrow interface (`prepare(turn_context) -> MiddlewareResult`) so new middleware can be slotted in without touching `Agent`.
3. **ToolInvocationManager**
   - Extract `_handle_tool_calls`, approval routing, and allow/deny list enforcement.
   - Expose structured events or callbacks so stats/logging can subscribe; keep business logic (duration tracking, error strings) out of the main agent loop.
   - _Status: Implemented in `vibe/core/tool_invocation_manager.py`; `Agent` now delegates tool execution to it._
4. **InteractionObserver**
   - Own `AgentStats` mutation + `InteractionLogger` persistence, driven by emitted events (`assistant`, `tool_call_start`, `tool_result`, `compact`, etc.).
   - This allows deterministic tests by swapping the observer with a stub.
5. **Acceptance**
   - Existing public APIs (`Agent.act`, `Agent.set_approval_callback`, stats usage) remain intact.
   - All events still flow through the dispatcher/bus via `AgentRunner`.
   - New components have targeted unit tests.

## Phase 2 – VibeApp & UI Controllers

**Goal:** Separate Textual view code from planner/agent orchestration while keeping the UI wired to the NNG bus.

1. **EventBusAdapter**
   - Wrap subscriber start/stop logic and dispatching to `TextualEventConsumer`.
   - Provide hooks for clean shutdown so tests can swap in-memory bus stubs.
2. **AppController**
   - Own agent lifecycle, planner controller bridging, approval prompts, thinking mode heuristics, and command execution.
   - VibeApp should forward widget events to the controller, receiving high-level callbacks (e.g., `render_message`, `update_sidebar`).
3. **Bottom Panel / Widget Managers**
   - Create small presenters for chat input, config view, approval view, and planner ticker that subscribe to `UIStateStore` updates.
   - Encourage unidirectional flow: store updates -> presenters -> widgets.
4. **Testing**
   - Controller logic becomes unit-testable via fake event bus + fake state store.
   - Textual integration still exists but is thinner.
5. **IPC**
   - Ensure NNG wiring happens inside `EventBusAdapter` so the UI never silently falls back.

## Phase 3 – CLI Entrypoint Decomposition

**Goal:** Turn `vibe/cli/entrypoint.py` into a small coordinator with reusable helpers.

1. **Setup helpers**
   - `ensure_config_files()`, `ensure_instructions_file()`, `ensure_history_file()`; unit-testable without running the CLI.
2. **Session management**
   - `SessionLockManager` context manager around acquire/release; emits clear errors if lock fails.
   - `ResumePrompt` helper: inspects `InteractionLogger`, prompts user, returns `ResumeSessionInfo`.
3. **Mode runners**
   - `run_programmatic_session(config, args, prompt)` (already largely implemented) should be called from `main`.
   - `run_textual_session(config, args, initial_prompt, session_info)` handles UI path.
4. **Config overrides**
   - Keep environment overrides (enabled tools, agents) in isolated helpers so we can reuse them in tests/other entrypoints.
5. **Outcome**
   - `main()` reads like: parse args → ensure setup → maybe resume → run programmatic or run interactive (inside lock) → handle errors.

## Phase 4 – IPC & Testing

**Goal:** Validate that every surface emits and consumes events exclusively over NNG and document the requirement.

1. **IPC conformance**
   - Audit new components to ensure they instantiate `EventBusPublisher/EventBusSubscriber` rather than ad-hoc messaging.
   - Provide simple stubs/mocks so unit tests can run without a live pynng socket.
2. **Testing strategy**
   - Add targeted tests for extracted components (ConversationState, MiddlewareRunner, AppController, CLI helpers).
   - Document how to run integration tests locally (e.g., `uv run pytest tests/...`).
3. **Documentation**
   - Update `docs/architecture.md` (already partially done) as components land.
   - Note the pynng requirement in README/installation instructions.
4. **Telemetry**
   - Ensure structured logs (planner confirmations, subagent flows, IPC start/stop) survive refactor; adjust log locations if needed.

## Tracking

- Use this file as the source-of-truth checklist; tick off items as PRs land.
- Keep the existing `docs/ui_refactor_plan.md` focused on UI/Planner specifics; this document covers core/CLI-wide work.
- Reference related issues/PRs beside each item when complete.
