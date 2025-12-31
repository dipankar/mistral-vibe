# UI & Planner Refactor Roadmap

This document tracks the refactor required before we expand the CLI UI or add new features. The work is split into five phases; each phase can ship independently but later phases depend on the earlier ones.

## Phase 1 – Unified UI State
- [x] Introduce a central `UIStateStore` that owns planner state, loading indicators, and bottom panel mode.
- [x] Replace scattered `self._foo` booleans in `VibeApp` with store accessors and reactive watchers.
- [x] Add unit tests for store mutations (no Textual rendering required).

## Phase 2 – Planner Controller
- [x] Extract planning logic (plan creation, persistence, executor, subagent coordination) into a `PlannerController`.
- [x] Update `VibeApp` to call the controller API instead of manipulating `PlannerAgent` directly.
- [x] Ensure the controller publishes plan/subagent updates back into the UI state store.

## Phase 3 – Planner Confirmation UX & Thinking Mode Integration
- [x] Present planner confirmation prompts using the bottom-panel state machine (approval-style yes/no).
- [x] Keep the chat input visible during confirmations and plan execution.
- [x] When thinking mode is enabled, default to planning/subagents while still allowing the user to decline.

## Phase 4 – NNG Event Bus
- [x] Introduce an `ipc.event_bus` module backed by NNG (pub/sub) for agent + planner events.
- [x] Adapt the current `EventDispatcher` to publish to the bus while maintaining in-process compatibility.
- [x] Update the Textual UI (and other consumers) to subscribe via the bus abstraction.

## Phase 5 – Cleanup & Instrumentation
- [x] Remove legacy planner wiring that bypasses the controller or state store.
- [x] Add structured logs/metrics around planner confirmations, subagent starts/completions, and IPC health.
- [x] Document the new architecture so future UI work targets the store/controller/bus seams.

Progress on this document should be kept up to date; we should not add new UI surface area until all items above are complete.
