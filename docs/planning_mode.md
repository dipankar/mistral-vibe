# Planning Mode Proposal

## Purpose

Add a Claude-style planning workflow to Vibe so users can opt into a deliberate “plan → execute” cycle. The planner should:

- Generate and maintain a structured plan (steps, owners, status) for a user goal.
- Ask for key decisions before executing risky or ambiguous steps.
- Delegate execution to scoped subagents that run existing Vibe tools.
- Surface progress in the UI by reusing the todo tab’s lower half when planning is active.

## User Flow

1. User runs `/plan <goal>` (or `/plan` to start an interactive prompt).
2. PlannerAgent drafts a plan: numbered steps, dependencies, open questions.
3. User reviews within the sidebar; planner highlights “decision checkpoints” and prompts for confirmations or parameter choices.
4. Once confirmed, planner executes steps sequentially (or concurrently when safe) by launching SubAgents.
5. Each subagent streams events (tool calls, outputs) into chat as usual. When completed, planner updates the plan status and stores a summarized artifact.
6. User can inspect progress via `/plan status`, pause via `/plan pause`, resume with `/plan resume`, or cancel (`/plan cancel`).

## Commands

| Command | Description |
|---------|-------------|
| `/plan [goal]` | Start planning for the provided goal (or prompt user if omitted). |
| `/plan status` | Show the current plan, step statuses, and outstanding decisions. |
| `/plan pause` | Pause the active planner (no new steps start). |
| `/plan resume` | Resume after a pause or after decisions have been provided. |
| `/plan cancel` | Abort planning mode, cleaning up planner/subagent state. |

All commands belong to `PlannerCommand` entries in `CommandRegistry` and display in `/help`.

## Planner Architecture

### PlannerAgent

New orchestrator module (`vibe/core/planner.py`). Responsibilities:

- **Planning Prompt**: uses a planner-specific system prompt emphasizing decomposition and decision capture.
- **Plan Graph**: Maintains `PlanState` containing:
  - `PlanMetadata`: goal, author, timestamps.
  - `PlanStep`: `id`, `description`, `status` (`pending`, `awaiting_decision`, `in_progress`, `blocked`, `done`), `owner`, `dependencies`, `artifacts`.
  - `DecisionRequest`: `id`, `question`, `options`, `context`. Planner blocks execution when a decision is pending.
- **Execution Loop**:
  1. `build_plan(goal) -> PlanState`
  2. Emit `PlanStartedEvent`
  3. For each runnable step:
     - If the step contains a decision checkpoint, emit `PlanDecisionEvent` and wait for `/plan decide <id> <choice>`.
     - Otherwise, spawn `SubAgentRunner` with step-specific instructions.
  4. On completion/error, update PlanState, emit `PlanStepUpdateEvent`.
  5. Once all steps succeed, emit `PlanCompletedEvent`.

- **Shared Memory & Context Budget**: Planner owns `PlanningMemory`, which references:
  - `session_memory` summaries (shared context).
  - Step artifacts (diffs, logs, generated files).
  - Lessons learned or constraints to inform later steps.
  - `context_budget`: remaining tokens relative to both session auto-compact threshold and planner-specific `max_context_tokens`.
  - `rate_limit_budget`: remaining retries/backoff state when upstream providers throttle requests.
  - Before launching a subagent, planner estimates token usage; if it would exceed the budget, planner either auto-compacts, trims history, or pauses to ask the user whether to discard older steps.
  - Planner tracks rate-limit errors per step and uses exponential backoff + jitter; if the retry budget is exhausted, the planner surfaces a decision prompt (“Wait longer?” / “Skip step?”).

### SubAgents

Reuse `vibe.core.agent.Agent`, but wrap it in `SubAgentRunner` that supplies:

- `subagent_id` (for auditing), `step_id`.
- Specialty-aware system prompt: `[PLANNER CONTEXT] Step #n: <description>` followed by `mode`-specific guidance (`code`, `test`, `research`, `design`, `docs`, `run`). Planner sets `PlanStep.mode`, letting the execution agent adapt instructions without changing core tools.
- Filtered tool permissions (can mirror user config or tighten).
- After completion, `SubAgentRunner` returns `SubAgentResult`:
  - `status`: success/failure
  - `summary`: short string for planner memory
  - `artifacts`: references to modified files, outputs, tool logs.
  - `context_delta`: tokens consumed and memory impact so planner can update the remaining context budget.

Specialists can be added later (e.g., `TestAgent`, `ResearchAgent`), but MVP uses a single subagent implementation with configurable modes.

## UI Integration

### Sidebar Reuse

- Todo sidebar is vertically split:
  - **Top Half**: existing todo feed.
  - **Bottom Half**: “Planner Panel” when planning mode is active.
- When planner is idle, lower half shows a CTA (“Run `/plan <goal>` to start planning”).
- Planner Panel contents:
  - Plan title + status.
  - Collapsible list of steps with badges (`Pending`, `Needs decision`, `Running`, `Done`, `Blocked`).
  - Decision prompts appear inline with buttons (e.g., “Approve plan”, “Choose option A/B”) or instruct the user to run `/plan decide`.
- If no plan is active, the bottom half reverts to todo content (existing behavior).

### Chat Stream

- New textual widgets for planning events:
  - `PlanStartedMessage`
  - `PlanStepMessage` (updates)
  - `PlanDecisionPrompt`
  - `PlanCompletedMessage`
- Planner decisions show as interactive prompts in chat plus mirrored in the sidebar to ensure visibility.

## Decision Checkpoints

Planner should explicitly ask before:

1. Executing destructive steps (e.g., rewriting large files).
2. Choosing between strategies (e.g., “Refactor vs rewrite”).
3. Running risky commands (shell operations).

Workflow:

- `PlanDecisionEvent` includes `question`, `options`, and `default`.
- User responds via `/plan decide <id> <option>` or by clicking an actionable button when UI support lands.
- Planner stores the decision in `PlanState.decisions` and resumes execution.

## Event & Type Additions

Extend `vibe/core/types.py` with:

- `class PlanEvent(BaseEvent)`
- `class PlanStartedEvent(PlanEvent)`
- `class PlanStepUpdateEvent(PlanEvent)`
- `class PlanDecisionEvent(PlanEvent)`
- `class PlanCompletedEvent(PlanEvent)`

Each carries enough data for UI rendering and logging (plan id, step id, status, summaries).

## Configuration

Add new config entries (`[planner]` section):

- `enabled` (default true)
- `auto_run` (whether to auto-start execution after plan generation)
- `max_subagents`
- `decision_timeout_seconds`
- `max_context_tokens` (planner-level cap across subagents)
- `retry_strategy` (controls rate-limit handling and backoffs)

Expose toggles in `/config` later if needed.

## Implementation Phases

1. **Spec & Data Structures**
   - Add planner module, plan models (`PlanState`, `PlanStep`, `DecisionRequest`).
   - Extend command registry with `/plan` commands.
   - Define new events and placeholder UI widgets.

2. **Plan Generation**
   - Implement `PlannerAgent.build_plan` using existing LLM backend.
   - Display plan in sidebar/chat without executing steps.

3. **Execution & Decisions**
   - Implement `SubAgentRunner`.
   - Handle decision prompts + `/plan decide`.
   - Update UI in real time; integrate with session memory.

4. **Polish**
- Persist plan state across reloads/saves.
- Add tests (unit for planner logic, snapshot for UI).
- Tie into logging/analytics.

## Persistence & Resume

- Plan state serializes to a per-workspace file under `~/.vibe/plans/<slug>.json` whenever the planner mutates; the directory is created on demand.
- Each workspace gets its own slugged file name where `<slug>` hashes the repo path so multiple projects never collide. The payload records the owning instance id + PID so we skip adopting plans that belong to other running CLIs.
- On app startup we attempt to load this file, ignore corrupt JSON, and resume execution automatically if the stored plan was `ACTIVE`.
- Snapshot tests stub `_load_plan_state()` to avoid touching the user filesystem.
- Future config can expose a toggle (e.g., `planner.enable_persistence`) so advanced users can opt out of disk writes entirely.

## Open Questions

- Should planner support parallel execution? (recommended to defer until sequential MVP is stable.)
- How to handle planner restarts after Vibe restarts? (Plan to serialize `PlanState` into history.)
- How to expose planner steps in transcripts/log exports? (Need log schema update.)
- How aggressive should context trimming be vs. auto-compaction? (Need heuristics.)
- What retry budgets should apply when upstream rate limits are hit repeatedly?
- How to surface rate-limit stalls to the user without spamming notifications?

## Next Steps

- Validate this plan with stakeholders.
- Once approved, implement Phase 1 (data structures + commands + basic UI toggle) before enabling execution.
