# Planning Mode

Planning mode enables a deliberate "plan → execute" workflow in Vibe. The planner decomposes complex goals into structured steps, manages dependencies, handles decision checkpoints, and orchestrates execution through isolated subagents.

## Quick Start

```bash
# Start planning for a goal
/plan Refactor the authentication module to use JWT tokens

# Check progress
/plan status

# Pause/resume execution
/plan pause
/plan resume

# Cancel and clean up
/plan cancel
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `/plan [goal]` | Start planning for the provided goal (prompts if omitted) |
| `/plan status` | Show current plan, step statuses, and outstanding decisions |
| `/plan pause` | Pause execution (no new steps start) |
| `/plan resume` | Resume after pause or after decisions provided |
| `/plan cancel` | Abort planning, clean up state |
| `/plan auto on\|off` | Toggle auto-start execution after plan generation |
| `/plan decide <id> <choice>` | Record a decision and unblock waiting steps |
| `/plan retry <step_id>` | Reset a blocked step to pending and re-run it |

## User Flow

1. **Initiate**: Run `/plan <goal>` to start planning
2. **Review**: PlannerAgent generates a structured plan with numbered steps, dependencies, and open questions
3. **Decide**: If the planner needs input, it presents decision prompts in the chat and sidebar
4. **Execute**: Steps execute sequentially or in parallel (when dependencies allow)
5. **Monitor**: Track progress via `/plan status` or the sidebar panel
6. **Complete**: Plan finalizes when all steps succeed

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        PlannerAgent                         │
│  vibe/core/planner.py                                       │
│  - Decomposes goals into PlanState with steps & decisions   │
│  - Validates dependencies (circular, missing, self-refs)    │
│  - Manages execution lifecycle (pause/resume/cancel)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         SubAgent                            │
│  vibe/core/subagent.py                                      │
│  - Isolated execution environment per step                  │
│  - Mode-aware prompts (code, test, research, etc.)          │
│  - Token budget management                                  │
│  - Tool filtering per step                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Mode Catalog                          │
│  vibe/core/modes.py                                         │
│  - Unified guidance for planner step creation               │
│  - Detailed instructions for subagent execution             │
└─────────────────────────────────────────────────────────────┘
```

### Data Structures

**PlanState** (`vibe/core/planner.py`):
- `plan_id`: Unique identifier
- `goal`: User's original objective
- `status`: `IDLE | ACTIVE | PAUSED | COMPLETED | CANCELLED`
- `steps`: List of `PlanStep` objects
- `decisions`: Dict of `PlanDecision` objects
- `created_at`, `updated_at`: Timestamps

**PlanStep**:
- `step_id`: Unique step identifier
- `title`: Human-readable description
- `status`: `PENDING | IN_PROGRESS | BLOCKED | NEEDS_DECISION | COMPLETED`
- `mode`: Execution mode (`code`, `test`, `research`, `design`, `docs`, `run`)
- `owner`: Specialist type
- `notes`: Execution notes/results
- `depends_on`: List of step IDs this step depends on
- `decision_id`: Optional link to a decision that must be resolved first

**PlanDecision**:
- `decision_id`: Unique identifier
- `header`: Short label for UI
- `question`: Full question text
- `options`: List of `DecisionOption` (label + description)
- `multi_select`: Whether multiple options can be selected
- `selections`: User's choices
- `resolved`: Whether the decision has been made

### Dependency Validation

Before execution, `PlanState.validate_dependencies()` checks for:

1. **Missing references**: Dependencies pointing to non-existent steps
2. **Self-dependencies**: Steps depending on themselves
3. **Circular dependencies**: Using DFS cycle detection

Invalid dependencies are automatically removed with warnings logged.

### Execution Modes

The unified mode catalog (`vibe/core/modes.py`) provides consistent guidance:

| Mode | Specialist Title | Purpose |
|------|------------------|---------|
| `code` | Implementation Specialist | Writing/editing code with best practices |
| `test` | Validation Specialist | Creating tests, running suites, coverage analysis |
| `research` | Research Strategist | Gathering information, reading files, summarizing |
| `design` | Design Architect | Planning approaches, analyzing trade-offs |
| `docs` | Documentation Specialist | Updating READMEs, comments, docstrings |
| `run` | Execution Specialist | Running commands, capturing output, validation |

Each mode provides:
- `planner_guidance`: Instructions for the planner when creating steps
- `subagent_guidance`: Detailed instructions for subagent execution
- `icon`: Visual indicator for UI display

## SubAgent Execution

SubAgents are isolated execution environments for individual plan steps:

### Features

- **Identity & Tracking**: Each subagent has a unique `subagent_id` and `step_id`
- **Event Tagging**: All events include `subagent_id` and `subagent_step_id` for traceability
- **Tool Filtering**: Uses `ToolManager.filter_tools(allowed, denied)` per step
- **Token Budget**: Checked both before AND after LLM responses
- **Approval Handling**: Inherits parent agent's approval callback (sync and async)

### SubAgentConfig

```python
SubAgentConfig(
    step_id="step_1",
    title="Implement JWT validation",
    mode="code",
    allowed_tools=["read_file", "write_file", "search_replace"],
    denied_tools=["bash"],
    max_tokens=50000,
    max_turns=10,
    system_prompt_suffix="Focus on security best practices."
)
```

### SubAgentResult

```python
SubAgentResult(
    status="success",  # or "failure"
    output="Final assistant response...",
    error=None,        # Error message if failed
    stats={
        "prompt_tokens": 1234,
        "completion_tokens": 567,
        "tool_calls": 5,
        "tool_successes": 5,
        "tool_failures": 0,
        "duration_seconds": 12.5
    },
    artifacts=[]       # References to modified files
)
```

## Decision Checkpoints

The planner requests user decisions before:

1. **Destructive operations**: Rewriting large files, deleting content
2. **Strategy choices**: "Refactor vs rewrite", "Library A vs B"
3. **Risky commands**: Shell operations with side effects

### Workflow

1. Planner emits `PlanDecisionEvent` with question and options
2. Decision appears in chat as an interactive prompt
3. User responds via `/plan decide <id> <choice>` or UI button
4. Planner stores decision and unblocks waiting steps

### Multi-Select Decisions

For decisions allowing multiple selections:

```bash
# Single selection
/plan decide auth_method jwt

# Multiple selections (comma-separated)
/plan decide features "auth,logging,metrics"
```

## UI Integration

### Sidebar Panel

When planning is active, the sidebar shows:
- Plan title and overall status
- Collapsible step list with status badges
- Progress bar (`[███░░░░░░░░░░░░] 3/10`)
- Decision prompts with actionable buttons

### Status Bar Ticker

The bottom bar displays:
- Current goal (truncated)
- Step progress
- Pending decisions count
- Active subagent indicator
- Context/rate limit warnings

### Chat Stream

Planning events render as:
- `PlanStartedMessage`: Plan overview with steps
- `PlanStepMessage`: Step status updates
- `PlanDecisionPrompt`: Interactive decision UI
- `PlanCompletedMessage`: Final summary

## Persistence

Plan state persists to `~/.vibe/plans/<slug>.json`:

- `<slug>` is derived from the workspace path hash
- Payload includes `instance_id` and `PID` to avoid conflicts
- On app startup, active plans can be resumed automatically
- Corrupt JSON is ignored gracefully

## Configuration

Add to `~/.vibe/config.toml`:

```toml
[planner]
enabled = true                    # Enable planning commands
auto_run = false                  # Auto-start execution after plan generation
max_parallel_steps = 1            # Max concurrent step executions
decision_timeout_seconds = 0      # 0 = no timeout
max_context_tokens = 100000       # Token budget for planner context
enable_persistence = true         # Save plan state to disk
```

Or toggle auto-start at runtime:
```bash
/plan auto on
/plan auto off
```

## Events

Planning emits these events for UI and logging:

| Event | Payload |
|-------|---------|
| `PlanStartedEvent` | plan_id, goal, steps |
| `PlanStepUpdateEvent` | plan_id, step_id, old_status, new_status, notes |
| `PlanDecisionEvent` | plan_id, decision_id, question, options |
| `PlanCompletedEvent` | plan_id, final_status, summary |

## Execution Flow

```
/plan <goal>
    │
    ▼
PlannerAgent.start_plan(goal)
    ├─ LLM generates structured plan
    └─ validate_dependencies() (DFS cycle check)
    │
    ▼
Emit PlanStartedEvent
    │
    ▼
Start plan executor (background worker)
    │
    ▼
┌─────────────────────────────────────┐
│  Main Execution Loop                │
│  ─────────────────────────────────  │
│  1. get_parallel_runnable_steps()   │
│  2. Filter decision-blocked steps   │
│  3. For each executable step:       │
│     - Create SubAgentConfig         │
│     - SubAgent.execute(prompt)      │
│     - Update step status            │
│     - Emit PlanStepUpdateEvent      │
│  4. Loop until all steps complete   │
└─────────────────────────────────────┘
    │
    ▼
Emit PlanCompletedEvent
```

## Roadmap

### Implemented

- [x] PlannerAgent with goal decomposition
- [x] Dependency validation (circular, missing, self-refs)
- [x] SubAgent isolated execution
- [x] Mode-aware system prompts
- [x] Decision checkpoints with multi-select
- [x] Parallel step execution
- [x] Plan persistence infrastructure
- [x] All `/plan` commands
- [x] UI widgets (plan panel, ticker)
- [x] Event system for UI updates
- [x] Context budget management for planner (token tracking with warning thresholds)
- [x] Rate-limit handling with exponential backoff and jitter
- [x] Decision timeout mechanism with configurable timeout
- [x] Plan export to session logs (`export_for_logging()`)
- [x] Resource warning events (PlanResourceWarningEvent)
- [x] Plan completion events (PlanCompletedEvent)
- [x] Auto-compaction trigger when context budget is critical (95%+)
- [x] Specialized subagent types (CodeSpecialist, TestSpecialist, ResearchSpecialist, etc.)
- [x] Plan scheduling/queuing with priority and dependency support

### Planned

- [ ] UI integration for plan scheduler commands
- [ ] Multi-plan workflow templates
- [ ] Plan analytics and reporting

## Handling Subagent Failures

Subagents can fail for many reasons (tool crashes, invalid plans, rate limits). Vibe now keeps that failure context visible and ensures every plan finishes with a clear outcome summary.

- When a step transitions to `BLOCKED`, the chat feed shows a red failure banner with the reason plus any notes from the planner.
- The sidebar/panel keeps the step marked as blocked until you retry it or cancel the plan. Completed steps automatically clear their prior failure entries.
- At the end of each run (completed or cancelled) the app posts a summary card that lists the final status, progress, and any outstanding blocked steps, including failure details.
- Use `/plan retry <step_id>` to reset a blocked step back to `PENDING` and re-run the subagent. This works for steps that are currently blocked or were previously retried into a pending state.
- If multiple steps fail in parallel, each one is tracked independently so you can review them before deciding to retry or cancel.

## Troubleshooting

### Plan stuck on "Needs Decision"

Check `/plan status` for pending decisions and respond with `/plan decide <id> <choice>`.

### Steps not executing in parallel

Steps only run in parallel when they have no mutual dependencies. Check the `depends_on` field in `/plan status`.

### Context limit warnings

If you see context warnings, the planner may be hitting token limits. Consider:
1. Breaking the goal into smaller sub-goals
2. Increasing `max_context_tokens` in config
3. Using `/plan cancel` and starting fresh

### Plan not resuming after restart

Ensure the plan was saved (check `~/.vibe/plans/`). Plans from different Vibe instances won't be adopted to avoid conflicts.
