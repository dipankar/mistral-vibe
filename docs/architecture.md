# Mistral Vibe Architecture

This document provides a technical overview of Mistral Vibe's internal architecture.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                      │
│  vibe/cli/                                                                  │
│  ├── textual_ui/app.py      Main Textual application                       │
│  ├── commands.py            Command registry (/plan, /help, etc.)          │
│  └── history_manager.py     Conversation history                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Core Layer                                     │
│  vibe/core/                                                                 │
│  ├── agent.py               Main Agent orchestration                        │
│  ├── planner.py             PlannerAgent for goal decomposition             │
│  ├── subagent.py            Isolated step execution                         │
│  ├── modes.py               Execution mode catalog                          │
│  ├── memory.py              Session memory management                       │
│  ├── config.py              Configuration system                            │
│  └── types.py               Core type definitions & events                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLM Layer                                      │
│  vibe/core/llm/                                                             │
│  ├── backend/mistral.py     Mistral API integration                        │
│  ├── backend/generic.py     OpenAI-compatible backends                     │
│  └── backend/factory.py     Backend selection                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Tools Layer                                    │
│  vibe/core/tools/                                                           │
│  ├── manager.py             Tool lifecycle & filtering                      │
│  ├── base.py                Base tool classes                               │
│  └── builtins/              Built-in tools (bash, read_file, etc.)          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Agent (`vibe/core/agent.py`)

The main orchestrator stitches together smaller collaborators:
- **ConversationState** – encapsulates history/memory management and message observers.
- **MiddlewareRunner** – enforces guards (max turns, max price) and emits compaction triggers.
- **ToolInvocationManager** – owns approval prompts, allow/deny checks, and event fan-out for every tool execution.
- Conversation loop with the active LLM backend and streaming handler.
- Event emission for UI updates.

```python
class Agent:
    def __init__(self, config: VibeConfig, ...):
        self.config = config
        self.tool_manager = ToolManager(...)
        self.backend = create_backend(...)
        self.middlewares = [...]

    async def run(self, user_message: str) -> AsyncIterator[BaseEvent]:
        # Main conversation loop
        # Yields events for UI rendering
```

### PlannerAgent (`vibe/core/planner.py`)

Orchestrates complex multi-step tasks:

```python
class PlannerAgent:
    """Decomposes goals into structured plans with steps and decisions."""

    def start_plan(self, goal: str) -> PlanState:
        """Generate a structured plan from a goal."""

    def get_runnable_steps(self) -> list[PlanStep]:
        """Get steps ready for execution (dependencies met)."""

    def update_step_status(self, step_id: str, status: PlanStepStatus):
        """Update step status and emit events."""

    def decide(self, decision_id: str, selections: list[str]):
        """Record user decision and unblock waiting steps."""
```

**Key Data Structures:**

```python
@dataclass
class PlanState:
    plan_id: str
    goal: str
    status: PlanRunStatus  # IDLE, ACTIVE, PAUSED, COMPLETED, CANCELLED
    steps: list[PlanStep]
    decisions: dict[str, PlanDecision]

@dataclass
class PlanStep:
    step_id: str
    title: str
    status: PlanStepStatus  # PENDING, IN_PROGRESS, BLOCKED, NEEDS_DECISION, COMPLETED
    mode: str  # code, test, research, design, docs, run
    depends_on: list[str]
    decision_id: str | None
```

### SubAgent (`vibe/core/subagent.py`)

Isolated execution environment for individual plan steps:

```python
class SubAgent:
    """Executes a single plan step in isolation."""

    def __init__(self, config: SubAgentConfig, ...):
        self.subagent_id = str(uuid.uuid4())
        self.config = config

    async def execute(self, prompt: str) -> SubAgentResult:
        """Run the subagent with the given prompt."""
        # Creates filtered tool manager
        # Runs isolated conversation
        # Tags all events with subagent_id
        # Enforces token budget

@dataclass
class SubAgentConfig:
    step_id: str
    title: str
    mode: str = "code"
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    max_tokens: int = 50000
    max_turns: int = 10
```

### Mode Catalog (`vibe/core/modes.py`)

Unified source of truth for execution modes:

```python
class ExecutionMode(StrEnum):
    CODE = "code"
    TEST = "test"
    RESEARCH = "research"
    DESIGN = "design"
    DOCS = "docs"
    RUN = "run"

@dataclass(frozen=True)
class ModeGuidance:
    title: str              # "Implementation Specialist"
    planner_guidance: str   # Instructions for plan step creation
    subagent_guidance: str  # Instructions for subagent execution
    icon: str               # UI icon

MODE_CATALOG: dict[ExecutionMode, ModeGuidance] = {...}
```

### ToolManager (`vibe/core/tools/manager.py`)

Manages tool lifecycle and access control:

```python
class ToolManager:
    def __init__(self, config: VibeConfig, ...):
        self._available: dict[str, type[BaseTool]] = {}
        self._instances: dict[str, BaseTool] = {}

    def discover_tools(self) -> None:
        """Discover built-in and custom tools."""

    def filter_tools(
        self,
        allowed: list[str] | None = None,
        denied: list[str] | None = None
    ) -> None:
        """Filter available tools by allow/deny lists."""

    async def execute_tool(self, name: str, args: dict) -> ToolResult:
        """Execute a tool with permission checking."""
```

## Event System

All components emit events for UI updates and logging:

```python
class BaseEvent(BaseModel, ABC):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    subagent_id: str | None = None      # Set when from subagent
    subagent_step_id: str | None = None

# Agent events
class AssistantEvent(BaseEvent): ...
class ToolCallEvent(BaseEvent): ...
class ToolResultEvent(BaseEvent): ...

# Planning events
class PlanStartedEvent(PlanEvent): ...
class PlanStepUpdateEvent(PlanEvent): ...
class PlanDecisionEvent(PlanEvent): ...
class PlanCompletedEvent(PlanEvent): ...

# Memory events
class MemoryEntryEvent(BaseEvent): ...
class CompactStartEvent(BaseEvent): ...
class CompactEndEvent(BaseEvent): ...
```

Each event carries a UUID-backed `event_id`, allowing the Textual UI to deliver local events immediately while still ignoring the matching payloads when they loop back over the IPC bus.

## Configuration System

### VibeConfig (`vibe/core/config.py`)

Central configuration with multiple sources:

```python
class VibeConfig(BaseSettings):
    # Model settings
    active_model: str = "devstral-2"
    models: list[ModelConfig]
    providers: list[ProviderConfig]

    # Context management
    auto_compact_threshold: int = 200_000
    memory_soft_limit_ratio: float = 0.95

    # Tool configuration
    tools: dict[str, BaseToolConfig]
    enabled_tools: list[str]
    disabled_tools: list[str]

    # Planner settings
    planner: PlannerConfig

    # MCP servers
    mcp_servers: list[MCPServer]
```

**Configuration Sources (priority order):**
1. Constructor arguments
2. Environment variables (`VIBE_*`)
3. TOML file (`~/.vibe/config.toml`)
4. Defaults

### PlannerConfig

```python
class PlannerConfig(BaseSettings):
    enabled: bool = True
    auto_run: bool = False
    max_parallel_steps: int = 1
    decision_timeout_seconds: int = 0
    max_context_tokens: int = 100000
    enable_persistence: bool = True
```

## LLM Backend System

### Backend Factory

```python
def create_backend(config: VibeConfig) -> LLMBackend:
    provider = config.get_provider_for_model(config.get_active_model())
    match provider.backend:
        case Backend.MISTRAL:
            return MistralBackend(config)
        case Backend.GENERIC:
            return GenericBackend(config)
```

### Backend Interface

```python
class LLMBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages to LLM and stream response."""
```

## UI Layer

### Textual Application (`vibe/cli/textual_ui/app.py`)

Main TUI application with:
- Chat display widget
- Sidebar (todo list, plan panel)
- Input area with autocompletion
- Status bar with planner ticker

### Widgets

```
vibe/cli/textual_ui/widgets/
├── chat_display.py      # Message rendering
├── input_area.py        # User input with completion
├── sidebar.py           # Collapsible sidebar
├── plan_panel.py        # Plan visualization
├── planner_ticker.py    # Status bar metrics
├── todo_panel.py        # Todo list display
└── status_icons.py      # Shared status icons
```

### Planner-Oriented UI Flow

- **State Store (`vibe/cli/textual_ui/state_store.py`)** – `UIStateStore` is the single source of truth for planner and bottom-panel state. Controllers update the store, while view-specific presenters subscribe to keys (bottom panel mode, collapsible sections, confirmation prompts) and reactively update widgets, eliminating scattered boolean flags.
- **Chat Input Presenter (`vibe/cli/textual_ui/presenters/chat_input_presenter.py`)** – Listens for planner confirmation changes and toggles the prompt banner inside `ChatInputContainer` so VibeApp no longer handles those transitions directly.
- **Bottom Panel Manager (`vibe/cli/textual_ui/bottom_panel_manager.py`)** – Owns the bottom panel widget stack (chat input, config UI, approval UI) and reacts to `UIStateStore` changes, so VibeApp asks it to “show config” or “return to input” instead of manually mounting/removing widgets.
- **App Controller (`vibe/cli/textual_ui/app_controller.py`)** – Mediates user messages, planner confirmations, and agent initialization. VibeApp forwards chat submissions to the controller, which decides whether to auto-plan, prompt, or run the agent.
- **Command Controller (`vibe/cli/textual_ui/command_controller.py`)** – Encapsulates the full slash-command surface (help, planner lifecycle/decisions, status/config/log/memory/compact, retries, exit) so the Textual app forwards intent handling to a single place rather than mixing orchestration into widget code.
- **Planner Controller (`vibe/cli/textual_ui/planner_controller.py`)** – Owns the `PlannerAgent`, handles plan persistence, orchestrates subagent execution, and feeds UI callbacks. The app never instantiates `PlannerAgent` directly; all plan lifecycle changes flow through the controller so they are centrally observable.
- **IPC Event Bus (`vibe/ipc/event_bus.py`)** – A required pynng-backed pub/sub bus deduplicates event fan-out. Every `EventDispatcher` publishes events to the bus, and the Textual UI subscribes to the same address, even when API-driven agents run out-of-process. The UI also streams events directly from the dispatcher and tags each payload with `event_id` so it can ignore the duplicate bus echo. If pynng cannot be initialized the CLI refuses to start, ensuring all surfaces observe the same IPC contract.
- **Headless Surfaces** – Programmatic runs (`vibe/core/programmatic.py`) and ACP sessions (`vibe/acp/acp_agent.py`) also publish every `BaseEvent` onto the same bus, so external dashboards or alternate UIs can subscribe without embedding Textual.

This separation keeps planning UX extensible: new features target the store/controller/bus seams without reaching into widget internals.

### Command Registry (`vibe/cli/commands.py`)

```python
class CommandRegistry:
    def register(self, name: str, handler: CommandHandler):
        """Register a slash command."""

    def execute(self, command: str) -> CommandResult:
        """Execute a registered command."""

# Built-in commands
/help           # Show help
/plan <goal>    # Start planning
/plan status    # Show plan status
/config         # Show configuration
/clear          # Clear chat
/compact        # Trigger context compaction
```

## Session Management

### Session Locking

```python
class SessionLock:
    """Prevents multiple Vibe instances in same workspace."""

    def acquire(self) -> bool:
        """Acquire lock using fcntl.flock()."""

    def release(self) -> None:
        """Release the lock."""
```

### Session Persistence

- Sessions saved to `~/.vibe/logs/`
- Plan state saved to `~/.vibe/plans/`
- Resume with `vibe -c` or `vibe --resume <id>`

## Middleware Chain

```python
class Agent:
    middlewares = [
        AutoCompactMiddleware(),     # Auto-compact when context too large
        ContextWarningMiddleware(),  # Warn when approaching limits
        PriceLimitMiddleware(),      # Track spending
        TurnLimitMiddleware(),       # Limit conversation turns
    ]
```

## File Structure

```
vibe/
├── __init__.py
├── __main__.py           # Entry point
├── cli/
│   ├── main.py           # CLI argument parsing
│   ├── commands.py       # Command registry
│   ├── history_manager.py
│   ├── clipboard.py
│   ├── autocompletion/
│   │   ├── completers.py
│   │   └── provider.py
│   └── textual_ui/
│       ├── app.py        # Main Textual app
│       ├── handlers/     # Event handlers
│       └── widgets/      # UI components
├── core/
│   ├── agent.py          # Main agent
│   ├── planner.py        # Planning orchestration
│   ├── subagent.py       # Step execution
│   ├── modes.py          # Mode catalog
│   ├── memory.py         # Session memory
│   ├── config.py         # Configuration
│   ├── types.py          # Type definitions
│   ├── middleware.py     # Middleware chain
│   ├── system_prompt.py  # Prompt building
│   ├── interaction_logger.py
│   ├── llm/
│   │   ├── backend/
│   │   │   ├── factory.py
│   │   │   ├── mistral.py
│   │   │   └── generic.py
│   │   ├── format.py
│   │   ├── types.py
│   │   └── exceptions.py
│   └── tools/
│       ├── manager.py
│       ├── base.py
│       └── builtins/
│           ├── bash.py
│           ├── read_file.py
│           ├── write_file.py
│           ├── search_replace.py
│           ├── grep.py
│           └── todo.py
└── acp/                  # Agent Communication Protocol
    └── acp_agent.py
```

## Key Design Decisions

### 1. Event-Driven Architecture
All components emit events rather than directly updating UI. This enables:
- Decoupled components
- Easy logging and debugging
- Flexible UI rendering

### 2. Isolated SubAgents
SubAgents run in isolation with:
- Separate message history
- Filtered tool access
- Token budgets
- Tagged events for traceability

### 3. Mode-Aware Execution
The mode catalog provides consistent guidance:
- Planner uses mode to describe steps
- SubAgents use mode for execution behavior
- UI uses mode for visual indicators

### 4. Dependency Validation
Plans validate dependencies using DFS:
- Detects circular references
- Removes invalid dependencies
- Enables parallel execution of independent steps

### 5. Configuration Precedence
Settings are layered for flexibility:
- CLI args > Environment > TOML > Defaults
- Agent configs can override base settings
- Tool permissions are per-tool configurable
