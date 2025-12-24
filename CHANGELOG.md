# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Session Resume Prompt**: Users are now prompted to resume their previous session when starting vibe
- **Session Locking**: Prevent multiple simultaneous vibe sessions with atomic file locking
- **Unified Mode Catalog** (`vibe/core/modes.py`): Single source of truth for execution modes (code, test, research, design, docs, run) with guidance for both planner and subagents
- **ThinkingModeConfig**: Configurable thinking depth options (minimal, standard, deep) with auto-generated instructions
- **Plan Dependency Validation**: Circular dependency detection using DFS cycle detection in planner
- **SubAgent Event Tagging**: Events from subagents now include `subagent_id` and `subagent_step_id` for better traceability
- **Shared Status Icons** (`vibe/cli/textual_ui/widgets/status_icons.py`): Centralized status icons, progress bars, and status summaries for UI consistency
- **PlannerConfig** (`[planner]` section in config.toml): New configuration section for planning mode with options for `enabled`, `auto_run`, `max_parallel_steps`, `decision_timeout_seconds`, `max_context_tokens`, and `enable_persistence`
- **Architecture Documentation** (`docs/architecture.md`): New technical architecture guide covering core components, event system, configuration, and design decisions
- **Planner Resource Management** (`vibe/core/planner_resources.py`): New module for managing planner resources including:
  - **Token Budget Tracking**: Track cumulative token usage across all plan steps with configurable warning thresholds (80% warning, 95% critical)
  - **Rate-Limit Handling**: Exponential backoff with jitter for rate-limited API calls, configurable retry budget
  - **Decision Timeout**: Track pending decisions with configurable timeout and warning notifications
- **PlanResourceWarningEvent**: New event type for resource constraint notifications (budget, rate_limit, decision_timeout)
- **PlanCompletedEvent**: New event type with completion summary including steps completed and resource usage
- **Plan Export for Logging**: `PlannerAgent.export_for_logging()` method for session log integration with resource summary and progress stats
- **Auto-Compaction Trigger**: `CompactionRequest` and `check_compaction_needed()` for automatic context compaction when budget reaches critical levels (95%+)
- **Specialized Subagent Types** (`vibe/core/specialist_agents.py`): Pre-configured specialist agents with tailored tool access and instructions:
  - `CodeSpecialist`: Implementation with full tool access
  - `TestSpecialist`: Testing/validation focus
  - `ResearchSpecialist`: Read-only research mode
  - `DesignSpecialist`: Architecture planning
  - `DocsSpecialist`: Documentation updates
  - `RunSpecialist`: Command execution
- **Plan Scheduler** (`vibe/core/plan_scheduler.py`): Queue-based plan management with:
  - Priority-based scheduling (higher priority plans execute first)
  - Plan dependencies (plans can depend on completion of other plans)
  - Pause/resume support
  - Concurrent execution limit
  - Statistics and serialization

### Changed

- **SubAgent Token Budget**: Now checks budget both before AND after LLM responses for more accurate budget enforcement
- **ToolManager Filtering**: Added `filter_tools()` method for cleaner tool filtering in subagents
- **Approval Callback Handling**: SubAgent now uses `asyncio.iscoroutinefunction()` for reliable callback type detection
- **Exception Handling**: Replaced bare `except Exception` catches with specific exception types across core modules
- **Planning Mode Documentation** (`docs/planning_mode.md`): Converted from proposal to implementation documentation with complete API reference, troubleshooting guide, and roadmap

### Fixed

- **Session Lock Race Condition**: Fixed TOCTOU vulnerability using `fcntl.flock()` for atomic locking
- **Circular Dependencies**: Planner now validates and auto-removes invalid dependencies before execution

## [1.1.3] - 2025-12-12

### Added

- Add more copy_to_clipboard methods to support all cases
- Add bindings to scroll chat history

### Changed

- Relax config to accept extra inputs
- Remove useless stats from assistant events
- Improve scroll actions while streaming
- Do not check for updates more than once a day
- Use PyPI in update notifier

### Fixed

- Fix tool permission handling for "allow always" option in ACP
- Fix security issue: prevent command injection in GitHub Action prompt handling
- Fix issues with vLLM

## [1.1.2] - 2025-12-11

### Changed

- add `terminal-auth` auth method to ACP agent only if the client supports it
- fix `user-agent` header when using Mistral backend, using SDK hook

## [1.1.1] - 2025-12-10

### Changed

- added `include_commit_signature` in `config.toml` to disable signing commits

## [1.1.0] - 2025-12-10

### Fixed

- fixed crash in some rare instances when copy-pasting

### Changed

- improved context length from 100k to 200k

## [1.0.6] - 2025-12-10

### Fixed

- add missing steps in bump_version script
- move `pytest-xdist` to dev dependencies
- take into account config for bash timeout

### Changed

- improve textual performance
- improve README:
  - improve windows installation instructions
  - update default system prompt reference
  - document MCP tool permission configuration

## [1.0.5] - 2025-12-10

### Fixed

- Fix streaming with OpenAI adapter

## [1.0.4] - 2025-12-09

### Changed

- Rename agent in distribution/zed/extension.toml to mistral-vibe

### Fixed

- Fix icon and description in distribution/zed/extension.toml

### Removed

- Remove .envrc file

## [1.0.3] - 2025-12-09

### Added

- Add LICENCE symlink in distribution/zed for compatibility with zed extension release process

## [1.0.2] - 2025-12-09

### Fixed

- Fix setup flow for vibe-acp builds

## [1.0.1] - 2025-12-09

### Fixed

- Fix update notification

## [1.0.0] - 2025-12-09

### Added

- Initial release
