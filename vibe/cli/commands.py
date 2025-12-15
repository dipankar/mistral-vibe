from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Command:
    aliases: frozenset[str]
    description: str
    handler: str
    exits: bool = False
    accepts_argument: bool = False


class CommandRegistry:
    def __init__(self, excluded_commands: list[str] | None = None) -> None:
        if excluded_commands is None:
            excluded_commands = []
        self.commands = {
            "help": Command(
                aliases=frozenset(["/help", "/h"]),
                description="Show help message",
                handler="_show_help",
            ),
            "status": Command(
                aliases=frozenset(["/status", "/stats"]),
                description="Display agent statistics",
                handler="_show_status",
            ),
            "config": Command(
                aliases=frozenset(["/config", "/cfg", "/theme", "/model"]),
                description="Edit config settings",
                handler="_show_config",
            ),
            "reload": Command(
                aliases=frozenset(["/reload", "/r"]),
                description="Reload configuration from disk",
                handler="_reload_config",
            ),
            "clear": Command(
                aliases=frozenset(["/clear", "/reset"]),
                description="Clear conversation history",
                handler="_clear_history",
            ),
            "log": Command(
                aliases=frozenset(["/log", "/logpath"]),
                description="Show path to current interaction log file",
                handler="_show_log_path",
            ),
            "compact": Command(
                aliases=frozenset(["/compact", "/summarize"]),
                description="Compact conversation history by summarizing",
                handler="_compact_history",
            ),
            "memory": Command(
                aliases=frozenset(["/memory", "/mem"]),
                description="Manage session memory (`/memory`, `/memory clear`)",
                handler="_memory_command",
                accepts_argument=True,
            ),
            "exit": Command(
                aliases=frozenset(["/exit", "/quit", "/q"]),
                description="Exit the application",
                handler="_exit_app",
                exits=True,
            ),
            "plan": Command(
                aliases=frozenset(["/plan"]),
                description="Start a planning session for a goal",
                handler="_start_planning",
                accepts_argument=True,
            ),
            "plan_status": Command(
                aliases=frozenset(["/plan status", "/plan-status"]),
                description="Show the current planning status",
                handler="_show_plan_status",
            ),
            "plan_pause": Command(
                aliases=frozenset(["/plan pause", "/plan-pause"]),
                description="Pause the active planning session",
                handler="_pause_plan",
            ),
            "plan_resume": Command(
                aliases=frozenset(["/plan resume", "/plan-resume"]),
                description="Resume a paused planning session",
                handler="_resume_plan",
            ),
            "plan_cancel": Command(
                aliases=frozenset(["/plan cancel", "/plan-cancel"]),
                description="Cancel the active planning session",
                handler="_cancel_plan",
            ),
            "plan_decide": Command(
                aliases=frozenset(["/plan decide", "/plan-decide"]),
                description="Respond to a planner decision prompt",
                handler="_handle_plan_decision",
                accepts_argument=True,
            ),
        }

        for command in excluded_commands:
            self.commands.pop(command, None)

        self._alias_map = {}
        for cmd_name, cmd in self.commands.items():
            for alias in cmd.aliases:
                self._alias_map[alias.lower()] = cmd_name
        self._sorted_aliases = sorted(self._alias_map.keys(), key=len, reverse=True)

    def find_command(self, user_input: str) -> tuple[Command, str | None] | None:
        stripped = user_input.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        for alias in self._sorted_aliases:
            if lowered == alias:
                cmd_name = self._alias_map[alias]
                return self.commands[cmd_name], None
            alias_prefix = alias + " "
            if lowered.startswith(alias_prefix):
                cmd_name = self._alias_map[alias]
                argument = stripped[len(alias) :].strip()
                return self.commands[cmd_name], argument or None
        return None

    def get_help_text(self) -> str:
        lines: list[str] = [
            "### Keyboard Shortcuts",
            "",
            "- `Enter` Submit message",
            "- `Ctrl+J` / `Shift+Enter` Insert newline",
            "- `Escape` Interrupt agent or close dialogs",
            "- `Ctrl+C` Quit (or clear input if text present)",
            "- `Ctrl+Shift+C` Copy selected text",
            "- `Ctrl+O` Toggle tool output view",
            "- `Ctrl+T` Toggle todo view",
            "- `Shift+Tab` Toggle auto-approve mode",
            "",
            "### Special Features",
            "",
            "- `!<command>` Execute bash command directly",
            "- `@path/to/file/` Autocompletes file paths",
            "",
            "### Commands",
            "",
        ]

        for cmd in self.commands.values():
            aliases = ", ".join(f"`{alias}`" for alias in sorted(cmd.aliases))
            lines.append(f"- {aliases}: {cmd.description}")
        return "\n".join(lines)
