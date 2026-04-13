from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CommandClass(str, Enum):
    """
    How a slash command is handled.

    GOVERNANCE  — intercepted by node, answered from envelope/trace
                  executor never sees it
    CONTEXT     — routed to context subsystem
                  executor sees updated ContextView, not the command
    PASSTHROUGH — forwarded to executor
                  only if policy allows it, always logged in trace
    """
    GOVERNANCE  = "governance"
    CONTEXT     = "context"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class SlashCommand:
    """
    A slash command from the user, before routing.

    name:    e.g. "compact", "tools", "clear"
    args:    anything after the command name
    source:  where it came from (session, test, etc.)
    """
    name: str
    args: str
    source: str
    raw: str             # original raw string e.g. "/compact --aggressive"


@dataclass(frozen=True)
class CommandResult:
    """
    Result of handling a slash command.
    Returned to session layer for display or routing.
    """
    command: SlashCommand
    command_class: CommandClass
    output: Any          # string for display, or structured for context updates
    allowed: bool        # False if passthrough was denied by policy
    denial_reason: str | None = None
