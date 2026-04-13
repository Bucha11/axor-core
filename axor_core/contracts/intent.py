from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentKind(str, Enum):
    """
    All privileged actions an executor may request.

    Executors never perform these directly.
    They surface as intents — the node decides what happens.
    """
    TOOL_CALL      = "tool_call"       # call an external tool
    SPAWN_CHILD    = "spawn_child"     # create a child GovernedNode
    EXPORT         = "export"          # export intermediate result
    EXPAND_CONTEXT = "expand_context"  # request more context visibility


@dataclass(frozen=True)
class Intent:
    """
    A privileged action request from an executor.

    The executor surfaces this (e.g. as a tool_use block in Claude SDK).
    The node intercepts it before execution and resolves it against policy.

    payload contents depend on kind:
    - tool_call:      {"tool": str, "args": dict}
    - spawn_child:    {"task": str, "context_hint": str}
    - export:         {"content": str, "mode": str}
    - expand_context: {"reason": str, "scope": str}
    """
    kind: IntentKind
    payload: dict[str, Any]
    node_id: str
    sequence: int = 0   # position in the intent stream for this node


@dataclass(frozen=True)
class ResolvedIntent:
    """
    An Intent after node resolution.
    Carries the original intent plus the decision and result.
    """
    intent: Intent
    approved: bool
    reason: str
    result: Any = None                    # tool result if approved
    transformed_payload: dict | None = None  # if decision was TRANSFORM
