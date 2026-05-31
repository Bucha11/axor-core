from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaintSource(str, Enum):
    """Origin of an external input that triggered a taint propagation."""
    WEB = "web"
    MCP = "mcp"
    FILE = "file"
    API = "api"
    CHILD_AGENT = "child_agent"
    MEMORY = "memory"
    PROVIDER_TOOL = "provider_tool"
    UNKNOWN_EXTERNAL = "unknown_external"


class TaintScope(str, Enum):
    """
    How widely taint propagates once triggered.

    INTENT       — affects only the current intent.
    NODE         — affects the current node for its lifetime.
    SUBTREE      — affects the node and all children spawned from it.
    SESSION      — affects the entire session (default for high-security).
    CROSS_SESSION — persists across sessions via Sentinel ReputationSnapshot;
                   widest possible scope. Used by axor-eval to measure
                   cross-session data-flow integrity (§7.1).
    """
    INTENT = "intent"
    NODE = "node"
    SUBTREE = "subtree"
    SESSION = "session"
    CROSS_SESSION = "cross_session"


@dataclass(frozen=True)
class ClearanceRecord:
    """Immutable record of a single taint clearance event."""
    clearance_id: str
    cleared_by: str
    authority_type: str
    timestamp: float
    reason_code: str
    authorized_by_principal_id: str
    audit_id: str = ""


@dataclass(frozen=True)
class TaintState:
    """
    Persistent taint state for a session or node.

    Taint is sticky by default — it does not decay on its own.
    Clearance requires explicit governance action recorded in clearance_history.

    sources           — set of TaintSource values that contributed to this state.
    scope             — widest scope across all propagations.
    sticky            — if True, taint persists until governance clears it.
    intent_age        — number of intents processed since taint was first set.
    wall_clock_age    — seconds since taint was first set (float epoch).
    parent_inherited  — True if taint was propagated from a parent node.
    clearance_history — ordered list of past clearance events.
    clearance_authority — principal that most recently cleared taint (or "").
    """
    sources: frozenset[TaintSource] = field(default_factory=frozenset)
    scope: TaintScope = TaintScope.SESSION
    sticky: bool = True
    intent_age: int = 0
    wall_clock_age: float = 0.0
    parent_inherited: bool = False
    clearance_history: tuple[ClearanceRecord, ...] = field(default_factory=tuple)
    clearance_authority: str = ""

    @property
    def is_tainted(self) -> bool:
        return bool(self.sources)
