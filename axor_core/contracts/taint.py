from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Carrier(str, Enum):
    """Imperative-channel lattice (TM1): can the *form* of a value carry an
    instruction?  ENDORSED ⊏ CLOSED_SCHEMA ⊏ FREE_TEXT, ⊤ = FREE_TEXT (fail-closed).

    Classification must be deterministic/structural, never a model (T0) — see
    security/carrier.py. A model classifier would let the projection be steered
    by the governed content's semantics (breaks K4/T0).
    """
    ENDORSED = "endorsed"          # structurally guaranteed instruction-free
    CLOSED_SCHEMA = "closed_schema"  # parses fully into a closed, verified schema
    FREE_TEXT = "free_text"        # ⊤ — may carry an instruction; fail-closed


# Carrier order for lattice comparisons (index = height; FREE_TEXT is ⊤).
_CARRIER_ORDER = (Carrier.ENDORSED, Carrier.CLOSED_SCHEMA, Carrier.FREE_TEXT)


def carrier_join(a: Carrier, b: Carrier) -> Carrier:
    """Least upper bound (the more imperative / less safe of the two)."""
    return _CARRIER_ORDER[max(_CARRIER_ORDER.index(a), _CARRIER_ORDER.index(b))]


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
    # Confidentiality label (TM2 dual labels). Integrity is implicit in `sources`
    # (any source ⇒ untrusted); `sensitive` is the independent confidentiality
    # axis — True if a sensitive source (e.g. a secret read) contributed, i.e. it
    # is harmful for this to leave. Drives the confidentiality/egress gate.
    sensitive: bool = False

    @property
    def is_tainted(self) -> bool:
        return bool(self.sources)
