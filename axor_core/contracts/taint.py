from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Carrier(str, Enum):
    """Can the *form* of a value carry an instruction? Ordered from safest to
    most dangerous: ENDORSED < CLOSED_SCHEMA < FREE_TEXT, with FREE_TEXT the
    fail-closed top of the order.

    Classification must be deterministic and structural, never decided by a
    model. A model classifier would let the result be steered by the governed
    content's own semantics, which is exactly what this guards against.
    """
    ENDORSED = "endorsed"          # structurally guaranteed instruction-free
    CLOSED_SCHEMA = "closed_schema"  # parses fully into a closed, verified schema
    FREE_TEXT = "free_text"        # may carry an instruction; treated as unsafe


# Carrier order from safest to most dangerous (index = height; FREE_TEXT is the top).
_CARRIER_ORDER = (Carrier.ENDORSED, Carrier.CLOSED_SCHEMA, Carrier.FREE_TEXT)


def carrier_join(a: Carrier, b: Carrier) -> Carrier:
    """Combine two carriers into the more imperative / less safe of the two."""
    return _CARRIER_ORDER[max(_CARRIER_ORDER.index(a), _CARRIER_ORDER.index(b))]


class TrustedOrigin(str, Enum):
    """Where a value the attacker cannot author came from.

    Used by the context-default integrity mode: once a node's context holds
    untrusted data, a model-generated value is clean only if it provably equals a
    value of one of these origins (docs/rfc-integrity-context-default.md).
    """
    TASK = "task"            # the user's task for this node
    OPERATOR = "operator"    # operator configuration (enum allowlist members)
    TOOL = "tool"            # output of a trusted tool
    ENDORSED = "endorsed"    # released by governance endorsement


# The two integrity defaults for a model-generated value that carries no
# registered untrusted fragment. "clean": trusted (legacy). "context": carries the
# node's context root unless it has a TrustedOrigin.
INTEGRITY_DEFAULTS = frozenset({"clean", "context"})


def resolve_integrity_default(requested: "str | None", *, strict: bool) -> str:
    """The integrity default a governed session or governor runs with.

    ``None`` means "the mode's default": ``"context"`` under STRICT, ``"clean"``
    otherwise. An explicit value wins. Choosing ``"clean"`` under STRICT is the
    legacy opt-out and is logged: it restores the gap where a re-encoded
    untrusted identifier passes the integrity gate on every sink without an enum
    allowlist (docs/rfc-integrity-context-default.md).
    """
    if requested is None:
        return "context" if strict else "clean"
    if requested not in INTEGRITY_DEFAULTS:
        raise ValueError(
            f"unknown integrity_default {requested!r}; expected one of "
            f"{sorted(INTEGRITY_DEFAULTS)}"
        )
    if strict and requested == "clean":
        import logging

        logging.getLogger("axor.taint").warning(
            "integrity_default='clean' under STRICT: model-generated values that "
            "carry no registered untrusted fragment are trusted, so a re-encoded "
            "attacker identifier passes the integrity gate on every sink without "
            "an enum allowlist. This is the legacy opt-out; STRICT defaults to "
            "'context'."
        )
    return requested


# Which TrustedOrigins may clear a value at an integrity sink.
# "request-only": only values the request itself fixed (plus operator config,
#   which the attacker cannot author either and is not a read).
# "any-trusted": any registered trusted origin, including a trusted tool's
#   output — that is, a value that legitimately arrived through a read.
INTEGRITY_ORIGIN_MODES = frozenset({"request-only", "any-trusted"})


def resolve_integrity_origins(requested: "str | None", *, strict: bool) -> str:
    """The integrity-origin mode a governed session runs with.

    ``None`` means the mode's default. NOTE: the default is deliberately
    ``"any-trusted"`` in both modes for now, so that turning this knob on does
    not silently change results measured before it existed. Flipping STRICT to
    ``"request-only"`` is a separate, explicit decision.
    """
    if requested is None:
        return "any-trusted"
    if requested not in INTEGRITY_ORIGIN_MODES:
        raise ValueError(
            f"integrity_origins must be one of {sorted(INTEGRITY_ORIGIN_MODES)}, "
            f"got {requested!r}"
        )
    return requested


def integrity_origins_for(mode: str) -> "frozenset[TrustedOrigin]":
    """The TrustedOrigins that clear a value at an integrity sink under ``mode``.

    ``request-only`` admits TASK (the request) and OPERATOR (operator config —
    not attacker-authorable and not a read); it deliberately excludes TOOL (a
    value that arrived through a read) and ENDORSED (a governance release, which
    must not relax the strict mode). ``any-trusted`` admits every origin.
    """
    if mode == "request-only":
        return frozenset({TrustedOrigin.TASK, TrustedOrigin.OPERATOR})
    return frozenset(TrustedOrigin)


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
    CROSS_SESSION — persists across sessions via the reputation snapshot;
                   widest possible scope. Used to measure cross-session
                   data-flow integrity.
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
    # Confidentiality label, independent of integrity. Integrity is implicit in
    # `sources` (any source means untrusted); `sensitive` is the separate
    # confidentiality axis — True if a sensitive source (e.g. a secret read)
    # contributed, i.e. it is harmful for this to leave. Drives the egress gate.
    sensitive: bool = False

    @property
    def is_tainted(self) -> bool:
        return bool(self.sources)
