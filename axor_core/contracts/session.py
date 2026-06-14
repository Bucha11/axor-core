"""Closed-session audit contract — the Core → Sentinel seam.

axor-sentinel runs a background audit cycle that consumes a summary of each closed
``GovernedSession`` to update its reputation graph (which agent ran, what tools it
invoked, whether the session was tainted). This module defines the record core
emits and the sink it calls on session close.

Counterpart to ``contracts/reputation.py`` (the *Sentinel → Core* seam, where
sentinel injects resource reputation into ``NormalizedIntent``). Together they are
the two-way attachment between the engine and its observer.

Deliberately plain types — ``str`` / ``Sequence`` / ``Mapping``, never core enums —
for two reasons:
  * the consumer (sentinel) compares ``event_kinds`` / ``taint_sources`` as strings
    and must not need to import core enums; and
  * sentinel attaches *structurally* (it defines its own matching Protocol and duck
    types against this shape), so core stays free of any import edge into sentinel
    (invariant P-34). Keeping the field types as primitives is what lets the two
    independently-defined shapes line up at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ToolInvocationRecord:
    """One tool call resolved within a session.

    ``executed`` is False when the intent was denied (or otherwise never ran); the
    audit cycle uses tool + args to derive the touched resource and ``executed`` to
    grade an export attempt as adjacent (ran) vs failed (blocked).
    """
    tool: str
    args: Mapping[str, Any]
    executed: bool


@dataclass(frozen=True)
class SessionAuditRecord:
    """Immutable summary of a closed ``GovernedSession``, emitted to sentinel.

    Fields:
        session_id:       the closed session's id
        agent_id:         the agent identity that ran (``AgentDefinition.name``; "" if
                          none). Sentinel keys poisoning-mitigation on this when no
                          authenticated ``source_class`` is attested (F1).
        started_at:       unix timestamp the session was constructed
        taint_active:     session-wide taint shadow at close (any value tainted)
        taint_sources:    ``TaintSource`` *values* (strings) observed this session
        event_kinds:      ``TraceEventKind`` *values* (strings) observed this session
        tool_invocations: every resolved tool call, in order
        source_class:     authenticated actor class core attests, or "" when it
                          cannot. NEVER set from an attacker-influenceable taint
                          label — sentinel falls back to ``agent_id`` when empty.
    """
    session_id: str
    agent_id: str
    started_at: float
    taint_active: bool
    taint_sources: Sequence[str]
    event_kinds: Sequence[str]
    tool_invocations: Sequence[ToolInvocationRecord]
    source_class: str = ""


@runtime_checkable
class SessionSink(Protocol):
    """Consumes closed-session records (Core → Sentinel).

    Implemented sentinel-side (``CoreSessionSink``); core defines it here and fires
    it from ``GovernedSession.aclose()``. The implementation MUST NOT raise — a
    failing observer must never disturb the governance path; core swallows and logs.
    """

    async def on_session_closed(self, record: SessionAuditRecord) -> None:
        ...
