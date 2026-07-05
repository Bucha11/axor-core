"""Trace bridge — one trace, two consumers (spec 12.0 point 4).

The plane subscribes to the SAME event feed Eval records; there is no second
instrumentation path. This module translates axor-core :class:`TraceEvent`s
into kernel-schema :class:`~axor_core.kernel.events.Event`s so the runtime's
recorded trace flows straight to the plane service (telemetry) and the replay
fold — no per-consumer schema.

The mapping is deliberately lossy in one direction only: a kernel Event keeps
the governance-load-bearing columns (kind, gate/category, verdict, causal_root,
severity for facts) and drops adapter-cosmetic detail (token counts, cache
stats, routing) into ``payload`` untouched. Anything the fold or the UI reasons
over is a first-class column; everything else rides along.
"""
from __future__ import annotations

from dataclasses import asdict

from axor_core.contracts.trace import (
    CancelledEvent,
    DegradationTransitionEvent,
    IntentDeniedEvent,
    SourceQuarantinedEvent,
    SuspiciousIntentEvent,
    TaintPropagatedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.kernel.events import Event, EventKind, Verdict

# axor-core DegradationLevel name -> kernel Fact severity (== the recompute's
# level index). Denials become facts that drive level = max(severity(uncovered)).
_LEVEL_SEVERITY = {
    "NORMAL": 0, "CAUTIOUS": 1, "RESTRICTED": 2, "LOCKED": 3, "TERMINAL": 4,
}


def _level_name(level: object) -> str:
    """DegradationLevel enum or str -> canonical NORMAL..TERMINAL name."""
    name = getattr(level, "name", None)
    return name if name is not None else str(level or "NORMAL")


def _ts(index: int) -> str:
    # Trace ordering is by seq; ts is informational. The kernel is clock-free,
    # so a synthetic monotone stamp keeps replay bit-reproducible.
    return f"seq:{index}"


def trace_event_to_kernel(event: TraceEvent, node_id: str | None = None) -> Event | None:
    """Translate one TraceEvent. Returns None for events with no governance
    meaning on the plane (they stay in the full local trace)."""
    nid = node_id or event.node_id
    seq = event.sequence

    if isinstance(event, IntentDeniedEvent):
        return Event(
            seq=seq, node_id=nid, kind=EventKind.DENIAL, ts=_ts(seq),
            gate=_category_of(event.reason), verdict=Verdict.DENY,
            payload={"reason": event.reason, "intent_kind": event.intent_kind},
        )
    if isinstance(event, DegradationTransitionEvent):
        d = asdict(event)
        new_level = _level_name(d.get("new_level"))
        # A transition upward is a fact for the recompute; keyed on the trigger.
        return Event(
            seq=seq, node_id=nid, kind=EventKind.FACT, ts=_ts(seq),
            causal_root=d.get("trigger_source_id"),
            payload={
                "fact_id": f"deg_{seq}",
                "fact_type": "degradation_transition",
                "severity": _LEVEL_SEVERITY.get(new_level, 0),
                "reason": d.get("reason", ""),
                "new_level": new_level,
                "previous_level": _level_name(d.get("previous_level")),
            },
        )
    if isinstance(event, SourceQuarantinedEvent):
        d = asdict(event)
        return Event(
            seq=seq, node_id=nid, kind=EventKind.FACT, ts=_ts(seq),
            causal_root=d.get("source_id"),
            payload={
                "fact_id": f"quar_{seq}",
                "fact_type": "source_quarantined",
                "severity": _LEVEL_SEVERITY["RESTRICTED"],
                "reason": d.get("reason", ""),
            },
        )
    if isinstance(event, TaintPropagatedEvent):
        return Event(
            seq=seq, node_id=nid, kind=EventKind.TOOL_RESULT, ts=_ts(seq),
            causal_root=asdict(event).get("source_id"),
            payload=dict(event.payload),
        )
    if isinstance(event, SuspiciousIntentEvent):
        denied = event.policy_action == "denied"
        return Event(
            seq=seq, node_id=nid,
            kind=EventKind.DENIAL if denied else EventKind.GATE_EVAL, ts=_ts(seq),
            gate="anomaly", verdict=Verdict.DENY if denied else Verdict.PASS,
            payload={"tool": event.tool, "score": event.score,
                     "reasons": list(event.reasons)},
        )
    if isinstance(event, CancelledEvent):
        return Event(
            seq=seq, node_id=nid, kind=EventKind.OPERATOR_INTERVENTION, ts=_ts(seq),
            payload={"reason": event.reason, "detail": event.detail},
        ) if "control plane" in event.detail else None

    # Approvals and cosmetic adapter events (tokens/cache/routing) are not
    # replayed as governance steps; they stay in the local trace only.
    if event.kind is TraceEventKind.INTENT_APPROVED:
        return Event(
            seq=seq, node_id=nid, kind=EventKind.TOOL_CALL, ts=_ts(seq),
            verdict=Verdict.PASS, payload=dict(event.payload),
        )
    return None


def _category_of(reason: str) -> str:
    """Coarse gate category from a denial reason (mirrors the runtime's own
    _classify_denial, kept in sync with the gate category strings)."""
    lowered = reason.lower()
    for needle, cat in (
        ("taint", "taint_enforcement"), ("carrier", "carrier_gate"),
        ("consequence", "consequence_gate"), ("ssrf", "ssrf_gate"),
        ("value policy", "value_policy"), ("capability", "capability"),
        ("budget", "budget"), ("degradation", "degradation"),
        ("unclassified", "unclassified_tool"), ("positional", "positional_gate"),
    ):
        if needle in lowered:
            return cat
    return "denial"


def trace_to_kernel(events: list[TraceEvent], node_id: str | None = None) -> list[Event]:
    """Translate a full trace, dropping non-governance events."""
    out = []
    for event in events:
        kernel = trace_event_to_kernel(event, node_id)
        if kernel is not None:
            out.append(kernel)
    return out
