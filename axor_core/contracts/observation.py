"""Neutral observation contract — the read-only seam external monitors attach to.

axor-core EMITS observations on its governance path; external packages
(axor-probe, axor-sentinel) attach as *structural* implementations of the
Protocols below. Core never imports those packages (the one-way dependency rule):
it defines the vocabulary here and fans observations out to whoever registered.

Two granularities, deliberately separate:

  * ``ContextTap.on_context_event(SessionContextView)`` — the LIVE per-turn context
    window, on the governance HOT path. A tap must return promptly and must not
    raise; heavy work (e.g. a probe inference cycle) is scheduled out-of-band by
    the consumer. Used by axor-probe to build drift snapshots.

  * ``SessionSink.on_session_closed(SessionAuditRecord)`` — a per-session summary
    of raw facts at close, on the COLD / audit path. Used by axor-sentinel to feed
    its reputation cycle.

All payloads are *raw facts*. Enum-valued fields are passed as their ``.value``
strings (taint sources, event kinds) so a consumer needs no core enums to read
them, and the consumer does its own bucketing into its vocabulary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("axor.observation")


# ── Shared leaf ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolInvocationRecord:
    """One tool call observed in a session: name, arguments, and whether it ran.

    ``args`` is the raw argument mapping (the consumer derives resource identity
    from it); ``executed`` is False when the invocation was denied/blocked.
    """
    tool: str
    args: dict[str, Any]
    executed: bool


# ── Live per-turn context (hot path) → axor-probe ContextTap ──────────────────

@dataclass(frozen=True)
class SessionContextView:
    """Read-only view of a node's context window at one governance event.

    Emitted on the hot path, so a ``ContextTap`` must return promptly.
    ``context_window`` is message-shaped (``{"role": ..., "content": ...}`` dicts)
    so a consumer can replay it directly. Structural buckets (sensitivity, etc.)
    are derived by the consumer — core hands over raw facts only.
    """
    session_id: str
    agent_id: str
    timestamp: float
    turn_index: int
    token_count: int
    context_window: tuple[dict[str, Any], ...]
    system_prompt_hash: str
    taint_active: bool
    external_read_count: int


@runtime_checkable
class ContextTap(Protocol):
    """Receiver of live per-turn context events, implemented structurally by an
    external monitor (e.g. axor-probe). Core never imports the implementor.

    Contract: ``on_context_event`` runs on the governance HOT path. It MUST return
    promptly and MUST NOT raise. Heavy work is scheduled out-of-band by the tap.
    """
    async def on_context_event(self, view: SessionContextView) -> None: ...


# ── Per-session-close audit (cold path) → axor-sentinel SessionSink ───────────

@dataclass(frozen=True)
class SessionAuditRecord:
    """Raw-facts summary of one closed session, emitted once at teardown.

    Consumers bucket these into their own vocabulary (e.g. sentinel's
    ``SessionSummary``). ``taint_sources`` and ``event_kinds`` are ``.value``
    strings (``TaintSource`` / ``TraceEventKind``) so the consumer needs no core
    enums to read them.
    """
    session_id: str
    agent_id: str
    started_at: float
    taint_active: bool
    taint_sources: tuple[str, ...]
    event_kinds: tuple[str, ...]
    tool_invocations: tuple[ToolInvocationRecord, ...]


@runtime_checkable
class SessionSink(Protocol):
    """Receiver of per-session-close audit records, implemented structurally by an
    external monitor (e.g. axor-sentinel). Core never imports the implementor.

    Contract: ``on_session_closed`` runs on the cold/audit path and MUST NOT raise.
    """
    async def on_session_closed(self, record: SessionAuditRecord) -> None: ...


# ── Neutral fan-out registry ──────────────────────────────────────────────────

class ObservationHub:
    """Holds registered taps/sinks and fans observations out to them.

    Core owns one hub (per session or process) and calls ``emit_context`` on the
    governance hot path and ``emit_session_closed`` at teardown. Observers attach
    structurally — the hub imports none of them.

    Fail-safe: a misbehaving observer must never break the governance path. The
    Protocols forbid raising, but ``emit_*`` defend in depth — an observer
    exception is logged and swallowed so one bad tap cannot stop the session or
    starve the other observers.
    """

    def __init__(self) -> None:
        self._taps: list[ContextTap] = []
        self._sinks: list[SessionSink] = []

    def register_tap(self, tap: ContextTap) -> None:
        self._taps.append(tap)

    def register_sink(self, sink: SessionSink) -> None:
        self._sinks.append(sink)

    async def emit_context(self, view: SessionContextView) -> None:
        for tap in self._taps:
            try:
                await tap.on_context_event(view)
            except Exception:
                log.warning("context tap raised on the hot path — swallowed", exc_info=True)

    async def emit_session_closed(self, record: SessionAuditRecord) -> None:
        for sink in self._sinks:
            try:
                await sink.on_session_closed(record)
            except Exception:
                log.warning("session sink raised on the audit path — swallowed", exc_info=True)
