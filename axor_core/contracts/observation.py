from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable


# ── Neutral observation DTOs ────────────────────────────────────────────────────
#
# axor-core owns these shapes. They carry *raw facts* only — no bucketing,
# normalization, or consumer-specific vocabulary. Downstream observers
# (axor-probe, axor-sentinel) translate them into their own types:
#
#   SessionContextView  → axor-probe StateSnapshot / CanonicalizedContextSummary
#   SessionAuditRecord  → axor-sentinel SessionSummary / ResourceAccess
#
# axor-core never imports the consumers — dependency direction is strictly
# one-way (P-34). Consumers attach by implementing the Protocols below.


@dataclass(frozen=True)
class ToolInvocationRecord:
    """
    A single tool invocation observed within a session.

    Raw material only. `args` is the unmodified tool argument mapping as seen by
    the capability executor — consumers derive resource identity / signal grading
    from it (axor-sentinel runs its own resource normalizer over `tool` + `args`).
    """

    tool: str
    args: Mapping[str, object]
    executed: bool  # True if the tool ran; False if the intent was denied before execution


@dataclass(frozen=True)
class SessionContextView:
    """
    Read-only, bounded, hash-only view of live session context at a trigger point.

    Owned by axor-core; never persisted by core (P-10). It carries the real
    bounded context window because an in-process probe instance needs it to
    measure drift — but it stays in-process: only redacted signals are exported
    downstream (P-11). The system prompt is exposed as a hash only, never text.

    Structural facts are raw. Consumers bucket them (e.g. axor-probe derives
    CanonicalizedContextSummary.data_sensitivity from `taint_sources` /
    `external_read_count`).
    """

    session_id: str
    agent_id: str
    timestamp: float
    turn_index: int
    token_count: int
    context_window: tuple[Mapping[str, object], ...]  # bounded slice; bound owned by core
    system_prompt_hash: str                           # hash only — never plaintext (P-11)
    taint_active: bool
    taint_sources: tuple[str, ...]                    # TaintSource.value
    taint_scope: str                                  # TaintScope.value
    taint_intent_age: int
    external_read_count: int


@dataclass(frozen=True)
class SessionAuditRecord:
    """
    Raw end-of-session facts, built from the session's DecisionTraces and the
    final TaintState. axor-core's vocabulary, not the consumer's — the mapping
    into axor-sentinel SessionSummary (had_export_attempt, had_failed_export,
    had_escalation, ResourceAccess grading) is done by the consumer.
    """

    session_id: str
    agent_id: str
    started_at: float
    ended_at: float
    taint_active: bool
    taint_sources: tuple[str, ...]                    # TaintSource.value
    event_kinds: tuple[str, ...]                      # distinct TraceEventKind.value seen this session
    tool_invocations: tuple[ToolInvocationRecord, ...]


# ── Observer Protocols ──────────────────────────────────────────────────────────


@runtime_checkable
class ContextTap(Protocol):
    """
    Mid-session, read-only context tap.

    axor-core calls on_context_event() at each turn boundary when at least one
    tap is registered. Gating (whether a probe should actually fire) lives in the
    consumer — core emits unconditionally and cheaply.

    Implementations MUST return promptly and MUST NOT block the governance path:
    out-of-band work (e.g. probe inference) must be scheduled on a separate task.
    Implementations MUST NOT raise — the caller catches and logs, but a slow or
    throwing tap still degrades the governed turn it rides on.

    Canonical implementation: axor_probe.integration.core_tap.CoreContextTap.
    """

    async def on_context_event(self, view: SessionContextView) -> None: ...


@runtime_checkable
class SessionSink(Protocol):
    """
    End-of-session audit sink.

    axor-core calls on_session_closed() once during GovernedSession.aclose()
    when at least one sink is registered. Naturally out-of-band — the consumer
    (e.g. axor-sentinel) buffers the record for its next background audit cycle.

    Implementations MUST NOT raise — failures are caught and logged by core.

    Canonical implementation: axor_sentinel.integration.core_sink.CoreSessionSink.
    """

    async def on_session_closed(self, record: SessionAuditRecord) -> None: ...
