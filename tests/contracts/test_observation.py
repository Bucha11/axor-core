"""The neutral observation contract must satisfy both attached monitors.

These tests pin the field shapes that axor-probe (live ContextTap) and
axor-sentinel (cold SessionSink) actually read — transcribed from their consumer
code — WITHOUT importing those packages (the one-way dependency rule). If the
contract drifts from what a consumer reads, a test here fails before the consumer
breaks at runtime.
"""
from __future__ import annotations

import asyncio

from axor_core.contracts.observation import (
    ContextTap,
    ObservationHub,
    SessionAuditRecord,
    SessionContextView,
    SessionSink,
    ToolInvocationRecord,
)


def _view() -> SessionContextView:
    return SessionContextView(
        session_id="s1", agent_id="a1", timestamp=123.0, turn_index=3,
        token_count=420,
        context_window=(
            {"role": "user", "content": "do the thing"},
            {"role": "tool", "content": "[doc] ..."},
        ),
        system_prompt_hash="deadbeef", taint_active=True, external_read_count=2,
    )


def _record() -> SessionAuditRecord:
    return SessionAuditRecord(
        session_id="s1", agent_id="a1", started_at=100.0, taint_active=True,
        taint_sources=("external_web",),
        event_kinds=("taint_propagated", "escalation_denied", "intent_denied"),
        tool_invocations=(
            ToolInvocationRecord(tool="read_file", args={"path": "/x"}, executed=True),
            ToolInvocationRecord(tool="send_email", args={"to": "a@b.c"}, executed=False),
        ),
    )


# ── probe (ContextTap) reads these off SessionContextView ─────────────────────

def test_context_view_supports_probe_snapshot_fields() -> None:
    v = _view()
    # ViewSnapshotFactory.create maps exactly these.
    assert isinstance(v.session_id, str) and isinstance(v.agent_id, str)
    assert isinstance(v.timestamp, float) and isinstance(v.turn_index, int)
    assert isinstance(v.token_count, int)
    assert isinstance(v.system_prompt_hash, str)
    assert isinstance(v.taint_active, bool) and isinstance(v.external_read_count, int)
    # context_window must be replayable message dicts (role/content).
    assert isinstance(v.context_window, tuple)
    assert all(set(m) >= {"role", "content"} for m in v.context_window)


# ── sentinel (SessionSink) buckets these off SessionAuditRecord ───────────────

def test_audit_record_supports_sentinel_bucketing() -> None:
    r = _record()
    # Replicate CoreSessionSink._map_record's exact derivations.
    event_kinds = tuple(r.event_kinds)
    had_taint = bool(r.taint_active) or ("taint_propagated" in event_kinds)
    taint_source = r.taint_sources[0] if r.taint_sources else "unknown_external"
    had_escalation = "escalation_granted" in event_kinds or "escalation_denied" in event_kinds

    export_tokens = ("export", "send", "upload", "email", "post", "write", "commit", "push", "share")
    had_export = any(
        any(tok in inv.tool.lower() for tok in export_tokens) for inv in r.tool_invocations
    )
    had_failed_export = had_export and ("intent_denied" in event_kinds)

    assert had_taint and had_escalation and had_export and had_failed_export
    assert taint_source == "external_web"
    # _map_access reads tool / args / executed off each invocation.
    inv = r.tool_invocations[1]
    assert inv.tool == "send_email" and inv.args["to"] == "a@b.c" and inv.executed is False


# ── ObservationHub fans out to structurally-attached observers ────────────────

def test_hub_fans_context_and_session_to_structural_observers() -> None:
    seen_views: list[SessionContextView] = []
    seen_records: list[SessionAuditRecord] = []

    class _Tap:  # structural ContextTap — not imported, just shaped
        async def on_context_event(self, view: SessionContextView) -> None:
            seen_views.append(view)

    class _Sink:  # structural SessionSink
        async def on_session_closed(self, record: SessionAuditRecord) -> None:
            seen_records.append(record)

    tap, sink = _Tap(), _Sink()
    assert isinstance(tap, ContextTap) and isinstance(sink, SessionSink)  # runtime_checkable

    hub = ObservationHub()
    hub.register_tap(tap)
    hub.register_sink(sink)
    asyncio.run(hub.emit_context(_view()))
    asyncio.run(hub.emit_session_closed(_record()))

    assert len(seen_views) == 1 and seen_views[0].session_id == "s1"
    assert len(seen_records) == 1 and seen_records[0].agent_id == "a1"


def test_hub_swallows_a_misbehaving_observer() -> None:
    delivered: list[str] = []

    class _Bad:
        async def on_context_event(self, view: SessionContextView) -> None:
            raise RuntimeError("observer blew up on the hot path")

    class _Good:
        async def on_context_event(self, view: SessionContextView) -> None:
            delivered.append(view.session_id)

    hub = ObservationHub()
    hub.register_tap(_Bad())
    hub.register_tap(_Good())
    # Must not raise — and the good tap still gets the event.
    asyncio.run(hub.emit_context(_view()))
    assert delivered == ["s1"]
