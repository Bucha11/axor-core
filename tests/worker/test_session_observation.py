"""Session observation taps/sinks — read-only context tap and end-of-session sink.

These verify the P-34-clean integration seam: core emits neutral SessionContextView
and SessionAuditRecord facts to structurally-attached observers, fail-safe and
out-of-band, without importing any consumer (axor-probe / axor-sentinel).
"""
from __future__ import annotations

import pytest

from axor_core import GovernedSession
from axor_core.contracts.observation import SessionAuditRecord, SessionContextView
from axor_core.contracts.trace import TraceConfig
from tests.conftest import EchoExecutor


class _CollectingTap:
    def __init__(self) -> None:
        self.views: list[SessionContextView] = []

    async def on_context_event(self, view: SessionContextView) -> None:
        self.views.append(view)


class _CollectingSink:
    def __init__(self) -> None:
        self.records: list[SessionAuditRecord] = []

    async def on_session_closed(self, record: SessionAuditRecord) -> None:
        self.records.append(record)


class _RaisingTap:
    async def on_context_event(self, view: SessionContextView) -> None:
        raise RuntimeError("tap boom")


class _RaisingSink:
    async def on_session_closed(self, record: SessionAuditRecord) -> None:
        raise RuntimeError("sink boom")


def _session(executor, cap_executor, **kw):
    return GovernedSession(
        executor=executor,
        capability_executor=cap_executor,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        **kw,
    )


class TestContextTap:
    @pytest.mark.asyncio
    async def test_tap_receives_one_view_per_turn(self, echo_executor, cap_executor):
        tap = _CollectingTap()
        session = _session(echo_executor, cap_executor, context_taps=[tap])
        await session.run("write a test")
        await session.run("explain this")
        assert len(tap.views) == 2
        assert tap.views[0].turn_index == 1
        assert tap.views[1].turn_index == 2

    @pytest.mark.asyncio
    async def test_view_carries_neutral_facts(self, echo_executor, cap_executor):
        tap = _CollectingTap()
        session = _session(echo_executor, cap_executor, context_taps=[tap])
        await session.run("write a test")
        view = tap.views[0]
        assert view.session_id == session.session_id()
        assert isinstance(view.context_window, tuple)
        assert view.token_count > 0
        # taint not raised on a clean run
        assert view.taint_active is False
        assert view.taint_scope  # non-empty TaintScope.value

    @pytest.mark.asyncio
    async def test_external_read_count_increments_on_read(self, cap_executor):
        executor = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
        tap = _CollectingTap()
        session = _session(executor, cap_executor, context_taps=[tap])
        await session.run("write a test for the auth module")
        assert tap.views[0].external_read_count >= 1

    @pytest.mark.asyncio
    async def test_no_tap_means_no_emission(self, echo_executor, cap_executor):
        # No taps registered → run completes normally, nothing to observe.
        session = _session(echo_executor, cap_executor)
        result = await session.run("write a test")
        assert result is not None

    @pytest.mark.asyncio
    async def test_raising_tap_does_not_break_run(self, echo_executor, cap_executor):
        session = _session(echo_executor, cap_executor, context_taps=[_RaisingTap()])
        result = await session.run("write a test")
        assert result is not None  # governance path unaffected by tap failure


class TestSessionSink:
    @pytest.mark.asyncio
    async def test_sink_receives_record_on_close(self, echo_executor, cap_executor):
        sink = _CollectingSink()
        session = _session(echo_executor, cap_executor, session_sinks=[sink])
        await session.run("write a test")
        assert sink.records == []  # not emitted mid-session
        await session.aclose()
        assert len(sink.records) == 1
        rec = sink.records[0]
        assert rec.session_id == session.session_id()
        assert isinstance(rec.event_kinds, tuple) and len(rec.event_kinds) > 0

    @pytest.mark.asyncio
    async def test_record_captures_tool_invocation(self, cap_executor):
        executor = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
        sink = _CollectingSink()
        session = _session(executor, cap_executor, session_sinks=[sink])
        await session.run("write a test for the auth module")
        await session.aclose()
        rec = sink.records[0]
        tools = [ti.tool for ti in rec.tool_invocations]
        assert "read" in tools
        read_inv = next(ti for ti in rec.tool_invocations if ti.tool == "read")
        assert read_inv.executed is True
        assert read_inv.args.get("path") == "auth.py"

    @pytest.mark.asyncio
    async def test_no_sink_means_no_emission(self, echo_executor, cap_executor):
        session = _session(echo_executor, cap_executor)
        await session.run("write a test")
        await session.aclose()  # must not raise without sinks

    @pytest.mark.asyncio
    async def test_raising_sink_does_not_break_close(self, echo_executor, cap_executor):
        session = _session(echo_executor, cap_executor, session_sinks=[_RaisingSink()])
        await session.run("write a test")
        await session.aclose()  # swallowed
