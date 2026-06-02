"""OBSERVE mode: governance resolves and records denials but never blocks.

In ExecutionMode.OBSERVE the IntentLoop still evaluates every intent against
policy/taint/degradation and records what it WOULD deny (INTENT_DENIED with
observed=True), but the tool executes anyway and returns its real result, so
evaluation measurement is uncontaminated by enforcement.
"""
from __future__ import annotations

from typing import Any

import pytest

from axor_core import GovernedSession, presets
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.trace import TraceEventKind
from axor_core.contracts.trace import TraceConfig
from tests.conftest import EchoExecutor


class _TrackingBash(ToolHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "bash"

    async def execute(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return "bash output"


def _session(handler, mode):
    cap = CapabilityExecutor()
    cap.register(handler)
    return GovernedSession(
        executor=EchoExecutor(tool_calls=[("bash", {"cmd": "ls"})]),
        capability_executor=cap,
        mode=mode,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )


def _events(session, kind):
    out = []
    for trace in session.all_traces():
        out.extend(e for e in trace.events if e.kind == kind)
    return out


def _denied_events(session):
    return _events(session, TraceEventKind.INTENT_DENIED)


def _approved_events(session):
    return _events(session, TraceEventKind.INTENT_APPROVED)


@pytest.mark.asyncio
async def test_enforce_mode_blocks_denied_tool():
    handler = _TrackingBash()
    session = _session(handler, ExecutionMode.LIBRARY)
    await session.run("do something", policy=presets.get("readonly"))  # readonly denies bash
    assert handler.calls == []                      # tool never executed
    denied = _denied_events(session)
    assert len(denied) >= 1
    assert denied[0].payload.get("observed") is False


@pytest.mark.asyncio
async def test_observe_mode_records_denial_but_executes():
    handler = _TrackingBash()
    session = _session(handler, ExecutionMode.OBSERVE)
    await session.run("do something", policy=presets.get("readonly"))
    # OBSERVE: the would-be-denied tool actually ran...
    assert handler.calls == [{"cmd": "ls"}]
    # ...and the denial is recorded as observed-only, plus an approval/execution.
    denied = _denied_events(session)
    assert len(denied) >= 1
    assert denied[0].payload.get("observed") is True
    assert len(_approved_events(session)) >= 1
