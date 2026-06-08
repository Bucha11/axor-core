"""T1 / pure-allow invariant (v4.12 Phase 1).

`allow` must be a pure function of (structural projection, policy): equal
projection + equal policy ⇒ equal decision. Detection-layer signals
(reputation float, anomaly score/class) must NOT enter the gate — they are
out-of-band (TM7) and may influence a decision only by tightening degradation,
which narrows *policy* (a separate, monotone, session-state input).

These tests pin the boundary: varying the anomaly verdict, with no degradation
engine wired, does not change the gate's decision.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop


def _detector(cls: AnomalyClass, score: float) -> AsyncMock:
    d = AsyncMock()
    d.score = AsyncMock(return_value=AnomalyResult(score=score, cls=cls, reasons=()))
    return d


async def _decide(loop: IntentLoop, envelope, tool_name: str) -> bool:
    async def _stream():
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": tool_name, "args": {}, "tool_use_id": "t"},
            node_id=envelope.node_id,
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": {"input_tokens": 1, "output_tokens": 1, "tool_tokens": 0}},
            node_id=envelope.node_id,
        )

    approved = None
    async for ev in loop.run(_stream(), envelope):
        if ev.kind == ExecutorEventKind.TEXT and "approved" in ev.payload:
            approved = ev.payload["approved"]
    return approved


@pytest.mark.asyncio
async def test_anomaly_verdict_does_not_change_gate_decision(make_envelope, cap_executor):
    """NORMAL / SUSPICIOUS / CRITICAL all yield the SAME gate decision on `read`
    when no degradation engine is wired — detection is not part of `allow`.
    """
    decisions = {}
    for cls, score in [
        (AnomalyClass.NORMAL, 0.1),
        (AnomalyClass.SUSPICIOUS, 0.6),
        (AnomalyClass.CRITICAL, 0.95),
    ]:
        loop = IntentLoop(
            capability_executor=cap_executor,
            trace_events=[],
            anomaly_detector=_detector(cls, score),
        )
        decisions[cls] = await _decide(loop, make_envelope(), "read")

    assert decisions[AnomalyClass.NORMAL] is True
    # All three identical — the gate did not read the anomaly verdict.
    assert len(set(decisions.values())) == 1


@pytest.mark.asyncio
async def test_gate_decision_matches_with_and_without_detector(make_envelope, cap_executor):
    """The gate decision for an allowed tool is identical whether or not a
    detector is present (detector cannot flip an allow/deny by itself)."""
    no_det = IntentLoop(capability_executor=cap_executor, trace_events=[])
    with_det = IntentLoop(
        capability_executor=cap_executor,
        trace_events=[],
        anomaly_detector=_detector(AnomalyClass.CRITICAL, 0.99),
    )
    assert await _decide(no_det, make_envelope(), "read") == await _decide(
        with_det, make_envelope(), "read"
    )
