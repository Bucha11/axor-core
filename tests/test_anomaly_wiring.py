"""
Tests for anomaly detection wiring in IntentLoop and GovernedSession.

Three cases:
  (a) CRITICAL → tool call denied
  (b) SUSPICIOUS → tool call approved + SuspiciousIntentEvent in trace
  (c) anomaly_detector=None → existing behavior unchanged
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult, NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import SuspiciousIntentEvent, TraceEventKind
from axor_core.node.intent_loop import IntentLoop


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ni(**kw) -> NormalizedIntent:
    defaults = dict(
        tool="read", operation="file_read", target_kind="workdir",
        destination_kind="none", provenance="repo",
        reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=False,
        after_secret_access=False, data_flow="none",
    )
    defaults.update(kw)
    return NormalizedIntent(**defaults)


def _mock_detector(cls: AnomalyClass, score: float = 0.9) -> AsyncMock:
    """Returns a mock AnomalyDetector that always returns the given class."""
    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=score,
        cls=cls,
        reasons=("test_reason",),
    ))
    return detector


async def _run_loop(
    loop: IntentLoop,
    envelope,
    tool_name: str = "read",
    tool_args: dict | None = None,
) -> tuple[list, list]:
    """Drive the loop with a single tool_use event, collect results and trace."""

    async def _stream():
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": tool_name, "args": tool_args or {}, "tool_use_id": "tid1"},
            node_id=envelope.node_id,
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": {"input_tokens": 10, "output_tokens": 5, "tool_tokens": 0}},
            node_id=envelope.node_id,
        )

    events = []
    async for ev in loop.run(_stream(), envelope):
        events.append(ev)
    return events, loop._trace_events


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critical_anomaly_denies_tool_call(make_envelope, cap_executor):
    """CRITICAL anomaly score → tool call is denied, approved=False."""
    trace_events: list = []
    detector = _mock_detector(AnomalyClass.CRITICAL, score=0.92)

    loop = IntentLoop(
        capability_executor=cap_executor,
        trace_events=trace_events,
        anomaly_detector=detector,
    )
    envelope = make_envelope()

    events, trace = await _run_loop(loop, envelope, tool_name="read")

    # Tool result event should be a denial
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert len(tool_results) == 1
    payload = tool_results[0].payload
    assert payload.get("approved") is False
    assert "anomaly" in str(payload.get("tool_result", "")).lower()

    # ANOMALY_FLAGGED trace event emitted
    anomaly_events = [e for e in trace if e.kind == TraceEventKind.ANOMALY_FLAGGED]
    assert len(anomaly_events) == 1
    ae = anomaly_events[0]
    assert isinstance(ae, SuspiciousIntentEvent)
    assert ae.anomaly_class == AnomalyClass.CRITICAL
    assert ae.policy_action == "denied"
    assert ae.score == pytest.approx(0.92)

    # INTENT_DENIED trace event also emitted
    denied_events = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
    assert len(denied_events) == 1


@pytest.mark.asyncio
async def test_suspicious_anomaly_flags_but_approves(make_envelope, cap_executor):
    """SUSPICIOUS anomaly score → tool call approved + SuspiciousIntentEvent in trace."""
    trace_events: list = []
    detector = _mock_detector(AnomalyClass.SUSPICIOUS, score=0.55)

    loop = IntentLoop(
        capability_executor=cap_executor,
        trace_events=trace_events,
        anomaly_detector=detector,
    )
    envelope = make_envelope()

    events, trace = await _run_loop(loop, envelope, tool_name="read")

    # Tool result event should be approved
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert len(tool_results) == 1
    payload = tool_results[0].payload
    assert payload.get("approved") is True

    # ANOMALY_FLAGGED trace event emitted with policy_action=flagged
    anomaly_events = [e for e in trace if e.kind == TraceEventKind.ANOMALY_FLAGGED]
    assert len(anomaly_events) == 1
    ae = anomaly_events[0]
    assert isinstance(ae, SuspiciousIntentEvent)
    assert ae.anomaly_class == AnomalyClass.SUSPICIOUS
    assert ae.policy_action == "flagged"

    # No INTENT_DENIED event (intent was approved)
    denied_events = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
    assert len(denied_events) == 0


@pytest.mark.asyncio
async def test_no_anomaly_detector_unchanged_behavior(make_envelope, cap_executor):
    """anomaly_detector=None → behavior identical to before (no anomaly events)."""
    trace_events: list = []

    loop = IntentLoop(
        capability_executor=cap_executor,
        trace_events=trace_events,
        anomaly_detector=None,
    )
    envelope = make_envelope()

    events, trace = await _run_loop(loop, envelope, tool_name="read")

    # Tool call is approved normally
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert len(tool_results) == 1
    assert tool_results[0].payload.get("approved") is True

    # No anomaly trace events
    anomaly_events = [e for e in trace if e.kind == TraceEventKind.ANOMALY_FLAGGED]
    assert len(anomaly_events) == 0


@pytest.mark.asyncio
async def test_normal_anomaly_no_trace_event(make_envelope, cap_executor):
    """NORMAL anomaly score → no SuspiciousIntentEvent emitted."""
    trace_events: list = []
    detector = _mock_detector(AnomalyClass.NORMAL, score=0.1)

    loop = IntentLoop(
        capability_executor=cap_executor,
        trace_events=trace_events,
        anomaly_detector=detector,
    )
    envelope = make_envelope()

    events, trace = await _run_loop(loop, envelope, tool_name="read")

    anomaly_events = [e for e in trace if e.kind == TraceEventKind.ANOMALY_FLAGGED]
    assert len(anomaly_events) == 0

    # Intent still approved
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert tool_results[0].payload.get("approved") is True


@pytest.mark.asyncio
async def test_anomaly_detector_exception_is_safe(make_envelope, cap_executor):
    """If the anomaly detector raises, the tool call is approved (fail-open)."""
    trace_events: list = []
    detector = AsyncMock()
    detector.score = AsyncMock(side_effect=RuntimeError("detector exploded"))

    loop = IntentLoop(
        capability_executor=cap_executor,
        trace_events=trace_events,
        anomaly_detector=detector,
    )
    envelope = make_envelope()

    events, trace = await _run_loop(loop, envelope, tool_name="read")

    # Should not raise; tool call approved
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert tool_results[0].payload.get("approved") is True

    # No anomaly event recorded when detector raises
    assert len([e for e in trace if e.kind == TraceEventKind.ANOMALY_FLAGGED]) == 0
