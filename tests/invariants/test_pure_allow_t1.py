"""T1 / pure-allow invariant (v4.12). Enforcement is purely structural — no
probabilistic component near the gate (ML/judge removed)."""

from __future__ import annotations

import inspect

import pytest

from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop


async def _decide(loop, envelope, tool_name):
    async def _stream():
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": tool_name, "args": {}, "tool_use_id": "t"}, node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP,
                            payload={"usage": {"input_tokens": 1, "output_tokens": 1, "tool_tokens": 0}}, node_id=envelope.node_id)
    approved = None
    async for ev in loop.run(_stream(), envelope):
        if ev.kind == ExecutorEventKind.TEXT and "approved" in ev.payload:
            approved = ev.payload["approved"]
    return approved


def test_no_probabilistic_component_in_the_loop():
    params = inspect.signature(IntentLoop.__init__).parameters
    assert "anomaly_detector" not in params
    assert "anomaly_window_size" not in params


@pytest.mark.asyncio
async def test_gate_decision_is_stable(make_envelope, cap_executor):
    decisions = [await _decide(IntentLoop(capability_executor=cap_executor, trace_events=[]),
                               make_envelope(), "read") for _ in range(3)]
    assert decisions[0] is True and len(set(decisions)) == 1


@pytest.mark.asyncio
async def test_unauthorized_tool_denied(make_envelope, cap_executor):
    approved = await _decide(IntentLoop(capability_executor=cap_executor, trace_events=[]),
                             make_envelope(), "bash")
    assert approved is False
