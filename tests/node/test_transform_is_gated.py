"""A TRANSFORM decision is gated on the arguments it produces, not the ones it replaced.

`PolicyDecision` admits a rewritten payload, and the executor honoured it. That
made a premise of the enforcement argument unexamined: a gate that ran on the
pre-transform arguments does not justify executing the post-transform ones. The
loop now resolves a transform BEFORE normalization and before the cascade, so one
argument set is normalized, gated, approved, re-checked and executed.

Nothing in core emits TRANSFORM today, which is why this is pinned here with an
injected decision rather than found by a failing user: a latent premise is still
a premise, and the whole point of the chain argument is that every link holds for
the call that actually runs.
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.engine import TaintEngine

_INTERNAL = "http://169.254.169.254/latest/meta-data/"
_BENIGN = "http://example.com/page"


class _Fetch(ToolHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fetch"

    async def execute(self, args: dict):
        self.calls.append(dict(args))
        return "ok"


def _loop(handler: _Fetch) -> IntentLoop:
    ex = CapabilityExecutor()
    ex.register(handler)
    return IntentLoop(ex, [], taint_engine=TaintEngine())


def _transforming(loop: IntentLoop, new_args: dict) -> None:
    """Make the capability axis return APPROVE with a rewritten payload."""
    original = loop._evaluate_tool_intent

    def _patched(intent, envelope):
        _decision, pending = original(intent, envelope)
        return PolicyDecision(
            kind=PolicyDecisionKind.TRANSFORM,
            reason="rewritten by policy",
            transformed_payload=dict(new_args),
        ), pending

    loop._evaluate_tool_intent = _patched  # type: ignore[method-assign]


def _env(make_envelope):
    env = make_envelope()
    return dataclasses.replace(
        env,
        capabilities=dataclasses.replace(
            env.capabilities, allowed_tools=frozenset({"fetch"})
        ),
    )


def _call(url: str, node_id: str) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": url}, "tool_use_id": "tu1"},
        node_id=node_id,
    )


@pytest.mark.asyncio
async def test_a_transform_that_makes_the_call_worse_is_denied(make_envelope):
    """The pre-transform call is admissible; the post-transform one targets cloud
    metadata. If the gates ran on the arguments they were handed rather than the
    ones that execute, this would be allowed."""
    handler = _Fetch()
    loop = _loop(handler)
    _transforming(loop, {"url": _INTERNAL})

    resolved = await loop._resolve_tool_intent(
        _call(_BENIGN, "n1"), _env(make_envelope)
    )
    assert not resolved.approved
    assert "internal" in resolved.reason.lower() or "metadata" in resolved.reason.lower()
    assert handler.calls == []


@pytest.mark.asyncio
async def test_a_transform_that_makes_the_call_admissible_is_allowed(make_envelope):
    """The converse, so the fix is not just "deny on transform": a sanitising
    rewrite is gated on its own output and goes through."""
    handler = _Fetch()
    loop = _loop(handler)
    _transforming(loop, {"url": _BENIGN})

    resolved = await loop._resolve_tool_intent(
        _call(_INTERNAL, "n1"), _env(make_envelope)
    )
    assert resolved.approved, resolved.reason
    assert handler.calls == [{"url": _BENIGN}]


@pytest.mark.asyncio
async def test_the_handler_and_the_trace_see_the_transformed_arguments(make_envelope):
    handler = _Fetch()
    loop = _loop(handler)
    _transforming(loop, {"url": _BENIGN, "added": "by-policy"})

    resolved = await loop._resolve_tool_intent(
        _call(_BENIGN, "n1"), _env(make_envelope)
    )
    assert resolved.approved
    assert handler.calls == [{"url": _BENIGN, "added": "by-policy"}]
    assert resolved.transformed_payload == {"url": _BENIGN, "added": "by-policy"}
    transformed = [
        e for e in loop._trace_events
        if e.kind == TraceEventKind.INTENT_TRANSFORMED
    ]
    assert len(transformed) == 1
