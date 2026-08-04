"""The two ways of wrapping an agent record the SAME thing.

`ToolCallGovernor` and `IntentLoop` are not competing designs — they are two
entry points to one kernel, chosen by who owns the agent loop. The framework
owning it gets the synchronous governor; the kernel owning it gets the streaming
loop. A consumer downstream should not be able to tell which was used, because
the verdict and the provenance it turned on are the same verdict and the same
provenance.

They did not record alike. The governor recorded `payload={"tool": name}` and
the loop recorded the same, with a denial recording no payload at all. Each had
computed the per-argument taint roots, the driving-arg join and the floor state
to REACH its verdict, and each threw all of it away — so anything wanting to
show an operator *why* a call was refused had to re-derive provenance from raw
arguments. That is how a second taint ledger gets built downstream, and it is
what happened: axor-lab grew one.

Both now call one shared builder (`policy.provenance.call_payload`). This file
drives the real kernel down both paths with the same attack and asserts the
recorded payloads are equal — the property that lets ONE bridge and ONE replay
fold serve both.
"""

from __future__ import annotations

import pytest

from typing import Any

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.governor import ToolCallGovernor


class _Handler(ToolHandler):
    def __init__(self, name: str, result: Any) -> None:
        self._name, self._result = name, result

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, args: dict[str, Any]) -> Any:
        return self._result


def _executor() -> CapabilityExecutor:
    """BOTH tools registered. An unregistered tool is refused by the capability
    gate before the taint cascade runs, which would compare a capability denial
    here against a taint denial there."""
    ex = CapabilityExecutor()
    ex.register(_Handler("read", {"description": TAINTED}))
    ex.register(_Handler("write", {"ok": True}))
    return ex

# one deployment taxonomy, used by both paths
# `read` / `write` because the stock test policy already admits those
# capabilities. The taxonomy below is what makes them a taint SOURCE and an
# egress SINK — the tool names themselves carry no meaning to the kernel.
UNTRUSTED = {"read"}
EGRESS = {"write"}
DRIVING = {"write": ["recipient"]}
TAINTED = "PAY DE89370400440532013000"

PROVENANCE_KEYS = {"tool", "arg_provenance", "driving_args", "driving_root", "floor_active"}


def _verdicts(events):
    return [
        e for e in events
        if e.kind in (TraceEventKind.INTENT_APPROVED, TraceEventKind.INTENT_DENIED)
    ]


def _path_b():
    """The synchronous path: the framework owns the loop."""
    governor = ToolCallGovernor(
        untrusted_sources=UNTRUSTED, egress_sinks=EGRESS, driving_args=DRIVING,
    )
    approved = governor.evaluate("read", {})
    governor.register_output(approved, {"description": TAINTED})
    governor.evaluate("write", {"recipient": TAINTED})
    return _verdicts(governor.trace_events)


async def _path_a(make_envelope, _focused):
    """The streaming path: the kernel owns the loop."""
    from axor_core.node.intent_loop import IntentLoop

    async def _stream():
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "read", "args": {}, "tool_use_id": "t1"},
            node_id="node_test_root",
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "write", "args": {"recipient": TAINTED},
                     "tool_use_id": "t2"},
            node_id="node_test_root",
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": {"input_tokens": 1, "output_tokens": 1, "tool_tokens": 0}},
            node_id="node_test_root",
        )

    events: list = []
    loop = IntentLoop(
        capability_executor=_executor(), trace_events=events,
        untrusted_sources=UNTRUSTED, egress_sinks=EGRESS, driving_args=DRIVING,
    )
    async for _ in loop.run(_stream(), make_envelope(policy=_focused)):
        pass
    return _verdicts(events)


class TestTheSynchronousPathRecordsProvenance:
    def test_every_verdict_carries_the_full_shape(self) -> None:
        for event in _path_b():
            assert PROVENANCE_KEYS <= set(event.payload), event.payload


@pytest.mark.asyncio
class TestTheStreamingPathRecordsProvenance:
    async def test_every_verdict_carries_the_full_shape(
        self, make_envelope, focused_policy,
    ) -> None:
        verdicts = await _path_a(make_envelope, focused_policy)
        assert verdicts, "the loop recorded no verdict at all"
        for event in verdicts:
            assert PROVENANCE_KEYS <= set(event.payload), event.payload

    async def test_a_denial_carries_what_it_was_denied_ON(
        self, make_envelope, focused_policy,
    ) -> None:
        """A denial used to record NO payload. A consumer could see THAT the
        kernel refused and never what it refused over — so an operator asking
        'why' got the reason string and nothing to anchor it to."""
        denials = [
            e for e in await _path_a(make_envelope, focused_policy)
            if e.kind is TraceEventKind.INTENT_DENIED
        ]
        assert denials, "the tainted egress was not denied"
        payload = denials[0].payload
        assert payload["driving_args"] == ["recipient"]
        assert payload["arg_provenance"]["recipient"]["sources"]
        assert payload["driving_root"]["sources"] == \
            payload["arg_provenance"]["recipient"]["sources"]


@pytest.mark.asyncio
class TestBothPathsAgree:
    async def test_the_recorded_provenance_is_identical(
        self, make_envelope, focused_policy,
    ) -> None:
        """The property the whole design rests on: a downstream consumer cannot
        tell which entry point produced the record, so one bridge and one replay
        fold serve both."""
        a = await _path_a(make_envelope, focused_policy)
        b = _path_b()
        assert [e.kind for e in a] == [e.kind for e in b]
        for ea, eb in zip(a, b):
            assert {k: ea.payload[k] for k in PROVENANCE_KEYS} == \
                   {k: eb.payload[k] for k in PROVENANCE_KEYS}

    async def test_both_deny_the_same_call(self, make_envelope, focused_policy) -> None:
        a = await _path_a(make_envelope, focused_policy)
        b = _path_b()
        assert [e.kind is TraceEventKind.INTENT_DENIED for e in a] == \
               [e.kind is TraceEventKind.INTENT_DENIED for e in b] == [False, True]
