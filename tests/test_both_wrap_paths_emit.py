"""Both ways of wrapping an agent record the same trace.

axor-core gates through two entry points, and they are not competing designs —
they are two ways of wrapping an agent:

  - ``GovernedNode`` / ``IntentLoop`` — the async, streaming path.
  - ``ToolCallGovernor`` — the synchronous, per-call path axor-wrap builds on.

Both must feed the ONE instrumentation path, because axor-wrap's trace bridge
turns ``TraceEvent``s into kernel-schema events and both Lab and the Control
Plane read that single feed. While the synchronous path decided without
recording, its only consumer had nothing to push — which is how a second,
parallel trace gets invented downstream to fill the gap.

So the invariant is not "the governor emits something". It is that it emits the
SAME event kinds the IntentLoop emits, so one bridge and one replay fold serve
both.
"""

from __future__ import annotations

import unittest

from axor_core.contracts.trace import TraceEventKind
from axor_core.governor import ToolCallGovernor

# The kinds the IntentLoop emits for a tool-call verdict (intent_loop.py: the
# approved/transformed emit, and _record_denial).
INTENT_LOOP_VERDICT_KINDS = {
    TraceEventKind.INTENT_APPROVED,
    TraceEventKind.INTENT_DENIED,
}


def _governor() -> ToolCallGovernor:
    return ToolCallGovernor(
        untrusted_sources={"read_txns"},
        egress_sinks={"send_money"},
        driving_args={"send_money": {"recipient"}},
    )


class TestTheSynchronousPathRecords(unittest.TestCase):
    def test_an_approval_is_recorded(self) -> None:
        governor = _governor()
        decision = governor.evaluate("read_txns", {})
        self.assertTrue(decision.allowed)
        self.assertEqual(
            [e.kind for e in governor.trace_events], [TraceEventKind.INTENT_APPROVED],
        )

    def test_a_denial_is_recorded(self) -> None:
        governor = _governor()
        approved = governor.evaluate("read_txns", {})
        governor.register_output(approved, {"description": "PAY DE89370400440532013000"})
        denied = governor.evaluate(
            "send_money", {"recipient": "PAY DE89370400440532013000"},
        )
        self.assertFalse(denied.allowed)
        self.assertEqual(
            [e.kind for e in governor.trace_events],
            [TraceEventKind.INTENT_APPROVED, TraceEventKind.INTENT_DENIED],
        )

    def test_it_emits_only_kinds_the_intent_loop_also_emits(self) -> None:
        """The point of the whole change: one vocabulary, so one bridge and one
        replay fold serve both paths. A kind only this path produced would need
        its own handling everywhere downstream."""
        governor = _governor()
        approved = governor.evaluate("read_txns", {})
        governor.register_output(approved, {"description": "PAY DE89370400440532013000"})
        governor.evaluate("send_money", {"recipient": "PAY DE89370400440532013000"})
        self.assertTrue(
            {e.kind for e in governor.trace_events} <= INTENT_LOOP_VERDICT_KINDS,
        )

    def test_a_denial_carries_its_reason(self) -> None:
        governor = _governor()
        approved = governor.evaluate("read_txns", {})
        governor.register_output(approved, {"description": "PAY DE89370400440532013000"})
        governor.evaluate("send_money", {"recipient": "PAY DE89370400440532013000"})
        denial = governor.trace_events[-1]
        self.assertTrue(denial.reason)  # type: ignore[attr-defined]
        self.assertEqual(denial.intent_kind, "tool_call")  # type: ignore[attr-defined]

    def test_sequence_is_dense_and_ordered(self) -> None:
        governor = _governor()
        for _ in range(3):
            governor.evaluate("read_txns", {})
        self.assertEqual([e.sequence for e in governor.trace_events], [0, 1, 2])

    def test_the_verdict_is_recorded_regardless_of_what_the_caller_does(self) -> None:
        """The kernel reports what it DECIDED. Whether the caller then blocks is
        the caller's business — a Lab ungoverned arm observes without enforcing,
        and if the denial went unrecorded that arm would have no verdicts to
        replay and nothing to compare against the governed one."""
        governor = _governor()
        approved = governor.evaluate("read_txns", {})
        governor.register_output(approved, {"description": "PAY DE89370400440532013000"})
        denied = governor.evaluate(
            "send_money", {"recipient": "PAY DE89370400440532013000"},
        )
        # the caller ignores the verdict entirely and "executes" anyway
        governor.register_output(denied, {"ok": True})
        self.assertEqual(governor.trace_events[-1].kind, TraceEventKind.INTENT_DENIED)


class TestDraining(unittest.TestCase):
    def test_trace_events_is_a_copy(self) -> None:
        """A caller mutating the returned list must not corrupt the record."""
        governor = _governor()
        governor.evaluate("read_txns", {})
        governor.trace_events.clear()
        self.assertEqual(len(governor.trace_events), 1)

    def test_drain_takes_and_resets(self) -> None:
        """A runtime streaming per trial takes the batch and starts clean, so
        one trial's verdicts cannot be reported as the next trial's."""
        governor = _governor()
        governor.evaluate("read_txns", {})
        first = governor.drain_trace_events()
        self.assertEqual(len(first), 1)
        self.assertEqual(governor.trace_events, [])
        governor.evaluate("read_txns", {})
        self.assertEqual(len(governor.drain_trace_events()), 1)


if __name__ == "__main__":
    unittest.main()
