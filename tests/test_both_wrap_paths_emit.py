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
# approved/transformed emit, and _record_denial) plus the source event it emits
# when a read arms taint (_register_value_taint → _record_taint_propagated).
INTENT_LOOP_VERDICT_KINDS = {
    TraceEventKind.INTENT_APPROVED,
    TraceEventKind.INTENT_DENIED,
    TraceEventKind.TAINT_PROPAGATED,
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
            [TraceEventKind.INTENT_APPROVED,
             # the read armed taint — the trace says WHERE it came from
             TraceEventKind.TAINT_PROPAGATED,
             TraceEventKind.INTENT_DENIED],
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


class TestTheVerdictCarriesItsProvenance(unittest.TestCase):
    """Recording only the tool name is what forces a consumer to build a second
    taint ledger: the trace says a call was denied but not what it was denied
    ON, so anything wanting to show the operator *why* has to re-derive the
    provenance the kernel already computed.

    It goes under ``arg_provenance``, not the schema's ``arg_refs``: that field
    is ``{arg: value_ref}`` — opaque ids the replay fold looks up and the
    subgraph walks as edges — and filling it with nested dicts crashed the fold
    (`TypeError: unhashable type: 'dict'`). See
    `test_governor_payload_survives_replay.py`, which puts these bytes through
    the real fold rather than comparing them to a docstring."""

    def _denial(self):
        governor = _governor()
        approved = governor.evaluate("read_txns", {})
        governor.register_output(approved, {"description": "PAY DE89370400440532013000"})
        governor.evaluate("send_money", {"recipient": "PAY DE89370400440532013000"})
        return governor.trace_events[-1].payload

    def test_an_approval_names_its_tool(self) -> None:
        governor = _governor()
        governor.evaluate("read_txns", {})
        self.assertEqual(governor.trace_events[0].payload["tool"], "read_txns")

    def test_each_argument_carries_the_sources_it_was_derived_from(self) -> None:
        payload = self._denial()
        self.assertEqual(sorted(payload["arg_provenance"]), ["recipient"])
        self.assertTrue(payload["arg_provenance"]["recipient"]["sources"])

    def test_the_driving_root_is_the_join_over_the_declared_driving_args(self) -> None:
        """The verdict turns on the driving root, not on every argument, so a
        consumer reading only the per-argument breakdown could not tell an
        incidental tainted argument from the one that caused the denial."""
        payload = self._denial()
        self.assertEqual(payload["driving_args"], ["recipient"])
        self.assertEqual(
            payload["driving_root"]["sources"], payload["arg_provenance"]["recipient"]["sources"],
        )

    def test_a_denial_carries_its_category(self) -> None:
        self.assertEqual(self._denial()["category"], "taint_enforcement")

    def test_untainted_arguments_report_no_sources_rather_than_being_omitted(self) -> None:
        """Absence and clean must be distinguishable — an omitted entry reads as
        'not recorded', which is exactly the ambiguity that makes a trace
        un-auditable."""
        governor = _governor()
        governor.evaluate("send_money", {"recipient": "self"})
        payload = governor.trace_events[-1].payload
        self.assertEqual(payload["arg_provenance"]["recipient"]["sources"], [])
        self.assertFalse(payload["driving_root"]["sensitive"])

    def test_the_floor_state_is_reported(self) -> None:
        self.assertIn("floor_active", self._denial())

    def test_sources_are_declared_tokens_not_python_reprs(self) -> None:
        """These tokens land in recorded governance artifacts that other
        languages read and hash. `str()` on a `(str, Enum)` member yields
        `TaintSource.WEB` — a Python implementation detail no Rust or TS
        producer would ever emit for the same value."""
        from axor_core.contracts.taint import TaintSource

        declared = {s.value for s in TaintSource}
        payload = self._denial()
        for token in payload["driving_root"]["sources"]:
            self.assertIn(token, declared, f"{token!r} is not a declared source value")
        self.assertTrue(payload["driving_root"]["sources"])


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
