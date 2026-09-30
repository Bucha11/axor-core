"""What the governor records must survive the fold that consumes it.

`kernel/events.py` documents the TOOL_CALL payload and `kernel/replay.py` folds
it. Nothing checked that the payload the governor actually emits is one that
fold can read — the two sides were only ever compared by reading prose.

They disagreed twice, in opposite directions.

First the governor filled ``arg_refs`` with a nested per-argument provenance
dict. ``arg_refs`` is ``{arg: value_ref}`` — opaque value IDS that
`_derive_driving_root` looks up in `state.tainted_refs` and that `subgraph`
walks as edges — so the fold did ``if ref in state.excised_refs`` on a dict and
died with `TypeError: unhashable type: 'dict'`. The per-argument breakdown moved
to ``arg_provenance``, where it belongs.

Then the governor recorded no refs at all, and that was the deeper gap: it has
no ref vocabulary of its own (it derives taint from content, not from minted
ids), so a recorded verdict named a tool and a root and nothing that BOUND the
denied argument to the read that tainted it. Replay coped — it falls back to
``driving_root`` — but every consumer reasoning over the value graph got an
empty graph, and the Control Plane's incident converter, which binds a sink's
driving arg through ``arg_refs``, could not reproduce a single recorded DENY and
refused whole runs.

So the governor now mints refs (`policy.provenance.ValueRefLedger`) with the
SAME content derivation the gates decide on, records a source event per arming
read, and binds them in ``arg_refs``.

These tests do not read a docstring. They take the bytes the governor emits and
put them through the real fold.
"""

from __future__ import annotations

import unittest

from axor_core.contracts.trace import TraceEventKind
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.replay import (
    GovernanceState,
    KernelConfig,
    _derive_driving_root,
    root_from_payload,
)

TAINTED = "PAY DE89370400440532013000"


def _session() -> ToolCallGovernor:
    """A real approve-then-deny session on the real kernel."""
    governor = ToolCallGovernor(
        untrusted_sources={"read"}, egress_sinks={"send"},
        driving_args={"send": ["to"]},
    )
    approved = governor.evaluate("read", {})
    governor.register_output(approved, {"description": TAINTED})
    governor.evaluate("send", {"to": TAINTED})
    return governor


def _payloads() -> list[dict]:
    return [dict(e.payload) for e in _session().trace_events]


def _fold_state() -> GovernanceState:
    """The state a consumer folding this session's SOURCE events arrives at —
    what `kernel.replay` builds from the TOOL_RESULT branch."""
    state = GovernanceState()
    for event in _session().trace_events:
        if event.kind is not TraceEventKind.TAINT_PROPAGATED:
            continue
        state.tainted_refs[str(event.payload["value_ref"])] = root_from_payload(
            event.payload["root"],
        )
    return state


class TestTheFoldCanReadWhatTheGovernorWrote(unittest.TestCase):
    def test_deriving_the_driving_root_does_not_raise(self) -> None:
        """It raised. `TypeError: unhashable type: 'dict'`, on every call."""
        for payload in _payloads():
            with self.subTest(tool=payload.get("tool")):
                _derive_driving_root(payload, _fold_state(), KernelConfig())

    def test_the_derived_root_carries_the_taint_the_governor_found(self) -> None:
        """Not merely 'does not crash': the fold has to arrive at the same taint
        the governor denied on, or a replay silently re-decides on nothing."""
        denial = _payloads()[-1]
        root = _derive_driving_root(denial, _fold_state(), KernelConfig())
        self.assertTrue(root.sources, "the fold derived a clean root for a tainted call")

    def test_driving_root_round_trips_through_root_from_payload(self) -> None:
        """`root_from_payload` builds `TaintSource(s)` from each token, so a
        token that is not a declared enum VALUE raises here — which is why the
        payload records `web`, not `TaintSource.WEB`."""
        denial = _payloads()[-1]
        root = root_from_payload(denial["driving_root"])
        self.assertTrue(root.sources)


class TestArgRefsBindTheSinkToItsSource(unittest.TestCase):
    """The binding a consumer needs to answer "denied on WHAT value" without
    re-deriving provenance from raw arguments."""

    def test_the_denied_argument_names_the_ref_the_read_minted(self) -> None:
        events = _session().trace_events
        source = next(
            e for e in events if e.kind is TraceEventKind.TAINT_PROPAGATED
        )
        denial = events[-1]
        self.assertEqual(
            denial.payload["arg_refs"]["to"], source.payload["value_ref"],
        )

    def test_the_per_argument_breakdown_keeps_its_own_key(self) -> None:
        """`arg_provenance` is a taint SUMMARY; `arg_refs` is a graph edge. They
        are not interchangeable and filling one with the other's shape is what
        crashed the fold."""
        denial = _payloads()[-1]
        self.assertIn("to", denial["arg_provenance"])
        self.assertTrue(denial["arg_provenance"]["to"]["sources"])
        self.assertIsInstance(denial["arg_refs"]["to"], str)

    def test_a_clean_argument_is_bound_to_no_ref(self) -> None:
        """A ref claims 'this argument carries that value'. Minting one for an
        argument the ledger does not derive from any registered read would be a
        graph edge that does not exist."""
        governor = ToolCallGovernor(
            untrusted_sources={"read"}, egress_sinks={"send"},
            driving_args={"send": ["to"]},
        )
        approved = governor.evaluate("read", {})
        governor.register_output(approved, {"description": TAINTED})
        governor.evaluate("send", {"to": "my-own-account"})
        self.assertNotIn("arg_refs", governor.trace_events[-1].payload)

    def test_a_real_arg_refs_still_takes_precedence_in_the_fold(self) -> None:
        """The documented path is unchanged: refs are resolved against the
        session's tainted values, not against the recorded root."""
        from axor_core.contracts.taint import TaintSource
        from axor_core.taint.causal_root import CausalRoot

        state = GovernanceState()
        state.tainted_refs["v1"] = CausalRoot.external_read(TaintSource.WEB)
        root = _derive_driving_root(
            {"arg_refs": {"to": "v1"}, "driving_root": {"sources": [], "sensitive": False}},
            state, KernelConfig(),
        )
        self.assertIn(TaintSource.WEB, root.sources)


class TestAnUnresolvableRefDoesNotFailOpen(unittest.TestCase):
    """The failure mode minting refs introduces, and the reason the fold has a
    fallback. Every ref is a promise that some earlier event registered it; a
    fold that never saw that event has NO information, and the pre-existing
    branch answered "no information" with "clean" — turning a recorded DENY into
    an ALLOW on a truncated trace."""

    def test_a_denial_whose_source_event_is_missing_stays_tainted(self) -> None:
        denial = _payloads()[-1]
        # the same payload, folded WITHOUT the source event
        root = _derive_driving_root(denial, GovernanceState(), KernelConfig())
        self.assertTrue(
            root.sources,
            "an unresolvable ref re-decided a tainted call as clean",
        )

    def test_excision_still_yields_a_clean_root(self) -> None:
        """Excision is the one case where 'no root' is the ANSWER, not missing
        data — the operator removed that value's future influence (spec 8.2.1),
        so falling back to the recorded root would undo the counterfactual."""
        denial = _payloads()[-1]
        state = _fold_state()
        state.excised_refs.update(denial["arg_refs"].values())
        root = _derive_driving_root(denial, state, KernelConfig())
        self.assertFalse(root.sources)
        self.assertFalse(root.sensitive)


class TestBothPathsStillRecordAlike(unittest.TestCase):
    def test_the_denial_still_names_its_gate_and_kind(self) -> None:
        """The rest of the payload is unaffected by the ref work."""
        denial = _session().trace_events[-1]
        self.assertEqual(denial.kind, TraceEventKind.INTENT_DENIED)
        # whichever gate fires first, the category must be one `gate_of` knows —
        # not a guess at WHICH gate, which is the kernel's business
        from axor_core.governor import DENIAL_CATEGORIES

        self.assertIn(denial.payload["category"], DENIAL_CATEGORIES)
        self.assertEqual(denial.payload["driving_args"], ["to"])


if __name__ == "__main__":
    unittest.main()
