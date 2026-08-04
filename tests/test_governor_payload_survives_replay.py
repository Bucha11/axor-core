"""What the governor records must survive the fold that consumes it.

`kernel/events.py` documents the TOOL_CALL payload and `kernel/replay.py` folds
it. Nothing checked that the payload the governor actually emits is one that
fold can read — the two sides were only ever compared by reading prose.

They disagreed. The schema's ``arg_refs`` is ``{arg: value_ref}``: opaque value
IDS that `_derive_driving_root` looks up in `state.tainted_refs` and that
`subgraph` walks as edges. The governor filled it with a nested per-argument
provenance dict, so the fold did

    for ref in arg_refs.values():
        if ref in state.excised_refs:

on a dict and died with `TypeError: unhashable type: 'dict'`. Every recorded
verdict from the synchronous path was unreplayable, and `subgraph` would have
stringified those dicts into nonsense node ids.

The governor has no value-ref vocabulary — it derives taint from content, not
from minted ids — so it leaves `arg_refs` alone and records `driving_root`,
which is exactly the fallback `_derive_driving_root` takes and which is what the
verdict turned on. The per-argument breakdown lives under `arg_provenance`.

So these tests do not read a docstring. They take the bytes the governor emits
and put them through the real fold.
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


def _payloads() -> list[dict]:
    """A real approve-then-deny session's recorded payloads."""
    governor = ToolCallGovernor(
        untrusted_sources={"read"}, egress_sinks={"send"},
        driving_args={"send": ["to"]},
    )
    approved = governor.evaluate("read", {})
    governor.register_output(approved, {"description": TAINTED})
    governor.evaluate("send", {"to": TAINTED})
    return [dict(e.payload) for e in governor.trace_events]


class TestTheFoldCanReadWhatTheGovernorWrote(unittest.TestCase):
    def test_deriving_the_driving_root_does_not_raise(self) -> None:
        """It raised. `TypeError: unhashable type: 'dict'`, on every call."""
        for payload in _payloads():
            with self.subTest(tool=payload.get("tool")):
                _derive_driving_root(payload, GovernanceState(), KernelConfig())

    def test_the_derived_root_carries_the_taint_the_governor_found(self) -> None:
        """Not merely 'does not crash': the fold has to arrive at the same taint
        the governor denied on, or a replay silently re-decides on nothing."""
        denial = _payloads()[-1]
        root = _derive_driving_root(denial, GovernanceState(), KernelConfig())
        self.assertTrue(root.sources, "the fold derived a clean root for a tainted call")

    def test_driving_root_round_trips_through_root_from_payload(self) -> None:
        """`root_from_payload` builds `TaintSource(s)` from each token, so a
        token that is not a declared enum VALUE raises here — which is why the
        payload records `web`, not `TaintSource.WEB`."""
        denial = _payloads()[-1]
        root = root_from_payload(denial["driving_root"])
        self.assertTrue(root.sources)


class TestArgRefsIsLeftForRefs(unittest.TestCase):
    def test_the_governor_does_not_populate_arg_refs(self) -> None:
        """It mints no value ids, so anything it put there would be a lie the
        fold and the subgraph both act on."""
        for payload in _payloads():
            with self.subTest(tool=payload.get("tool")):
                self.assertNotIn("arg_refs", payload)

    def test_the_per_argument_breakdown_is_under_its_own_key(self) -> None:
        denial = _payloads()[-1]
        self.assertIn("to", denial["arg_provenance"])
        self.assertTrue(denial["arg_provenance"]["to"]["sources"])

    def test_a_real_arg_refs_still_takes_precedence_in_the_fold(self) -> None:
        """The documented path is unchanged: a producer that DOES mint refs gets
        them resolved against the session's tainted values."""
        from axor_core.contracts.taint import TaintSource
        from axor_core.taint.causal_root import CausalRoot

        state = GovernanceState()
        state.tainted_refs["v1"] = CausalRoot.external_read(TaintSource.WEB)
        root = _derive_driving_root(
            {"arg_refs": {"to": "v1"}, "driving_root": {"sources": [], "sensitive": False}},
            state, KernelConfig(),
        )
        self.assertIn(TaintSource.WEB, root.sources)


class TestBothPathsStillRecordAlike(unittest.TestCase):
    def test_the_denial_still_names_its_gate_and_kind(self) -> None:
        """The rest of the payload is unaffected by moving the breakdown."""
        governor = ToolCallGovernor(
            untrusted_sources={"read"}, egress_sinks={"send"},
            driving_args={"send": ["to"]},
        )
        approved = governor.evaluate("read", {})
        governor.register_output(approved, {"description": TAINTED})
        governor.evaluate("send", {"to": TAINTED})
        denial = governor.trace_events[-1]
        self.assertEqual(denial.kind, TraceEventKind.INTENT_DENIED)
        # whichever gate fires first, the category must be one `gate_of` knows —
        # not a guess at WHICH gate, which is the kernel's business
        from axor_core.governor import DENIAL_CATEGORIES

        self.assertIn(denial.payload["category"], DENIAL_CATEGORIES)
        self.assertEqual(denial.payload["driving_args"], ["to"])


if __name__ == "__main__":
    unittest.main()
