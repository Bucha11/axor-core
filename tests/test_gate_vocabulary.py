"""Every denial category the kernel can emit names a gate, and is recorded.

Two failures motivated this file, and both were silent.

1. Consumers kept private category→gate maps. axor-lab's was keyed on `ssrf`,
   `consequence`, `positional`, `carrier` — names this kernel does not emit (it
   emits `ssrf_gate`, `consequence_gate`, …). Seven of eleven categories fell
   through unmapped, and the raw category landed in a field that only accepts a
   gate name, producing traces that failed schema validation. Nothing caught it
   because no test ever denied for those reasons end-to-end.

2. The STRICT unclassified-tool denial returned directly instead of going
   through the recording path, so the kernel's most security-relevant verdict —
   an undeclared or renamed tool — left NO trace event. Replay saw an intent
   with no decision; a consumer never learned anything was blocked.

The guard against (1) recurring is that the mapping lives here and is checked
against the categories actually present in the source. The guard against (2) is
that a denial with no recorded event fails a test.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from axor_core.contracts.trace import TraceEventKind
from axor_core.governor import (
    DENIAL_CATEGORIES,
    GATE_OF_CATEGORY,
    ToolCallGovernor,
    gate_of,
)

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "axor_core"

# the nine-gate sequence a recorded verdict may name
KNOWN_GATES = frozenset({
    "capability", "consequence", "value_policies", "degradation", "ssrf",
    "positional", "carrier", "taint_floor", "adjudicator", "message", "budget",
})


def _categories_in_source() -> set[str]:
    """Every string literal passed as `category=` anywhere in the package.

    Read from the AST rather than by running the kernel: a category only some
    exotic configuration can reach is exactly the one a hand-written list
    forgets, and it is the one whose trace turns out to be invalid in
    production.
    """
    found: set[str] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg == "category" and isinstance(keyword.value, ast.Constant):
                    if isinstance(keyword.value.value, str):
                        found.add(keyword.value.value)
    return found


class TestTheVocabularyIsComplete(unittest.TestCase):
    def test_every_category_in_the_source_names_a_gate(self) -> None:
        missing = _categories_in_source() - DENIAL_CATEGORIES
        self.assertEqual(
            missing, set(),
            f"category/categories {sorted(missing)} can be emitted but name no gate — "
            f"a consumer writing them into a trace would produce an invalid one",
        )

    def test_every_gate_named_is_a_real_gate(self) -> None:
        unknown = set(GATE_OF_CATEGORY.values()) - KNOWN_GATES
        self.assertEqual(unknown, set(), f"not gates: {sorted(unknown)}")

    def test_an_unknown_category_raises_rather_than_passing_through(self) -> None:
        """Passing the raw category through is what turned an unrecognised
        denial into an invalid trace instead of a loud error."""
        with self.assertRaises(ValueError) as ctx:
            gate_of("something_new")
        self.assertIn("unknown denial category", str(ctx.exception))

    def test_known_categories_map(self) -> None:
        self.assertEqual(gate_of("taint_enforcement"), "taint_floor")
        self.assertEqual(gate_of("unclassified_tool"), "capability")


class TestEveryDenialIsRecorded(unittest.TestCase):
    def test_the_unclassified_tool_denial_emits_its_event(self) -> None:
        """STRICT mode's refusal of an undeclared tool is the kernel's
        fail-closed default. It returned without recording, so the one verdict
        most worth auditing was the one that left no evidence."""
        governor = ToolCallGovernor(
            untrusted_sources={"read"}, egress_sinks={"send"},
            driving_args={"send": {"to"}}, require_tool_roles=True,
        )
        decision = governor.evaluate("mystery_tool", {})
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "unclassified_tool")
        self.assertEqual(
            [e.kind for e in governor.trace_events], [TraceEventKind.INTENT_DENIED],
        )

    def test_that_event_carries_its_category_and_reason(self) -> None:
        governor = ToolCallGovernor(
            untrusted_sources={"read"}, egress_sinks={"send"},
            driving_args={"send": {"to"}}, require_tool_roles=True,
        )
        governor.evaluate("mystery_tool", {})
        event = governor.trace_events[-1]
        self.assertEqual(event.payload["category"], "unclassified_tool")
        self.assertIn("no declared data-flow role", event.reason)  # type: ignore[attr-defined]
        self.assertEqual(gate_of(event.payload["category"]), "capability")

    def test_a_denied_call_never_leaves_the_trace_empty(self) -> None:
        """The general invariant: whatever the kernel refuses, it records."""
        governor = ToolCallGovernor(
            untrusted_sources={"read"}, egress_sinks={"send"},
            driving_args={"send": {"to"}}, require_tool_roles=True,
        )
        for tool, args in (("mystery_tool", {}), ("send", {"to": "x"})):
            with self.subTest(tool=tool):
                before = len(governor.trace_events)
                governor.evaluate(tool, args)
                self.assertGreater(len(governor.trace_events), before)


if __name__ == "__main__":
    unittest.main()
