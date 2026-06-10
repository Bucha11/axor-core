"""Phase 2 group 3 — M9 (budget register_node wired) and M11 (restrict_export
actually enforced). Only the genuinely-live inherited findings; M1/M2/C1/C2/C3/M7/M8
verified already-resolved and M3/M14 superseded by the v4.12 observe-only design."""

from __future__ import annotations

import logging

import pytest

from axor_core import GovernedSession, presets
from axor_core.budget.tracker import BudgetTracker
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.policy import ExportMode
from axor_core.contracts.trace import TraceConfig
from axor_core.node.wrapper import _more_restrictive_export
from tests.conftest import EchoExecutor

pytestmark = pytest.mark.adversarial


# ── M9: register_node ──────────────────────────────────────────────────────────

def test_register_node_idempotent_preserves_counters():
    # The risk in the fix: register before every record must NOT wipe accumulated
    # tokens. Re-registering updates metadata only.
    tr = BudgetTracker()
    tr.register_node("n1", parent_id=None, depth=0)
    tr.record("n1", input_tokens=100, output_tokens=50)
    tr.register_node("n1", parent_id=None, depth=0)   # idempotent re-register
    snap = tr.snapshot()["n1"]
    assert snap.input_tokens == 100 and snap.output_tokens == 50


def test_depth_and_subtree_accounting_correct():
    tr = BudgetTracker()
    tr.register_node("root", parent_id=None, depth=0)
    tr.record("root", input_tokens=10, output_tokens=0)
    tr.register_node("child", parent_id="root", depth=1)
    tr.record("child", input_tokens=5, output_tokens=0)
    assert tr.depth_tokens(0) == 10
    assert tr.depth_tokens(1) == 5
    assert "child" in tr._subtree_ids("root")


class _Read(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args) -> str:
        return "FILE CONTENT " * 40


def _cap() -> CapabilityExecutor:
    ex = CapabilityExecutor()
    ex.register(_Read())
    return ex


@pytest.mark.asyncio
async def test_session_run_registers_node_no_warning(caplog):
    # Wiring: the session path must register the node before recording, so the
    # tracker never falls back to its 'unregistered node' warn-and-default.
    ex = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
    sess = GovernedSession(
        executor=ex, capability_executor=_cap(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    with caplog.at_level(logging.WARNING):
        result = await sess.run("write a test for the auth module")
    assert all("unregistered node" not in r.message for r in caplog.records)
    # the node is registered at depth 0
    assert result.node_id in sess._tracker.snapshot()
    assert sess._tracker.snapshot()[result.node_id].depth == 0


# ── M11: restrict_export enforced ──────────────────────────────────────────────

def test_more_restrictive_export_ordering():
    F, S, FI, R = (ExportMode.FULL, ExportMode.SUMMARY,
                   ExportMode.FILTERED, ExportMode.RESTRICTED)
    assert _more_restrictive_export(F, S) == S        # narrow full -> summary
    assert _more_restrictive_export(R, S) == R        # never widen restricted
    assert _more_restrictive_export(FI, S) == FI      # filtered already tighter
    assert _more_restrictive_export(None, S) == S
    assert _more_restrictive_export(S, None) == S


@pytest.mark.asyncio
async def test_budget_restrict_export_narrows_full_export():
    # A FULL-export policy whose budget crosses restrict_export must actually narrow
    # to SUMMARY — previously the decision was only logged.
    async def _run(limit):
        ex = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
        sess = GovernedSession(
            executor=ex, capability_executor=_cap(),
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
            soft_token_limit=limit,
        )
        return await sess.run("research the auth module deeply",
                              policy=presets.get("research"))   # FULL export

    unrestricted = await _run(None)
    restricted = await _run(1)        # tiny budget → restrict_export fires
    # FULL leaves the full payload; the budget-restricted run is narrowed to the
    # SUMMARY shape (output-only).
    assert sorted(restricted.export_payload.keys()) == ["output"]
    assert sorted(unrestricted.export_payload.keys()) != ["output"]
