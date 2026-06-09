"""Per-value provenance crosses the parent→child spawn boundary (v4.12):
the value-ledger is inherited, the session-taint flag is NOT."""

from __future__ import annotations

import inspect

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine
from axor_core.taint.ledger import ValueTaintLedger

SECRET = "SECRET_TOKEN_abcdef123456"


def test_ledger_merge_folds_parent_fragments():
    parent = ValueTaintLedger()
    parent.register(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    child = ValueTaintLedger()
    assert child.derive(SECRET).is_tainted is False
    child.merge(parent)
    r = child.derive(f"x={SECRET}")
    assert r.is_tainted and r.sensitive


def test_child_engine_inherits_per_value_provenance():
    parent = TaintEngine(node_id="parent")
    parent.register_value(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    child = TaintEngine(node_id="child")
    assert child.derive_value(SECRET).is_tainted is False
    child.inherit_value_ledger(parent)
    assert child.derive_value(f"leak {SECRET}").sensitive is True


def test_inheritance_is_additive_not_shared_upward():
    parent = TaintEngine(node_id="parent")
    parent.register_value(SECRET, CausalRoot.external_read(TaintSource.WEB))
    child = TaintEngine(node_id="child")
    child.inherit_value_ledger(parent)
    child.register_value("CHILD_ONLY_fragment_xyz", CausalRoot.external_read(TaintSource.WEB))
    assert parent.derive_value("CHILD_ONLY_fragment_xyz").is_tainted is False


def test_spawn_inherits_ledger_not_session_taint():
    from axor_core.node.wrapper import GovernedNode
    src = inspect.getsource(GovernedNode._handle_spawn_child)
    assert "inherit_value_ledger" in src
    assert "inherit_from_parent" not in src
    assert "consequence_overrides=self._consequence_overrides" in src
