from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.errors.exceptions import TaintClearanceError
from axor_core.taint.engine import TaintEngine


def test_initial_state_not_tainted():
    engine = TaintEngine()
    assert not engine.state.is_tainted


def test_propagate_adds_source():
    engine = TaintEngine()
    state = engine.propagate(TaintSource.WEB)
    assert TaintSource.WEB in state.sources
    assert state.is_tainted


def test_propagate_accumulates_sources():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB)
    state = engine.propagate(TaintSource.MCP)
    assert TaintSource.WEB in state.sources
    assert TaintSource.MCP in state.sources


def test_sticky_by_default():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB)
    assert engine.state.sticky is True


def test_taint_persists_across_many_intents():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB, TaintScope.SESSION)
    for _ in range(50):
        engine.tick_intent()
    assert engine.state.is_tainted
    assert engine.state.intent_age == 50


def test_worker_cannot_clear_taint():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB)
    with pytest.raises(TaintClearanceError):
        engine.attempt_clear_by_worker()
    # Taint must still be active
    assert engine.state.is_tainted


def test_governance_clear_removes_taint():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB)
    state = engine.clear_by_governance(
        authority="operator",
        authority_type="human_operator",
        reason_code="session_reset",
        authorized_by_principal_id="op-1",
        audit_id="audit-123",
    )
    assert not state.is_tainted


def test_governance_clear_records_audit_fields():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB)
    engine.clear_by_governance(
        authority="operator",
        authority_type="human_operator",
        reason_code="session_reset",
        authorized_by_principal_id="op-1",
        audit_id="audit-xyz",
    )
    history = engine.state.clearance_history
    assert len(history) == 1
    record = history[0]
    assert record.cleared_by == "operator"
    assert record.authority_type == "human_operator"
    assert record.reason_code == "session_reset"
    assert record.authorized_by_principal_id == "op-1"
    assert record.audit_id == "audit-xyz"
    assert record.clearance_id.startswith("clr_")
    assert record.timestamp > 0


def test_scope_widens_to_broadest():
    engine = TaintEngine()
    engine.propagate(TaintSource.WEB, TaintScope.NODE)
    state = engine.propagate(TaintSource.MCP, TaintScope.SESSION)
    assert state.scope == TaintScope.SESSION


def test_child_inherits_parent_taint():
    parent_engine = TaintEngine()
    parent_engine.propagate(TaintSource.WEB, TaintScope.SESSION)

    child_engine = TaintEngine()
    state = child_engine.inherit_from_parent(parent_engine.state)
    assert state.is_tainted
    assert state.parent_inherited is True
    assert TaintSource.WEB in state.sources


def test_child_inherits_clean_parent_stays_clean():
    parent_engine = TaintEngine()
    child_engine = TaintEngine()
    state = child_engine.inherit_from_parent(parent_engine.state)
    assert not state.is_tainted
