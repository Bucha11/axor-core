"""
Tests for taint inheritance in child node spawning — Step 19.

Verifies:
- Parent tainted → child inherits taint via inherit_from_parent()
- Isolation boundary does NOT clear parent taint
- Clean parent → child starts clean
"""
from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.taint.engine import TaintEngine


def _tainted_engine() -> TaintEngine:
    engine = TaintEngine(node_id="parent")
    engine.propagate(TaintSource.WEB, TaintScope.SESSION)
    return engine


def _clean_engine() -> TaintEngine:
    return TaintEngine(node_id="parent")


# ── inherit_from_parent() propagation ──────────────────────────────────────────

def test_child_inherits_taint_from_parent():
    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert child.state.is_tainted


def test_child_inherits_taint_source():
    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert TaintSource.WEB in child.state.sources


def test_child_inherits_taint_scope():
    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert child.state.scope == TaintScope.SESSION


def test_child_taint_marked_parent_inherited():
    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert child.state.parent_inherited


# ── isolation boundary does NOT clear parent taint ────────────────────────────

def test_isolation_does_not_clear_inherited_taint():
    """
    Even if a child node has isolation=True conceptually, inheriting taint
    from a tainted parent means the child is tainted from the start.
    The child cannot clear taint inherited from parent via worker path.
    """
    from axor_core.errors.exceptions import TaintClearanceError

    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    # worker cannot clear inherited taint
    with pytest.raises(TaintClearanceError):
        child.attempt_clear_by_worker()
    assert child.state.is_tainted


# ── clean parent → clean child ────────────────────────────────────────────────

def test_clean_parent_child_starts_clean():
    parent = _clean_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    # Inheriting clean state keeps child clean
    assert not child.state.is_tainted


# ── multiple taint sources inherited ──────────────────────────────────────────

def test_child_inherits_multiple_taint_sources():
    parent = TaintEngine(node_id="parent")
    parent.propagate(TaintSource.WEB, TaintScope.SESSION)
    parent.propagate(TaintSource.MCP, TaintScope.SESSION)

    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert TaintSource.WEB in child.state.sources
    assert TaintSource.MCP in child.state.sources
