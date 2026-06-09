"""TaintEngine — per-value provenance (TM2 / ValueProvenance) + governance release.

Session-taint state was removed (dead in enforcement); the engine is per-value:
register a value's causal_root, derive it at a sink, release by governance
(endorse one value, or clear all).
"""
from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintSource
from axor_core.errors.exceptions import TaintClearanceError
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

V = "TAINTED_FRAGMENT_abcdef12"


def test_clean_by_default():
    assert TaintEngine().derive_value("anything at all").is_tainted is False


def test_register_then_derive_tainted():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    assert e.derive_value(f"prefix {V} suffix").is_tainted is True


def test_clean_value_stays_clean_even_with_taint_registered():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    assert e.derive_value("nothing tainted here").is_tainted is False


def test_worker_cannot_clear():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    with pytest.raises(TaintClearanceError):
        e.attempt_clear_by_worker()
    assert e.derive_value(V).is_tainted is True  # still tainted


def test_governance_clear_removes_all():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    e.clear_by_governance("operator", "human_operator", "reviewed")
    assert e.derive_value(V).is_tainted is False


def test_governance_clear_rejects_bad_authority():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    with pytest.raises(TaintClearanceError):
        e.clear_by_governance("", "worker", "")
    assert e.derive_value(V).is_tainted is True


def test_endorse_releases_one_value_under_governance():
    e = TaintEngine()
    e.register_value(V, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    removed = e.endorse_value(V, "operator", "human_operator", "ok")
    assert removed >= 1
    assert e.derive_value(V).is_tainted is False


def test_child_inherits_value_ledger():
    parent = TaintEngine()
    parent.register_value(V, CausalRoot.external_read(TaintSource.WEB))
    child = TaintEngine()
    assert child.derive_value(V).is_tainted is False
    child.inherit_value_ledger(parent)
    assert child.derive_value(V).is_tainted is True
