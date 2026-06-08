"""DensityMeter math + the per_value ⊆ session_sticky invariant (TM3.3)."""

from __future__ import annotations

from axor_core.taint.density import DensityMeter


def test_empty_report_is_zero():
    r = DensityMeter().report()
    assert r.high_stakes_firings == 0
    assert r.session_sticky_density == 0.0
    assert r.per_value_density == 0.0
    assert r.gap == 0.0


def test_densities_and_gap():
    m = DensityMeter()
    # 4 firings: 2 sticky-only, 1 sticky+value, 1 clean
    m.record("execute_generated_code", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=True)
    m.record("file_write", session_tainted=False, value_tainted=False)
    r = m.report()
    assert r.high_stakes_firings == 4
    assert r.session_sticky_tainted == 3
    assert r.per_value_tainted == 1
    assert r.session_sticky_density == 0.75
    assert r.per_value_density == 0.25
    assert r.gap == 0.5


def test_value_tainted_implies_session_tainted():
    # A tainted driving value implies a prior external read -> session tainted.
    # The meter enforces this so gap can never go negative.
    m = DensityMeter()
    m.record("execute_generated_code", session_tainted=False, value_tainted=True)
    r = m.report()
    assert r.session_sticky_tainted == 1
    assert r.per_value_tainted == 1
    assert r.gap == 0.0


def test_per_value_never_exceeds_session_sticky():
    m = DensityMeter()
    for st, vt in [(True, False), (True, True), (False, False), (False, True)]:
        m.record("file_write", session_tainted=st, value_tainted=vt)
    r = m.report()
    assert r.per_value_density <= r.session_sticky_density
    assert r.gap >= 0.0


def test_by_operation_breakdown():
    m = DensityMeter()
    m.record("file_write", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=True)
    r = m.report()
    assert r.by_operation["file_write"] == (1, 1, 0)
    assert r.by_operation["execute_generated_code"] == (1, 1, 1)
    assert "GAP" in r.render()
