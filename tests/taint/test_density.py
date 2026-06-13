"""DensityMeter math, per-axis split, and honest (non-masking) recording."""

from __future__ import annotations

from axor_core.taint.density import DensityMeter


def test_empty_report_is_zero():
    r = DensityMeter().report()
    assert r.high_stakes_firings == 0
    assert r.integrity.session_sticky_density == 0.0
    assert r.integrity.per_value_density == 0.0
    assert r.integrity.gap == 0.0
    assert r.sensitivity.gap == 0.0


def test_integrity_densities_and_gap():
    m = DensityMeter()
    # 4 firings: 2 sticky-only, 1 sticky+value, 1 clean
    m.record("execute_generated_code", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=True)
    m.record("file_write", session_tainted=False, value_tainted=False)
    r = m.report()
    assert r.high_stakes_firings == 4
    assert r.integrity.session_sticky_tainted == 3
    assert r.integrity.per_value_tainted == 1
    assert r.integrity.session_sticky_density == 0.75
    assert r.integrity.per_value_density == 0.25
    assert r.integrity.gap == 0.5
    assert r.integrity.invariant_violations == 0


def test_confidentiality_axis_is_independent():
    m = DensityMeter()
    # An integrity-tainted but non-sensitive firing, plus a sensitive one.
    m.record("send", session_tainted=True, value_tainted=True,
             session_sensitive=False, value_sensitive=False)
    m.record("send", session_tainted=True, value_tainted=True,
             session_sensitive=True, value_sensitive=True)
    r = m.report()
    assert r.integrity.per_value_tainted == 2
    assert r.sensitivity.per_value_tainted == 1
    assert r.sensitivity.session_sticky_tainted == 1


def test_violation_is_counted_not_masked():
    # value tainted but session not — a measurement bug. The meter must record
    # it as a violation, NOT silently rewrite session to True.
    m = DensityMeter()
    m.record("execute_generated_code", session_tainted=False, value_tainted=True)
    r = m.report()
    assert r.integrity.session_sticky_tainted == 0       # NOT rewritten to 1
    assert r.integrity.per_value_tainted == 1
    assert r.integrity.invariant_violations == 1
    # gap is allowed to be negative — that is the signal the measurement is off.
    assert r.integrity.gap < 0
    assert "INVARIANT VIOLATIONS" in r.render()


def test_no_violation_when_invariant_holds():
    m = DensityMeter()
    for st, vt in [(True, False), (True, True), (False, False)]:
        m.record("file_write", session_tainted=st, value_tainted=vt)
    r = m.report()
    assert r.integrity.invariant_violations == 0
    assert r.integrity.gap >= 0.0


def test_by_operation_breakdown():
    m = DensityMeter()
    m.record("file_write", session_tainted=True, value_tainted=False)
    m.record("execute_generated_code", session_tainted=True, value_tainted=True,
             session_sensitive=True, value_sensitive=True)
    r = m.report()
    # (firings, sess_integ, val_integ, sess_sens, val_sens)
    assert r.by_operation["file_write"] == (1, 1, 0, 0, 0)
    assert r.by_operation["execute_generated_code"] == (1, 1, 1, 1, 1)
    assert "GAP" in r.render()
