"""Opt-in detection that can tighten degradation but never loosen it.

Detection is observe-only by default. When an operator registers a reputation
threshold, a reading at or below it (but above zero) is a clear crossing fact that
may tighten the degradation level; it never lowers the level or turns a deny into
an allow. Isolation between tenants is structural: each session has its own
engine."""

from __future__ import annotations

import pytest

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.trace import TraceEventKind
from axor_core.degradation.engine import DegradationEngine

pytestmark = pytest.mark.adversarial


def _ni(resource_rep=0.0, container_rep=0.0, tool="bash"):
    return NormalizedIntent(
        tool=tool, operation="execute", target_kind="workdir",
        destination_kind="none", provenance="user",
        reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=False,
        after_secret_access=False, data_flow="none",
        target_resource_reputation=resource_rep,
        target_container_reputation=container_rep,
    )


def test_detection_off_by_default_is_noop():
    eng = DegradationEngine()  # no detection_floor
    t = eng.record_detection(_ni(resource_rep=0.01))
    assert t is None
    assert eng.state.level == DegradationLevel.NORMAL
    assert eng.drain_events() == []


def test_crossing_tightens_and_emits_event():
    eng = DegradationEngine(detection_floor=0.3)
    t = eng.record_detection(_ni(resource_rep=0.1))   # 0.1 at/below floor 0.3 → crossing
    assert t is not None
    assert eng.state.level == DegradationLevel.RESTRICTED
    evs = eng.drain_events()
    sig = [e for e in evs if e.kind == TraceEventKind.DETECTION_SIGNAL]
    assert sig and sig[0].verdict == "crossing" and sig[0].fed_degradation is True


def test_above_threshold_does_not_tighten():
    eng = DegradationEngine(detection_floor=0.3)
    t = eng.record_detection(_ni(resource_rep=0.9))   # 0.9 above floor 0.3 → no crossing
    assert t is None
    assert eng.state.level == DegradationLevel.NORMAL
    sig = [e for e in eng.drain_events() if e.kind == TraceEventKind.DETECTION_SIGNAL]
    assert sig and sig[0].verdict == "flagged" and sig[0].fed_degradation is False


def test_unknown_reputation_zero_never_crosses():
    # 0.0 is 'unknown', not a low score — must not trigger a false crossing.
    eng = DegradationEngine(detection_floor=0.3)
    t = eng.record_detection(_ni(resource_rep=0.0, container_rep=0.0))
    assert t is None
    assert eng.state.level == DegradationLevel.NORMAL


def test_worst_of_resource_and_container_used():
    eng = DegradationEngine(detection_floor=0.3)
    # resource ok (0.8) but container bad (0.05) → crossing on the worst.
    t = eng.record_detection(_ni(resource_rep=0.8, container_rep=0.05))
    assert t is not None
    assert eng.state.level == DegradationLevel.RESTRICTED


def test_per_tenant_isolation():
    # Poisoning tenant A's reputation must not tighten tenant B (separate engines).
    a = DegradationEngine(detection_floor=0.3, node_id="tenantA")
    b = DegradationEngine(detection_floor=0.3, node_id="tenantB")
    a.record_detection(_ni(resource_rep=0.01))
    assert a.state.level == DegradationLevel.RESTRICTED
    assert b.state.level == DegradationLevel.NORMAL


def test_detection_only_tightens_never_loosens():
    eng = DegradationEngine(detection_floor=0.3)
    eng.record_detection(_ni(resource_rep=0.1))       # → RESTRICTED
    eng.drain_events()
    # a subsequent good-reputation reading must NOT lower the level (monotone).
    eng.record_detection(_ni(resource_rep=0.95))
    assert eng.state.level == DegradationLevel.RESTRICTED
