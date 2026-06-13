"""Tests for governance metrics aggregation."""

from __future__ import annotations

from axor_core.contracts.trace import (
    DegradationTransitionEvent,
    IntentDeniedEvent,
    SourceQuarantinedEvent,
    TaintPropagatedEvent,
    TraceEventKind,
)
from axor_core.trace.metrics import GovernanceMetrics


def _events():
    return [
        IntentDeniedEvent(kind=TraceEventKind.INTENT_DENIED, node_id="n", sequence=0,
                          intent_kind="tool_call", reason="bash not allowed"),
        IntentDeniedEvent(kind=TraceEventKind.INTENT_DENIED, node_id="n", sequence=1,
                          intent_kind="tool_call", reason="path outside allowlist"),
        TaintPropagatedEvent(kind=TraceEventKind.TAINT_PROPAGATED, node_id="n", sequence=2,
                             taint_source="web", taint_scope="session"),
        TaintPropagatedEvent(kind=TraceEventKind.TAINT_PROPAGATED, node_id="n", sequence=3,
                             taint_source="file", taint_scope="session"),
        DegradationTransitionEvent(kind=TraceEventKind.DEGRADATION_TRANSITION, node_id="n",
                                   sequence=4, previous_level="NORMAL", new_level="RESTRICTED",
                                   trigger_source_id="s", trigger_intent="bash", reason="x"),
        SourceQuarantinedEvent(kind=TraceEventKind.SOURCE_QUARANTINED, node_id="n", sequence=5,
                               source_id="s", quarantined_at=0.0, reason="pressure"),
    ]


def test_counts_aggregate_correctly():
    m = GovernanceMetrics.from_events(_events())
    assert m.denials == 2
    assert m.taint_propagations_by_source["web"] == 1
    assert m.taint_propagations_by_source["file"] == 1
    assert m.degradation_transitions_by_level["RESTRICTED"] == 1
    assert m.sources_quarantined == 1


def test_prometheus_exposition_format():
    text = GovernanceMetrics.from_events(_events()).to_prometheus()
    assert "axor_governance_denials_total 2" in text
    assert 'axor_governance_taint_propagations_total{source="web"} 1' in text
    assert 'axor_governance_degradation_transitions_total{level="RESTRICTED"} 1' in text
    # Well-formed: every metric line has a HELP and TYPE header.
    assert text.count("# TYPE") >= 5


def test_empty_events():
    m = GovernanceMetrics.from_events([])
    assert m.denials == 0
    assert m.to_prometheus().count("denials_total") >= 1
