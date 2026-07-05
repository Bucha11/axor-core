"""Trace bridge: axor-core TraceEvents -> kernel Events -> replay fold.

One trace, two consumers (spec 12.0 point 4): denials become kernel DENIAL
events with the right gate category, degradation transitions become facts that
drive the recompute, and the translated stream folds through the kernel replay
that the platform's scrubber/regression use.
"""
from __future__ import annotations

from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.trace import (
    DegradationTransitionEvent,
    IntentDeniedEvent,
    TokensSpentEvent,
    TraceEventKind,
)
from axor_core.kernel.events import EventKind, Verdict
from axor_core.kernel.replay import replay
from axor_core.plane.bridge import trace_event_to_kernel, trace_to_kernel


def test_intent_denied_maps_to_kernel_denial_with_category() -> None:
    ev = IntentDeniedEvent(
        kind=TraceEventKind.INTENT_DENIED, node_id="n0", sequence=3,
        intent_kind="tool_call",
        reason="taint enforcement (per-value): tainted value into export",
    )
    kernel = trace_event_to_kernel(ev)
    assert kernel is not None
    assert kernel.kind is EventKind.DENIAL
    assert kernel.verdict is Verdict.DENY
    assert kernel.gate == "taint_enforcement"


def test_degradation_transition_becomes_a_fact_severity() -> None:
    ev = DegradationTransitionEvent(
        kind=TraceEventKind.DEGRADATION_TRANSITION, node_id="n0", sequence=5,
        previous_level=DegradationLevel.NORMAL, new_level=DegradationLevel.RESTRICTED,
        trigger_source_id="src_1", trigger_intent="bash", reason="tainted exec",
    )
    kernel = trace_event_to_kernel(ev)
    assert kernel is not None
    assert kernel.kind is EventKind.FACT
    assert kernel.payload["severity"] == int(DegradationLevel.RESTRICTED)
    assert kernel.causal_root == "src_1"


def test_cosmetic_events_are_dropped() -> None:
    ev = TokensSpentEvent(
        kind=TraceEventKind.TOKENS_SPENT, node_id="n0", sequence=1,
        input_tokens=10, output_tokens=5,
    )
    assert trace_event_to_kernel(ev) is None


def test_translated_trace_folds_and_recomputes_level() -> None:
    trace = [
        IntentDeniedEvent(
            kind=TraceEventKind.INTENT_DENIED, node_id="n0", sequence=0,
            intent_kind="tool_call", reason="carrier gate: imperative channel",
        ),
        DegradationTransitionEvent(
            kind=TraceEventKind.DEGRADATION_TRANSITION, node_id="n0", sequence=1,
            previous_level=DegradationLevel.NORMAL,
            new_level=DegradationLevel.RESTRICTED,
            trigger_source_id="src_1", trigger_intent="bash", reason="tainted",
        ),
    ]
    events = trace_to_kernel(trace)
    assert len(events) == 2
    result = replay(events)  # scrubber-mode fold
    # the degradation fact drives the recompute to RESTRICTED
    assert result.steps[-1].state.level is DegradationLevel.RESTRICTED
