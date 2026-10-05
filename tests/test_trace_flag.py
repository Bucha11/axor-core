"""trace flag on ToolCallGovernor: off skips trace-event building (the bulk of
per-call latency — a full provenance payload for replay/audit), decisions
unchanged. Default stays on so the control-plane / replay path is unaffected."""
from __future__ import annotations

from axor_core import ToolCallGovernor
from axor_core.contracts.canonical import ConsequenceClass as C


def _gov(trace):
    g = ToolCallGovernor(
        require_tool_roles=True, integrity_default="context", trace=trace,
        egress_sinks={"send_money"}, integrity_sinks=set(),
        driving_args={"send_money": ["recipient"]},
        untrusted_sources={"read_txns"}, sensitive_sources=set(),
        benign_tools={"get_balance", "read_txns"},
        consequence_overrides={"send_money": C.CONSEQUENTIAL,
                               "get_balance": C.BENIGN, "read_txns": C.BENIGN},
    )
    g.register_task("pay 1800 to GB29NWBK60161331926819")
    d = g.evaluate("read_txns", {})
    g.register_output(d, "- recipient: GB29NWBK60161331926819")
    return g


CALLS = [
    ("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 1800}),  # allow
    ("send_money", {"recipient": "US133", "amount": 1800}),                   # deny
    ("get_balance", {}),                                                      # benign
    ("wire_v2", {"to": "x"}),                                                 # unclassified deny
]


def test_decisions_identical_with_trace_on_or_off():
    on, off = _gov(True), _gov(False)
    for tool, args in CALLS:
        a, b = on.evaluate(tool, args), off.evaluate(tool, args)
        assert (a.allowed, a.category) == (b.allowed, b.category), (tool, a, b)


def test_trace_off_emits_no_events():
    g = _gov(False)
    for tool, args in CALLS:
        g.evaluate(tool, args)
    g.register_output(g.evaluate("read_txns", {}), "- recipient: GB29NWBK60161331926819")
    assert g.trace_events == []


def test_trace_on_is_default_and_records():
    g = _gov(True)
    for tool, args in CALLS:
        g.evaluate(tool, args)
    assert len(g.trace_events) >= len(CALLS)  # one verdict event per call (+ the warm-up read)
    # default (no trace kwarg) is on
    assert ToolCallGovernor()._trace is True
