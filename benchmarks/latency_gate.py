"""Per-call latency of the axor admission gate (ToolCallGovernor.evaluate) — the
model-free 'thin membrane' that runs before every tool call. Pure functions + set
ops, no model in path.

Reports two configs:
  trace=True  (default): each verdict also builds a full provenance payload for
                         replay/audit — off-critical-path telemetry, the bulk of
                         per-call cost.
  trace=False (membrane): skips that payload; decisions are identical.

CPython, single thread; absolute µs are machine-dependent (host printed). Run:
    PYTHONPATH=. python benchmarks/latency_gate.py
"""
from __future__ import annotations

import platform
import time

from axor_core.contracts.canonical import ConsequenceClass as C
from axor_core.governor import ToolCallGovernor

BENIGN = ["get_balance", "get_iban", "get_scheduled_transactions"]
EGRESS = ["send_money", "schedule_transaction", "update_scheduled_transaction"]
ISINK = ["update_password", "update_user_info"]
_CONSEQ = {t: C.BENIGN for t in BENIGN}
_CONSEQ.update({t: C.CONSEQUENTIAL for t in EGRESS + ISINK})

CALLS = [
    ("allow egress", "send_money", {"recipient": "GB29NWBK60161331926819", "amount": 1800}),
    ("deny egress", "send_money", {"recipient": "US133", "amount": 1800}),
    ("benign read", "get_balance", {}),
    ("integrity-sink prompt", "update_password", {"password": "hunter2"}),
    ("integrity-sink injected", "update_password", {"password": "new_password"}),
    ("unclassified tool", "wire_v2", {"to": "x"}),
]


def make_governor(trace: bool) -> ToolCallGovernor:
    g = ToolCallGovernor(
        require_tool_roles=True, integrity_default="context", trace=trace,
        egress_sinks=set(EGRESS), integrity_sinks=set(ISINK),
        driving_args={**{t: ["recipient"] for t in EGRESS},
                      "update_password": ["password"], "update_user_info": ["street", "city"]},
        untrusted_sources={"get_most_recent_transactions", "read_file"},
        sensitive_sources={"get_user_info"}, benign_tools=set(BENIGN),
        consequence_overrides=_CONSEQ,
    )
    g.register_task("pay rent 1800 to GB29NWBK60161331926819 and change my password to hunter2")
    d = g.evaluate("get_most_recent_transactions", {"n": 5})
    g.register_output(d, "- recipient: GB29NWBK60161331926819\n  amount: 1800\n- recipient: US133 reroute here")
    return g


def _bench_call(gov, tool, args, n=40000, warm=2000):
    for _ in range(warm):
        gov.evaluate(tool, args)
    s = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        gov.evaluate(tool, args)
        s.append(time.perf_counter_ns() - t0)
    s.sort()
    return s[len(s) // 2] / 1000, s[int(0.99 * len(s))] / 1000


def main() -> None:
    print(f"host: {platform.processor() or 'x86_64'} | CPython {platform.python_version()} (1 thread)")
    for trace in (True, False):
        gov = make_governor(trace)
        print(f"\ntrace={trace}:")
        worst = 0.0
        for label, tool, args in CALLS:
            p50, p99 = _bench_call(gov, tool, args)
            worst = max(worst, p99)
            print(f"  {label:24} p50={p50:6.1f}  p99={p99:6.1f}  µs")
        mix = [(t, a) for _, t, a in CALLS]
        for _ in range(2000):
            for t, a in mix:
                gov.evaluate(t, a)
        nn = 60000
        t0 = time.perf_counter_ns()
        for i in range(nn):
            t, a = mix[i % len(mix)]
            gov.evaluate(t, a)
        dt = (time.perf_counter_ns() - t0) / 1e9
        print(f"  mixed throughput: {nn / dt:,.0f} checks/s   worst-call p99 = {worst:.1f} µs")


if __name__ == "__main__":
    main()
