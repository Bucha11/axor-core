"""Latency of task-scoped grants: match (find a covering grant) and charge
(reserve + commit), standalone and inside ToolCallGovernor.evaluate on a grant
sink. Same method as latency_gate.py: CPython, one thread, p50/p99 over 40 000
calls after 2 000 warm-up calls.

    PYTHONPATH=. python benchmarks/latency_grants.py
"""
from __future__ import annotations

import platform
import time

from axor_core.capability.grant_store import LockedGrantStore
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.grants import GrantMinter, GrantStore
from axor_core.kernel.handles import HandleTable

IBAN = "GB29NWBK60161331926819"
GA = {"send_money": {"recipient": "iban", "amount": "money"},
      "send_email": {"recipients": "email", "cc": "email?", "bcc": "email?"}}
CALL = {"recipient": IBAN, "amount": 1800, "subject": "rent"}
PROMPT = f"pay rent 1800 to {IBAN}; email bob@x.io and carol@y.io the receipt"


def bench(fn, n=40000, warm=2000):
    for _ in range(warm):
        fn()
    s = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        s.append(time.perf_counter_ns() - t0)
    s.sort()
    return s[len(s) // 2] / 1000, s[int(0.99 * len(s))] / 1000


def main() -> None:
    print(f"host: {platform.processor() or 'x86_64'} | CPython {platform.python_version()} (1 thread)")
    store, table = GrantStore(), HandleTable()
    GrantMinter(store, table, GA, default_uses=None).from_request(PROMPT)
    print(f"grants in store: {len(store.grants())}")
    p = bench(lambda: store.find("send_money", CALL, table))
    print(f"  match (find)                p50={p[0]:6.1f}  p99={p[1]:6.1f}  µs")
    locked = LockedGrantStore(store)

    def charge():
        r = locked.reserve("send_money", CALL, table)
        locked.commit(r.key)
    p = bench(charge)
    print(f"  charge (reserve+commit)     p50={p[0]:6.1f}  p99={p[1]:6.1f}  µs")
    for trace in (True, False):
        g = ToolCallGovernor(egress_sinks={"send_money"}, driving_args={"send_money": ["recipient"]},
                             integrity_default="context", integrity_origins="request-only",
                             grant_args=GA, trace=trace)
        g.register_task(PROMPT)
        g.grants.default_uses = None
        g.grants.from_request(PROMPT)
        p = bench(lambda: g.evaluate("send_money", CALL))
        print(f"  evaluate on a grant sink trace={trace!s:5}  p50={p[0]:6.1f}  p99={p[1]:6.1f}  µs")

        def full():
            d = g.evaluate("send_money", CALL)
            g.grant_store.commit(g.spend(d, CALL))
        p = bench(full)
        print(f"  evaluate + charge trace={trace!s:5}      p50={p[0]:6.1f}  p99={p[1]:6.1f}  µs")


if __name__ == "__main__":
    main()
