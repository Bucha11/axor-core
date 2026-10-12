"""E4: does O5 hold under races and faults? A microbenchmark of the grant spend path.

    python benchmarks/grant_spend.py            # prints a JSON report
    python benchmarks/grant_spend.py --quick

Three parts, all in-process (the store is the one the daemon would host; the lock
here stands in for the daemon's serialization):

  race    T threads fire the same covered call at a grant with `uses` = k.
          Pass: effects == k exactly, and no spend key was handed to the handler
          twice (the handler dedups, so a re-run attempt is the failure signal).
  faults  every call's handler may die before the effect, die after it, or hang
          with an outcome the handler cannot report. Reservations are then
          reconciled against the handler's log. Pass: no key is re-run, effects
          never exceed uses, and every reservation whose outcome is knowable ends
          COMMITTED or RELEASED ("lost" counts the ones left RESERVED — by design
          only those whose outcome nobody can report).
  latency p50/p99 of reserve+commit next to ToolCallGovernor.evaluate() on the
          same call.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from axor_core.capability.grant_store import LockedGrantStore, NotRun, execute_granted  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402
from axor_core.kernel.grants import GrantMinter, SpendState  # noqa: E402
from axor_core.kernel.handles import HandleTable  # noqa: E402

GA = {"send_money": {"recipient": "iban", "amount": "money"}}
IBAN = "GB29NWBK60161331926819"
CALL = {"recipient": IBAN, "amount": 100}


class Ledger:
    """A handler whose effects are counted per key, with injectable faults."""

    def __init__(self, faults: dict[str, float] | None = None, seed: int = 0) -> None:
        self.runs: dict[str, int] = {}
        self.lock = threading.Lock()
        self.faults = faults or {}
        self.rng = random.Random(seed)
        self.silent: set[str] = set()
        self.reruns = 0  # a key handed to run() again after it already ran

    def _roll(self) -> str:
        with self.lock:
            r = self.rng.random()
        acc = 0.0
        for kind, p in self.faults.items():
            acc += p
            if r < acc:
                return kind
        return "ok"

    def run(self, args, key):
        f = self._roll()
        if f == "before":
            raise TimeoutError("killed before the effect")
        if f == "refuse":
            raise NotRun("refused before the effect")
        with self.lock:
            if key in self.runs:  # dedup: an idempotent handler never re-runs a key
                self.reruns += 1
                return "dup"
            self.runs[key] = 1
        if f == "after":
            raise TimeoutError("killed after the effect")
        if f == "silent":
            with self.lock:
                self.silent.add(key)
            raise TimeoutError("outcome unknown")
        return "ok"

    def executed(self, key):
        with self.lock:
            if key in self.silent:
                return None
            return key in self.runs


def _store(uses: int) -> tuple[LockedGrantStore, HandleTable]:
    store, table = LockedGrantStore(), HandleTable()
    GrantMinter(store.inner, table, GA, default_uses=uses).from_request(f"send 100 to {IBAN}")
    return store, table


def _fire(store, table, handler, threads: int, calls_each: int):
    barrier = threading.Barrier(threads)
    outs: list = []
    lk = threading.Lock()

    def worker(i):
        barrier.wait()
        for j in range(calls_each):
            o = execute_granted(store, table, "send_money", CALL, handler)
            with lk:
                outs.append(o)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return outs


def race(trials: int, threads: int, uses: int) -> dict:
    bad = 0
    for s in range(trials):
        store, table = _store(uses)
        h = Ledger(seed=s)
        _fire(store, table, h, threads, 4)
        effects = sum(h.runs.values())
        if effects != uses or h.reruns:
            bad += 1
    return {"trials": trials, "threads": threads, "uses": uses, "violations": bad}


def faults(trials: int, threads: int, uses: int, p: dict[str, float]) -> dict:
    double = over = lost = lost_knowable = calls = committed = released = 0
    for s in range(trials):
        store, table = _store(uses)
        h = Ledger(p, seed=1000 + s)
        outs = _fire(store, table, h, threads, 4)
        calls += len(outs)
        double += h.reruns
        if sum(h.runs.values()) > uses:
            over += 1
        for r in store.inner.reservations():
            if r.state is SpendState.RESERVED:
                lost += 1
                if r.key not in h.silent:
                    lost_knowable += 1
            elif r.state is SpendState.COMMITTED:
                committed += 1
            else:
                released += 1
    return {"trials": trials, "threads": threads, "uses": uses, "fault_p": p, "calls": calls,
            "same_key_rerun_attempts": double, "effects_over_uses": over,
            "committed": committed, "released": released,
            "lost_unknowable": lost - lost_knowable, "lost_knowable": lost_knowable}


def latency(n: int) -> dict:
    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    spend, ev = [], []
    store, table = _store(n + 10)
    g = ToolCallGovernor(grant_args=GA, egress_sinks={"send_money"},
                         driving_args={"send_money": ["recipient"]}, trace=False,
                         grant_store=store, handle_table=table)
    for _ in range(n):
        t = time.perf_counter_ns()
        g.evaluate("send_money", CALL)
        ev.append(time.perf_counter_ns() - t)
        t = time.perf_counter_ns()
        r = store.reserve("send_money", CALL, table)
        store.commit(r.key)
        spend.append(time.perf_counter_ns() - t)
    us = lambda v: round(v / 1000, 1)  # noqa: E731
    return {"n": n,
            "evaluate_us": {"p50": us(pct(ev, .5)), "p99": us(pct(ev, .99)), "mean": us(statistics.mean(ev))},
            "reserve_commit_us": {"p50": us(pct(spend, .5)), "p99": us(pct(spend, .99)),
                                  "mean": us(statistics.mean(spend))}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    t = 20 if a.quick else 200
    report = {
        "race": [race(t, th, u) for th, u in ((8, 1), (32, 1), (32, 5))],
        "faults": [
            faults(t, 16, 3, {"before": .1, "after": .1}),
            faults(t, 16, 3, {"before": .1, "after": .1, "silent": .05, "refuse": .05}),
        ],
        "latency": latency(2000 if a.quick else 20000),
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
