"""The premise of the check–dispatch lemma (docs/effect-spec-sketch.md §3a),
tested directly rather than through its consequences.

Premise: between the last evaluation of the mutable-state checks for a call and
the entry of that call's handler, control never returns to the event loop. If it
did, another coroutine could change the budget, the secret floor or the approval
tables between the check and the effect.

Instrument: a ticker task that increments a counter every time the event loop
gets control. A coroutine that does not suspend cannot let it run, so the counter
read at the check and read again at handler entry are equal iff no suspension
happened in between. "Last check" is the capability/lease evaluation at the start
of the cascade for a call that never waits, and the post-approval re-check for a
call that waited on an approver.
"""
from __future__ import annotations

import asyncio
import contextvars
from unittest import mock

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import axor_core.node.intent_loop as il
from axor_core.capability.executor import CapabilityExecutor
import tests.spec.test_loop_refines_effect_spec as H

_last_check: contextvars.ContextVar = contextvars.ContextVar("last_check", default=None)


class _Ticker:
    def __init__(self) -> None:
        self.n = 0
        self._stop = False

    async def run(self) -> None:
        while not self._stop:
            self.n += 1
            await asyncio.sleep(0)

    def stop(self) -> None:
        self._stop = True


async def _observe(program, kw, extra_patch=None):
    ticker = _Ticker()
    violations: list[str] = []
    orig_eval = il.IntentLoop._evaluate_tool_intent
    orig_recheck = il.IntentLoop._recheck_volatile_state
    orig_exec = H._Recorder.execute

    def evaluate(self, intent, envelope):
        _last_check.set(("evaluate", ticker.n))
        return orig_eval(self, intent, envelope)

    def recheck(self, *a, **k):
        _last_check.set(("recheck", ticker.n))
        return orig_recheck(self, *a, **k)

    async def handler_entry(self, args):
        mark = _last_check.get()
        if mark is not None and mark[1] != ticker.n:
            violations.append(
                f"{self.name}: event loop ran {ticker.n - mark[1]} time(s) between the "
                f"last check ({mark[0]}) and handler entry"
            )
        return await orig_exec(self, args)

    patches = [
        mock.patch.object(il.IntentLoop, "_evaluate_tool_intent", evaluate),
        mock.patch.object(il.IntentLoop, "_recheck_volatile_state", recheck),
        mock.patch.object(H._Recorder, "execute", handler_entry),
    ]
    if extra_patch is not None:
        patches.append(extra_patch)
    for p in patches:
        p.start()
    try:
        tick = asyncio.ensure_future(ticker.run())
        await asyncio.sleep(0)
        await H._run(program, **kw)
        ticker.stop()
        await tick
    finally:
        for p in reversed(patches):
            p.stop()
    return violations


_cases = dict(
    program=st.lists(H._step, max_size=10),
    send_in_rights=st.booleans(),
    call_approval=st.booleans(),
    budget=st.one_of(st.none(), st.integers(0, 6)),
    answers=st.lists(st.sampled_from([True, True, False]), min_size=1, max_size=6),
    ticks=st.lists(st.integers(0, 8), min_size=1, max_size=6),
)


@settings(max_examples=300, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(**_cases)
def test_no_suspension_between_last_check_and_handler_entry(
    program, send_in_rights, call_approval, budget, answers, ticks
):
    kw = dict(send_in_rights=send_in_rights, call_approval=call_approval,
              budget=budget, answers=answers, doc_text="pay bob", ticks=ticks)
    assert asyncio.run(_observe(program, kw)) == []


def test_the_instrument_sees_a_suspension_in_the_critical_segment():
    """Not vacuous: an `await asyncio.sleep(0)` inserted between the dispatch
    decision and the handler (here: at executor entry) is reported."""
    orig = CapabilityExecutor.execute

    async def yielding_execute(self, intent, caps):
        await asyncio.sleep(0)
        return await orig(self, intent, caps)

    kw = dict(send_in_rights=True, call_approval=False, budget=None,
              answers=[True], doc_text="", ticks=[0])
    program = [("call", H.READ_DOC, {})]
    clean = asyncio.run(_observe(program, kw))
    broken = asyncio.run(_observe(
        program, kw, mock.patch.object(CapabilityExecutor, "execute", yielding_execute)
    ))
    assert clean == []
    assert broken and "between the last check" in broken[0]
