"""Runtime messaging: envelopes with labels, the in-memory bus, signing,
spawn-level inheritance, and the node_stale death path.

Spec v2 Ch.4 — orchestration around the pure kernel, zero governance logic in
the orchestration; Ch.1 §1 — transport is irrelevant to semantics.
"""
from __future__ import annotations

from typing import AsyncIterator

import pytest
from axor_core import GovernedSession, presets
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceConfig, TraceEventKind
from axor_core.degradation.engine import DegradationEngine
from axor_core.federation.signing import HmacSigner
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.node.messaging import (
    InMemoryMessageBus,
    MessageDenied,
    MessageRejected,
    make_envelope,
)
from axor_core.node.spawn import inherit_degradation
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine
from tests.conftest import EchoExecutor


def _bus_with(*nodes: str, **kw) -> tuple[InMemoryMessageBus, dict[str, TaintEngine], list[Event]]:
    events: list[Event] = []
    bus = InMemoryMessageBus(emit=events.append, **kw)
    engines = {}
    for n in nodes:
        engines[n] = TaintEngine(node_id=n)
        bus.register(n, engines[n])
    return bus, engines, events


def test_carried_not_laundered() -> None:
    """A tainted value crossing a lateral edge arrives still tainted at the
    sibling — one hop through a sibling washes nothing."""
    bus, eng, events = _bus_with("a", "b")
    value = "fabricated result"
    eng["a"].register_value(value, CausalRoot.cross_process_in())

    env = make_envelope(eng["a"], "a", "b", "lateral", value)
    bus.send(env)

    assert eng["b"].derive_value(value).is_tainted
    kinds = [(e.node_id, e.kind, e.verdict) for e in events]
    assert ("a", EventKind.MESSAGE_SENT, Verdict.PASS) in kinds
    assert ("b", EventKind.MESSAGE_RECEIVED, Verdict.PASS) in kinds


def test_clean_value_stays_clean_intra() -> None:
    """The envelope never edits labels: a clean value crosses clean (per-value
    fidelity is the intra win — no blanket re-mint inside one federation)."""
    bus, eng, _ = _bus_with("a", "b")
    value = "constant text"
    env = make_envelope(eng["a"], "a", "b", "lateral", value)
    bus.send(env)
    root = eng["b"].derive_value(value)
    assert not root.is_tainted and not root.sensitive


def test_sensitive_value_arms_receiver_floor() -> None:
    bus, eng, _ = _bus_with("a", "b")
    secret = "API_KEY_123456789"
    eng["a"].register_value(secret, CausalRoot(sources=frozenset(), sensitive=True))
    bus.send(make_envelope(eng["a"], "a", "b", "lateral", secret))
    assert eng["b"].confidentiality_floor_active() is True


def test_locked_sender_denied_and_traced() -> None:
    bus, eng, events = _bus_with("a", "b")
    bus.register("a", eng["a"], level_of=lambda: DegradationLevel.LOCKED)
    with pytest.raises(MessageDenied):
        bus.send(make_envelope(eng["a"], "a", "b", "lateral", "x"))
    deny = [e for e in events if e.kind is EventKind.MESSAGE_SENT][-1]
    assert deny.verdict is Verdict.DENY and deny.gate == "degradation"
    assert bus.inbox("b") == []  # containment at the source: never delivered


def test_undeclared_peer_edge_denied() -> None:
    bus, eng, _ = _bus_with("a", "b")
    with pytest.raises(MessageDenied):
        bus.send(make_envelope(eng["a"], "a", "b", "peer", "x"))


def test_cycle_cannot_launder() -> None:
    """A→B→A: after the round trip the origin still sees the taint (monotone
    re-fold by causal-root identity, not hop count)."""
    bus, eng, _ = _bus_with("a", "b")
    value = "poisoned doc"
    eng["a"].register_value(value, CausalRoot.cross_process_in())
    bus.send(make_envelope(eng["a"], "a", "b", "lateral", value))
    bus.send(make_envelope(eng["b"], "b", "a", "lateral", value))
    assert eng["a"].derive_value(value).is_tainted


def test_signed_envelope_verifies_and_tamper_rejects() -> None:
    signer = HmacSigner(shared_key=b"fed-key-32-bytes-long-xxxxxxxxxx")
    bus, eng, _ = _bus_with(
        "a", "b", verifier_for=lambda peer: HmacSigner(shared_key=b"fed-key-32-bytes-long-xxxxxxxxxx")
    )
    env = make_envelope(eng["a"], "a", "b", "lateral", "v", signer=signer)
    bus.send(env)  # valid signature → delivered
    assert len(bus.inbox("b")) == 1

    forged = make_envelope(eng["a"], "a", "b", "lateral", "v2",
                           signer=HmacSigner(shared_key=b"wrong-key-32-bytes-long-xxxxxxxxx"))
    with pytest.raises(MessageRejected):
        bus.send(forged)


# ── spawn inheritance (spec v2 Ch.4 §3) ────────────────────────────────────────


def test_child_inherits_parent_level_narrow_or_preserve() -> None:
    parent = DegradationEngine(node_id="p")
    parent.tighten(DegradationLevel.CAUTIOUS, reason="test", trigger_intent="t")
    child = inherit_degradation(parent, "c")
    assert child.state.level is DegradationLevel.CAUTIOUS
    assert child is not parent  # own engine — no shared governance state


def test_child_of_normal_parent_starts_normal() -> None:
    parent = DegradationEngine(node_id="p")
    child = inherit_degradation(parent, "c")
    assert child.state.level is DegradationLevel.NORMAL


def test_child_engine_is_independent_after_spawn() -> None:
    parent = DegradationEngine(node_id="p")
    child = inherit_degradation(parent, "c")
    child.tighten(DegradationLevel.RESTRICTED, reason="child fault",
                  trigger_intent="t")
    assert parent.state.level is DegradationLevel.NORMAL  # no upward bleed


# ── death: node_stale (spec v2 Ch.4 §4) ────────────────────────────────────────


class _CrashingChild(Invokable):
    async def stream(self, envelope) -> AsyncIterator[ExecutorEvent]:
        raise RuntimeError("child process disappeared")
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_child_crash_is_a_fact_not_a_clean_return() -> None:
    parent = EchoExecutor(tool_calls=[("spawn_child", {"task": "doomed"})])
    sess = GovernedSession(
        executor=parent,
        capability_executor=__import__(
            "axor_core.capability.executor", fromlist=["CapabilityExecutor"]
        ).CapabilityExecutor(),
        child_executor=_CrashingChild(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    result = await sess.run("delegate", policy=presets.get("federated"))
    # The parent survived; the child's absence is recorded, not silently clean.
    stale = [
        e
        for trace in sess.all_traces()
        for e in getattr(trace, "events", [])
        if getattr(e, "kind", None) is TraceEventKind.CHILD_STALE
    ]
    assert stale, "child crash must leave a CHILD_STALE trace event"
    assert sess._degradation_engine.state.level >= DegradationLevel.CAUTIOUS
    assert "node_stale" in (result.output or "")
