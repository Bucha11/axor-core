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


def test_declared_peer_edge_passes_the_send_gate() -> None:
    events: list[Event] = []
    bus = InMemoryMessageBus(emit=events.append,
                             peer_declared=lambda p: p == "partner")
    eng = TaintEngine(node_id="a")
    bus.register("a", eng)
    bus.register("partner", TaintEngine(node_id="partner"))
    bus.send(make_envelope(eng, "a", "partner", "peer", "hello"))
    sent = [e for e in events if e.kind is EventKind.MESSAGE_SENT][-1]
    assert sent.verdict is Verdict.PASS


# ── inter-federation inbound: the ladder in the live path (spec v2 Ch.1 §2) ───


def _peer_bus(level: str, *, key: bytes = b"fed-key-32-bytes-long-xxxxxxxxxx"):
    from axor_core.federation.ladder import L2, PeerDeclaration

    events: list[Event] = []
    decl = PeerDeclaration(
        peer_id="partner", level=level,
        verifier=HmacSigner(shared_key=key),
        discount_classes=frozenset({"default"}) if level == L2 else frozenset(),
    )
    bus = InMemoryMessageBus(
        emit=events.append,
        peer_declaration_for=lambda p: decl if p == "partner" else None,
    )
    ours = TaintEngine(node_id="ours")
    theirs = TaintEngine(node_id="partner")
    bus.register("ours", ours)
    bus.register("partner", theirs)
    return bus, ours, theirs, events


def test_peer_inbound_l1_remints_despite_clean_claim() -> None:
    """The peer asserts 'clean'; at L1 that is attribution, not trust — the
    value registers TAINTED locally (labels are claims across keysets)."""
    from axor_core.federation.ladder import L1

    bus, ours, theirs, events = _peer_bus(L1)
    value = "their report"
    env = make_envelope(theirs, "partner", "ours", "peer", value)  # clean at sender
    bus.send(env)
    assert ours.derive_value(value).is_tainted
    received = [e for e in events if e.kind is EventKind.MESSAGE_RECEIVED][-1]
    # the foreign root survives as an opaque forensic ref; the local root is minted
    assert received.payload["foreign_root"]["sources"] == []
    assert received.payload["local_root"]["sources"] != []


def test_peer_inbound_l2_signed_assertion_discounts_never_clean() -> None:
    from axor_core.federation.ladder import L2

    signer = HmacSigner(shared_key=b"fed-key-32-bytes-long-xxxxxxxxxx")
    bus, ours, theirs, events = _peer_bus(L2)
    value = "their web summary"
    theirs.register_value(value, CausalRoot.cross_process_in())
    env = make_envelope(theirs, "partner", "ours", "peer", value, signer=signer)
    bus.send(env)
    root = ours.derive_value(value)
    assert root.is_tainted  # discounted is still tainted — never to clean
    received = [e for e in events if e.kind is EventKind.MESSAGE_RECEIVED][-1]
    assert "l2_discount_applied" in received.payload["ladder_evidence"]


def test_peer_inbound_forged_assertion_falls_to_l0_evidenced() -> None:
    from axor_core.federation.ladder import L2

    bus, ours, theirs, events = _peer_bus(L2)
    value = "forged-labels value"
    env = make_envelope(theirs, "partner", "ours", "peer", value,
                        signer=HmacSigner(shared_key=b"WRONG-key-32-bytes-xxxxxxxxxxxxxx"))
    bus.send(env)
    assert ours.derive_value(value).is_tainted  # L0 handling: full re-mint
    received = [e for e in events if e.kind is EventKind.MESSAGE_RECEIVED][-1]
    assert any("assertion_forged_fell_to_l0" in e
               for e in received.payload["ladder_evidence"])
