"""
Adversarial tests: burstfire attack — Step 23.

An adversary inserts N benign intents between an external read (which taints the
session) and a privileged action, hoping taint will decay.  With sticky=True these
tests confirm taint does not decay regardless of N.

Reverse-verify invariant: if TaintEngine.sticky were disabled (tick_intent cleared
sources when sticky=False), all three tests would fail.
"""
from __future__ import annotations

from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.taint.engine import TaintEngine


# ── helpers ───────────────────────────────────────────────────────────────────

def _tainted_engine() -> TaintEngine:
    """Session tainted by a web read."""
    engine = TaintEngine(node_id="node-adv")
    engine.propagate(TaintSource.WEB, TaintScope.SESSION)
    return engine


def _assert_privileged_action_blocked(engine: TaintEngine) -> None:
    """
    Simulate the governance check that runs before any privileged action.

    In production, IntentLoop checks engine.state.is_tainted and escalates or
    denies.  Here we assert the precondition the policy would reject.
    """
    assert engine.state.is_tainted, (
        "taint decayed — privileged action would not be blocked"
    )
    assert engine.state.sticky is True, (
        "sticky cleared — governance cannot rely on taint persistence"
    )


# ── burstfire N=1 ─────────────────────────────────────────────────────────────

def test_burstfire_n1_taint_survives():
    """
    1 benign intent between external read and privileged action.

    Taint must remain active; privileged action must be blocked.
    """
    engine = _tainted_engine()
    assert engine.state.is_tainted

    # Adversary injects 1 benign intent
    engine.tick_intent()

    _assert_privileged_action_blocked(engine)
    assert TaintSource.WEB in engine.state.sources


# ── burstfire N=10 ────────────────────────────────────────────────────────────

def test_burstfire_n10_taint_survives():
    """
    10 benign intents between external read and privileged action.

    Taint must remain; intent_age must be tracked (showing N intents elapsed).
    """
    engine = _tainted_engine()

    for _ in range(10):
        engine.tick_intent()

    _assert_privileged_action_blocked(engine)
    assert engine.state.intent_age == 10
    assert TaintSource.WEB in engine.state.sources


# ── burstfire N=50 ────────────────────────────────────────────────────────────

def test_burstfire_n50_taint_survives():
    """
    50 benign intents between external read and privileged action.

    Even a large burst cannot wash taint; intent_age is 50.
    """
    engine = _tainted_engine()

    for _ in range(50):
        engine.tick_intent()

    _assert_privileged_action_blocked(engine)
    assert engine.state.intent_age == 50
    assert TaintSource.WEB in engine.state.sources
