"""
Adversarial tests: warm-up attack — Step 23.

An adversary builds trust with benign behaviour before introducing a privileged
action.  Three delay variants: same-task, unrelated-task, child-agent.

In all cases taint — once set — persists across the delay because sticky=True.

Reverse-verify invariant: if TaintEngine.sticky were disabled these tests would
fail because taint would not survive the delay.
"""
from __future__ import annotations

from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.taint.engine import TaintEngine


# ── same-task delayed attack ───────────────────────────────────────────────────

def test_warmup_same_task_taint_persists():
    """
    Same-task delay: adversary warms up with benign intents, reads external
    content (taints session), continues with benign intents — taint persists.
    """
    engine = TaintEngine(node_id="node-adv")

    # Warm-up phase: 20 benign intents before any external read
    for _ in range(20):
        engine.tick_intent()

    # External read — taints session
    engine.propagate(TaintSource.WEB, TaintScope.SESSION)
    assert engine.state.is_tainted

    # Continued benign intents (adversary hopes delay washes taint)
    for _ in range(10):
        engine.tick_intent()

    # Taint must persist
    assert engine.state.is_tainted
    assert engine.state.sticky is True
    assert TaintSource.WEB in engine.state.sources


# ── unrelated-task delayed attack ─────────────────────────────────────────────

def test_warmup_unrelated_task_taint_persists():
    """
    Unrelated-task delay: intents use different tool categories before and after
    the taint-producing external read — taint persists regardless of tool variety.
    """
    engine = TaintEngine(node_id="node-adv")

    # 15 diverse benign intents (simulated as ticks)
    for _ in range(15):
        engine.tick_intent()

    # External read from a different source (MCP, not WEB)
    engine.propagate(TaintSource.MCP, TaintScope.SESSION)
    assert engine.state.is_tainted

    # 15 further "unrelated" intents
    for _ in range(15):
        engine.tick_intent()

    # Taint from MCP persists regardless of subsequent tool mix
    assert engine.state.is_tainted
    assert TaintSource.MCP in engine.state.sources
    assert engine.state.sticky is True
    assert engine.state.intent_age == 15  # age since taint, not total intents


# ── child-agent delayed execution ─────────────────────────────────────────────

def test_warmup_child_agent_taint_persists():
    """
    Child-agent delay: taint set in parent, passed to child via inherit_from_parent;
    child then executes 30 intents — inherited taint must persist throughout.
    """
    parent = TaintEngine(node_id="parent")
    parent.propagate(TaintSource.WEB, TaintScope.SESSION)

    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)

    # Child executes 30 intents after inheriting taint
    for _ in range(30):
        child.tick_intent()

    # Inherited taint must persist — not cleared by child's intent history
    assert child.state.is_tainted
    assert child.state.parent_inherited is True
    assert child.state.sticky is True
    assert TaintSource.WEB in child.state.sources
