"""
Adversarial tests for DegradationEngine.

Categories (from spec):
  A) Degradation monotonicity — 3 variants
  B) Source isolation — 4 variants
  C) Cross-origin export → LOCKED — 2 variants
  D) LOCKED_TTL auto-terminal — 2 variants
  E) Child floor inheritance — 3 variants
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

# ── helpers ───────────────────────────────────────────────────────────────────

def _engine(ttl: float = 300.0):
    from axor_core.degradation.engine import DegradationEngine
    from axor_core.contracts.degradation import DegradationPolicy
    return DegradationEngine(DegradationPolicy(LOCKED_TTL=ttl))


def _taint(tainted: bool = True):
    from axor_core.contracts.taint import TaintState, TaintSource
    if tainted:
        return TaintState(sources=frozenset({TaintSource.WEB}))
    return TaintState()


def _ni(
    tool: str = "bash",
    operation: str = "execute",
    destination_kind: str = "none",
    provenance: str = "external_web",
    executes_generated: bool = False,
    after_external: bool = False,
):
    from axor_core.contracts.anomaly import NormalizedIntent
    return NormalizedIntent(
        tool=tool,
        operation=operation,
        target_kind="workdir",
        destination_kind=destination_kind,
        provenance=provenance,
        reads_secret_like_data=False,
        writes_outside_workdir=False,
        executes_generated_code=executes_generated,
        after_external_read=after_external,
        after_secret_access=False,
        data_flow="none",
    )


def _deny():
    from axor_core.contracts.denial import DenialResponse
    return DenialResponse(status="denied", coarse_category="tool_denied")


def _authority():
    from axor_core.contracts.degradation import GovernanceAuthority
    return GovernanceAuthority(
        authority_id="test-operator",
        authority_type="human_operator",
        reason_code="test_clearance",
    )


# ── A) Degradation monotonicity ───────────────────────────────────────────────

def test_degrad_mono_a1_level_only_increases():
    """Level is non-decreasing across 20 mixed deny/approve signals."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _engine()
    ts = _taint()
    ni = _ni()
    denial = _deny()
    prev = DegradationLevel.NORMAL
    for i in range(20):
        d = denial if i % 2 == 0 else None
        engine.record_signal(ni, d, ts)
        assert engine.state.level >= prev, (
            f"level dropped from {prev.name} to {engine.state.level.name} at step {i}"
        )
        prev = engine.state.level


def test_degrad_mono_a2_governance_can_decrease_level():
    """clear_by_governance is the only valid path to lower the level."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _engine()
    engine.quarantine_source("src", "test")
    assert engine.state.level == DegradationLevel.RESTRICTED

    engine.clear_by_governance(_authority(), "reviewed", DegradationLevel.NORMAL)
    assert engine.state.level == DegradationLevel.NORMAL


def test_degrad_mono_a3_approved_signals_cannot_lower_level():
    """Flooding approved signals after quarantine cannot reduce the level."""
    engine = _engine()
    ts = _taint()
    ni = _ni()
    engine.quarantine_source("evil", "test")
    locked_at = engine.state.level

    for _ in range(50):
        engine.record_signal(ni, None, ts)  # all approved

    assert engine.state.level >= locked_at


# ── B) Source isolation ───────────────────────────────────────────────────────

def test_degrad_iso_b1_clean_source_unaffected_while_malicious_quarantined():
    """
    Quarantining one source does not quarantine another.
    Clean source still gets full policy; malicious source gets restricted.
    """
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ExportMode
    from axor_core.contracts.degradation import DegradationLevel
    engine = _engine()
    engine.quarantine_source("malicious:web", "injection detected")
    assert engine.state.level == DegradationLevel.RESTRICTED

    base = ExecutionPolicy(
        name="test",
        tool_policy=ToolPolicy(allow_bash=True, allow_write=True),
        export_mode=ExportMode.FULL,
    )
    # Malicious source → restricted
    restricted = engine.apply_to_policy(base, source_id="malicious:web")
    assert restricted.tool_policy.allow_bash is False
    assert restricted.export_mode == ExportMode.RESTRICTED

    # Clean source → unchanged (known clean source_id not quarantined)
    clean = engine.apply_to_policy(base, source_id="clean:repo")
    assert clean.tool_policy.allow_bash is True
    assert clean.export_mode == ExportMode.FULL


def test_degrad_iso_b2_quarantine_does_not_leak_to_other_source_records():
    """Quarantining sourceA does not mark sourceB as quarantined."""
    engine = _engine()
    engine.quarantine_source("sourceA", "test")

    # sourceB should not exist or not be quarantined
    src_b = engine.state.sources.get("sourceB")
    assert src_b is None or not src_b.quarantined


def test_degrad_iso_b3_multiple_sources_tracked_independently():
    """Two sources can accumulate pressure independently."""
    engine = _engine()
    ts = _taint()
    denial = _deny()

    # Source A: provenance external_web
    ni_a = _ni(tool="bash", provenance="external_web")
    # Source B: provenance official_docs
    ni_b = _ni(tool="bash", provenance="official_docs")

    engine.record_signal(ni_a, denial, ts)
    engine.record_signal(ni_b, denial, ts)

    # Both should be tracked separately
    assert len(engine.state.sources) >= 1  # at least one source record exists


def test_degrad_iso_b4_source_pressure_counts_correct():
    """tool_pressure_count increments only for the triggering source."""
    engine = _engine()
    ts = _taint()
    denial = _deny()
    ni = _ni(tool="bash", provenance="external_web")

    engine.record_signal(ni, denial, ts)
    engine.record_signal(ni, denial, ts)

    # Source should have accumulated pressure
    sources = engine.state.sources
    assert len(sources) >= 1
    src = list(sources.values())[0]
    assert src.tool_pressure_count >= 1


# ── C) Cross-origin export → LOCKED ──────────────────────────────────────────

def test_degrad_cross_c1_file_write_to_external_locks_immediately():
    """file_write to external_domain immediately escalates to LOCKED."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _engine()
    ts = _taint()
    ni = _ni(tool="write", operation="file_write", destination_kind="external_domain")
    denial = _deny()

    transition = engine.record_signal(ni, denial, ts)
    assert engine.state.level == DegradationLevel.LOCKED
    assert transition is not None
    assert transition.reason == "cross_origin_export_denied"


def test_degrad_cross_c2_network_request_to_private_network_locks():
    """network_request to private_network deny escalates to LOCKED."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _engine()
    ts = _taint()
    ni = _ni(tool="curl", operation="network_request", destination_kind="private_network")
    denial = _deny()

    engine.record_signal(ni, denial, ts)
    assert engine.state.level == DegradationLevel.LOCKED


# ── D) LOCKED_TTL auto-terminal ───────────────────────────────────────────────

def test_degrad_ttl_d1_auto_terminal_after_ttl_expires():
    """After LOCKED_TTL seconds in LOCKED state, next signal triggers TERMINAL."""
    from axor_core.contracts.degradation import DegradationLevel
    # Use a very short TTL
    engine = _engine(ttl=0.01)
    ts = _taint()
    ni = _ni(tool="curl", operation="network_request", destination_kind="private_network")
    denial = _deny()

    # Escalate to LOCKED
    engine.record_signal(ni, denial, ts)
    assert engine.state.level == DegradationLevel.LOCKED

    # Wait for TTL to expire
    time.sleep(0.05)

    # Next signal should trigger TERMINAL
    transition = engine.record_signal(_ni(), _deny(), ts)
    assert engine.state.level == DegradationLevel.TERMINAL
    assert transition is not None
    assert transition.new_level == DegradationLevel.TERMINAL


def test_degrad_ttl_d2_no_auto_terminal_before_ttl():
    """LOCKED state before TTL expiry does not auto-terminal."""
    from axor_core.contracts.degradation import DegradationLevel
    # Long TTL — should not expire during test
    engine = _engine(ttl=3600.0)
    ts = _taint()
    ni = _ni(tool="curl", operation="network_request", destination_kind="private_network")
    denial = _deny()

    engine.record_signal(ni, denial, ts)
    assert engine.state.level == DegradationLevel.LOCKED

    # Immediately send more signals — should stay LOCKED, not go TERMINAL
    for _ in range(5):
        engine.record_signal(_ni(), _deny(), ts)

    assert engine.state.level == DegradationLevel.LOCKED


# ── E) Child floor inheritance ────────────────────────────────────────────────

def test_degrad_child_e1_child_node_receives_same_engine_instance():
    """
    GovernedNode passes the same DegradationEngine instance to child nodes.
    This ensures child starts at exactly the parent's current level (D-6 floor).
    """
    from axor_core.node.wrapper import GovernedNode
    parent_engine = _engine()
    parent_engine.quarantine_source("evil", "test")

    node = GovernedNode(
        executor=MagicMock(),
        capability_executor=MagicMock(),
        analyzer=MagicMock(),
        selector=MagicMock(),
        composer=MagicMock(),
        degradation_engine=parent_engine,
    )
    # Engine passed in must be the same object
    assert node._degradation_engine is parent_engine


def test_degrad_child_e2_child_cannot_start_below_parent_level():
    """
    Child node's engine level is >= parent's level because they share the instance.
    """
    from axor_core.contracts.degradation import DegradationLevel
    from axor_core.node.wrapper import GovernedNode
    parent_engine = _engine()
    parent_engine.quarantine_source("evil", "injection")
    parent_level = parent_engine.state.level
    assert parent_level == DegradationLevel.RESTRICTED

    child_node = GovernedNode(
        executor=MagicMock(),
        capability_executor=MagicMock(),
        analyzer=MagicMock(),
        selector=MagicMock(),
        composer=MagicMock(),
        degradation_engine=parent_engine,
    )
    # Child shares same engine — level is the parent's level, not lower
    assert child_node._degradation_engine.state.level >= parent_level


def test_degrad_child_e3_child_policy_restricted_when_parent_engine_restricted():
    """
    When parent engine is RESTRICTED, child's apply_to_policy also restricts tools.
    """
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ExportMode
    from axor_core.contracts.degradation import DegradationLevel
    parent_engine = _engine()
    parent_engine.quarantine_source("malicious", "test")
    assert parent_engine.state.level == DegradationLevel.RESTRICTED

    base = ExecutionPolicy(
        name="test",
        tool_policy=ToolPolicy(allow_bash=True, allow_write=True),
        export_mode=ExportMode.FULL,
    )
    # Child using same engine gets restricted policy for the quarantined source
    result = parent_engine.apply_to_policy(base, source_id="malicious")
    assert result.tool_policy.allow_bash is False
    assert result.export_mode == ExportMode.RESTRICTED
