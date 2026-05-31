"""
Tests for behavioral drift observer wiring.

Covers:
  - TaintEngineDriftObserver propagates correct TaintScope per action
  - GovernedSession.notify_behavioral_drift() calls observer
  - GovernedSession works without observer (observer=None)
  - Observer failure is swallowed and logged — does not propagate
"""
from __future__ import annotations

import pytest

from axor_core.contracts.drift import BehavioralDriftObserver
from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.node.drift_observer import TaintEngineDriftObserver
from axor_core.taint.engine import TaintEngine


# ── TaintEngineDriftObserver ──────────────────────────────────────────────────

async def test_elevated_review_propagates_taint() -> None:
    engine = TaintEngine(node_id="test-node")
    observer = TaintEngineDriftObserver(engine)

    await observer.on_drift("sess-1", "agent-1", "elevated_review")

    assert engine.state.is_tainted
    assert TaintSource.UNKNOWN_EXTERNAL in engine.state.sources


async def test_restricted_mode_propagates_taint() -> None:
    engine = TaintEngine(node_id="test-node")
    observer = TaintEngineDriftObserver(engine)

    await observer.on_drift("sess-1", "agent-1", "restricted_mode")

    assert engine.state.is_tainted
    assert TaintSource.UNKNOWN_EXTERNAL in engine.state.sources


async def test_both_actions_produce_session_scope() -> None:
    # TaintState default scope is SESSION; propagate(SESSION) → SESSION regardless of action.
    for action in ("elevated_review", "restricted_mode"):
        engine = TaintEngine(node_id="test-node")
        observer = TaintEngineDriftObserver(engine)
        await observer.on_drift("sess-1", "agent-1", action)
        assert engine.state.scope == TaintScope.SESSION


async def test_repeated_drift_calls_remain_tainted() -> None:
    engine = TaintEngine(node_id="test-node")
    observer = TaintEngineDriftObserver(engine)

    await observer.on_drift("sess-1", "agent-1", "elevated_review")
    await observer.on_drift("sess-1", "agent-1", "restricted_mode")

    assert engine.state.is_tainted
    assert engine.state.scope == TaintScope.SESSION


# ── BehavioralDriftObserver Protocol structural check ─────────────────────────

def test_taint_engine_observer_satisfies_protocol() -> None:
    engine = TaintEngine(node_id="test-node")
    observer = TaintEngineDriftObserver(engine)
    assert isinstance(observer, BehavioralDriftObserver)


# ── GovernedSession.notify_behavioral_drift ───────────────────────────────────

async def test_governed_session_calls_observer_on_notify() -> None:
    import sys
    sys.path.insert(0, "axor-core")
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.worker.session import GovernedSession
    from unittest.mock import AsyncMock, MagicMock

    taint_engine = TaintEngine(node_id="sess")
    observer = TaintEngineDriftObserver(taint_engine)

    mock_executor = MagicMock()
    mock_cap_executor = MagicMock(spec=CapabilityExecutor)
    mock_cap_executor.register_post_callback = MagicMock()

    session = GovernedSession(
        executor=mock_executor,
        capability_executor=mock_cap_executor,
        behavioral_drift_observer=observer,
    )

    await session.notify_behavioral_drift(agent_id="agent-1", action="restricted_mode")

    assert taint_engine.state.is_tainted
    assert taint_engine.state.scope == TaintScope.SESSION


async def test_governed_session_no_observer_is_noop() -> None:
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.worker.session import GovernedSession
    from unittest.mock import MagicMock

    mock_executor = MagicMock()
    mock_cap_executor = MagicMock(spec=CapabilityExecutor)
    mock_cap_executor.register_post_callback = MagicMock()

    session = GovernedSession(
        executor=mock_executor,
        capability_executor=mock_cap_executor,
        # no behavioral_drift_observer
    )

    # Must not raise
    await session.notify_behavioral_drift(agent_id="agent-1", action="restricted_mode")


async def test_governed_session_observer_failure_is_swallowed() -> None:
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.worker.session import GovernedSession
    from unittest.mock import AsyncMock, MagicMock

    broken_observer = MagicMock(spec=BehavioralDriftObserver)
    broken_observer.on_drift = AsyncMock(side_effect=RuntimeError("observer crashed"))

    mock_executor = MagicMock()
    mock_cap_executor = MagicMock(spec=CapabilityExecutor)
    mock_cap_executor.register_post_callback = MagicMock()

    session = GovernedSession(
        executor=mock_executor,
        capability_executor=mock_cap_executor,
        behavioral_drift_observer=broken_observer,
    )

    # Must not raise — observer failures are swallowed
    await session.notify_behavioral_drift(agent_id="agent-1", action="restricted_mode")
    broken_observer.on_drift.assert_called_once()
