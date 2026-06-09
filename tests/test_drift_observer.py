"""probe behavioral-drift observer — v4.12: probe is a WATCHER (non-enforcing)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.drift import BehavioralDriftObserver
from axor_core.node.drift_observer import TaintEngineDriftObserver
from axor_core.taint.engine import TaintEngine
from axor_core.worker.session import GovernedSession


async def test_drift_does_not_taint_enforcement_state():
    engine = TaintEngine(node_id="n")
    observer = TaintEngineDriftObserver(engine)
    await observer.on_drift("s", "a", "elevated_review")
    await observer.on_drift("s", "a", "restricted_mode")
    assert engine.state.is_tainted is False  # watcher: untouched


async def test_drift_observer_runs_without_engine():
    await TaintEngineDriftObserver().on_drift("s", "a", "elevated_review")  # no raise


def test_observer_satisfies_protocol():
    assert isinstance(TaintEngineDriftObserver(TaintEngine(node_id="n")), BehavioralDriftObserver)


def _session(observer):
    cap = MagicMock(spec=CapabilityExecutor)
    cap.register_post_callback = MagicMock()
    return GovernedSession(executor=MagicMock(), capability_executor=cap, behavioral_drift_observer=observer)


async def test_session_calls_observer_on_notify():
    observer = TaintEngineDriftObserver(TaintEngine(node_id="s"))
    observer.on_drift = AsyncMock()  # type: ignore[method-assign]
    await _session(observer).notify_behavioral_drift(agent_id="a", action="restricted_mode")
    observer.on_drift.assert_awaited_once()


async def test_session_no_observer_is_noop():
    cap = MagicMock(spec=CapabilityExecutor)
    cap.register_post_callback = MagicMock()
    s = GovernedSession(executor=MagicMock(), capability_executor=cap)
    await s.notify_behavioral_drift(agent_id="a", action="restricted_mode")  # no raise


async def test_session_observer_failure_is_swallowed():
    observer = TaintEngineDriftObserver()
    observer.on_drift = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    await _session(observer).notify_behavioral_drift(agent_id="a", action="restricted_mode")  # no raise
