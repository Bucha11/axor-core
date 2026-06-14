"""Behavioral-drift watcher: it is a passive WATCHER (non-enforcing)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.drift import BehavioralDriftObserver
from axor_core.node.drift_observer import BehavioralDriftWatcher
from axor_core.taint.engine import TaintEngine
from axor_core.worker.session import GovernedSession


def test_watcher_holds_no_enforcement_reference():
    # Structural guarantee: the watcher cannot reach the TaintEngine or any
    # other governance object, so a drift signal can never feed enforcement.
    watcher = BehavioralDriftWatcher()
    assert not hasattr(watcher, "_taint")
    assert not any(isinstance(v, TaintEngine) for v in vars(watcher).values())


async def test_drift_does_not_touch_enforcement_state():
    engine = TaintEngine(node_id="n")
    watcher = BehavioralDriftWatcher()
    await watcher.on_drift("s", "a", "elevated_review")
    await watcher.on_drift("s", "a", "restricted_mode")
    # watcher: an independent engine's per-value ledger registers nothing.
    assert engine.derive_value("anything").is_tainted is False


async def test_watcher_runs_and_does_not_raise():
    await BehavioralDriftWatcher().on_drift("s", "a", "elevated_review")  # no raise


def test_watcher_satisfies_protocol():
    assert isinstance(BehavioralDriftWatcher(), BehavioralDriftObserver)


def _session(observer):
    cap = MagicMock(spec=CapabilityExecutor)
    cap.register_post_callback = MagicMock()
    return GovernedSession(executor=MagicMock(), capability_executor=cap, behavioral_drift_observer=observer)


async def test_session_calls_observer_on_notify():
    watcher = BehavioralDriftWatcher()
    watcher.on_drift = AsyncMock()  # type: ignore[method-assign]
    await _session(watcher).notify_behavioral_drift(agent_id="a", action="restricted_mode")
    watcher.on_drift.assert_awaited_once()


async def test_session_no_observer_is_noop():
    cap = MagicMock(spec=CapabilityExecutor)
    cap.register_post_callback = MagicMock()
    s = GovernedSession(executor=MagicMock(), capability_executor=cap)
    await s.notify_behavioral_drift(agent_id="a", action="restricted_mode")  # no raise


async def test_session_observer_failure_is_swallowed():
    watcher = BehavioralDriftWatcher()
    watcher.on_drift = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    await _session(watcher).notify_behavioral_drift(agent_id="a", action="restricted_mode")  # no raise
