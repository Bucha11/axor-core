"""Stateful trajectory observers tighten degradation (the stove-on-too-long class).

A trajectory observer is fed every executed (tool, args, result), carries state, and
may only tighten degradation — never authorise. This verifies the engine entry point,
the observer contract, and the end-to-end wiring through GovernedSession.
"""
from __future__ import annotations

from typing import Any

import pytest

from axor_core import GovernedSession
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.policy import (
    ChildMode, CompressionMode, ContextMode, ExecutionPolicy, ExportMode,
    TaskComplexity, ToolPolicy,
)
from axor_core.contracts.trajectory import TrajectorySignal
from axor_core.degradation.engine import DegradationEngine
from tests.conftest import EchoExecutor


# ── engine.tighten ───────────────────────────────────────────────────────────

def test_tighten_raises_level_monotonically():
    eng = DegradationEngine()
    assert eng.state.level == DegradationLevel.NORMAL
    eng.tighten(DegradationLevel.LOCKED, reason="stove")
    assert eng.state.level == DegradationLevel.LOCKED
    # a lower target never lowers the level
    eng.tighten(DegradationLevel.CAUTIOUS, reason="noise")
    assert eng.state.level == DegradationLevel.LOCKED


def test_tighten_emits_a_transition_event():
    eng = DegradationEngine()
    eng.tighten(DegradationLevel.RESTRICTED, reason="metric not improving")
    events = eng.drain_events()
    assert any("DEGRADATION" in type(e).__name__.upper() for e in events)


# ── an observer (inline; mirrors examples/trajectory/stove.py) ──────────────────

class _StoveObserver:
    def __init__(self, threshold=30):
        self.threshold = threshold
        self._fired = False

    def observe(self, tool: str, args: dict, result: Any):
        if tool == "check_stove" and isinstance(result, dict) and not self._fired:
            if result.get("on") and int(result.get("minutes", 0)) > self.threshold:
                self._fired = True
                return TrajectorySignal(DegradationLevel.LOCKED, reason="stove on too long")
        return None


def test_observer_fires_only_past_threshold():
    obs = _StoveObserver(threshold=30)
    assert obs.observe("check_stove", {}, {"on": True, "minutes": 10}) is None
    sig = obs.observe("check_stove", {}, {"on": True, "minutes": 45})
    assert sig is not None and sig.target_level == DegradationLevel.LOCKED
    # latched — does not re-fire
    assert obs.observe("check_stove", {}, {"on": True, "minutes": 60}) is None


# ── end-to-end through GovernedSession ──────────────────────────────────────────

class _CheckStove(ToolHandler):
    def __init__(self, minutes): self._m = minutes
    @property
    def name(self): return "check_stove"
    async def execute(self, args): return {"on": True, "minutes": self._m}


class _StartOven(ToolHandler):
    def __init__(self): self.calls = 0
    @property
    def name(self): return "start_oven"
    async def execute(self, args): self.calls += 1; return "oven on"


def _policy():
    return ExecutionPolicy(
        name="kitchen", derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL, compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED, max_child_depth=0,
        tool_policy=ToolPolicy(extra_allowed=frozenset({"check_stove", "start_oven"})),
        export_mode=ExportMode.SUMMARY,
    )


@pytest.mark.asyncio
async def test_trajectory_pressure_freezes_the_surface_end_to_end():
    from axor_core.contracts.trace import TraceConfig
    oven = _StartOven()
    cap = CapabilityExecutor()
    cap.register(_CheckStove(minutes=45))   # already over threshold
    cap.register(oven)
    # the executor checks the stove, then tries to start the oven
    ex = EchoExecutor(tool_calls=[("check_stove", {}), ("start_oven", {})])
    session = GovernedSession(
        executor=ex, capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        trajectory_observers=[_StoveObserver(threshold=30)],
    )
    await session.run("manage the kitchen", policy=_policy())

    # the stove check tightened the session to LOCKED ...
    assert session.current_degradation_level() == DegradationLevel.LOCKED.value
    # ... so the subsequent start_oven was frozen and never executed
    assert oven.calls == 0


@pytest.mark.asyncio
async def test_no_observer_no_tightening():
    oven = _StartOven()
    cap = CapabilityExecutor()
    cap.register(_CheckStove(minutes=45))
    cap.register(oven)
    from axor_core.contracts.trace import TraceConfig
    ex = EchoExecutor(tool_calls=[("check_stove", {}), ("start_oven", {})])
    session = GovernedSession(
        executor=ex, capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )  # no observers
    await session.run("manage the kitchen", policy=_policy())
    assert session.current_degradation_level() == DegradationLevel.NORMAL.value
    assert oven.calls == 1  # not frozen
