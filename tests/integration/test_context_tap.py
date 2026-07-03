"""Core hot path: GovernedSession emits a live SessionContextView to a ContextTap.

Drives the full public stack (GovernedSession.run) with a recording ContextTap
and asserts the per-turn context view the probe consumes is shaped and populated
(non-empty, message-shaped context_window carrying the task). The contract lives
in axor_core.contracts.observation; the emit is node/context_observation.
"""
from __future__ import annotations

import pytest

from axor_core import GovernedSession
from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.observation import ContextTap, SessionContextView
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
from axor_core.contracts.trace import TraceConfig
from tests.conftest import EchoExecutor

_POLICY = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_read=True))


class _RecordingTap:
    """A ContextTap that captures the views it receives."""

    def __init__(self) -> None:
        self.views: list[SessionContextView] = []

    async def on_context_event(self, view: SessionContextView) -> None:
        self.views.append(view)


def _session(tap=None) -> GovernedSession:
    return GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        context_taps=[tap] if tap is not None else None,
    )


@pytest.mark.asyncio
async def test_context_tap_receives_live_view() -> None:
    tap = _RecordingTap()
    session = _session(tap=tap)

    await session.run("summarize the quarterly report", policy=_POLICY)

    assert len(tap.views) >= 1
    v = tap.views[0]
    assert v.session_id == session.session_id()
    assert v.context_window                                       # non-empty
    assert all(set(m) >= {"role", "content"} for m in v.context_window)
    # the task text is carried into the context window as a fragment
    assert any("summarize the quarterly report" in m["content"] for m in v.context_window)


@pytest.mark.asyncio
async def test_no_tap_is_noop() -> None:
    session = _session(tap=None)
    result = await session.run("do x", policy=_POLICY)
    assert result.output                                          # run unaffected


@pytest.mark.asyncio
async def test_tap_failure_does_not_break_run() -> None:
    class _Bad:
        async def on_context_event(self, view: SessionContextView) -> None:
            raise RuntimeError("tap exploded on the hot path")

    session = _session(tap=_Bad())
    result = await session.run("do x", policy=_POLICY)            # must not raise
    assert result.output


def test_tap_protocol_is_structural() -> None:
    assert isinstance(_RecordingTap(), ContextTap)

    class _NotATap:
        pass

    assert not isinstance(_NotATap(), ContextTap)
