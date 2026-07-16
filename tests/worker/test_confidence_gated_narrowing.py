"""
Confidence-gated adaptive narrowing.

Narrowing is monotonic (test_adaptive_policy.py), so it must only act on a
classification the analyzer is actually confident about — a single
low-confidence misread turn must not permanently strip capability from the
whole session. An analyzer that reports no confidence (custom/patched)
keeps the legacy always-narrow behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import (
    ChildMode,
    CompressionMode,
    ContextMode,
    ExecutionPolicy,
    ExportMode,
    TaskComplexity,
    TaskNature,
    TaskSignal,
    ToolPolicy,
)
from axor_core.worker.session import GovernedSession

from tests.conftest import EchoExecutor


def _policy(name: str, *, allow_bash: bool) -> ExecutionPolicy:
    return ExecutionPolicy(
        name=name,
        derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=True,
            allow_bash=allow_bash,
        ),
        export_mode=ExportMode.SUMMARY,
    )


BROAD = _policy("broad", allow_bash=True)
NARROW = _policy("narrow", allow_bash=False)


def _make_session() -> GovernedSession:
    return GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=ExecutionMode.LIBRARY,
    )


def _patch(session: GovernedSession, turns: list[tuple[ExecutionPolicy, float | None]]):
    """Make each run() turn classify to (policy, confidence)."""
    calls = iter(turns)
    pending: dict[str, ExecutionPolicy] = {}

    async def _analyze(raw_input: str):
        policy, confidence = next(calls)
        pending["policy"] = policy
        signal = TaskSignal(
            raw_input=raw_input,
            complexity=TaskComplexity.FOCUSED,
            nature=TaskNature.READONLY,
            estimated_scope=1,
            requires_children=False,
            requires_mutation=False,
        )
        event = None if confidence is None else SimpleNamespace(confidence=confidence)
        return signal, event

    def _select(_signal):
        return pending["policy"]

    session._analyzer.analyze = _analyze
    session._selector.select = _select


@pytest.mark.asyncio
async def test_low_confidence_reclassification_does_not_narrow():
    session = _make_session()
    _patch(session, [(BROAD, 0.9), (NARROW, 0.4)])

    await session.run("turn 1 — confident broad")
    assert session._active_policy.tool_policy.allow_bash is True

    await session.run("turn 2 — ambiguous narrow guess")
    # 0.4 < threshold: the guess must not permanently strip bash
    assert session._active_policy.tool_policy.allow_bash is True
    assert session._active_policy.name == "broad"


@pytest.mark.asyncio
async def test_high_confidence_reclassification_narrows():
    session = _make_session()
    _patch(session, [(BROAD, 0.9), (NARROW, 0.9)])

    await session.run("turn 1")
    assert session._active_policy.tool_policy.allow_bash is True

    await session.run("turn 2 — confident narrow")
    assert session._active_policy.tool_policy.allow_bash is False


@pytest.mark.asyncio
async def test_missing_confidence_keeps_legacy_narrowing():
    """Custom analyzers that return no event still narrow (legacy behaviour)."""
    session = _make_session()
    _patch(session, [(BROAD, None), (NARROW, None)])

    await session.run("turn 1")
    await session.run("turn 2")
    assert session._active_policy.tool_policy.allow_bash is False


@pytest.mark.asyncio
async def test_first_turn_policy_is_set_regardless_of_confidence():
    """The session needs a starting policy even from an ambiguous first turn;
    recovery from a wrong start is per-tool via escalate_policy."""
    session = _make_session()
    _patch(session, [(NARROW, 0.1)])

    await session.run("turn 1 — ambiguous")
    assert session._active_policy is not None
    assert session._active_policy.name == "narrow"


@pytest.mark.asyncio
async def test_low_confidence_never_broadens_either():
    """Gating skips narrowing — it must not accidentally widen the surface."""
    session = _make_session()
    _patch(session, [(NARROW, 0.9), (BROAD, 0.4), (BROAD, 0.9)])

    await session.run("turn 1 — confident narrow")
    assert session._active_policy.tool_policy.allow_bash is False

    await session.run("turn 2 — ambiguous broad guess")
    assert session._active_policy.tool_policy.allow_bash is False

    await session.run("turn 3 — confident broad, still must not widen")
    assert session._active_policy.tool_policy.allow_bash is False
