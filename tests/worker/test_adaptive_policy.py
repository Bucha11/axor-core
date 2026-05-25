"""
Tests for adaptive policy narrowing in GovernedSession.

Invariant: capability surface can only decrease automatically across turns.
Automatic broadening is not allowed — only explicit operator overrides widen policy.
"""
from __future__ import annotations

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
    ToolPolicy,
)
from axor_core.worker.session import GovernedSession

from tests.conftest import EchoExecutor


# ── Policies ───────────────────────────────────────────────────────────────────

def _policy(
    name: str,
    *,
    allow_bash: bool = False,
    allow_write: bool = True,
    allow_spawn: bool = False,
    child_mode: ChildMode = ChildMode.DENIED,
    max_child_depth: int = 0,
    context_mode: ContextMode = ContextMode.MINIMAL,
    export_mode: ExportMode = ExportMode.SUMMARY,
    compression_mode: CompressionMode = CompressionMode.BALANCED,
) -> ExecutionPolicy:
    return ExecutionPolicy(
        name=name,
        derived_from=TaskComplexity.FOCUSED,
        context_mode=context_mode,
        compression_mode=compression_mode,
        child_mode=child_mode,
        max_child_depth=max_child_depth,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=allow_write,
            allow_bash=allow_bash,
            allow_spawn=allow_spawn,
        ),
        export_mode=export_mode,
    )


EXPANSIVE = _policy(
    "expansive",
    allow_bash=True,
    allow_spawn=True,
    child_mode=ChildMode.ALLOWED,
    max_child_depth=3,
    context_mode=ContextMode.BROAD,
    export_mode=ExportMode.FULL,
    compression_mode=CompressionMode.LIGHT,
)

FOCUSED = _policy(
    "focused_readonly",
    allow_bash=False,
    allow_write=False,
    allow_spawn=False,
    child_mode=ChildMode.DENIED,
    context_mode=ContextMode.MINIMAL,
    export_mode=ExportMode.SUMMARY,
    compression_mode=CompressionMode.AGGRESSIVE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_session() -> GovernedSession:
    return GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=ExecutionMode.LIBRARY,
    )


def _patch_selector(session: GovernedSession, policies: list[ExecutionPolicy]):
    """Make PolicySelector.select() return policies in sequence."""
    calls = iter(policies)

    async def _analyze(raw_input: str):
        from axor_core.contracts.policy import TaskSignal, TaskNature
        signal = TaskSignal(
            raw_input=raw_input,
            complexity=TaskComplexity.FOCUSED,
            nature=TaskNature.READONLY,
            estimated_scope=1,
            requires_children=False,
            requires_mutation=False,
        )
        return signal, None  # session.run() discards the event

    def _select(_signal):
        return next(calls)

    session._analyzer.analyze = _analyze
    session._selector.select = _select


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_turn_sets_active_policy():
    session = _make_session()
    assert session._active_policy is None

    _patch_selector(session, [EXPANSIVE])
    await session.run("rewrite the entire repo")

    assert session._active_policy is not None
    assert session._active_policy.name == "expansive"


@pytest.mark.asyncio
async def test_second_turn_narrower_policy_is_applied():
    """If turn 2 classifies as more restrictive, _active_policy narrows."""
    session = _make_session()
    _patch_selector(session, [EXPANSIVE, FOCUSED])

    await session.run("rewrite the entire repo")
    assert session._active_policy.tool_policy.allow_bash is True  # expansive

    await session.run("explain a function")
    # narrowed: FOCUSED disables bash
    assert session._active_policy.tool_policy.allow_bash is False


@pytest.mark.asyncio
async def test_second_turn_broader_policy_does_not_broaden():
    """If turn 2 classifies as more permissive, _active_policy stays narrow."""
    session = _make_session()
    _patch_selector(session, [FOCUSED, EXPANSIVE])

    await session.run("explain a function")
    assert session._active_policy.tool_policy.allow_bash is False  # focused

    await session.run("rewrite the entire repo")
    # must NOT broaden: bash stays denied
    assert session._active_policy.tool_policy.allow_bash is False


@pytest.mark.asyncio
async def test_narrowing_is_permanent():
    """Once narrowed, subsequent broader turns do not restore capability."""
    session = _make_session()
    _patch_selector(session, [EXPANSIVE, FOCUSED, EXPANSIVE, EXPANSIVE])

    await session.run("turn 1 — expansive")
    assert session._active_policy.tool_policy.allow_bash is True

    await session.run("turn 2 — narrows to focused")
    assert session._active_policy.tool_policy.allow_bash is False

    await session.run("turn 3 — expansive again, stays narrow")
    assert session._active_policy.tool_policy.allow_bash is False

    await session.run("turn 4 — still narrow")
    assert session._active_policy.tool_policy.allow_bash is False


@pytest.mark.asyncio
async def test_explicit_operator_override_bypasses_adaptive():
    """An explicit policy= argument to run() skips adaptive logic entirely."""
    session = _make_session()
    # Selector called zero times — policy is provided explicitly each turn
    _patch_selector(session, [])  # would raise StopIteration if called

    await session.run("any task", policy=FOCUSED)
    # _active_policy stays None — operator overrides don't feed the adaptive tracker
    assert session._active_policy is None


@pytest.mark.asyncio
async def test_adaptive_then_operator_override_uses_explicit():
    """Operator override after adaptive turns uses the given policy, not the accumulated one."""
    session = _make_session()
    _patch_selector(session, [FOCUSED])  # only called on the first (adaptive) turn

    await session.run("adaptive turn — focused")
    assert session._active_policy.tool_policy.allow_bash is False

    # operator explicitly provides expansive — should be used as-is
    result = await session.run("operator turn", policy=EXPANSIVE)
    # The result used expansive (check via echo output)
    assert "expansive" in result.output


@pytest.mark.asyncio
async def test_child_mode_narrows_across_turns():
    """child_mode and max_child_depth also narrow correctly."""
    session = _make_session()
    _patch_selector(session, [EXPANSIVE, FOCUSED])

    await session.run("turn 1")
    assert session._active_policy.child_mode == ChildMode.ALLOWED
    assert session._active_policy.max_child_depth == 3

    await session.run("turn 2")
    assert session._active_policy.child_mode == ChildMode.DENIED
    assert session._active_policy.max_child_depth == 0
