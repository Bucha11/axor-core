"""
GovernedSession wiring for the two operator guards:

  1. escalation_callback= — threaded into every node's IntentLoop so
     require_human escalations can actually be approved.
  2. default_policy= — session-wide explicit policy; the classifier is
     bypassed entirely (recommended PRODUCTION posture, nudged by a
     one-time warning when PRODUCTION derives policy from classification).
"""
from __future__ import annotations

import logging

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


EXPLICIT = ExecutionPolicy(
    name="operator_explicit",
    derived_from=TaskComplexity.FOCUSED,
    context_mode=ContextMode.MINIMAL,
    compression_mode=CompressionMode.BALANCED,
    child_mode=ChildMode.DENIED,
    max_child_depth=0,
    tool_policy=ToolPolicy(allow_read=True, allow_write=True),
    export_mode=ExportMode.SUMMARY,
)

PER_CALL = ExecutionPolicy(
    name="per_call_override",
    derived_from=TaskComplexity.FOCUSED,
    context_mode=ContextMode.MINIMAL,
    compression_mode=CompressionMode.BALANCED,
    child_mode=ChildMode.DENIED,
    max_child_depth=0,
    tool_policy=ToolPolicy(allow_read=True),
    export_mode=ExportMode.SUMMARY,
)


def _session(mode=ExecutionMode.LIBRARY, **kwargs) -> GovernedSession:
    return GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=mode,
        **kwargs,
    )


def _forbid_classification(session: GovernedSession) -> None:
    async def _fail(_raw_input: str):
        raise AssertionError("classifier must not run when policy is explicit")

    session._analyzer.analyze = _fail


# ── escalation_callback wiring ──────────────────────────────────────────────────

def test_escalation_callback_reaches_the_node():
    async def _cb(tool_use_id, tool, paths, max_ops) -> bool:
        return True

    session = _session(escalation_callback=_cb)
    node = session._make_node(session._context_manager)
    assert node._escalation_callback is _cb


def test_no_callback_by_default():
    session = _session()
    node = session._make_node(session._context_manager)
    assert node._escalation_callback is None


# ── default_policy ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_default_policy_bypasses_classifier():
    session = _session(default_policy=EXPLICIT)
    _forbid_classification(session)

    result = await session.run("любая задача на любом языке")

    assert "operator_explicit" in result.output  # EchoExecutor echoes policy name
    # explicit policy never feeds the adaptive tracker
    assert session._active_policy is None


@pytest.mark.asyncio
async def test_per_call_policy_wins_over_default():
    session = _session(default_policy=EXPLICIT)
    _forbid_classification(session)

    result = await session.run("task", policy=PER_CALL)

    assert "per_call_override" in result.output


@pytest.mark.asyncio
async def test_without_default_policy_classifier_still_runs():
    session = _session()
    result = await session.run("explain what the function does")
    assert session._active_policy is not None


# ── PRODUCTION nudge ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_production_warns_once_on_classifier_derived_policy(caplog):
    session = _session(mode=ExecutionMode.PRODUCTION)
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        await session.run("explain what the function does")
        await session.run("explain another function")
    warnings = [r for r in caplog.records if "deriving policy from task" in r.message]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_production_with_default_policy_does_not_warn(caplog):
    session = _session(mode=ExecutionMode.PRODUCTION, default_policy=EXPLICIT)
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        await session.run("task")
    assert not [r for r in caplog.records if "deriving policy" in r.message]


@pytest.mark.asyncio
async def test_library_mode_does_not_warn(caplog):
    session = _session(mode=ExecutionMode.LIBRARY)
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        await session.run("explain what the function does")
    assert not [r for r in caplog.records if "deriving policy" in r.message]
