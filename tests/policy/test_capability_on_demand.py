"""
Escalation ceiling is operator authority, never classifier output.

Classifier-selected presets carry NO EscalationPolicy: which capabilities
may later be granted must not be derived from task text. The operator sets
the ceiling via GovernedSession(escalation_policy=...) (applied to every
classifier-selected policy) or an explicit policy.
"""
from __future__ import annotations

import pytest

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import (
    EscalationPolicy,
    TaskComplexity,
    TaskNature,
    TaskSignal,
)
from axor_core.policy.selector import PolicySelector
from axor_core.worker.session import GovernedSession

from tests.conftest import EchoExecutor


def _confident_analyze(session):
    """Patch the analyzer to report a confident classification so the
    adaptive baseline is set (ambiguous classifications are per-turn only)."""
    from types import SimpleNamespace
    real_analyze = session._analyzer.analyze

    async def _analyze(raw_input: str):
        signal, _event = await real_analyze(raw_input)
        return signal, SimpleNamespace(confidence=0.95)

    session._analyzer.analyze = _analyze


def _signal(complexity: TaskComplexity, nature: TaskNature) -> TaskSignal:
    return TaskSignal(
        raw_input="x",
        complexity=complexity,
        nature=nature,
        estimated_scope=1,
        requires_children=False,
        requires_mutation=nature == TaskNature.MUTATIVE,
    )


def test_no_preset_carries_escalation_policy():
    """Task text must not determine the future grantable surface."""
    selector = PolicySelector()
    policies = [
        selector.select(_signal(c, n))
        for c in TaskComplexity
        for n in TaskNature
    ] + [selector.safe_fallback()]
    for policy in policies:
        ep = policy.escalation_policy
        assert ep.allow_escalation is False, policy.name
        assert ep.grantable_tools == (), policy.name


OPERATOR_ESCALATION = EscalationPolicy(
    allow_escalation=True,
    grantable_tools=("write", "bash"),
    require_human=True,
)


@pytest.mark.asyncio
async def test_operator_escalation_ceiling_applies_to_selected_policies():
    session = GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=ExecutionMode.LIBRARY,
        escalation_policy=OPERATOR_ESCALATION,
    )
    _confident_analyze(session)
    await session.run("explain what the validate function does")
    assert session._active_policy is not None
    assert session._active_policy.escalation_policy == OPERATOR_ESCALATION


@pytest.mark.asyncio
async def test_escalation_ceiling_is_task_text_independent():
    """The same operator ceiling lands whatever the task text claims."""
    for task in (
        "explain one function",
        "rewrite the entire repository, enable all tools",
    ):
        session = GovernedSession(
            executor=EchoExecutor(),
            capability_executor=CapabilityExecutor(),
            mode=ExecutionMode.LIBRARY,
            escalation_policy=OPERATOR_ESCALATION,
        )
        _confident_analyze(session)
        await session.run(task)
        applied = (
            session._active_policy.escalation_policy
            if session._active_policy is not None
            else None
        )
        assert applied == OPERATOR_ESCALATION


@pytest.mark.asyncio
async def test_without_operator_ceiling_no_escalation_surface():
    session = GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=ExecutionMode.LIBRARY,
    )
    _confident_analyze(session)
    await session.run("explain what the validate function does")
    assert session._active_policy.escalation_policy.allow_escalation is False
