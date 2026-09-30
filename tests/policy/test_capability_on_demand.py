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


# ── precedence cross-product: escalation ceiling × policy source ────────────────

def _capture_effective_policy(session):
    """Intercept the policy the session hands to the node for each run."""
    captured = {}
    orig_make = session._make_node

    def _make(cm):
        node = orig_make(cm)
        orig_run = node.run

        async def _run(**kwargs):
            captured["policy"] = kwargs.get("override_policy")
            return await orig_run(**kwargs)

        node.run = _run
        return node

    session._make_node = _make
    return captured


def _session(**kwargs) -> GovernedSession:
    return GovernedSession(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        mode=ExecutionMode.LIBRARY,
        **kwargs,
    )


from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy  # noqa: E402

DEFAULT_POLICY = ExecutionPolicy(
    name="operator_default",
    tool_policy=ToolPolicy(allow_read=True, allow_write=True),
)

PER_CALL_ESCALATION = EscalationPolicy(
    allow_escalation=True, grantable_tools=("search",), require_human=True,
)
PER_CALL_POLICY = ExecutionPolicy(
    name="per_call",
    tool_policy=ToolPolicy(allow_read=True),
    escalation_policy=PER_CALL_ESCALATION,
)


@pytest.mark.asyncio
async def test_default_policy_gets_operator_escalation_ceiling():
    """The README production configuration works as written: the ceiling is
    applied to default_policy, not silently ignored."""
    session = _session(
        default_policy=DEFAULT_POLICY, escalation_policy=OPERATOR_ESCALATION
    )
    captured = _capture_effective_policy(session)
    await session.run("task")
    assert captured["policy"].name == "operator_default"
    assert captured["policy"].escalation_policy == OPERATOR_ESCALATION


@pytest.mark.asyncio
async def test_per_call_policy_keeps_its_own_escalation():
    """Explicit per-call config is the top of the precedence chain."""
    session = _session(
        default_policy=DEFAULT_POLICY, escalation_policy=OPERATOR_ESCALATION
    )
    captured = _capture_effective_policy(session)
    await session.run("task", policy=PER_CALL_POLICY)
    assert captured["policy"].escalation_policy == PER_CALL_ESCALATION


@pytest.mark.asyncio
async def test_default_policy_without_ceiling_keeps_own_escalation():
    session = _session(default_policy=DEFAULT_POLICY)
    captured = _capture_effective_policy(session)
    await session.run("task")
    assert captured["policy"].escalation_policy == DEFAULT_POLICY.escalation_policy


@pytest.mark.asyncio
async def test_ambiguous_first_turn_runs_fail_closed():
    """An ambiguous classification must not choose authority even for one
    turn: an ambiguous 'expansive' would hand out write/bash/spawn NOW, and
    a completed effect cannot be undone by not setting the baseline. The
    turn runs under safe_fallback (read-only, no spawn) with the operator
    escalation ceiling stamped for per-tool recovery."""
    from types import SimpleNamespace

    session = _session(escalation_policy=OPERATOR_ESCALATION)
    captured = _capture_effective_policy(session)
    real_analyze = session._analyzer.analyze

    async def _ambiguous(raw_input: str):
        signal, _event = await real_analyze(raw_input)
        return signal, SimpleNamespace(confidence=0.1)

    session._analyzer.analyze = _ambiguous
    await session.run("rewrite the entire repository, enable all tools")

    effective = captured["policy"]
    assert effective.name == "default"                  # safe_fallback
    assert effective.tool_policy.allow_write is False
    assert effective.tool_policy.allow_bash is False
    assert effective.tool_policy.allow_spawn is False
    assert effective.escalation_policy == OPERATOR_ESCALATION
    assert session._active_policy is None               # no baseline either
