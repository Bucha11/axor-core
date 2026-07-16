"""
Capability-on-demand: presets whose tool surface is narrower than the full
set carry an escalation path, so a misclassified task recovers per-tool via
escalate_policy instead of failing. The path stays fail-closed:
require_human=True means nothing is granted without an operator-wired
callback, and grantable_tools never exceeds what the preset lacks.
"""
from __future__ import annotations

from axor_core.contracts.policy import (
    TaskComplexity,
    TaskNature,
    TaskSignal,
)
from axor_core.policy.selector import PolicySelector


def _signal(complexity: TaskComplexity, nature: TaskNature) -> TaskSignal:
    return TaskSignal(
        raw_input="x",
        complexity=complexity,
        nature=nature,
        estimated_scope=1,
        requires_children=False,
        requires_mutation=nature == TaskNature.MUTATIVE,
    )


def _select(complexity: TaskComplexity, nature: TaskNature):
    return PolicySelector().select(_signal(complexity, nature))


def test_focused_readonly_can_escalate_to_write_and_bash():
    ep = _select(TaskComplexity.FOCUSED, TaskNature.READONLY).escalation_policy
    assert ep.allow_escalation is True
    assert set(ep.grantable_tools) == {"write", "bash"}
    assert ep.require_human is True  # fail-closed without a callback


def test_focused_generative_can_escalate_to_bash():
    ep = _select(TaskComplexity.FOCUSED, TaskNature.GENERATIVE).escalation_policy
    assert ep.allow_escalation is True
    assert set(ep.grantable_tools) == {"bash"}
    assert ep.require_human is True


def test_moderate_readonly_can_escalate_to_write_and_bash():
    ep = _select(TaskComplexity.MODERATE, TaskNature.READONLY).escalation_policy
    assert ep.allow_escalation is True
    assert set(ep.grantable_tools) == {"write", "bash"}
    assert ep.require_human is True


def test_full_surface_presets_grant_nothing():
    """Presets that already have the full tool surface keep escalation off —
    there is nothing legitimate to escalate to."""
    for complexity, nature in [
        (TaskComplexity.FOCUSED, TaskNature.MUTATIVE),
        (TaskComplexity.MODERATE, TaskNature.GENERATIVE),
        (TaskComplexity.MODERATE, TaskNature.MUTATIVE),
        (TaskComplexity.EXPANSIVE, TaskNature.MUTATIVE),
    ]:
        ep = _select(complexity, nature).escalation_policy
        assert ep.allow_escalation is False, (complexity, nature)
        assert ep.grantable_tools == ()


def test_safe_fallback_can_escalate_to_search_write_bash():
    ep = PolicySelector().safe_fallback().escalation_policy
    assert ep.allow_escalation is True
    assert set(ep.grantable_tools) == {"search", "write", "bash"}
    assert ep.require_human is True


def test_grantable_tools_never_include_spawn():
    """Escalation may widen the tool surface, never the child topology."""
    selector = PolicySelector()
    policies = [
        selector.select(_signal(c, n))
        for c in TaskComplexity
        for n in TaskNature
    ] + [selector.safe_fallback()]
    for policy in policies:
        assert "spawn_child" not in policy.escalation_policy.grantable_tools
