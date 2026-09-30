"""
PR A of the authority/plan split (RFC: task classification must not
determine authority): the new types and the legacy adapter.

Invariants pinned here:
  * split/merge round-trips every shipped preset byte-identically (§22.7);
  * every legacy field lands on exactly one side, and ExecutionPlan carries
    no authority-bearing field at all;
  * ChildMode decomposes into permission (ChildAuthorityPolicy) + shape
    (DecompositionPreference), with the one documented normalization;
  * ExportMode is a total order (the precondition for a single max_mode
    export ceiling, RFC Q1).
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.contracts.authority import (
    AuthorityPolicy,
    ChildAuthorityPolicy,
    ExportAuthorityPolicy,
)
from axor_core.contracts.planning import (
    NEUTRAL_PLAN,
    DecompositionPreference,
    ExecutionPlan,
    ResourceBudget,
    RetrievalBreadth,
)
from axor_core.contracts.policy import (
    ChildMode,
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    TaskComplexity,
    TaskNature,
    TaskSignal,
    ToolPolicy,
)
from axor_core.policy.legacy import merge_to_legacy_policy, split_legacy_policy
from axor_core.policy.selector import PolicySelector
from axor_core.policy import presets as policy_presets


def _signal(complexity: TaskComplexity, nature: TaskNature) -> TaskSignal:
    return TaskSignal(
        raw_input="x",
        complexity=complexity,
        nature=nature,
        estimated_scope=1,
        requires_children=False,
        requires_mutation=nature == TaskNature.MUTATIVE,
    )


def _all_shipped_presets() -> list[ExecutionPolicy]:
    selector = PolicySelector()
    from_selector = [
        selector.select(_signal(c, n))
        for c in TaskComplexity
        for n in TaskNature
    ] + [selector.safe_fallback()]
    from_presets = [
        policy_presets.get(name)
        for name in ("readonly", "sandboxed", "standard", "federated",
                     "research", "support", "analysis")
    ]
    return from_selector + from_presets


# ── split/merge equivalence (§22.7) ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "policy", _all_shipped_presets(), ids=lambda p: p.name
)
def test_split_merge_round_trips_every_shipped_preset(policy):
    authority, plan = split_legacy_policy(policy)
    assert merge_to_legacy_policy(authority, plan) == policy


def test_shallow_with_deep_ceiling_normalizes_to_allowed():
    """The one documented lossy case: SHALLOW with max_child_depth > 1 is
    contradictory legacy config (SHALLOW means depth ≤ 1); it normalizes to
    ALLOWED at the same depth — permission-equivalent for the spawn gate."""
    contradictory = ExecutionPolicy(
        name="weird", child_mode=ChildMode.SHALLOW, max_child_depth=3,
    )
    authority, plan = split_legacy_policy(contradictory)
    merged = merge_to_legacy_policy(authority, plan)
    assert merged.child_mode == ChildMode.ALLOWED
    assert merged.max_child_depth == 3
    assert dataclasses.replace(merged, child_mode=ChildMode.SHALLOW) == contradictory


# ── field placement ─────────────────────────────────────────────────────────────

_AUTHORITY_BEARING_FIELDS = {
    "tool_policy", "allowed_paths", "max_unattended_consequence",
    "escalation_policy", "allow_spawn", "allowed_passthrough_commands",
    "allow_model_switch", "export_mode", "max_mode", "require_human",
}


def test_execution_plan_carries_no_authority_fields():
    plan_fields = {f.name for f in dataclasses.fields(ExecutionPlan)}
    leaked = plan_fields & _AUTHORITY_BEARING_FIELDS
    assert not leaked, f"authority-bearing fields leaked into ExecutionPlan: {leaked}"


def test_split_puts_every_authority_field_in_authority():
    policy = ExecutionPolicy(
        name="probe",
        tool_policy=ToolPolicy(allow_read=True, allow_write=True, allow_bash=True),
        allowed_paths=("/workspace",),
        child_mode=ChildMode.ALLOWED,
        max_child_depth=2,
        export_mode=ExportMode.FILTERED,
        escalation_policy=EscalationPolicy(
            allow_escalation=True, grantable_tools=("bash",),
        ),
        allowed_passthrough_commands=("status",),
        allow_model_switch=True,
    )
    authority, plan = split_legacy_policy(policy)

    assert authority.tool_policy == policy.tool_policy
    assert authority.allowed_paths == ("/workspace",)
    assert authority.max_unattended_consequence == policy.max_unattended_consequence
    assert authority.escalation_policy == policy.escalation_policy
    assert authority.export_policy == ExportAuthorityPolicy(max_mode=ExportMode.FILTERED)
    assert authority.child_authority == ChildAuthorityPolicy(
        allow_spawn=True, max_depth=2,
    )
    assert authority.allowed_passthrough_commands == ("status",)
    assert authority.allow_model_switch is True


def test_split_puts_planning_fields_in_plan():
    policy = PolicySelector().select(
        _signal(TaskComplexity.MODERATE, TaskNature.GENERATIVE)
    )
    _, plan = split_legacy_policy(policy)

    assert plan.context_mode == policy.context_mode
    assert plan.compression_mode == policy.compression_mode
    assert plan.child_context_fraction == policy.child_context_fraction
    assert plan.suggested_child_depth == policy.max_child_depth
    assert plan.retrieval_breadth == RetrievalBreadth.MODERATE
    assert plan.decomposition == DecompositionPreference.ALLOW
    assert plan.expected_scope == 5
    assert plan.source == "legacy_split"


# ── child mode decomposition ────────────────────────────────────────────────────

@pytest.mark.parametrize("child_mode,depth,allow_spawn,decomposition", [
    (ChildMode.DENIED, 0, False, DecompositionPreference.AVOID),
    (ChildMode.SHALLOW, 1, True, DecompositionPreference.ALLOW),
    (ChildMode.ALLOWED, 3, True, DecompositionPreference.PREFER),
])
def test_child_mode_splits_into_permission_and_shape(
    child_mode, depth, allow_spawn, decomposition
):
    policy = ExecutionPolicy(child_mode=child_mode, max_child_depth=depth)
    authority, plan = split_legacy_policy(policy)
    assert authority.child_authority.allow_spawn is allow_spawn
    assert authority.child_authority.max_depth == depth
    assert plan.decomposition == decomposition
    assert plan.suggested_child_depth == depth


# ── export lattice (RFC Q1 precondition) ────────────────────────────────────────

def test_export_mode_is_a_total_order():
    """A single max_mode export ceiling is sound only while ExportMode's
    allowed-field sets form a chain and token caps are monotone."""
    from axor_core.node.envelope import _EXPORT_ALLOWED_FIELDS, _EXPORT_MAX_TOKENS

    chain = [ExportMode.RESTRICTED, ExportMode.SUMMARY,
             ExportMode.FILTERED, ExportMode.FULL]
    for narrower, wider in zip(chain, chain[1:]):
        assert _EXPORT_ALLOWED_FIELDS[narrower] < _EXPORT_ALLOWED_FIELDS[wider], (
            f"{narrower} must expose strictly fewer fields than {wider}"
        )
        n_cap = _EXPORT_MAX_TOKENS[narrower]
        w_cap = _EXPORT_MAX_TOKENS[wider]
        assert w_cap is None or (n_cap is not None and n_cap < w_cap), (
            f"{narrower} token cap must be below {wider}"
        )


# ── defaults ────────────────────────────────────────────────────────────────────

def test_neutral_plan_is_moderate_everything():
    assert NEUTRAL_PLAN.name == "neutral"
    assert NEUTRAL_PLAN.retrieval_breadth == RetrievalBreadth.MODERATE
    assert NEUTRAL_PLAN.decomposition == DecompositionPreference.ALLOW
    assert NEUTRAL_PLAN.suggested_child_depth == 0


def test_default_authority_is_restrictive():
    authority = AuthorityPolicy()
    assert authority.child_authority.allow_spawn is False
    assert authority.escalation_policy.allow_escalation is False
    assert authority.export_policy.max_mode == ExportMode.SUMMARY
    assert authority.allow_model_switch is False


def test_resource_budget_defaults_to_unlimited():
    budget = ResourceBudget()
    assert all(
        getattr(budget, f.name) is None for f in dataclasses.fields(ResourceBudget)
    )


def test_new_types_are_frozen():
    for instance in (AuthorityPolicy(), ChildAuthorityPolicy(),
                     ExportAuthorityPolicy(), ExecutionPlan(), ResourceBudget()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, "name", "x")
