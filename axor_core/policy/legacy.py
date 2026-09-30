"""Legacy adapter between ExecutionPolicy and (AuthorityPolicy, ExecutionPlan).

The legacy :class:`ExecutionPolicy` mixes trusted authority with advisory
execution planning in one object. This module is the single place that knows
how to split it — every field lands on exactly one side:

Authority fields (security-sensitive):
    tool_policy, allowed_paths, max_unattended_consequence,
    escalation_policy, allowed_passthrough_commands, allow_model_switch,
    the permission portion of child_mode/max_child_depth,
    the ceiling portion of export_mode.

Plan fields (optimization-only):
    context_mode, compression_mode, child_context_fraction,
    the shape portion of child_mode/max_child_depth (decomposition
    preference / suggested depth), derived_from (as scope telemetry).

``merge_to_legacy_policy`` is the inverse, used by the migration-window
compatibility layer and by the equivalence tests: for every shipped preset,
``merge(split(p)) == p`` exactly.

Known, documented normalization (not representable in the new model):
``ChildMode.SHALLOW`` with ``max_child_depth > 1`` — SHALLOW's authority
meaning IS depth ≤ 1, so such a legacy policy is contradictory; it
normalizes to ALLOWED at the same depth (permission-equivalent: the depth
ceiling is what the spawn gate enforces).
"""
from __future__ import annotations

from axor_core.contracts.authority import (
    AuthorityPolicy,
    ChildAuthorityPolicy,
    ExportAuthorityPolicy,
)
from axor_core.contracts.planning import (
    DecompositionPreference,
    ExecutionPlan,
    RetrievalBreadth,
)
from axor_core.contracts.policy import (
    ChildMode,
    ContextMode,
    ExecutionPolicy,
    TaskComplexity,
)

# derived_from is a TaskComplexity tag with a canonical scope estimate (the
# same table HeuristicClassifier and TaskSignalClassifier use). The plan
# carries the scope number; the inverse map restores the tag on merge.
_COMPLEXITY_TO_SCOPE: dict[TaskComplexity, int] = {
    TaskComplexity.FOCUSED: 1,
    TaskComplexity.MODERATE: 5,
    TaskComplexity.EXPANSIVE: 999,
}
_SCOPE_TO_COMPLEXITY = {v: k for k, v in _COMPLEXITY_TO_SCOPE.items()}

_CONTEXT_TO_BREADTH: dict[ContextMode, RetrievalBreadth] = {
    ContextMode.MINIMAL: RetrievalBreadth.NARROW,
    ContextMode.MODERATE: RetrievalBreadth.MODERATE,
    ContextMode.BROAD: RetrievalBreadth.BROAD,
}

_CHILD_MODE_TO_DECOMPOSITION: dict[ChildMode, DecompositionPreference] = {
    ChildMode.DENIED: DecompositionPreference.AVOID,
    ChildMode.SHALLOW: DecompositionPreference.ALLOW,
    ChildMode.ALLOWED: DecompositionPreference.PREFER,
}


def split_legacy_policy(
    policy: ExecutionPolicy,
) -> tuple[AuthorityPolicy, ExecutionPlan]:
    """Split a legacy ExecutionPolicy into its authority and plan halves."""
    authority = AuthorityPolicy(
        name=policy.name,
        tool_policy=policy.tool_policy,
        allowed_paths=tuple(policy.allowed_paths or ()),
        max_unattended_consequence=policy.max_unattended_consequence,
        child_authority=ChildAuthorityPolicy(
            allow_spawn=policy.child_mode != ChildMode.DENIED,
            max_depth=policy.max_child_depth,
        ),
        escalation_policy=policy.escalation_policy,
        export_policy=ExportAuthorityPolicy(max_mode=policy.export_mode),
        allowed_passthrough_commands=tuple(policy.allowed_passthrough_commands or ()),
        allow_model_switch=policy.allow_model_switch,
    )
    plan = ExecutionPlan(
        name=policy.name,
        context_mode=policy.context_mode,
        compression_mode=policy.compression_mode,
        retrieval_breadth=_CONTEXT_TO_BREADTH[policy.context_mode],
        decomposition=_CHILD_MODE_TO_DECOMPOSITION[policy.child_mode],
        suggested_child_depth=policy.max_child_depth,
        child_context_fraction=policy.child_context_fraction,
        expected_scope=_COMPLEXITY_TO_SCOPE.get(policy.derived_from),
        source="legacy_split",
    )
    return authority, plan


def merge_to_legacy_policy(
    authority: AuthorityPolicy,
    plan: ExecutionPlan,
) -> ExecutionPolicy:
    """Inverse of :func:`split_legacy_policy` (up to the documented SHALLOW
    normalization). Exists for the migration-window compatibility layer and
    the split/merge equivalence tests."""
    if not authority.child_authority.allow_spawn:
        child_mode = ChildMode.DENIED
    elif authority.child_authority.max_depth <= 1:
        child_mode = ChildMode.SHALLOW
    else:
        child_mode = ChildMode.ALLOWED
    return ExecutionPolicy(
        name=authority.name,
        derived_from=_SCOPE_TO_COMPLEXITY.get(
            plan.expected_scope, TaskComplexity.FOCUSED
        ),
        context_mode=plan.context_mode,
        compression_mode=plan.compression_mode,
        child_mode=child_mode,
        max_child_depth=authority.child_authority.max_depth,
        tool_policy=authority.tool_policy,
        allowed_paths=authority.allowed_paths,
        export_mode=authority.export_policy.max_mode,
        child_context_fraction=plan.child_context_fraction,
        max_unattended_consequence=authority.max_unattended_consequence,
        escalation_policy=authority.escalation_policy,
        allowed_passthrough_commands=authority.allowed_passthrough_commands,
        allow_model_switch=authority.allow_model_switch,
    )
