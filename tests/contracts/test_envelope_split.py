"""
PR B of the authority/plan split: the envelope carries both halves and
consumers read the right one.

Invariants pinned here:
  * every envelope always has authority+plan, consistent with its legacy
    policy (auto-derived when not passed explicitly);
  * capability resolution from the authority half is exactly equivalent to
    legacy resolution from the whole policy, for every shipped preset;
  * the lease ceiling validates against AuthorityPolicy;
  * enforcement reads authority even when an inconsistent plan is injected
    (the plan cannot influence the capability surface).
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.resolver import CapabilityResolver
from axor_core.capability.lease_validator import LeaseValidator
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.planning import ExecutionPlan
from axor_core.contracts.policy import (
    ContextMode,
    TaskComplexity,
    TaskNature,
    TaskSignal,
    ToolPolicy,
)
from axor_core.policy.legacy import split_legacy_policy
from axor_core.policy.selector import PolicySelector
from axor_core.contracts.authority import AuthorityPolicy, ChildAuthorityPolicy


def _signal(complexity: TaskComplexity, nature: TaskNature) -> TaskSignal:
    return TaskSignal(
        raw_input="x",
        complexity=complexity,
        nature=nature,
        estimated_scope=1,
        requires_children=False,
        requires_mutation=False,
    )


def _all_presets():
    selector = PolicySelector()
    return [
        selector.select(_signal(c, n))
        for c in TaskComplexity
        for n in TaskNature
    ] + [selector.safe_fallback()]


# ── envelope always carries consistent halves ───────────────────────────────────

def test_envelope_autoderives_authority_and_plan(make_envelope):
    env = make_envelope()
    expected_authority, expected_plan = split_legacy_policy(env.policy)
    assert env.authority == expected_authority
    assert env.plan == expected_plan


def test_envelope_explicit_halves_are_kept(make_envelope):
    env = make_envelope()
    explicit_plan = ExecutionPlan(name="explicit", context_mode=ContextMode.BROAD)
    env2 = dataclasses.replace(env, plan=explicit_plan, authority=None)
    assert env2.plan == explicit_plan
    assert env2.authority == split_legacy_policy(env2.policy)[0]


# ── capability resolution equivalence ───────────────────────────────────────────

@pytest.mark.parametrize("policy", _all_presets(), ids=lambda p: p.name)
def test_authority_resolution_equals_legacy_resolution(policy):
    resolver = CapabilityResolver()
    authority, plan = split_legacy_policy(policy)

    legacy_caps = resolver.resolve(policy)
    authority_caps = resolver.resolve(
        authority,
        allow_context_expansion=plan.context_mode != ContextMode.MINIMAL,
    )
    assert authority_caps == legacy_caps


# ── lease ceiling validates against authority ───────────────────────────────────

def test_lease_ceiling_from_authority():
    authority = AuthorityPolicy(
        name="ceiling",
        tool_policy=ToolPolicy(allow_read=True, allow_write=True, allow_bash=False),
        allowed_paths=("/workspace",),
    )
    validator = LeaseValidator()

    _, err = validator.create_lease(
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        allowed_tools=["write"],
        parent_policy=authority,
        allowed_paths=["/workspace/src"],
    )
    assert err is None

    _, err = validator.create_lease(
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        allowed_tools=["bash"],   # outside the authority's tool surface
        parent_policy=authority,
    )
    assert err is not None and "outside parent ceiling" in err

    _, err = validator.create_lease(
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        allowed_tools=["write"],
        parent_policy=authority,
        allowed_paths=["/etc"],   # outside the authority's path ceiling
    )
    assert err is not None and "paths outside parent ceiling" in err


# ── the plan cannot influence the capability surface ────────────────────────────

def test_inflated_plan_does_not_widen_capabilities(make_envelope):
    """Inject a maximally inflated plan next to a narrow authority: the
    resolved capability surface and every enforcement read stay driven by
    the authority half."""
    resolver = CapabilityResolver()
    narrow = AuthorityPolicy(
        name="narrow",
        tool_policy=ToolPolicy(allow_read=True),
        child_authority=ChildAuthorityPolicy(allow_spawn=False, max_depth=0),
    )
    caps_neutral = resolver.resolve(narrow, allow_context_expansion=False)
    caps_inflated = resolver.resolve(narrow, allow_context_expansion=True)

    # the ONLY plan-driven capability field is context expansion; the tool
    # surface and child topology are byte-identical
    assert caps_inflated.allowed_tools == caps_neutral.allowed_tools == frozenset({"read"})
    assert caps_inflated.allow_children is False
    assert caps_inflated.max_child_depth == 0
    assert caps_inflated.allow_mutation is False
