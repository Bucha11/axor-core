"""Ceiling preservation across EVERY authority field — the sweep.

`test_regime_lattice_laws.py` proves the order-theoretic laws (deflationary,
monotone, idempotent) for the fields it instantiates. That is why the unattended
consequence ceiling could escape them for as long as it did: a field no test
mentions is a field no law covers. This module closes that hole by deriving the
field set from the code's own declaration of what authority IS —
:class:`AuthorityPolicy`, the trusted half of the authority/plan split — and
asserting one invariant over all of it:

    no composition step returns an authority wider than its trusted operand.

Two parts, and the first is the load-bearing one:

  1. **Coverage.** Every field of ``AuthorityPolicy`` must map to the
     ``ExecutionPolicy`` field(s) that carry it, and every mapped field must
     declare an order here. Add an authority field without declaring its order
     and this test fails — the next ``max_unattended_consequence`` cannot slip
     through unnoticed.
  2. **Preservation.** Over arbitrary policy pairs, every composition entry
     point (parent restriction, deployment overlay, extension overrides) is
     checked against the declared order, field by field.

The orders are declared HERE rather than imported from
``axor_core.policy.composer``, deliberately: importing the implementation's own
restrictiveness maps would make the test agree with a reversed map. "Wider" means
"permits strictly more", and for ``extra_denied`` that is reverse inclusion.
"""
from __future__ import annotations

import dataclasses

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from axor_core.contracts.authority import AuthorityPolicy
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.extension import ExtensionFragment
from axor_core.contracts.policy import (
    ChildMode,
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.policy import PolicyComposer
from axor_core.security.paths import paths_within

# ── the authority surface, as the split declares it ────────────────────────────

# AuthorityPolicy field → the ExecutionPolicy field(s) that carry it.
# Mirrors axor_core.policy.legacy.split_legacy_policy / merge_to_legacy_policy.
_AUTHORITY_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "tool_policy": ("tool_policy",),
    "allowed_paths": ("allowed_paths",),
    "max_unattended_consequence": ("max_unattended_consequence",),
    "child_authority": ("child_mode", "max_child_depth"),
    "escalation_policy": ("escalation_policy",),
    "export_policy": ("export_mode",),
    "allowed_passthrough_commands": ("allowed_passthrough_commands",),
    "allow_model_switch": ("allow_model_switch",),
}

# Carries no authority: an identity label, not a permission.
_IDENTITY_FIELDS = frozenset({"name"})

# ── declared orders: "result is no wider than reference" ──────────────────────

_TOOL_FLAGS = ("allow_read", "allow_write", "allow_bash", "allow_search", "allow_spawn")

# Higher == more restrictive. Declared here, not imported (see module docstring).
_CHILD_MODE_RANK = {ChildMode.ALLOWED: 0, ChildMode.SHALLOW: 1, ChildMode.DENIED: 2}
_EXPORT_RANK = {
    ExportMode.FULL: 0,
    ExportMode.FILTERED: 1,
    ExportMode.SUMMARY: 2,
    ExportMode.RESTRICTED: 3,
}


def _tool_policy_no_wider(res: ExecutionPolicy, ref: ExecutionPolicy) -> bool:
    a, b = res.tool_policy, ref.tool_policy
    if any(getattr(a, f) and not getattr(b, f) for f in _TOOL_FLAGS):
        return False
    if not set(a.extra_allowed) <= set(b.extra_allowed):
        return False
    # contravariant: a denial list may only grow
    return set(a.extra_denied) >= set(b.extra_denied)


def _paths_no_wider(res: ExecutionPolicy, ref: ExecutionPolicy) -> bool:
    """Empty means UNRESTRICTED everywhere it is consumed (the filesystem gate in
    node/intent_loop, _restrict_allowed_paths, LeaseValidator.check_path_allowed),
    so empty is the TOP of this order, never a deny-all sentinel."""
    r, b = tuple(res.allowed_paths or ()), tuple(ref.allowed_paths or ())
    if not b:
        return True
    if not r:
        return False
    return paths_within(r, b)


def _escalation_no_wider(res: ExecutionPolicy, ref: ExecutionPolicy) -> bool:
    a, b = res.escalation_policy, ref.escalation_policy
    if not b.allow_escalation:
        # reference grants no escalation at all: the only no-wider result is off
        return not a.allow_escalation
    if not a.allow_escalation:
        return True  # off is strictly stricter; its other fields are inert
    if not set(a.grantable_tools) <= set(b.grantable_tools):
        return False
    if a.max_escalations > b.max_escalations:
        return False
    if a.max_ops_per_grant > b.max_ops_per_grant:
        return False
    # dropping a human gate the reference requires is a widening
    if b.require_human and not a.require_human:
        return False
    # effect_approval turns a denial into a question, so gaining it is a widening
    return (not a.effect_approval) or b.effect_approval


_NO_WIDER: dict[str, callable] = {
    "tool_policy": _tool_policy_no_wider,
    "allowed_paths": _paths_no_wider,
    "max_unattended_consequence": lambda r, b: (
        r.max_unattended_consequence <= b.max_unattended_consequence
    ),
    "child_mode": lambda r, b: (
        _CHILD_MODE_RANK[r.child_mode] >= _CHILD_MODE_RANK[b.child_mode]
    ),
    "max_child_depth": lambda r, b: r.max_child_depth <= b.max_child_depth,
    "escalation_policy": _escalation_no_wider,
    "export_mode": lambda r, b: _EXPORT_RANK[r.export_mode] >= _EXPORT_RANK[b.export_mode],
    "allowed_passthrough_commands": lambda r, b: (
        set(r.allowed_passthrough_commands) <= set(b.allowed_passthrough_commands)
    ),
    "allow_model_switch": lambda r, b: (not r.allow_model_switch) or b.allow_model_switch,
}


def _assert_no_wider(result: ExecutionPolicy, reference: ExecutionPolicy, what: str):
    widened = [f for f, le in _NO_WIDER.items() if not le(result, reference)]
    assert not widened, (
        f"{what} widened {widened}\n"
        f"  result    = {[(f, getattr(result, f)) for f in widened]}\n"
        f"  reference = {[(f, getattr(reference, f)) for f in widened]}"
    )


# ── strategies ────────────────────────────────────────────────────────────────

_ROOTS = ("/srv/a", "/srv/a/b", "/srv/c", "/opt/x")
_TOOLS = ("read", "write", "bash", "fetch", "send")
_CMDS = ("ls", "pwd", "git status")

_tool_policies = st.builds(
    ToolPolicy,
    allow_read=st.booleans(),
    allow_write=st.booleans(),
    allow_bash=st.booleans(),
    allow_search=st.booleans(),
    allow_spawn=st.booleans(),
    extra_allowed=st.lists(st.sampled_from(_TOOLS), max_size=3, unique=True).map(tuple),
    extra_denied=st.lists(st.sampled_from(_TOOLS), max_size=2, unique=True).map(tuple),
)

_escalation_policies = st.builds(
    EscalationPolicy,
    allow_escalation=st.booleans(),
    grantable_tools=st.lists(st.sampled_from(_TOOLS), max_size=3, unique=True).map(tuple),
    max_escalations=st.integers(min_value=0, max_value=5),
    max_ops_per_grant=st.integers(min_value=0, max_value=20),
    require_human=st.booleans(),
    effect_approval=st.booleans(),
)

_policies = st.builds(
    ExecutionPolicy,
    name=st.just("p"),
    tool_policy=_tool_policies,
    allowed_paths=st.lists(st.sampled_from(_ROOTS), max_size=3, unique=True).map(tuple),
    max_unattended_consequence=st.sampled_from(list(ConsequenceClass)),
    child_mode=st.sampled_from(list(ChildMode)),
    max_child_depth=st.integers(min_value=0, max_value=4),
    export_mode=st.sampled_from(list(ExportMode)),
    escalation_policy=_escalation_policies,
    allowed_passthrough_commands=st.lists(
        st.sampled_from(_CMDS), max_size=2, unique=True
    ).map(tuple),
    allow_model_switch=st.booleans(),
)

_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ── 1. coverage: the field set comes from the code, not from a wish list ──────


class TestCoverage:

    def test_every_authority_field_is_mapped(self):
        declared = {f.name for f in dataclasses.fields(AuthorityPolicy)}
        covered = set(_AUTHORITY_FIELD_MAP) | _IDENTITY_FIELDS
        assert declared == covered, (
            "AuthorityPolicy changed: map every authority field to its "
            "ExecutionPolicy carrier(s) and declare its order in _NO_WIDER "
            f"(unmapped={sorted(declared - covered)}, stale={sorted(covered - declared)})"
        )

    def test_every_mapped_carrier_declares_an_order(self):
        policy_fields = {f.name for f in dataclasses.fields(ExecutionPolicy)}
        carriers = {c for cs in _AUTHORITY_FIELD_MAP.values() for c in cs}
        assert carriers <= policy_fields, sorted(carriers - policy_fields)
        assert carriers == set(_NO_WIDER), (
            f"orders missing for {sorted(carriers - set(_NO_WIDER))}, "
            f"stale orders for {sorted(set(_NO_WIDER) - carriers)}"
        )

    def test_the_order_is_reflexive(self):
        """A sanity floor: a policy is never wider than itself. Catches an order
        declared with a flipped comparison."""
        p = ExecutionPolicy(
            name="p",
            allowed_paths=("/srv/a",),
            max_unattended_consequence=ConsequenceClass.REVERSIBLE,
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=("write",), require_human=True
            ),
        )
        _assert_no_wider(p, p, "identity")


# ── 2. preservation at every composition entry point ─────────────────────────


class TestParentRestriction:

    @_SETTINGS
    @given(child=_policies, parent=_policies)
    def test_never_widens_the_parent(self, child, parent):
        narrowed = PolicyComposer().apply_parent_restrictions(child, parent)
        _assert_no_wider(narrowed, parent, "apply_parent_restrictions")

    @_SETTINGS
    @given(child=_policies, parent=_policies)
    def test_is_idempotent(self, child, parent):
        once = PolicyComposer().apply_parent_restrictions(child, parent)
        twice = PolicyComposer().apply_parent_restrictions(once, parent)
        _assert_no_wider(twice, once, "re-restriction")
        _assert_no_wider(once, twice, "re-restriction (converse)")

    @_SETTINGS
    @given(child=_policies, parent=_policies)
    def test_only_allowed_paths_may_exceed_the_child(self, child, parent):
        """The one deliberate asymmetry, asserted rather than left implicit:
        when the child's roots fall entirely outside the parent ceiling,
        _restrict_allowed_paths falls back to the PARENT roots instead of an
        empty (= unrestricted) set. That is wider than the child asked for and
        still within the parent, so the ceiling invariant holds; every other
        field stays no wider than both operands."""
        narrowed = PolicyComposer().apply_parent_restrictions(child, parent)
        widened_vs_child = [
            f for f, le in _NO_WIDER.items() if not le(narrowed, child)
        ]
        assert widened_vs_child in ([], ["allowed_paths"]), widened_vs_child


class TestDeploymentOverlay:

    @_SETTINGS
    @given(policy=_policies, overlay=_policies)
    def test_overlay_never_widens_the_per_task_policy(self, policy, overlay):
        """Every field but allowed_paths: no wider than the per-task policy. The
        path exception is the disjoint case, covered by the two tests below —
        same declared asymmetry as the parent restriction, for the same reason
        (an empty allowlist means UNRESTRICTED downstream, so "confine to
        nothing" has no representation and the composition falls back to the
        trusted operand)."""
        composed = PolicyComposer(
            consequence_ceiling=overlay.max_unattended_consequence,
            escalation_policy=overlay.escalation_policy,
            allowed_paths=overlay.allowed_paths or None,
        ).compose(policy, [])
        widened = [f for f, le in _NO_WIDER.items() if not le(composed, policy)]
        assert widened in ([], ["allowed_paths"]), widened

    @_SETTINGS
    @given(policy=_policies, overlay=_policies)
    def test_overlay_is_always_a_ceiling_on_paths(self, overlay, policy):
        """Whatever the overlay does to allowed_paths, the result stays inside the
        operator workspace and is never unrestricted while the workspace is set."""
        if not overlay.allowed_paths:
            return
        composed = PolicyComposer(allowed_paths=overlay.allowed_paths).compose(
            policy, []
        )
        assert composed.allowed_paths, "filesystem ceiling was dropped"
        assert paths_within(composed.allowed_paths, overlay.allowed_paths)

    def test_disjoint_workspace_does_not_drop_the_filesystem_ceiling(self):
        """Regression: intersect_allowlist returns () for disjoint allowlists, and
        () is read as "unrestricted" by every consumer — so the intersection
        removed the ceiling entirely, widening BOTH operands. It now falls back to
        the operator workspace: wider than the per-task roots (which lie outside
        the workspace and cannot be honoured), never wider than the operator's."""
        policy = ExecutionPolicy(name="task", allowed_paths=("/srv/task",))
        composed = PolicyComposer(allowed_paths=("/srv/other",)).compose(policy, [])
        assert composed.allowed_paths, "filesystem ceiling was dropped"
        assert paths_within(composed.allowed_paths, ("/srv/other",))
        assert not paths_within(composed.allowed_paths, ("/srv/task",))

    @_SETTINGS
    @given(parent=_policies, child=_policies, overlay=_policies)
    def test_overlay_ceiling_holds_transitively(self, parent, child, overlay):
        """The deployment shape: parent and child are composed under the SAME
        overlay, so the operator ceiling reaches the child through the parent.
        (A parent composed under a different overlay is not a case the runtime
        produces, and the parent-restriction fallback would not preserve it.)"""
        composer = PolicyComposer(
            consequence_ceiling=overlay.max_unattended_consequence,
            escalation_policy=overlay.escalation_policy,
            allowed_paths=overlay.allowed_paths or None,
        )
        parent_eff = composer.compose(parent, [])
        child_eff = composer.compose(child, [], parent_policy=parent_eff)
        _assert_no_wider(child_eff, parent_eff, "compose(parent=...)")
        assert child_eff.max_unattended_consequence <= overlay.max_unattended_consequence
        if overlay.allowed_paths:
            assert paths_within(child_eff.allowed_paths, overlay.allowed_paths)


class TestExtensionOverrides:

    @_SETTINGS
    @given(
        base=_policies,
        requests=st.lists(
            st.fixed_dictionaries(
                {
                    "allow_read": st.booleans(),
                    "allow_write": st.booleans(),
                    "allow_bash": st.booleans(),
                    "allow_search": st.booleans(),
                    "allow_spawn": st.booleans(),
                    "extra_allowed_tools": st.lists(
                        st.sampled_from(_TOOLS), max_size=4, unique=True
                    ),
                }
            ),
            max_size=3,
        ),
    )
    def test_extensions_never_widen_the_base(self, base, requests):
        """A skill or plugin requests; core grants only what base already allows.
        Extension fragments are the one composition input a third party authors,
        so this is the same invariant under a hostile operand."""
        fragments = [
            ExtensionFragment(
                name=f"ext{i}",
                context_fragment="",
                required_tools=(),
                policy_overrides=dict(r),
                source="test",
            )
            for i, r in enumerate(requests)
        ]
        composed = PolicyComposer().compose(base, fragments)
        _assert_no_wider(composed, base, "apply_extension_overrides")
