"""Build an :class:`~axor_core.kernel.authority.AuthorityView` from live objects.

The invariant and the legal transitions are defined in Ring 0
(:mod:`axor_core.kernel.authority`). This module is the Ring 1 half: it READS the
runtime objects that actually decide — the policy, the capability resolver, the
degradation engine's narrowed policy, the escalation manager's own tables — and
presents them in the shape the invariant is stated over.

The split is the point. A state machine that kept its own copy of what is in
force would prove things about the copy; this one has no state of its own, so a
violation it reports is a violation in the objects the enforcement path uses.
"""

from __future__ import annotations

from axor_core.capability.lease_validator import LeaseValidator
from axor_core.capability.resolver import CapabilityResolver
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.kernel.authority import AuthorityView

_resolver = CapabilityResolver()
_leases = LeaseValidator()


def authority_view(
    *,
    base_policy: ExecutionPolicy,
    effective_policy: ExecutionPolicy | None = None,
    escalation: object | None = None,
    degradation: object | None = None,
) -> AuthorityView:
    """Snapshot the authority situation.

    ``base_policy`` is the trusted policy for the turn — it defines ``B``
    (``A₀ ∪ G`` plus the path and consequence ceilings). ``effective_policy`` is
    what is in force after degradation has narrowed it (defaults to the base).
    ``escalation`` is an :class:`~axor_core.node.escalation.EscalationManager`,
    or None when no authority has been granted. ``degradation`` is the
    :class:`~axor_core.degradation.engine.DegradationEngine`, consulted for the
    level's own tool rules — which a grant does NOT reopen, so a tool the level
    refuses is not in force however it was granted. Asked of the engine rather
    than re-derived here: a probe that re-implemented the rules could say a tool
    is in force while the enforcement path refuses it, which is the one way a
    containment check becomes decorative.
    """
    effective = effective_policy if effective_policy is not None else base_policy
    records = (
        escalation.authority_records()
        if escalation is not None and hasattr(escalation, "authority_records")
        else None
    )
    class_grants = records.class_grants if records else frozenset()
    unspent = records.unspent_calls if records else frozenset()
    spent = records.spent_calls if records else frozenset()

    in_force = frozenset(_resolver.resolve(effective).allowed_tools)
    # Live authority extends what is in force: a class grant for its tool, an
    # unspent call authorisation for the one call it names.
    in_force |= frozenset(class_grants) | {tool for tool, _ in unspent}
    # ...and the degradation level takes back what it refuses by name, grant or
    # no grant.
    if degradation is not None and hasattr(degradation, "tool_denial"):
        in_force = frozenset(
            t for t in in_force
            if degradation.tool_denial(t, effective_policy or base_policy) is None
        )

    esc = getattr(base_policy, "escalation_policy", None)
    return AuthorityView(
        ceiling_tools=_leases.delegation_ceiling(base_policy),
        grantable_tools=frozenset(getattr(esc, "grantable_tools", ()) or ()),
        ceiling_paths=tuple(base_policy.allowed_paths or ()),
        ceiling_consequence=base_policy.max_unattended_consequence,
        in_force_tools=in_force,
        in_force_paths=tuple(effective.allowed_paths or ()),
        in_force_consequence=effective.max_unattended_consequence,
        class_grants=frozenset(class_grants),
        unspent_calls=frozenset(unspent),
        spent_calls=frozenset(spent),
    )
