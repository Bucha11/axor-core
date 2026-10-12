from __future__ import annotations

import time


from axor_core.capability.lease_validator import LeaseValidator
from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy


def _make_lease(
    tools=("write",),
    max_uses=5,
    ttl=300.0,
    paths=(),
    non_transitive=True,
) -> CapabilityLease:
    now = time.time()
    return CapabilityLease(
        grant_id="lease_test",
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution",
        allowed_tools=frozenset(tools),
        allowed_operations=frozenset(),
        allowed_paths=tuple(paths),
        allowed_providers=frozenset(),
        allowed_child_depth=0,
        creation_time=now,
        expiration_time=now + ttl,
        max_uses=max_uses,
        used_count=0,
        non_transitive=non_transitive,
    )


def _FakePolicy() -> ExecutionPolicy:
    return ExecutionPolicy(
        name="standard",
        tool_policy=ToolPolicy(allow_read=True, allow_write=True),
    )


def test_valid_lease():
    validator = LeaseValidator()
    lease = _make_lease()
    assert validator.is_valid(lease)


def test_expired_lease_denied():
    validator = LeaseValidator()
    lease = _make_lease(ttl=-1.0)
    assert not validator.is_valid(lease)
    assert not validator.check_tool_allowed(lease, "write")


def test_exhausted_lease_denied():
    validator = LeaseValidator()
    lease = _make_lease(max_uses=3)
    lease.used_count = 3
    assert not validator.is_valid(lease)
    assert not validator.check_tool_allowed(lease, "write")


def test_tool_in_lease_allowed():
    validator = LeaseValidator()
    lease = _make_lease(tools=["bash"])
    assert validator.check_tool_allowed(lease, "bash")
    assert not validator.check_tool_allowed(lease, "write")


def test_path_restriction():
    validator = LeaseValidator()
    lease = _make_lease(paths=["/tmp/safe/"])
    assert validator.check_path_allowed(lease, "/tmp/safe/file.py")
    assert not validator.check_path_allowed(lease, "/etc/passwd")


def test_path_restriction_rejects_prefix_sibling():
    validator = LeaseValidator()
    lease = _make_lease(paths=["/tmp/safe"])
    assert validator.check_path_allowed(lease, "/tmp/safe/file.py")
    assert not validator.check_path_allowed(lease, "/tmp/safe-secret/file.py")


def test_path_restriction_rejects_parent_traversal():
    validator = LeaseValidator()
    lease = _make_lease(paths=["/tmp/safe"])
    assert not validator.check_path_allowed(lease, "/tmp/safe/../safe-secret/file.py")


def test_path_restriction_rejects_missing_path():
    validator = LeaseValidator()
    lease = _make_lease(paths=["/tmp/safe"])
    assert not validator.check_path_allowed(lease, "")


def test_lease_exceeding_parent_ceiling_rejected():
    validator = LeaseValidator()
    # Policy only allows read+write; lease tries to grant bash too
    lease = _make_lease(tools=["read", "write", "bash"])
    error = validator.validate_against_policy_ceiling(lease, _FakePolicy())
    assert error is not None
    assert "bash" in error


def test_lease_within_parent_ceiling_ok():
    validator = LeaseValidator()
    lease = _make_lease(tools=["write"])
    error = validator.validate_against_policy_ceiling(lease, _FakePolicy())
    assert error is None


def test_create_lease_rejects_ceiling_violation():
    validator = LeaseValidator()
    lease, error = validator.create_lease(
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        allowed_tools=["write", "bash"],
        parent_policy=_FakePolicy(),
    )
    assert error is not None
    assert "bash" in error


def test_non_transitive_by_default():
    lease = _make_lease()
    assert lease.non_transitive is True


def test_increment_use():
    lease = _make_lease(max_uses=2)
    lease.increment_use()
    assert lease.used_count == 1
    assert lease.is_valid
    lease.increment_use()
    assert lease.is_exhausted


# ── the delegation ceiling: A₀ ∪ G, not A₀ ───────────────────────────────────
#
# "Callable now" and "may be granted" are different sets. A lease exists precisely
# to put a grantable tool in force for a bounded while, so validating it against
# the CURRENT capability set rejected every grant that had a point — which is what
# made documented capability-on-demand unusable. Validating against A₀ ∪ G fixes
# that without making G callable: G stays out of resolve().

def _ro_policy(grantable=("write",)):
    from axor_core.contracts.policy import EscalationPolicy, ExecutionPolicy, ToolPolicy
    return ExecutionPolicy(
        name="ro",
        tool_policy=ToolPolicy(allow_read=True),
        escalation_policy=EscalationPolicy(
            allow_escalation=True, grantable_tools=tuple(grantable)
        ),
    )


def test_delegation_ceiling_is_available_plus_grantable():
    from axor_core.capability.resolver import CapabilityResolver

    policy = _ro_policy()
    available = CapabilityResolver().resolve(policy).allowed_tools
    ceiling = LeaseValidator().delegation_ceiling(policy)
    assert "write" not in available          # G is NOT callable before a grant
    assert ceiling == frozenset(available) | {"write"}


def test_a_lease_for_a_grantable_tool_is_accepted():
    from axor_core.contracts.lease import LeaseAuthorityType

    lease, err = LeaseValidator().create_lease(
        "operator", LeaseAuthorityType.HUMAN_OPERATOR, ["write"], _ro_policy()
    )
    assert err is None, err
    assert lease.allowed_tools == frozenset({"write"})


def test_a_lease_for_a_tool_in_neither_set_is_rejected():
    from axor_core.contracts.lease import LeaseAuthorityType

    _lease, err = LeaseValidator().create_lease(
        "operator", LeaseAuthorityType.HUMAN_OPERATOR, ["bash"], _ro_policy()
    )
    assert err is not None and "delegation ceiling" in err


def test_the_delegation_ceiling_follows_the_narrowed_escalation_policy():
    """G is an authority field: parent and deployment composition narrow it like
    every other one, so the delegation ceiling is bounded by the trusted ceiling."""
    from axor_core.policy import PolicyComposer

    child = _ro_policy(grantable=("write", "bash"))
    parent = _ro_policy(grantable=("write",))
    narrowed = PolicyComposer().apply_parent_restrictions(child, parent)
    assert "bash" not in LeaseValidator().delegation_ceiling(narrowed)
