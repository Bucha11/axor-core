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
