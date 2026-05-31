"""
Adversarial tests: lease ceiling — Step 26.

CapabilityLease cannot grant capabilities that exceed the parent policy ceiling,
and cannot bypass TTL or max-use limits.

Three variants:
  1. Tool expansion — lease claiming tools outside parent policy is rejected
  2. Path expansion — lease cannot authorize paths the parent restricts
  3. TTL/max-use bypass — expired or exhausted lease is treated as no-lease

Reverse-verify: removing LeaseValidator.validate_against_policy_ceiling breaks
variant 1; removing is_valid check breaks variants 2 and 3.
"""
from __future__ import annotations

import time

import pytest

from axor_core.capability.lease_validator import LeaseValidator
from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy


# ── helpers ───────────────────────────────────────────────────────────────────

def _policy(allowed_tools: frozenset[str] | None = None) -> ExecutionPolicy:
    p = ExecutionPolicy(name="test")
    if allowed_tools is not None:
        object.__setattr__(p, "allowed_tools", allowed_tools)
    return p


def _lease(
    tools: frozenset[str],
    paths: tuple[str, ...] = (),
    ttl_offset: float = 300.0,
    max_uses: int = 10,
    used_count: int = 0,
) -> CapabilityLease:
    now = time.time()
    return CapabilityLease(
        grant_id="test-lease",
        granted_by="operator",
        authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution",
        allowed_tools=tools,
        allowed_operations=frozenset(),
        allowed_paths=paths,
        allowed_providers=frozenset(),
        allowed_child_depth=0,
        creation_time=now,
        expiration_time=now + ttl_offset,
        max_uses=max_uses,
        used_count=used_count,
        non_transitive=True,
    )


VALIDATOR = LeaseValidator()


# ── variant 1: tool expansion blocked ─────────────────────────────────────────

def test_lease_cannot_grant_tools_outside_parent_ceiling():
    """
    A lease requesting tools not in the parent policy ceiling must be rejected.

    Parent allows only {"read"}; lease requests {"read", "execute_shell"}.
    """
    parent = _policy(allowed_tools=frozenset(["read"]))
    lease = _lease(tools=frozenset(["read", "execute_shell"]))

    error = VALIDATOR.validate_against_policy_ceiling(lease, parent)
    assert error is not None, "lease expanding tools beyond parent should be rejected"
    assert "execute_shell" in error


def test_lease_within_parent_ceiling_is_valid():
    """Lease requesting a tool subset of parent ceiling is allowed."""
    parent = _policy(allowed_tools=frozenset(["read", "write", "bash"]))
    lease = _lease(tools=frozenset(["read"]))

    error = VALIDATOR.validate_against_policy_ceiling(lease, parent)
    assert error is None, "lease within ceiling should be accepted"


# ── variant 2: path expansion blocked ─────────────────────────────────────────

def test_lease_path_restriction_enforced():
    """
    A lease restricted to /safe/path must not authorize access to /etc/passwd.
    """
    lease = _lease(
        tools=frozenset(["write"]),
        paths=("/safe/path",),
    )
    assert VALIDATOR.check_path_allowed(lease, "/safe/path/file.txt")
    assert not VALIDATOR.check_path_allowed(lease, "/etc/passwd"), (
        "path outside lease restriction must be denied"
    )
    assert not VALIDATOR.check_path_allowed(lease, "/safe/path/../etc/passwd"), (
        "path traversal outside restriction must be denied"
    )


def test_lease_invalid_does_not_authorize_path():
    """Expired lease must not authorize any path."""
    expired = _lease(
        tools=frozenset(["read"]),
        paths=("/safe/",),
        ttl_offset=-1.0,  # already expired
    )
    assert expired.is_expired
    assert not VALIDATOR.check_path_allowed(expired, "/safe/file.txt"), (
        "expired lease must not authorize any path"
    )


# ── variant 3: TTL / max-use bypass blocked ────────────────────────────────────

def test_expired_lease_is_invalid():
    """An expired lease must not be valid."""
    expired = _lease(tools=frozenset(["read"]), ttl_offset=-10.0)  # expired 10s ago
    assert expired.is_expired
    assert not expired.is_valid
    assert not VALIDATOR.is_valid(expired)
    assert not VALIDATOR.check_tool_allowed(expired, "read")


def test_exhausted_lease_is_invalid():
    """A lease at max_uses must not be valid."""
    exhausted = _lease(tools=frozenset(["read"]), max_uses=5, used_count=5)
    assert exhausted.is_exhausted
    assert not exhausted.is_valid
    assert not VALIDATOR.is_valid(exhausted)
    assert not VALIDATOR.check_tool_allowed(exhausted, "read")
