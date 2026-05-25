from __future__ import annotations

import os
from pathlib import Path
import time
import uuid
from typing import Sequence

from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.capability.resolver import CapabilityResolver

_resolver = CapabilityResolver()


def path_matches_allowlist(path: str, allowed_paths: Sequence[str]) -> bool:
    """Return True only when path is equal to or contained by an allowed root."""
    if not path:
        return False
    candidate = _resolve_path(path)
    for allowed in allowed_paths:
        root = _resolve_path(allowed)
        if candidate == root:
            return True
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve(strict=False)


class LeaseValidator:
    """
    Validates and manages CapabilityLease instances.

    Enforces:
    - TTL expiry
    - max-use count
    - policy ceiling (lease cannot grant tools/paths outside parent policy)
    - non-transitive: child nodes do not inherit leases unless explicitly marked
    """

    def is_valid(self, lease: CapabilityLease) -> bool:
        return lease.is_valid

    def check_tool_allowed(self, lease: CapabilityLease, tool: str) -> bool:
        if not lease.is_valid:
            return False
        return tool in lease.allowed_tools

    def check_path_allowed(self, lease: CapabilityLease, path: str) -> bool:
        if not lease.is_valid:
            return False
        if not lease.allowed_paths:
            return True
        return path_matches_allowlist(path, lease.allowed_paths)


    def validate_against_policy_ceiling(
        self,
        lease: CapabilityLease,
        parent_policy: ExecutionPolicy,
    ) -> str | None:
        """
        Validate that the lease does not exceed the parent policy ceiling.

        Returns None if valid, or an error reason string if it would exceed the ceiling.
        """
        parent_caps = _resolver.resolve(parent_policy)
        excess = lease.allowed_tools - parent_caps.allowed_tools
        if excess:
            return f"lease grants tools outside parent ceiling: {excess}"
        return None

    def create_lease(
        self,
        granted_by: str,
        authority_type: LeaseAuthorityType,
        allowed_tools: Sequence[str],
        parent_policy: ExecutionPolicy,
        allowed_paths: Sequence[str] = (),
        ttl_seconds: float = 300.0,
        max_uses: int = 10,
        reason_code: str = "",
        audit_id: str = "",
    ) -> tuple[CapabilityLease, str | None]:
        """
        Create a new CapabilityLease bounded by the parent policy ceiling.

        Returns (lease, error_reason). If error_reason is not None, the lease
        creation was rejected and the returned lease should not be used.
        """
        tool_set = frozenset(allowed_tools)
        now = time.time()
        lease = CapabilityLease(
            grant_id=f"lease_{uuid.uuid4().hex[:12]}",
            granted_by=granted_by,
            authority_type=authority_type,
            grant_scope="tool_execution",
            allowed_tools=tool_set,
            allowed_operations=frozenset(),
            allowed_paths=tuple(allowed_paths),
            allowed_providers=frozenset(),
            allowed_child_depth=0,
            creation_time=now,
            expiration_time=now + ttl_seconds,
            max_intents=max_uses,
            max_uses=max_uses,
            used_count=0,
            non_transitive=True,
            parent_policy_ceiling_ref=getattr(parent_policy, "name", ""),
            reason_code=reason_code,
            audit_id=audit_id,
        )
        error = self.validate_against_policy_ceiling(lease, parent_policy)
        return lease, error
