from __future__ import annotations

import time
import uuid
from typing import Sequence

from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.capability.resolver import CapabilityResolver
# Canonical path primitives — single source of truth (axor_core.security.paths).
# Re-exported here for backward compatibility with existing imports.
from axor_core.security.paths import (
    path_matches_allowlist,
    paths_within,
)

_resolver = CapabilityResolver()


def extract_path_arg(args: dict) -> str:
    """Extract the path-like argument from tool args.

    Covers the common argument names across adapters (path, file_path, file,
    url, uri) so path enforcement cannot be sidestepped by using an alias.
    """
    for key in ("path", "file_path", "file", "url", "uri"):
        value = args.get(key)
        if value:
            return str(value)
    return ""


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
        # Path ceiling: when the parent policy restricts paths, the lease may not
        # grant access to paths outside that ceiling.
        parent_paths = getattr(parent_policy, "allowed_paths", ()) or ()
        if parent_paths and lease.allowed_paths:
            if not paths_within(lease.allowed_paths, parent_paths):
                return (
                    f"lease grants paths outside parent ceiling: "
                    f"{lease.allowed_paths!r} not within {tuple(parent_paths)!r}"
                )
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
            max_uses=max_uses,
            used_count=0,
            non_transitive=True,
            parent_policy_ceiling_ref=getattr(parent_policy, "name", ""),
            reason_code=reason_code,
            audit_id=audit_id,
        )
        error = self.validate_against_policy_ceiling(lease, parent_policy)
        return lease, error
