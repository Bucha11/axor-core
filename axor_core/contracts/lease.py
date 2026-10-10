from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class LeaseAuthorityType(str, Enum):
    HUMAN_OPERATOR = "human_operator"
    AUTOMATED_POLICY = "automated_policy"
    GOVERNANCE_RESET = "governance_reset"


@dataclass
class CapabilityLease:
    """
    Scoped, expiring, auditable capability grant.

    Created when a human or governance boundary approves an escalation.
    Replaces the simple flag-based EscalationPolicy grant.

    Invariants enforced by LeaseValidator:
    - leases expire by wall-clock TTL
    - leases expire by intent-count (max_uses)
    - leases are non-transitive by default (child cannot inherit)
    - leases cannot exceed parent policy ceiling
    - worker cannot mutate an issued lease
    """

    grant_id: str
    granted_by: str
    authority_type: LeaseAuthorityType
    grant_scope: str
    allowed_tools: frozenset[str]
    allowed_operations: frozenset[str]
    allowed_paths: tuple[str, ...]
    allowed_providers: frozenset[str]
    allowed_child_depth: int
    creation_time: float
    expiration_time: float
    max_uses: int
    used_count: int = field(default=0)
    non_transitive: bool = True
    parent_policy_ceiling_ref: str = ""
    reason_code: str = ""
    audit_id: str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiration_time

    @property
    def is_exhausted(self) -> bool:
        return self.used_count >= self.max_uses

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.is_exhausted

    def increment_use(self) -> None:
        self.used_count += 1
