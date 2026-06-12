"""EscalationManager — the session's escalation grants and capability leases.

Extracted from :class:`~axor_core.node.intent_loop.IntentLoop` so the loop
orchestrates rather than implements this concern. It owns three things and nothing
else:

  • the active grants / leases and the flood guard (state);
  • :meth:`resolve` — does a lease or grant cover this tool call? (the lease/grant
    branch of the loop's policy resolution);
  • :meth:`grant_from_intent` — turn an ``ESCALATE_POLICY`` intent into a grant +
    lease, behind the policy's flood guard and human-approval hook;
  • :meth:`covers` — the consequence gate's "is there a governance gate for this
    tool?" query.

Pure move of the loop's logic — same order, same decisions, same trace events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axor_core.capability.flood import EscalationFloodGuard
from axor_core.capability.lease_validator import (
    LeaseValidator,
    extract_path_arg,
    path_matches_allowlist,
)
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.contracts.result import ExecutorEvent
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.trace import (
    EscalationDeniedEvent,
    EscalationGrantedEvent,
    TraceEventKind,
)

if TYPE_CHECKING:
    from axor_core.contracts.lease import CapabilityLease
    from axor_core.node.intent_loop import EscalationCallback


def _safe_int(value: object, default: int) -> int:
    """Parse an int from untrusted args without raising on bad input."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class _GrantedEscalation:
    tool: str
    paths: list[str]  # empty = no path restriction
    ops_remaining: int


class EscalationManager:
    """Owns escalation grants + capability leases for one session."""

    def __init__(self, *, escalation_callback: "EscalationCallback | None" = None) -> None:
        self._granted_escalations: dict[str, _GrantedEscalation] = {}
        self._capability_leases: dict[str, "CapabilityLease"] = {}
        self._lease_validator = LeaseValidator()
        self._escalation_count = 0
        self._flood_guard = EscalationFloodGuard()
        self._escalation_callback = escalation_callback

    def covers(self, tool_name: str) -> bool:
        """True if an active escalation grant or capability lease covers this tool —
        the governance gate the consequence axis accepts for an over-ceiling sink."""
        return (
            tool_name in self._granted_escalations
            or tool_name in self._capability_leases
        )

    def resolve(self, tool_name: str, tool_args: dict) -> PolicyDecision | None:
        """The lease/grant branch of policy resolution.

        Returns a :class:`PolicyDecision` when a lease or grant applies (APPROVE via
        a grant, or DENY on an expired/exhausted lease or a path violation), else
        ``None`` — no lease/grant for this tool, so the caller falls through to the
        policy's ``allowed_tools``.
        """
        lease = self._capability_leases.get(tool_name)
        if lease is not None:
            if not self._lease_validator.is_valid(lease):
                del self._capability_leases[tool_name]
                if tool_name in self._granted_escalations:
                    del self._granted_escalations[tool_name]
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=f"capability lease for '{tool_name}' has expired or been exhausted",
                )
            tool_path = extract_path_arg(tool_args)
            if not self._lease_validator.check_path_allowed(lease, tool_path):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=f"lease for '{tool_name}' restricts to paths {lease.allowed_paths!r}",
                )
            lease.increment_use()

        grant = self._granted_escalations.get(tool_name)
        if grant is not None:
            if grant.paths and not lease:
                tool_path = extract_path_arg(tool_args)
                if not path_matches_allowlist(tool_path, grant.paths):
                    return PolicyDecision(
                        kind=PolicyDecisionKind.DENY,
                        reason=f"escalation grant for '{tool_name}' restricts to paths {grant.paths!r}",
                    )
            grant.ops_remaining -= 1
            remaining = grant.ops_remaining
            if remaining <= 0:
                del self._granted_escalations[tool_name]
            return PolicyDecision(
                kind=PolicyDecisionKind.APPROVE,
                reason=f"approved via escalation grant ({remaining} ops remaining)",
            )

        return None

    async def grant_from_intent(
        self,
        event: ExecutorEvent,
        envelope: ExecutionEnvelope,
        trace_events: list,
    ) -> dict:
        """Turn an ``ESCALATE_POLICY`` intent into a grant + capability lease, behind
        the policy's grantable-tools list, flood guard, and human-approval hook.
        Appends the granted/denied trace event. Returns the result dict."""
        args = event.payload.get("args", {})
        tool = args.get("tool", "")
        reason = args.get("reason", "")
        paths = args.get("paths", [])
        max_ops = min(
            _safe_int(args.get("max_ops", 10), default=10),
            envelope.policy.escalation_policy.max_ops_per_grant,
        )
        ep = envelope.policy.escalation_policy
        node_id = envelope.node_id
        tool_use_id = event.payload.get("tool_use_id", "")

        def _deny(deny_reason: str) -> dict:
            trace_events.append(
                EscalationDeniedEvent(
                    kind=TraceEventKind.ESCALATION_DENIED,
                    node_id=node_id,
                    sequence=len(trace_events),
                    tool=tool,
                    reason=deny_reason,
                )
            )
            return {"error": "escalation_denied", "reason": deny_reason}

        if not ep.allow_escalation:
            return _deny("escalation not allowed by policy")

        if tool not in ep.grantable_tools:
            return _deny(f"tool '{tool}' is not in grantable_tools")

        if max_ops <= 0:
            return _deny("escalation max_ops must be a positive integer")

        if self._escalation_count >= ep.max_escalations:
            return _deny(f"max escalations reached ({ep.max_escalations})")

        flood_denial = self._flood_guard.check(
            tool=tool,
            paths=paths,
            reason=reason,
            session_id=envelope.node_id,
            node_id=envelope.node_id,
        )
        if flood_denial is not None:
            return _deny(flood_denial)

        auto_approved = True
        if ep.require_human:
            if self._escalation_callback is None:
                return _deny(
                    "escalation requires human approval but no callback is configured"
                )
            approved = await self._escalation_callback(
                tool_use_id, tool, paths, max_ops
            )
            auto_approved = False
            if not approved:
                return _deny("human denied escalation request")

        # Create the CapabilityLease first — if it fails the grant is not stored,
        # preventing a grant-without-TTL bypass.
        lease, lease_err = self._lease_validator.create_lease(
            granted_by="operator" if ep.require_human else "auto_policy",
            authority_type=(
                LeaseAuthorityType.HUMAN_OPERATOR
                if ep.require_human
                else LeaseAuthorityType.AUTOMATED_POLICY
            ),
            allowed_tools=[tool],
            parent_policy=envelope.policy,
            allowed_paths=paths,
            ttl_seconds=300.0,
            max_uses=max_ops,
            reason_code=reason,
        )
        if lease_err is not None:
            return _deny(f"escalation rejected: lease creation failed ({lease_err})")

        self._capability_leases[tool] = lease
        self._granted_escalations[tool] = _GrantedEscalation(
            tool=tool,
            paths=paths,
            ops_remaining=max_ops,
        )
        self._escalation_count += 1
        self._flood_guard.record_approval()

        trace_events.append(
            EscalationGrantedEvent(
                kind=TraceEventKind.ESCALATION_GRANTED,
                node_id=node_id,
                sequence=len(trace_events),
                tool=tool,
                paths=paths,
                max_ops=max_ops,
                reason=reason,
                auto_approved=auto_approved,
            )
        )
        return {
            "granted": True,
            "tool": tool,
            "max_ops": max_ops,
            "paths": paths,
        }
