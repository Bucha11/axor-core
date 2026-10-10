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
    tool?" query;
  • :meth:`grant_for_effect` — the just-in-time path: one approved call becomes a
    one-shot grant bound to that call's effect digest.

Scope is explicit on every grant (:class:`~axor_core.contracts.approval.AuthorityScope`).
A CLASS grant is the original shape — the tool, within bounds, for ``max_ops``
uses — and is all that an ``ESCALATE_POLICY`` intent naming no arguments can
produce. An EFFECT grant is bound to one call's effect digest and is one-shot: a
different recipient or a different amount is a different effect and is not
covered. Binding uses the ALL-ARGUMENTS digest, not the declared effect
signature, so it does not rely on ``driving_args`` being declared correctly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

_log = logging.getLogger("axor.escalation")

import inspect
import warnings

from axor_core.capability.flood import EscalationFloodGuard
from axor_core.capability.lease_validator import (
    LeaseValidator,
    extract_path_arg,
    path_matches_allowlist,
)
from axor_core.contracts.approval import ApprovalRequest, AuthorityScope
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.contracts.result import ExecutorEvent
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.policy.effect import effect_digest, effect_signature
from axor_core.contracts.trace import (
    EscalationDeniedEvent,
    EscalationGrantedEvent,
    TraceEventKind,
)

if TYPE_CHECKING:
    from axor_core.contracts.lease import CapabilityLease
    from axor_core.node.intent_loop import EscalationCallback


def adapt_escalation_callback(cb):
    """Accept both callback protocols: the current ``(ApprovalRequest) -> bool``
    and the legacy ``(tool_use_id, tool, paths, max_ops) -> bool``.

    The legacy shape cannot express an effect-scoped question (it has nowhere to
    put the arguments), so an adapted callback is only ever asked CLASS-scoped
    questions — :meth:`EscalationManager.grant_for_effect` refuses to put an
    effect request to one rather than silently downgrading it to a class answer.
    """
    if cb is None:
        return None
    try:
        params = list(inspect.signature(cb).parameters.values())
    except (TypeError, ValueError):  # un-introspectable: assume current protocol
        return cb
    positional = [
        prm for prm in params
        if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(prm.kind is prm.VAR_POSITIONAL for prm in params)
    if variadic or len(positional) < 4:
        return cb  # current protocol (one ApprovalRequest) or free-form
    warnings.warn(
        "EscalationCallback(tool_use_id, tool, paths, max_ops) is deprecated: take "
        "a single ApprovalRequest instead. The legacy shape cannot carry the call's "
        "arguments, so it can only answer CLASS-scoped (tool-wide) questions and "
        "effect-scoped approval is unavailable to it.",
        DeprecationWarning,
        stacklevel=3,
    )

    async def _adapted(request: ApprovalRequest) -> bool:
        return await cb(
            request.tool_use_id, request.tool, list(request.paths), request.max_ops
        )

    _adapted.__axor_legacy_callback__ = True  # type: ignore[attr-defined]
    return _adapted


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
    # Empty = CLASS scope (any call of the tool within bounds). Set = EFFECT
    # scope: only a call whose effect digest equals this.
    effect_digest: str = ""


@dataclass
class _PendingConsumption:
    """A lease/grant consumption decided but not yet applied. The loop commits it
    only when the whole call is finally approved, so a call denied by a later
    data-flow gate does not burn a lease use or a grant op."""

    manager: "EscalationManager"
    lease: "CapabilityLease | None"
    grant: "_GrantedEscalation | None"
    grant_tool: str | None

    def commit(self) -> None:
        if self.lease is not None:
            self.lease.increment_use()
        if self.grant is not None:
            self.grant.ops_remaining -= 1
            if self.grant.ops_remaining <= 0:
                self.manager._granted_escalations.pop(self.grant_tool, None)


class EscalationManager:
    """Owns escalation grants + capability leases for one session."""

    def __init__(
        self,
        *,
        escalation_callback: "EscalationCallback | None" = None,
        driving_args: "dict[str, object] | None" = None,
    ) -> None:
        # Reported to the approver as the operator's declaration of which args are
        # effect-determining. NOT used for binding: the digest covers every
        # argument, so a mis-declared driving_args cannot widen an approval.
        self._driving_args = {
            k: tuple(v) for k, v in (driving_args or {}).items()
        }
        self._granted_escalations: dict[str, _GrantedEscalation] = {}
        self._capability_leases: dict[str, "CapabilityLease"] = {}
        self._lease_validator = LeaseValidator()
        self._escalation_count = 0
        self._flood_guard = EscalationFloodGuard()
        self._escalation_callback = adapt_escalation_callback(escalation_callback)

    def covers(self, tool_name: str, tool_args: dict | None = None) -> bool:
        """True if an active grant or lease covers this call — the governance gate
        the consequence axis accepts for an over-ceiling sink.

        A CLASS-scoped grant/lease covers every call of the tool. An EFFECT-bound
        one covers only the call it was issued for, so the digest must match;
        without ``tool_args`` (no call in hand) only class-scoped authority counts.
        """
        grant = self._granted_escalations.get(tool_name)
        lease = self._capability_leases.get(tool_name)
        candidates = [c for c in (grant, lease) if c is not None]
        if not candidates:
            return False
        bindings = [
            getattr(c, "effect_digest", "") or getattr(c, "bound_effect", "")
            for c in candidates
        ]
        if any(b == "" for b in bindings):
            return True  # at least one class-scoped authority
        if tool_args is None:
            return False
        digest = effect_digest(tool_name, tool_args)
        return any(b == digest for b in bindings)

    def evaluate(
        self, tool_name: str, tool_args: dict
    ) -> tuple[PolicyDecision | None, "_PendingConsumption | None"]:
        """The lease/grant branch of policy resolution — decided WITHOUT consuming.

        Returns ``(decision, pending)``:
          • ``decision`` is a DENY on an expired/exhausted lease or a path violation
            (``pending=None`` — nothing to consume), an APPROVE when a grant covers
            the tool, or ``None`` when no grant applies and the caller should fall
            through to the policy's ``allowed_tools``.
          • ``pending`` carries the lease use / grant op to apply *only* once the
            whole call is finally approved (see :class:`_PendingConsumption`), so a
            call denied by a later data-flow gate burns nothing. An expired lease is
            still cleaned up here (that is not a consumption).
        """
        lease = self._capability_leases.get(tool_name)
        pending_lease: "CapabilityLease | None" = None
        if lease is not None:
            if not self._lease_validator.is_valid(lease):
                del self._capability_leases[tool_name]
                if tool_name in self._granted_escalations:
                    del self._granted_escalations[tool_name]
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=f"capability lease for '{tool_name}' has expired or been exhausted",
                ), None
            tool_path = extract_path_arg(tool_args)
            if not self._lease_validator.check_path_allowed(lease, tool_path):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=f"lease for '{tool_name}' restricts to paths {lease.allowed_paths!r}",
                ), None
            # An effect-bound lease is evidence for ONE call. A different call of
            # the same tool is simply not covered by it — it falls through to the
            # ordinary policy rather than being denied, because a bound
            # authorisation is additive evidence for its own effect, never a new
            # restriction on the tool.
            if lease.bound_effect and not self._lease_validator.check_effect_allowed(
                lease, effect_digest(tool_name, tool_args)
            ):
                lease = None
            else:
                pending_lease = lease  # consume on commit, not now

        grant = self._granted_escalations.get(tool_name)
        if grant is not None and grant.effect_digest and (
            grant.effect_digest != effect_digest(tool_name, tool_args)
        ):
            grant = None  # bound to another effect — not evidence for this call
        if grant is not None:
            if grant.paths and not lease:
                tool_path = extract_path_arg(tool_args)
                if not path_matches_allowlist(tool_path, grant.paths):
                    return PolicyDecision(
                        kind=PolicyDecisionKind.DENY,
                        reason=f"escalation grant for '{tool_name}' restricts to paths {grant.paths!r}",
                    ), None
            remaining_after = grant.ops_remaining - 1
            pending = _PendingConsumption(
                manager=self, lease=pending_lease, grant=grant, grant_tool=tool_name
            )
            return PolicyDecision(
                kind=PolicyDecisionKind.APPROVE,
                reason=f"approved via escalation grant ({remaining_after} ops remaining)",
            ), pending

        if pending_lease is not None:
            # Valid lease, no grant: fall through to allowed_tools, but the lease use
            # is still consumed on final approval.
            return None, _PendingConsumption(
                manager=self, lease=pending_lease, grant=None, grant_tool=None
            )

        return None, None

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
        # An escalation request MAY name the call it is for. When it does, the
        # grant is bound to that effect and is one-shot: the approver is asked
        # about a concrete call instead of a tool-wide class, and nothing else can
        # be spent on the grant. Omitted (the original shape) → CLASS scope.
        intended = args.get("args")
        effect_bound = isinstance(intended, dict)
        digest = effect_digest(tool, intended) if effect_bound else ""
        scope = AuthorityScope.EFFECT if effect_bound else AuthorityScope.CLASS
        max_ops = 1 if effect_bound else min(
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
            # Fail-closed exception boundary: a crashing approver (terminal
            # EOF, broken operator callback) is a denial with a proper trace
            # event, never an exception escaping governed execution.
            request = ApprovalRequest(
                tool_use_id=tool_use_id,
                tool=tool,
                scope=scope,
                paths=tuple(paths),
                max_ops=max_ops,
                reason=reason,
                args=intended if effect_bound else {},
                effect_digest=digest,
                effect_signature=(
                    effect_signature(tool, intended, self._driving_args)
                    if effect_bound else ()
                ),
            )
            try:
                approved = await self._escalation_callback(request)
            except Exception:
                _log.exception(
                    "escalation approval callback raised for tool %r — denying",
                    tool,
                )
                return _deny("escalation approval callback failed")
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
            bound_effect=digest,
        )
        if lease_err is not None:
            return _deny(f"escalation rejected: lease creation failed ({lease_err})")

        self._capability_leases[tool] = lease
        self._granted_escalations[tool] = _GrantedEscalation(
            tool=tool,
            paths=paths,
            ops_remaining=max_ops,
            effect_digest=digest,
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
                scope=scope.value,
                effect_digest=digest,
            )
        )
        return {
            "granted": True,
            "tool": tool,
            "max_ops": max_ops,
            "paths": paths,
            "scope": scope.value,
            "effect_digest": digest,
        }

    async def grant_for_effect(
        self,
        *,
        tool: str,
        tool_args: dict,
        envelope: ExecutionEnvelope,
        tool_use_id: str,
        consequence: str,
        denial_reason: str,
        trace_events: list,
    ) -> str | None:
        """Just-in-time, effect-bound approval for ONE call.

        Called when every other gate has passed and the consequence axis is the
        only thing standing in the way (an action class above the unattended
        ceiling with no governance gate). Returns ``None`` when the call is now
        authorised, or a denial reason.

        Deliberately the LAST question asked, not the first: an approver is only
        ever shown a call the structural gates already admit, so the arguments it
        renders (untrusted content, by definition) cannot be used to talk past a
        gate that would have denied anyway.

        Every ceiling that bounds a requested escalation bounds this too — the
        master switch, ``grantable_tools``, the flood guard and
        ``max_escalations`` — so the path cannot reach further than an
        ``ESCALATE_POLICY`` the agent could have issued itself. What it adds is
        precision: the grant is one-shot and bound to this call's effect digest.
        """
        ep = envelope.policy.escalation_policy

        def _deny(reason: str) -> str:
            trace_events.append(
                EscalationDeniedEvent(
                    kind=TraceEventKind.ESCALATION_DENIED,
                    node_id=envelope.node_id,
                    sequence=len(trace_events),
                    tool=tool,
                    reason=reason,
                )
            )
            return reason

        if not ep.effect_approval:
            return denial_reason  # feature off: the consequence denial stands as-is
        if not ep.allow_escalation:
            return _deny(f"{denial_reason} (effect approval: escalation not allowed by policy)")
        if tool not in ep.grantable_tools:
            return _deny(f"{denial_reason} (effect approval: '{tool}' is not in grantable_tools)")
        if self._escalation_count >= ep.max_escalations:
            return _deny(
                f"{denial_reason} (effect approval: max escalations reached "
                f"({ep.max_escalations}))"
            )
        if self._escalation_callback is None:
            return _deny(f"{denial_reason} (effect approval: no approval callback configured)")
        if getattr(self._escalation_callback, "__axor_legacy_callback__", False):
            # A legacy 4-arg callback has nowhere to receive the call's arguments,
            # so it cannot answer "did the principal order THIS". Downgrading the
            # question to a class-scoped one would take a tool-wide answer as
            # authority for this effect — refuse instead.
            return _deny(
                f"{denial_reason} (effect approval: the configured callback uses the "
                f"legacy signature and cannot be asked about a specific call)"
            )

        flood_denial = self._flood_guard.check(
            tool=tool,
            paths=[],
            reason=denial_reason,
            session_id=envelope.node_id,
            node_id=envelope.node_id,
        )
        if flood_denial is not None:
            return _deny(f"{denial_reason} (effect approval: {flood_denial})")

        digest = effect_digest(tool, tool_args)
        request = ApprovalRequest(
            tool_use_id=tool_use_id,
            tool=tool,
            scope=AuthorityScope.EFFECT,
            max_ops=1,
            reason=denial_reason,
            args=tool_args,
            effect_digest=digest,
            effect_signature=effect_signature(tool, tool_args, self._driving_args),
            consequence=consequence,
        )
        try:
            approved = await self._escalation_callback(request)
        except Exception:
            _log.exception(
                "effect approval callback raised for tool %r — denying", tool
            )
            return _deny(f"{denial_reason} (effect approval: approval callback failed)")
        if not approved:
            return _deny(f"{denial_reason} (effect approval: not approved)")

        # Validate against the policy ceiling the same way a requested escalation
        # does — then DISCARD the lease and keep only the grant. A one-shot
        # authority consumed in this same call needs no TTL or use counter, and an
        # exhausted lease left in the table would deny the NEXT call of the tool
        # ("lease expired or exhausted") instead of asking about it afresh.
        _lease, lease_err = self._lease_validator.create_lease(
            granted_by="operator",
            authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
            allowed_tools=[tool],
            parent_policy=envelope.policy,
            allowed_paths=(),
            ttl_seconds=300.0,
            max_uses=1,
            reason_code=f"effect_approval:{consequence}",
            bound_effect=digest,
        )
        if lease_err is not None:
            return _deny(
                f"{denial_reason} (effect approval: outside the policy ceiling "
                f"({lease_err}))"
            )

        self._granted_escalations[tool] = _GrantedEscalation(
            tool=tool, paths=[], ops_remaining=1, effect_digest=digest
        )
        self._escalation_count += 1
        self._flood_guard.record_approval()
        trace_events.append(
            EscalationGrantedEvent(
                kind=TraceEventKind.ESCALATION_GRANTED,
                node_id=envelope.node_id,
                sequence=len(trace_events),
                tool=tool,
                paths=(),
                max_ops=1,
                reason=f"effect_approval:{consequence}",
                auto_approved=False,
                scope=AuthorityScope.EFFECT.value,
                effect_digest=digest,
            )
        )
        return None
