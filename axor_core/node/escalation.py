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
  • :meth:`grant_for_call` — the just-in-time path: one approved call becomes a
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
from axor_core.policy.effect import call_digest, effect_signature
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
    questions — :meth:`EscalationManager.grant_for_call` refuses to put an
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
    """CLASS-scoped authority: the tool, within bounds, for N ops."""
    tool: str
    paths: list[str]  # empty = no path restriction
    ops_remaining: int


@dataclass(frozen=True)
class AuthorityRecords:
    """What authority a manager holds, as data (see
    :meth:`EscalationManager.authority_records`)."""
    class_grants: frozenset[str]
    unspent_calls: frozenset[tuple[str, str]]
    spent_calls: frozenset[tuple[str, str]]


@dataclass
class _CallEvidence:
    """CALL-scoped authority: one approved call, not yet spent.

    Keyed by call digest rather than by tool, so two different calls of the same
    tool cannot overwrite each other's authorisation, and ``claimed`` makes
    spending it single-use: the claim and the decision to execute happen with no
    await in between, so two concurrent calls cannot both spend one approval.
    """
    tool: str
    digest: str
    claimed: bool = False


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
        # CALL authority, keyed by call digest: one approved call each, spent once.
        self._call_evidence: dict[str, _CallEvidence] = {}
        # Digests with an approval question outstanding — a second concurrent call
        # for the same digest is refused rather than asked a second time.
        self._inflight_calls: set[str] = set()
        self._capability_leases: dict[str, "CapabilityLease"] = {}
        self._lease_validator = LeaseValidator()
        self._escalation_count = 0
        self._flood_guard = EscalationFloodGuard()
        self._escalation_callback = adapt_escalation_callback(escalation_callback)

    def authority_records(self) -> "AuthorityRecords":
        """A read-only snapshot of what authority this manager currently holds.

        Exists so the authority-state probe
        (:mod:`axor_core.node.authority_probe`) reads the real tables instead of
        keeping a parallel copy of them. Expired or exhausted leases are left out:
        they authorise nothing, and counting them would make the state machine
        believe authority is in force when it is not.
        """
        live_leases = frozenset(
            tool for tool, lease in self._capability_leases.items()
            if self._lease_validator.is_valid(lease)
        )
        return AuthorityRecords(
            class_grants=frozenset(self._granted_escalations) | live_leases,
            unspent_calls=frozenset(
                (ev.tool, digest)
                for digest, ev in self._call_evidence.items()
                if not ev.claimed
            ),
            spent_calls=frozenset(
                (ev.tool, digest)
                for digest, ev in self._call_evidence.items()
                if ev.claimed
            ),
        )

    def covers(
        self, tool_name: str, tool_args: dict | None = None, *, scope: str = ""
    ) -> bool:
        """True if an active authorisation covers this call — the governance gate
        the consequence axis accepts for an over-ceiling sink.

        CLASS authority (a grant or lease) covers every call of the tool. CALL
        authority covers only the call it was issued for and only while unspent,
        so the digest must match; without ``tool_args`` (no call in hand) only
        class authority counts.
        """
        if (
            tool_name in self._granted_escalations
            or tool_name in self._capability_leases
        ):
            return True
        if tool_args is None:
            return False
        ev = self._call_evidence.get(
            call_digest(tool_name, tool_args, scope=scope)
        )
        return ev is not None and ev.tool == tool_name and not ev.claimed

    def claim_call(
        self, tool_name: str, tool_args: dict, *, scope: str = ""
    ) -> bool | None:
        """Spend the CALL authorisation for this call, atomically.

        Returns ``None`` when no call authorisation is involved (nothing to
        spend), ``True`` when this caller got it, and ``False`` when evidence for
        this call exists but is already spent — a second concurrent claimant, which
        must be denied rather than executed.

        There is no await between the check and the mutation, so this is atomic
        with respect to the loop: exactly one caller can win.
        """
        ev = self._call_evidence.get(
            call_digest(tool_name, tool_args, scope=scope)
        )
        if ev is None or ev.tool != tool_name:
            return None
        if ev.claimed:
            return False
        ev.claimed = True
        # Kept, not deleted: a second claimant must be able to learn that the
        # authorisation was already spent (and a repeat of the same call then asks
        # a fresh question, which overwrites this record).
        return True

    def revoke_call(
        self, tool_name: str, tool_args: dict, *, scope: str = ""
    ) -> None:
        """Discard an unspent CALL authorisation.

        An approval that cannot be used *now* is not banked: if the state-dependent
        re-check after the answer denies the call, the evidence is dropped rather
        than left for a later attempt under conditions nobody approved.
        """
        digest = call_digest(tool_name, tool_args, scope=scope)
        ev = self._call_evidence.get(digest)
        if ev is not None and not ev.claimed:
            del self._call_evidence[digest]

    def evaluate(
        self, tool_name: str, tool_args: dict, *, scope: str = ""
    ) -> tuple[PolicyDecision | None, "_PendingConsumption | None"]:
        """The lease/grant/call-authority branch of policy resolution — decided
        WITHOUT consuming.

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
            pending_lease = lease  # consume on commit, not now

        grant = self._granted_escalations.get(tool_name)
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

        # CALL authority: an unspent approval for exactly this call. It is what
        # lets a tool in `grantable_tools` but NOT in the current capability set be
        # reached at all — and only for the one call it was issued for. It is NOT
        # consumed here: the spend happens at :meth:`claim_call`, atomically with
        # the decision to execute, after the state-dependent re-check.
        ev = self._call_evidence.get(
            call_digest(tool_name, tool_args, scope=scope)
        )
        if ev is not None and ev.tool == tool_name and not ev.claimed:
            return PolicyDecision(
                kind=PolicyDecisionKind.APPROVE,
                reason="approved via call authorisation (one use, bound to this call)",
            ), None

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
        call_bound = isinstance(intended, dict)
        # The digest binds the sink, the authority scope (this node) and EVERY
        # argument of the named call.
        digest = (
            call_digest(tool, intended, scope=envelope.node_id) if call_bound else ""
        )
        scope = AuthorityScope.CALL if call_bound else AuthorityScope.CLASS
        max_ops = 1 if call_bound else min(
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
                args=intended if call_bound else {},
                call_digest=digest,
                effect_signature=(
                    effect_signature(tool, intended, self._driving_args)
                    if call_bound else ()
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
        )
        if lease_err is not None:
            return _deny(f"escalation rejected: lease creation failed ({lease_err})")

        if call_bound:
            # CALL authority: one approved call, spent once, keyed by its digest.
            # The lease above was created only to validate the ceiling (a lease
            # stored for a one-shot authority would, once exhausted, deny the NEXT
            # call of the tool instead of letting it be asked about afresh).
            self._call_evidence[digest] = _CallEvidence(tool=tool, digest=digest)
        else:
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
                scope=scope.value,
                call_digest=digest,
            )
        )
        return {
            "granted": True,
            "tool": tool,
            "max_ops": max_ops,
            "paths": paths,
            "scope": scope.value,
            "call_digest": digest,
        }

    async def grant_for_call(
        self,
        *,
        tool: str,
        tool_args: dict,
        envelope: ExecutionEnvelope,
        tool_use_id: str,
        consequence: str,
        denial_reason: str,
        trace_events: list,
        scope: str = "",
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

        if not ep.call_approval:
            return denial_reason  # feature off: the consequence denial stands as-is
        if not ep.allow_escalation:
            return _deny(f"{denial_reason} (call approval: escalation not allowed by policy)")
        if tool not in ep.grantable_tools:
            return _deny(f"{denial_reason} (call approval: '{tool}' is not in grantable_tools)")
        if self._escalation_count >= ep.max_escalations:
            return _deny(
                f"{denial_reason} (call approval: max escalations reached "
                f"({ep.max_escalations}))"
            )
        if self._escalation_callback is None:
            return _deny(f"{denial_reason} (call approval: no approval callback configured)")
        if getattr(self._escalation_callback, "__axor_legacy_callback__", False):
            # A legacy 4-arg callback has nowhere to receive the call's arguments,
            # so it cannot answer "did the principal order THIS". Downgrading the
            # question to a class-scoped one would take a tool-wide answer as
            # authority for this effect — refuse instead.
            return _deny(
                f"{denial_reason} (call approval: the configured callback uses the "
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
            return _deny(f"{denial_reason} (call approval: {flood_denial})")

        digest = call_digest(tool, tool_args, scope=scope)
        existing = self._call_evidence.get(digest)
        if existing is not None and not existing.claimed:
            return None  # already authorised and unspent — ask nobody twice
        if digest in self._inflight_calls:
            # A concurrent call with the identical digest is already being asked
            # about. Asking again would put the same question to the approver
            # twice and could mint two authorisations for one call.
            return _deny(
                f"{denial_reason} (call approval: an approval for this exact call "
                f"is already outstanding)"
            )
        request = ApprovalRequest(
            tool_use_id=tool_use_id,
            tool=tool,
            scope=AuthorityScope.CALL,
            max_ops=1,
            reason=denial_reason,
            args=tool_args,
            call_digest=digest,
            effect_signature=effect_signature(tool, tool_args, self._driving_args),
            consequence=consequence,
        )
        self._inflight_calls.add(digest)
        try:
            approved = await self._escalation_callback(request)
        except Exception:
            _log.exception(
                "call approval callback raised for tool %r — denying", tool
            )
            return _deny(f"{denial_reason} (call approval: approval callback failed)")
        finally:
            self._inflight_calls.discard(digest)
        if not approved:
            return _deny(f"{denial_reason} (call approval: not approved)")

        # Validate against the delegation ceiling the same way a requested
        # escalation does — then DISCARD the lease and keep only the call evidence.
        # A one-shot authority consumed in this same call needs no TTL or use
        # counter, and an exhausted lease left in the table would deny the NEXT
        # call of the tool ("lease expired or exhausted") instead of letting it be
        # asked about afresh.
        _lease, lease_err = self._lease_validator.create_lease(
            granted_by="operator",
            authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
            allowed_tools=[tool],
            parent_policy=envelope.policy,
            allowed_paths=(),
            ttl_seconds=300.0,
            max_uses=1,
            reason_code=f"call_approval:{consequence}",
        )
        if lease_err is not None:
            return _deny(
                f"{denial_reason} (call approval: outside the policy ceiling "
                f"({lease_err}))"
            )

        self._call_evidence[digest] = _CallEvidence(tool=tool, digest=digest)
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
                reason=f"call_approval:{consequence}",
                auto_approved=False,
                scope=AuthorityScope.CALL.value,
                call_digest=digest,
            )
        )
        return None
