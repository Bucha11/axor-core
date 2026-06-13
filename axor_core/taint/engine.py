from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.fingerprint import content_fingerprint
from axor_core.taint.ledger import ValueTaintLedger
from axor_core.contracts.degradation import GovernanceAuthority
from axor_core.contracts.trace import (
    TaintClearanceAttemptedEvent,
    TaintClearedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.errors.exceptions import TaintClearanceError

log = logging.getLogger("axor.taint")

# Cap on distinct outstanding sensitive reads tracked by fingerprint. Past it the
# floor goes STICKY (forced active) instead of growing the map without bound — a
# flood of distinct secret reads bounds memory while failing CLOSED (the floor
# stays up). Only governance (clear_by_governance) resets it.
_MAX_OUTSTANDING_SECRETS = 4096

if TYPE_CHECKING:
    pass

# Authority types permitted to clear/endorse governance state. "worker" (and
# anything outside this set) is rejected — workers must never clear their own taint.
_VALID_GOVERNANCE_AUTHORITY_TYPES = frozenset({
    "human_operator",
    "automated_policy",
    "trusted_boundary",
})


def _is_valid_governance_authority(authority: GovernanceAuthority) -> bool:
    """A governance authority is valid when it is an unforgeable GovernanceAuthority
    capability (constructible only by the host, never materialised from a worker's
    tool-call args) carrying a non-empty principal, a non-empty reason, and an
    authority_type from the allowed set. The TYPE is the capability — a worker
    cannot fabricate one from JSON/strings — and the field checks reject a blank or
    worker-labelled instance."""
    return bool(
        isinstance(authority, GovernanceAuthority)
        and authority.authority_id
        and authority.reason_code
        and authority.authority_type in _VALID_GOVERNANCE_AUTHORITY_TYPES
    )


class TaintEngine:
    """
    Per-value taint tracker. Implements the ValueProvenance contract.

    Enforcement is per-value: a sink decides on the driving argument's own
    causal root (content-derivation ledger), not a session-wide flag. There is no
    session-taint state — provenance lives on values, released by governance
    endorsement (per value) or cleared wholesale.

    Thread-safety: not thread-safe. Each session has its own instance.
    """

    def __init__(self, node_id: str = "") -> None:
        self._node_id = node_id
        self._pending_events: list[TraceEvent] = []
        self._ledger = ValueTaintLedger()
        # Session-wide SHADOW (observe-only, for the density comparison):
        # "has any tainted / any sensitive value ever been registered this session?"
        # This is what a coarse session-scoped model would gate on; it never feeds
        # `allow`, it only lets the density meter compare the session-wide flag
        # against per-value tracking honestly.
        self._session_any_tainted = False
        self._session_any_sensitive = False
        # Confidentiality floor (sound). An IDENTITY-BOUND registry of outstanding
        # sensitive reads: fingerprint(secret) -> count of un-released reads of THAT
        # exact secret. While non-empty the session is egress-restricted — soundly,
        # on the FACT of the read, content-blind (a paraphrased/re-encoded secret
        # cannot escape, unlike the per-value derivation gate). Keyed on a whole-
        # content fingerprint, NOT on the leaky ≥12-char ledger, so: (a) a secret
        # shorter than the ledger segment minimum still arms AND can be released
        # (the ledger can't represent it, the fingerprint can); (b) endorsing a
        # DIFFERENT value that merely shares a fragment cannot lift another secret's
        # floor (different fingerprint). This decouples the floor from the ledger's
        # derive(), removing the count/ledger desync.
        self._outstanding: dict[str, int] = {}
        # Sticky fail-closed flag: set once the outstanding map hits the cap, it
        # forces the floor active regardless of the map, so a flood of distinct
        # secret reads cannot grow memory without bound and endorsing the few tracked
        # secrets cannot lower the floor while untracked ones are still outstanding.
        self._floor_saturated = False

    # ── Per-value provenance (ValueProvenance) ────────────────────────────────

    def register_value(self, content: object, root: CausalRoot) -> None:
        """Record that a value with the given causal_root produced this content."""
        self._ledger.register(content, root)
        if root.is_tainted:
            self._session_any_tainted = True
        if root.sensitive:
            self._session_any_sensitive = True
            # Arm the floor on the READ fact, keyed by the secret's fingerprint —
            # regardless of whether the ledger stored a fragment (sub-threshold
            # secrets still count, and remain releasable by fingerprint).
            fp = content_fingerprint(content)
            if fp in self._outstanding or len(self._outstanding) < _MAX_OUTSTANDING_SECRETS:
                self._outstanding[fp] = self._outstanding.get(fp, 0) + 1
            elif not self._floor_saturated:
                # Cap reached on a NEW secret: do not grow the map; go sticky so the
                # floor stays up for this (untracked) read. Fail-closed.
                self._floor_saturated = True
                log.warning(
                    "confidentiality floor saturated at %d distinct outstanding "
                    "secrets — floor forced active until governance clears it",
                    _MAX_OUTSTANDING_SECRETS,
                )

    def confidentiality_floor_active(self) -> bool:
        """Sound egress floor: True while a secret read is outstanding (not
        governance-released). An ENFORCEMENT input — unlike session_shadow. Once the
        outstanding map saturates it stays True (sticky) until governance clears."""
        return self._floor_saturated or bool(self._outstanding)

    def session_shadow(self) -> tuple[bool, bool]:
        """(any_tainted, any_sensitive) for the session-wide shadow model.

        Observe-only, for the density comparison. NOT an enforcement input —
        `allow` is per-value; this only exists to measure what a coarse
        session-scoped model would have done.
        """
        return (self._session_any_tainted, self._session_any_sensitive)

    def derive_value(self, value: object) -> CausalRoot:
        """Per-value causal root of `value` by content derivation. Clean (constant)
        if it carries no registered tainted/sensitive content — the per-value win.
        """
        return self._ledger.derive(value)

    def inherit_value_ledger(self, parent: "TaintEngine") -> None:
        """Inherit the parent's per-value provenance into this (child) engine so
        the child's per-value gate sees values the parent marked tainted/sensitive."""
        self._ledger.merge(parent._ledger)
        # Inherit the session-wide shadow too, so child density measurement is
        # comparable to the parent's (observe-only).
        self._session_any_tainted = self._session_any_tainted or parent._session_any_tainted
        self._session_any_sensitive = (
            self._session_any_sensitive or parent._session_any_sensitive
        )
        # Inherit the confidentiality floor: a child of a session that read a secret
        # is egress-restricted too (else the child is a floor bypass). Merge per
        # fingerprint so distinct secrets are not conflated and re-inheritance does
        # not double-count a secret the child already carries.
        for fp, count in parent._outstanding.items():
            self._outstanding[fp] = max(self._outstanding.get(fp, 0), count)
        # A saturated parent floor is inherited sticky (else the child is a bypass).
        self._floor_saturated = self._floor_saturated or parent._floor_saturated

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear pending trace events for the trace collector."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    # ── Governance release ────────────────────────────────────────────────────

    def attempt_clear_by_worker(self) -> None:
        """Workers may not clear taint. Always raises TaintClearanceError."""
        self._pending_events.append(TaintClearanceAttemptedEvent(
            kind=TraceEventKind.TAINT_CLEARANCE_ATTEMPTED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            attempted_by="worker",
        ))
        raise TaintClearanceError(
            "worker attempted to clear taint — only governance may do this"
        )

    def clear_by_governance(self, authority: GovernanceAuthority) -> None:
        """Clear ALL per-value provenance under governance authority.

        Requires an unforgeable :class:`GovernanceAuthority` capability — a worker
        cannot materialise one from its tool-call args, so this path is not
        worker-reachable, and the object's fields are validated besides. For
        releasing a single value, use :meth:`endorse_value`.
        """
        if not _is_valid_governance_authority(authority):
            self._pending_events.append(TaintClearanceAttemptedEvent(
                kind=TraceEventKind.TAINT_CLEARANCE_ATTEMPTED,
                node_id=self._node_id,
                sequence=len(self._pending_events),
                attempted_by=getattr(authority, "authority_id", "") or "unknown",
            ))
            raise TaintClearanceError(
                "taint clearance rejected: requires a valid GovernanceAuthority "
                f"capability (got {authority!r})"
            )
        self._ledger = ValueTaintLedger()
        self._session_any_tainted = False
        self._session_any_sensitive = False
        self._outstanding = {}
        self._floor_saturated = False
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority.authority_id,
            authority_type=authority.authority_type,
            reason_code=authority.reason_code,
            audit_id=authority.audit_id,
        ))

    def endorse_value(self, content: object, authority: GovernanceAuthority) -> int:
        """Governed structural release of one specific value.

        Removes the value's fragments from the per-value ledger so ``derive_value``
        no longer flags it, AND lifts the confidentiality floor for THAT exact
        secret (by fingerprint). Attests release of this value/lineage (schema/
        transform/bounded use), NOT a semantic "safe" judgement, and NOT the whole
        session. Requires an unforgeable :class:`GovernanceAuthority`. Returns
        ledger fragments released.
        """
        if not _is_valid_governance_authority(authority):
            raise TaintClearanceError(
                "endorsement rejected: requires a valid GovernanceAuthority "
                f"capability (got {authority!r})"
            )
        # Lift the floor for THIS exact secret, identity-bound by fingerprint —
        # governance names the value it releases. A different value (even one that
        # shares a ledger fragment) has a different fingerprint and cannot lift
        # another secret's floor; a sub-threshold secret the ledger never stored is
        # still releasable here. This is what keeps the floor and the ledger from
        # desynchronising — the floor is released by identity, not by derive().
        self._outstanding.pop(content_fingerprint(content), None)
        removed = self._ledger.unregister(content)
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority.authority_id,
            authority_type=authority.authority_type,
            reason_code=f"endorsement:{authority.reason_code}",
            audit_id=authority.audit_id,
        ))
        return removed
