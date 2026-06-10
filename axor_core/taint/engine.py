from __future__ import annotations

from typing import TYPE_CHECKING

from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.ledger import ValueTaintLedger
from axor_core.contracts.trace import (
    TaintClearanceAttemptedEvent,
    TaintClearedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.errors.exceptions import TaintClearanceError

if TYPE_CHECKING:
    pass

# Authority types permitted to clear/endorse governance state. "worker" (and
# anything outside this set) is rejected — workers must never clear their own taint.
_VALID_GOVERNANCE_AUTHORITY_TYPES = frozenset({
    "human_operator",
    "automated_policy",
    "trusted_boundary",
})


def _is_valid_governance_authority(
    authority: str,
    authority_type: str,
    reason_code: str,
) -> bool:
    return bool(
        authority
        and reason_code
        and authority_type in _VALID_GOVERNANCE_AUTHORITY_TYPES
    )


class TaintEngine:
    """
    Per-value taint tracker (TM2). Implements the ValueProvenance contract.

    Enforcement is per-value: a sink decides on the driving argument's own
    causal_root (content-derivation ledger), not a session-wide flag. There is no
    session-taint state — provenance lives on values, released by governance
    endorsement (per value) or cleared wholesale.

    Thread-safety: not thread-safe. Each session has its own instance.
    """

    def __init__(self, node_id: str = "") -> None:
        self._node_id = node_id
        self._pending_events: list[TraceEvent] = []
        self._ledger = ValueTaintLedger()
        # Session-sticky SHADOW (observe-only, for the TM3.3 density experiment):
        # "has any tainted / any sensitive value ever been registered this session?"
        # This is what a coarse session-scoped model would gate on; it never feeds
        # `allow`, it only lets the density meter compare session-sticky vs
        # per-value honestly.
        self._session_any_tainted = False
        self._session_any_sensitive = False
        # Confidentiality SOUND FLOOR (TM4, 1.1b). Counts outstanding sensitive
        # reads: a secret entered the session and has NOT been governance-released.
        # While > 0 the session is egress-restricted — coarsely and SOUNDLY, on the
        # FACT of the read, independent of the egress value's content (so a
        # paraphrased / re-encoded secret cannot escape, unlike the per-value
        # content-derivation gate). Incremented on a sensitive read, decremented
        # only by governance endorsement of the secret value, reset by clearance.
        self._outstanding_sensitive = 0

    # ── Per-value provenance (TM2 / ValueProvenance) ──────────────────────────

    def register_value(self, content: object, root: CausalRoot) -> None:
        """Record that a value with the given causal_root produced this content."""
        self._ledger.register(content, root)
        if root.is_tainted:
            self._session_any_tainted = True
        if root.sensitive:
            self._session_any_sensitive = True
            # Activate the confidentiality floor on the READ fact — regardless of
            # whether the ledger stored a fragment (short secrets, NM1, still count).
            self._outstanding_sensitive += 1

    def confidentiality_floor_active(self) -> bool:
        """Sound egress floor (TM4): True while a secret read is outstanding (not
        governance-released). An ENFORCEMENT input — unlike session_shadow."""
        return self._outstanding_sensitive > 0

    def session_shadow(self) -> tuple[bool, bool]:
        """(any_tainted, any_sensitive) for the session-sticky shadow model.

        Observe-only (TM3.3 density). NOT an enforcement input — `allow` is
        per-value; this only exists to measure what a session model would have done.
        """
        return (self._session_any_tainted, self._session_any_sensitive)

    def derive_value(self, value: object) -> CausalRoot:
        """Per-value causal_root of `value` by content derivation. Clean (constant)
        if it carries no registered tainted/sensitive content — the per-value win.
        """
        return self._ledger.derive(value)

    def inherit_value_ledger(self, parent: "TaintEngine") -> None:
        """Inherit the parent's per-value provenance into this (child) engine so
        the child's per-value gate sees values the parent marked tainted/sensitive."""
        self._ledger.merge(parent._ledger)
        # Inherit the session-sticky shadow too, so child density measurement is
        # comparable to the parent's (observe-only).
        self._session_any_tainted = self._session_any_tainted or parent._session_any_tainted
        self._session_any_sensitive = (
            self._session_any_sensitive or parent._session_any_sensitive
        )
        # Inherit the confidentiality floor: a child of a session that read a secret
        # is egress-restricted too (else the child is a floor bypass).
        self._outstanding_sensitive += parent._outstanding_sensitive

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

    def clear_by_governance(
        self,
        authority: str,
        authority_type: str,
        reason_code: str,
        authorized_by_principal_id: str = "",
        audit_id: str = "",
    ) -> None:
        """Clear ALL per-value provenance under governance authority.

        Rejects clearance unless it carries a verifiable governance authority — a
        non-empty principal and an authority_type from the allowed set — so a
        worker-reachable path cannot launder taint. For releasing a single value,
        use endorse_value().
        """
        if not _is_valid_governance_authority(authority, authority_type, reason_code):
            self._pending_events.append(TaintClearanceAttemptedEvent(
                kind=TraceEventKind.TAINT_CLEARANCE_ATTEMPTED,
                node_id=self._node_id,
                sequence=len(self._pending_events),
                attempted_by=authority or "unknown",
            ))
            raise TaintClearanceError(
                "taint clearance rejected: requires a valid governance authority "
                f"(authority={authority!r}, authority_type={authority_type!r})"
            )
        self._ledger = ValueTaintLedger()
        self._session_any_tainted = False
        self._session_any_sensitive = False
        self._outstanding_sensitive = 0
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority,
            authority_type=authority_type,
            reason_code=reason_code,
            audit_id=audit_id,
        ))

    def endorse_value(
        self,
        content: object,
        authority: str,
        authority_type: str,
        reason_code: str,
        audit_id: str = "",
    ) -> int:
        """Endorsement (TM4) — governed STRUCTURAL release of one specific value.

        Removes the value's fragments from the per-value ledger so `derive_value`
        no longer flags it. Attests release of *this value/lineage* (schema/
        transform/bounded use), NOT a semantic "safe" judgement, and NOT the whole
        session. Requires a valid governance authority. Returns fragments released.
        """
        if not _is_valid_governance_authority(authority, authority_type, reason_code):
            raise TaintClearanceError(
                "endorsement rejected: requires a valid governance authority "
                f"(authority={authority!r}, authority_type={authority_type!r})"
            )
        # Lift one unit of the confidentiality floor iff this content is a currently
        # registered SENSITIVE value (governance attests its release, TM4). A
        # paraphrase that does not match the registered secret does not derive as
        # sensitive, so it cannot lift the floor — the floor stays sound.
        if self._outstanding_sensitive > 0 and self._ledger.derive(content).sensitive:
            self._outstanding_sensitive -= 1
        removed = self._ledger.unregister(content)
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority,
            authority_type=authority_type,
            reason_code=f"endorsement:{reason_code}",
            audit_id=audit_id,
        ))
        return removed
