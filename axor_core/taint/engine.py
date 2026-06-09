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

    # ── Per-value provenance (TM2 / ValueProvenance) ──────────────────────────

    def register_value(self, content: object, root: CausalRoot) -> None:
        """Record that a value with the given causal_root produced this content."""
        self._ledger.register(content, root)

    def derive_value(self, value: object) -> CausalRoot:
        """Per-value causal_root of `value` by content derivation. Clean (constant)
        if it carries no registered tainted/sensitive content — the per-value win.
        """
        return self._ledger.derive(value)

    def inherit_value_ledger(self, parent: "TaintEngine") -> None:
        """Inherit the parent's per-value provenance into this (child) engine so
        the child's per-value gate sees values the parent marked tainted/sensitive."""
        self._ledger.merge(parent._ledger)

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
