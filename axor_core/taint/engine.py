from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from axor_core.contracts.taint import (
    ClearanceRecord,
    TaintScope,
    TaintSource,
    TaintState,
)
from axor_core.contracts.trace import (
    TaintClearanceAttemptedEvent,
    TaintClearedEvent,
    TaintPropagatedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.errors.exceptions import TaintClearanceError

if TYPE_CHECKING:
    pass

# Authority types permitted to clear governance state. "worker" (and anything
# outside this set) is rejected — workers must never clear their own taint.
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
    Per-session taint tracker.

    Taint is sticky — once set it persists until cleared by governance.
    Workers may not clear taint; only governance (operator, trusted boundary)
    may call clear_by_governance().

    Thread-safety: not thread-safe. Each session has its own instance.
    """

    def __init__(self, node_id: str = "") -> None:
        self._state = TaintState()
        self._first_taint_time: float | None = None
        self._intent_count_at_first_taint: int = 0
        self._intent_count: int = 0
        self._node_id = node_id
        self._pending_events: list[TraceEvent] = []

    @property
    def state(self) -> TaintState:
        return self._state

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear pending trace events for the trace collector."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def tick_intent(self) -> None:
        """Called by IntentLoop on each intent to advance intent_age."""
        self._intent_count += 1
        if self._state.is_tainted:
            age = self._intent_count - self._intent_count_at_first_taint
            wall = time.monotonic() - (self._first_taint_time or 0.0)
            self._state = TaintState(
                sources=self._state.sources,
                scope=self._state.scope,
                sticky=self._state.sticky,
                intent_age=age,
                wall_clock_age=wall,
                parent_inherited=self._state.parent_inherited,
                clearance_history=self._state.clearance_history,
                clearance_authority=self._state.clearance_authority,
                sensitive=self._state.sensitive,
            )

    def propagate(
        self,
        source: TaintSource,
        scope: TaintScope = TaintScope.SESSION,
        sensitive: bool = False,
    ) -> TaintState:
        """
        Record that an external input from `source` was received.

        Widens scope if the new scope is broader than the current one.
        `sensitive` (TM2 confidentiality label) is monotonically OR-ed in — once
        a sensitive source has contributed, the session stays sensitive until
        governance clears it. Updates first-taint timestamp on first propagation.
        """
        now = time.monotonic()
        if not self._state.is_tainted:
            self._first_taint_time = now
            self._intent_count_at_first_taint = self._intent_count

        new_sources = self._state.sources | {source}
        new_scope = self._wider_scope(self._state.scope, scope)
        self._state = TaintState(
            sources=frozenset(new_sources),
            scope=new_scope,
            sticky=True,
            intent_age=self._state.intent_age,
            wall_clock_age=self._state.wall_clock_age,
            parent_inherited=self._state.parent_inherited,
            clearance_history=self._state.clearance_history,
            clearance_authority=self._state.clearance_authority,
            sensitive=self._state.sensitive or sensitive,
        )
        self._pending_events.append(TaintPropagatedEvent(
            kind=TraceEventKind.TAINT_PROPAGATED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            taint_source=source.value,
            taint_scope=scope.value,
        ))
        return self._state

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
    ) -> TaintState:
        """
        Clear taint under governance authority.

        Records a ClearanceRecord in clearance_history.
        Returns the new (clean) TaintState.

        Rejects clearance unless it carries a verifiable governance authority —
        a non-empty principal and an authority_type from the allowed set. This
        prevents a worker-reachable path from clearing taint by passing blank or
        worker-labelled authority strings.
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
        record = ClearanceRecord(
            clearance_id=f"clr_{uuid.uuid4().hex[:12]}",
            cleared_by=authority,
            authority_type=authority_type,
            timestamp=time.time(),
            reason_code=reason_code,
            authorized_by_principal_id=authorized_by_principal_id,
            audit_id=audit_id,
        )
        self._state = TaintState(
            sources=frozenset(),
            scope=TaintScope.SESSION,
            sticky=False,
            intent_age=0,
            wall_clock_age=0.0,
            parent_inherited=False,
            clearance_history=self._state.clearance_history + (record,),
            clearance_authority=authority,
        )
        self._first_taint_time = None
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority,
            authority_type=authority_type,
            reason_code=reason_code,
            audit_id=audit_id,
        ))
        return self._state

    # ── Cross-session persistence (§7.1) ──────────────────────────────────────

    def cross_session_persist(self, snapshot_dir: Path) -> None:
        """
        Persist current taint state across sessions via Sentinel ReputationSnapshot.

        Writes a suspicion_score of 1.0 for this node so a subsequent session
        can detect the mark via load_cross_session().
        """
        from axor_sentinel.sentinel.snapshot import ReputationSnapshot, atomic_swap  # type: ignore[import-untyped]
        import time as _time

        existing_snapshot = self._load_snapshot(snapshot_dir)
        existing_reputation = dict(existing_snapshot.resource_reputation) if existing_snapshot else {}
        existing_reputation[self._node_id] = 1.0

        snapshot = ReputationSnapshot(
            version=(existing_snapshot.version + 1 if existing_snapshot else 1),
            generated_at=_time.time(),
            resource_reputation=existing_reputation,
            container_reputation=dict(existing_snapshot.container_reputation) if existing_snapshot else {},
        ).with_checksum()
        atomic_swap(snapshot_dir, snapshot)

    def load_cross_session(self, snapshot_dir: Path) -> TaintState:
        """
        Load cross-session taint mark. If this node_id has a non-zero
        reputation score in the snapshot, propagate CROSS_SESSION taint.
        """
        snapshot = self._load_snapshot(snapshot_dir)
        if snapshot and snapshot.resource_reputation.get(self._node_id, 0.0) > 0:
            return self.propagate(TaintSource.MEMORY, TaintScope.CROSS_SESSION)
        return self._state

    @staticmethod
    def _load_snapshot(snapshot_dir: Path):  # type: ignore[return]
        try:
            from axor_sentinel.sentinel.snapshot import load_snapshot  # type: ignore[import-untyped]
            return load_snapshot(snapshot_dir)
        except ImportError:
            return None

    def inherit_from_parent(self, parent_state: TaintState) -> TaintState:
        """
        Copy parent taint into this engine (for child node construction).

        Child inherits all parent sources and the wider scope.
        """
        if not parent_state.is_tainted:
            return self._state
        new_sources = self._state.sources | parent_state.sources
        new_scope = self._wider_scope(self._state.scope, parent_state.scope)
        self._state = TaintState(
            sources=frozenset(new_sources),
            scope=new_scope,
            sticky=True,
            intent_age=parent_state.intent_age,
            wall_clock_age=parent_state.wall_clock_age,
            parent_inherited=True,
            clearance_history=parent_state.clearance_history,
            clearance_authority=parent_state.clearance_authority,
            sensitive=self._state.sensitive or parent_state.sensitive,
        )
        if not self._first_taint_time:
            self._first_taint_time = time.monotonic()
            self._intent_count_at_first_taint = self._intent_count
        return self._state

    @staticmethod
    def _wider_scope(a: TaintScope, b: TaintScope) -> TaintScope:
        _order = [
            TaintScope.INTENT,
            TaintScope.NODE,
            TaintScope.SUBTREE,
            TaintScope.SESSION,
            TaintScope.CROSS_SESSION,
        ]
        return _order[max(_order.index(a), _order.index(b))]
