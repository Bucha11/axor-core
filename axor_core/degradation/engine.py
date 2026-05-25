from __future__ import annotations

import time
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING

from axor_core.contracts.degradation import (
    DegradationLevel,
    DegradationPolicy,
    DegradationState,
    DegradationTransition,
    GovernanceAuthority,
    SourceRecord,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.trace import (
    DegradationTransitionEvent,
    SourceQuarantinedEvent,
    TraceEventKind,
)
from axor_core.errors.exceptions import DegradationClearanceError

if TYPE_CHECKING:
    from axor_core.contracts.anomaly import NormalizedIntent
    from axor_core.contracts.denial import DenialResponse
    from axor_core.contracts.taint import TaintState
    from axor_core.contracts.trace import TraceEvent

# Tools that count as "write/bash/export" pressure when denied.
_WRITE_BASH_EXPORT_TOOLS = frozenset({
    "bash", "write", "edit", "multiedit", "export",
    "computer", "execute", "run", "shell",
})

# Tools always permitted at LOCKED level.
_LOCKED_ALLOWED_TOOLS = frozenset({"read", "escalate", "escalate_policy"})

# destination_kinds that indicate cross-origin export risk.
_CROSS_ORIGIN_DESTINATIONS = frozenset({"external_domain", "private_network"})


def _is_write_bash_export(tool: str) -> bool:
    return tool.lower() in _WRITE_BASH_EXPORT_TOOLS


def _is_cross_origin_export(intent: "NormalizedIntent") -> bool:
    return (
        intent.destination_kind in _CROSS_ORIGIN_DESTINATIONS
        and intent.operation in ("network_request", "file_write", "execute_generated_code")
    )


class DegradationEngine:
    """
    Source-aware, taint-integrated degradation state machine.

    Thread-safety: not thread-safe. Each session has its own instance.
    """

    def __init__(self, policy: DegradationPolicy | None = None) -> None:
        self._policy = policy or DegradationPolicy()
        self._state = DegradationState()
        self._pending_events: list[TraceEvent] = []
        self._locked_at: float | None = None  # wall time when LOCKED was entered

    # ── Public properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> DegradationState:
        return self._state

    def drain_events(self) -> list[TraceEvent]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    # ── Core API ───────────────────────────────────────────────────────────────

    def record_signal(
        self,
        intent: "NormalizedIntent",
        denial: "DenialResponse | None",
        taint_state: "TaintState",
    ) -> DegradationTransition | None:
        """
        Called after every intent evaluation (pass or deny).
        Returns DegradationTransition if level changed, else None.
        """
        # TERMINAL → accept no more signals.
        if self._state.level == DegradationLevel.TERMINAL:
            return None

        # Check LOCKED_TTL before processing the new signal.
        ttl_transition = self._check_locked_ttl()
        if ttl_transition is not None:
            return ttl_transition

        if denial is None:
            return None

        # Determine source_id from taint provenance.
        source_id = self.derive_source_id(intent, taint_state)
        source = self._get_or_create_source(source_id, taint_state)

        self._state.session_deny_count += 1
        source.last_signal = time.time()

        # Classify signal type.
        tool_name = intent.tool.lower()
        cross_origin = _is_cross_origin_export(intent)
        is_tool_pressure = _is_write_bash_export(tool_name)
        is_instr_pressure = intent.executes_generated_code or intent.after_external_read

        # Update pressure counters before any early returns.
        if is_tool_pressure:
            source.tool_pressure_count += 1
        if is_instr_pressure:
            source.instruction_pressure_count += 1

        transition: DegradationTransition | None = None

        # Rule: cross-origin export deny → LOCKED immediately.
        if cross_origin:
            t = self._transition_to(
                DegradationLevel.LOCKED,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="cross_origin_export_denied",
            )
            if t is not None:
                return t
            # Already at LOCKED or TERMINAL — fall through to check other rules.

        # Rule: source tool pressure threshold → quarantine + RESTRICTED.
        if (
            not source.quarantined
            and source.tool_pressure_count >= self._policy.SOURCE_TOOL_PRESSURE_THRESHOLD
        ):
            transition = self._quarantine_and_restrict(source, source_id, intent.tool)

        # Rule: source instruction pressure threshold → quarantine + RESTRICTED.
        elif (
            not source.quarantined
            and source.instruction_pressure_count >= self._policy.SOURCE_INSTR_PRESSURE_THRESHOLD
        ):
            transition = self._quarantine_and_restrict(source, source_id, intent.tool)

        # First suspicious signal without immediate quarantine → CAUTIOUS.
        elif not source.quarantined and self._state.level == DegradationLevel.NORMAL:
            transition = self._transition_to(
                DegradationLevel.CAUTIOUS,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="first_suspicious_signal",
            )

        # Rule: session deny threshold → LOCKED.
        if self._state.session_deny_count >= self._policy.SESSION_DENY_THRESHOLD:
            t = self._transition_to(
                DegradationLevel.LOCKED,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="session_deny_threshold",
            )
            if t is not None:
                transition = t

        return transition

    def attempt_clear_by_worker(self) -> None:
        """Workers may not lower degradation level. Always raises DegradationClearanceError."""
        raise DegradationClearanceError(
            "worker attempted to clear degradation — only governance may do this"
        )

    def apply_to_policy(
        self,
        base_policy: ExecutionPolicy,
        source_id: str | None,
    ) -> ExecutionPolicy:
        """
        Return a narrowed ExecutionPolicy for the current degradation state.
        """
        level = self._state.level

        if level == DegradationLevel.NORMAL:
            return base_policy

        if level == DegradationLevel.CAUTIOUS:
            # Policy unchanged — document soft-blocked at call site.
            return base_policy

        if level == DegradationLevel.RESTRICTED:
            # Any quarantined source in the session → narrow the policy.
            # When source_id is known and quarantined, or when source_id is None
            # but quarantined sources exist, we cannot verify the call is clean.
            is_quarantined_call = (
                source_id is not None
                and source_id in self._state.sources
                and self._state.sources[source_id].quarantined
            ) or (
                source_id is None
                and any(s.quarantined for s in self._state.sources.values())
            )
            if is_quarantined_call:
                narrowed_tool = _dc_replace(
                    base_policy.tool_policy,
                    allow_bash=False,
                    allow_write=False,
                )
                return _dc_replace(
                    base_policy,
                    tool_policy=narrowed_tool,
                    export_mode=ExportMode.RESTRICTED,
                )
            return base_policy

        if level in (DegradationLevel.LOCKED, DegradationLevel.TERMINAL):
            # All tools frozen except read + escalate; export forced RESTRICTED.
            locked_tool = ToolPolicy(
                allow_read=True,
                allow_write=False,
                allow_bash=False,
                allow_search=False,
                allow_spawn=False,
                extra_allowed=tuple(_LOCKED_ALLOWED_TOOLS - {"read"}),
                extra_denied=tuple(
                    t for t in (base_policy.tool_policy.extra_allowed or ())
                    if t not in _LOCKED_ALLOWED_TOOLS
                ),
            )
            return _dc_replace(
                base_policy,
                tool_policy=locked_tool,
                export_mode=ExportMode.RESTRICTED,
            )

        return base_policy  # fallback

    def quarantine_source(self, source_id: str, reason: str) -> None:
        """Manually quarantine a source (e.g. from human review)."""
        source = self._state.sources.get(source_id)
        if source is None:
            source = SourceRecord(source_id=source_id, taint_source="unknown")
            self._state.sources[source_id] = source
        if not source.quarantined:
            source.quarantined = True
            self._emit_quarantine_event(source_id, reason)
        self._transition_to(
            DegradationLevel.RESTRICTED,
            source_id=source_id,
            trigger_intent="manual",
            reason=f"manual_quarantine: {reason}",
        )

    def clear_by_governance(
        self,
        authority: GovernanceAuthority,
        reason: str,
        target_level: DegradationLevel = DegradationLevel.NORMAL,
    ) -> None:
        """
        Lower degradation level. Requires GovernanceAuthority object.
        Records clearance in level_history.

        Worker path must call attempt_clear_by_worker() — this method requires
        a GovernanceAuthority and is only accessible on the governance path.
        """
        now = time.time()
        prev = self._state.level
        self._state.level = target_level
        self._state.tools_frozen = target_level >= DegradationLevel.LOCKED
        if target_level < DegradationLevel.LOCKED:
            self._locked_at = None
        entry = (now, target_level, f"governance_clearance:{authority.authority_id}:{reason}")
        self._state.level_history.append(entry)
        self._emit_transition_event(
            prev,
            target_level,
            source_id=None,
            trigger_intent="governance_clearance",
            reason=reason,
            ts=now,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _check_locked_ttl(self) -> DegradationTransition | None:
        if self._state.level == DegradationLevel.LOCKED and self._locked_at is not None:
            elapsed = time.time() - self._locked_at
            if elapsed >= self._policy.LOCKED_TTL:
                return self._transition_to(
                    DegradationLevel.TERMINAL,
                    source_id=None,
                    trigger_intent="ttl_check",
                    reason="locked_ttl_expired",
                )
        return None

    def derive_source_id(
        self,
        intent: "NormalizedIntent",
        taint_state: "TaintState",
    ) -> str:
        # Use provenance as primary key; fall back to taint sources.
        provenance = getattr(intent, "provenance", "") or ""
        if provenance and provenance not in ("user", "unknown"):
            return f"provenance:{provenance}"
        if taint_state.sources:
            # Use the alphabetically-first source for determinism.
            first = sorted(s.value for s in taint_state.sources)[0]
            return f"taint:{first}"
        return "unknown"

    def _get_or_create_source(
        self,
        source_id: str,
        taint_state: "TaintState",
    ) -> SourceRecord:
        if source_id not in self._state.sources:
            taint_src = "unknown"
            if taint_state.sources:
                taint_src = sorted(s.value for s in taint_state.sources)[0]
            self._state.sources[source_id] = SourceRecord(
                source_id=source_id,
                taint_source=taint_src,
            )
        return self._state.sources[source_id]

    def _quarantine_and_restrict(
        self,
        source: SourceRecord,
        source_id: str,
        tool_name: str,
    ) -> DegradationTransition | None:
        source.quarantined = True
        self._emit_quarantine_event(source_id, f"pressure_threshold:{tool_name}")
        return self._transition_to(
            DegradationLevel.RESTRICTED,
            source_id=source_id,
            trigger_intent=tool_name,
            reason="tool_pressure_threshold",
        )

    def _transition_to(
        self,
        target: DegradationLevel,
        *,
        source_id: str | None,
        trigger_intent: str,
        reason: str,
    ) -> DegradationTransition | None:
        current = self._state.level
        if target <= current:
            return None  # monotonic — never decrease
        now = time.time()
        self._state.level = target
        self._state.tools_frozen = target >= DegradationLevel.LOCKED
        if target == DegradationLevel.LOCKED and self._locked_at is None:
            self._locked_at = now
        entry = (now, target, reason)
        self._state.level_history.append(entry)
        self._emit_transition_event(current, target, source_id, trigger_intent, reason, now)
        return DegradationTransition(
            previous_level=current,
            new_level=target,
            trigger_source_id=source_id,
            trigger_intent=trigger_intent,
            reason=reason,
            timestamp=now,
        )

    def _emit_transition_event(
        self,
        prev: DegradationLevel,
        new: DegradationLevel,
        source_id: str | None,
        trigger_intent: str,
        reason: str,
        ts: float,
    ) -> None:
        self._pending_events.append(DegradationTransitionEvent(
            kind=TraceEventKind.DEGRADATION_TRANSITION,
            node_id="",
            sequence=len(self._pending_events),
            previous_level=prev.name,
            new_level=new.name,
            trigger_source_id=source_id or "",
            trigger_intent=trigger_intent,
            reason=reason,
        ))

    def _emit_quarantine_event(self, source_id: str, reason: str) -> None:
        self._pending_events.append(SourceQuarantinedEvent(
            kind=TraceEventKind.SOURCE_QUARANTINED,
            node_id="",
            sequence=len(self._pending_events),
            source_id=source_id,
            quarantined_at=time.time(),
            reason=reason,
        ))
