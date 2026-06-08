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
from axor_core.contracts.mode import ExecutionMode
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

# Authority types permitted to lower degradation level. Anything outside this
# set (notably "worker") is rejected.
_VALID_GOVERNANCE_AUTHORITY_TYPES = frozenset({
    "human_operator",
    "automated_policy",
    "trusted_boundary",
})

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

    In observe mode (observe=True or from_mode(ExecutionMode.OBSERVE)):
    - All signals are processed and trace events are emitted.
    - self._state.level is never mutated — the agent is never policy-restricted.
    - A shadow level (_shadow_level) tracks the level that *would* have been
      reached so that the monotonicity check still applies and each transition
      is emitted exactly once.
    """

    def __init__(
        self,
        policy: DegradationPolicy | None = None,
        node_id: str = "",
        observe: bool = False,
    ) -> None:
        self._policy = policy or DegradationPolicy()
        self._state = DegradationState()
        self._pending_events: list[TraceEvent] = []
        self._locked_at: float | None = None  # wall time when LOCKED was entered
        self._node_id = node_id
        self._observe = observe
        self._shadow_level: DegradationLevel = DegradationLevel.NORMAL

    @classmethod
    def from_mode(
        cls,
        mode: ExecutionMode,
        policy: DegradationPolicy | None = None,
        node_id: str = "",
    ) -> "DegradationEngine":
        """Construct a DegradationEngine with observe=True when mode is OBSERVE."""
        return cls(policy=policy, node_id=node_id, observe=(mode == ExecutionMode.OBSERVE))

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

        v4.12 TM8: transitions are driven **only by decidable facts** (parameter-
        free Booleans over the current intent and its causal structure), never by
        counters or thresholds. The fact table:

          cross-origin export of an untrusted-root value through a sink  → LOCKED
          deny where the driving value's causal_root is tainted          → RESTRICTED
          deny on intent that executes_generated_code / after_external   → RESTRICTED
          a further untrusted-root *dangerous* fact while in LOCKED       → TERMINAL

        Counters (session_deny_count, *_pressure_count) are kept as **telemetry**
        only — they no longer drive a transition (accumulation → detection, X2).
        LOCKED_TTL remains as an orchestrator "don't hang forever" hook, not a
        guarantee fact.
        """
        # TERMINAL → accept no more signals.
        if self._state.level == DegradationLevel.TERMINAL:
            return None

        # Orchestrator hook (not a guarantee fact): LOCKED "don't hang forever".
        ttl_transition = self._check_locked_ttl()
        if ttl_transition is not None:
            return ttl_transition

        if denial is None:
            return None

        source_id = self.derive_source_id(intent, taint_state)
        source = self._get_or_create_source(source_id, taint_state)
        source.last_signal = time.time()

        # Telemetry counters — recorded, but NOT transition drivers (TM8 / X2).
        tool_name = intent.tool.lower()
        self._state.session_deny_count += 1
        if _is_write_bash_export(tool_name):
            source.tool_pressure_count += 1
        if intent.executes_generated_code or intent.after_external_read:
            source.instruction_pressure_count += 1

        untrusted_root = self._is_untrusted_root(intent, taint_state)
        dangerous = (
            intent.executes_generated_code
            or intent.after_external_read
            or _is_write_bash_export(tool_name)
        )

        # Fact 4: a further untrusted-root dangerous fact while LOCKED → TERMINAL.
        if self._current_level() >= DegradationLevel.LOCKED and untrusted_root and dangerous:
            t = self._transition_to(
                DegradationLevel.TERMINAL,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="untrusted_root_dangerous_while_locked",
            )
            if t is not None:
                return t

        # Fact 1: cross-origin export of an untrusted-root value → LOCKED.
        if _is_cross_origin_export(intent) and untrusted_root:
            t = self._transition_to(
                DegradationLevel.LOCKED,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="cross_origin_export_denied",
            )
            if t is not None:
                return t

        # Facts 2/3: deny where causal_root is tainted, or the intent executes
        # generated code / followed an external read → quarantine + RESTRICTED.
        if untrusted_root or intent.executes_generated_code or intent.after_external_read:
            if not source.quarantined:
                return self._quarantine_and_restrict(source, source_id, intent.tool)
            return None

        # A benign denial (no untrusted root, not dangerous) is a soft signal:
        # first one → CAUTIOUS (non-narrowing). Decidable fact ("a denial
        # occurred"), parameter-free.
        if self._current_level() == DegradationLevel.NORMAL:
            return self._transition_to(
                DegradationLevel.CAUTIOUS,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason="denial_observed",
            )
        return None

    def _is_untrusted_root(
        self,
        intent: "NormalizedIntent",
        taint_state: "TaintState",
    ) -> bool:
        """Decidable: does the driving value have an untrusted causal_root?"""
        if taint_state.is_tainted:
            return True
        if (intent.provenance or "") in ("external_web", "unknown"):
            return True
        return bool(intent.after_external_read)

    def _current_level(self) -> DegradationLevel:
        """Effective level (shadow in observe mode, real otherwise)."""
        return self._shadow_level if self._observe else self._state.level


    def record_detection_crossing(
        self,
        intent: "NormalizedIntent",
        taint_state: "TaintState",
        *,
        reason: str,
    ) -> DegradationTransition | None:
        """Ingest a detection-layer threshold-crossing as a tightening-only fact (TM7.1).

        Detection (reputation, anomaly) never gates `allow` directly — that would
        break T1, since reputation/score are not part of the structural projection.
        Instead a *decidable crossing* (a Boolean, not the underlying score) raises
        degradation, which narrows `policy`; `allow` then enforces the narrowed
        policy purely. This is fact-driven (a crossing Boolean, not a counter) and
        strictly monotone — it may only tighten, never loosen.

        Cross-origin export crossing → LOCKED; otherwise quarantine the source and
        move to RESTRICTED.
        """
        if self._state.level == DegradationLevel.TERMINAL:
            return None
        ttl_transition = self._check_locked_ttl()
        if ttl_transition is not None:
            return ttl_transition

        source_id = self.derive_source_id(intent, taint_state)
        source = self._get_or_create_source(source_id, taint_state)
        source.last_signal = time.time()

        if _is_cross_origin_export(intent):
            return self._transition_to(
                DegradationLevel.LOCKED,
                source_id=source_id,
                trigger_intent=intent.tool,
                reason=f"detection_crossing:{reason}",
            )
        return self._quarantine_and_restrict(source, source_id, intent.tool)

    def attempt_clear_by_worker(self) -> None:
        """Workers may not lower degradation level. Always raises DegradationClearanceError."""
        raise DegradationClearanceError(
            "worker attempted to clear degradation — only governance may do this"
        )

    def check_ttl(self) -> DegradationTransition | None:
        """
        Explicitly check LOCKED_TTL and auto-transition to TERMINAL if elapsed.

        Called by GovernedSession.run() so that idle sessions that reached LOCKED
        but received no further intents still transition to TERMINAL on the next run.
        """
        return self._check_locked_ttl()

    def apply_to_policy(
        self,
        base_policy: ExecutionPolicy,
        source_id: str | None,
    ) -> ExecutionPolicy:
        """
        Return a narrowed ExecutionPolicy for the current degradation state.
        """
        # Check TTL on every policy application so LOCKED transitions to TERMINAL
        # even when record_signal is not called (e.g. non-denied intents).
        self._check_locked_ttl()
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

        The authority is validated: a blank principal or an authority_type
        outside the allowed governance set is rejected, so a worker cannot lower
        degradation by passing an empty or worker-labelled authority.
        """
        if (
            not authority.authority_id
            or not reason
            or authority.authority_type not in _VALID_GOVERNANCE_AUTHORITY_TYPES
        ):
            raise DegradationClearanceError(
                "degradation clearance rejected: requires a valid governance "
                f"authority (authority_id={authority.authority_id!r}, "
                f"authority_type={authority.authority_type!r})"
            )
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
        if self._observe:
            # Shadow monotonicity: each transition emitted exactly once.
            if target <= self._shadow_level:
                return None
            self._shadow_level = target
        else:
            if target <= current:
                return None  # monotonic — never decrease
            self._state.level = target
            self._state.tools_frozen = target >= DegradationLevel.LOCKED
            if target == DegradationLevel.LOCKED and self._locked_at is None:
                self._locked_at = time.time()
            self._state.level_history.append((time.time(), target, reason))
        now = time.time()
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
            node_id=self._node_id,
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
            node_id=self._node_id,
            sequence=len(self._pending_events),
            source_id=source_id,
            quarantined_at=time.time(),
            reason=reason,
        ))
