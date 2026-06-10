from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from axor_core.capability.executor import CapabilityExecutor
from axor_core.capability.flood import EscalationFloodGuard
from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent, IntentKind, ResolvedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.policy.consequence import consequence_class
from axor_core.policy.value_policy import check_value_policies
from axor_core.security.carrier import classify_carrier
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import (
    CancelledEvent,
    EscalationDeniedEvent,
    EscalationGrantedEvent,
    IntentDeniedEvent,
    SinkDensityEvent,
    TokensSpentEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.contracts.denial import DenialResponse
from axor_core.errors.exceptions import (
    ToolNotAllowedError,
    ToolNotFoundError,
)
from axor_core.capability.lease_validator import (
    LeaseValidator,
    extract_path_arg,
    path_matches_allowlist,
)
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.taint import Carrier, TaintSource
from axor_core.degradation.engine import _LOCKED_ALLOWED_TOOLS
from axor_core.node.normalizer import IntentNormalizer
from axor_core.taint.causal_root import CausalRoot

if TYPE_CHECKING:
    from axor_core.contracts.lease import CapabilityLease
    from axor_core.contracts.reputation import ReputationEnricher
    from axor_core.degradation.engine import DegradationEngine
    from axor_core.contracts.provenance import ValueProvenance

log = logging.getLogger("axor.intent_loop")

# Exception types from a tool execution that are expected and should be
# converted to a structured denial. Anything outside this set is a real
# programming error that we still convert to a denial (so the user-facing
# conversation continues) but log loudly so the bug isn't invisible.
_KNOWN_TOOL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ToolNotAllowedError,
    ToolNotFoundError,
    ValueError,
    OSError,
    PermissionError,
    FileNotFoundError,
    TimeoutError,
)

# Optional callback fired after each tool executes.
# Signature: (tool_use_id, tool_name, result, approved) → None
# Used by ClaudeCodeExecutor to push results into ToolResultBus
# without intent_loop knowing about adapter internals.
ToolResultCallback = Callable[[str, str, Any, bool], Awaitable[None]]

# Optional callback for spawn_child intents.
# Signature: (tool_use_id, task, context_hint) → str (child result)
# Implemented in wrapper.py — intent_loop stays provider-agnostic.
SpawnCallback = Callable[[str, str, str], Awaitable[str]]

# Optional callback for ESCALATE_POLICY intents requiring human approval.
# Signature: (tool_use_id, tool, paths, max_ops) → approved bool
EscalationCallback = Callable[[str, str, list[str], int], Awaitable[bool]]


@dataclass
class _GrantedEscalation:
    tool: str
    paths: list[str]  # empty = no path restriction
    ops_remaining: int


class IntentLoop:
    """
    Core of governed execution.

    Intercepts every event from the executor stream and processes it
    before anything reaches the outside world.

    For each event:
        tool_use  → convert to Intent → resolve against policy → execute or deny
        text      → pass through (export filter applied later in export.py)
        stop      → finalize token accounting, yield as-is
        error     → yield as-is, let wrapper handle

    The executor never knows it is being intercepted.
    It receives tool results exactly as if it called the tools directly.
    A denied tool returns a structured denial reason as the tool result —
    the executor sees this as a tool response, not a governance event.

    Token accounting happens here — after every tool result,
    running totals are updated and yielded as trace events.
    """

    def __init__(
        self,
        capability_executor: CapabilityExecutor,
        trace_events: list,
        current_depth: int = 0,
        tool_result_callback: ToolResultCallback | None = None,
        spawn_callback: SpawnCallback | None = None,
        escalation_callback: EscalationCallback | None = None,
        taint_engine: "ValueProvenance | None" = None,
        degradation_engine: "DegradationEngine | None" = None,
        reputation_enricher: "ReputationEnricher | None" = None,
        max_intents_per_session: int | None = None,
        max_total_spawns: int | None = None,
        value_policies: "dict | None" = None,
        consequence_overrides: "dict | None" = None,
        positional_sinks: "frozenset[str] | set[str] | None" = None,
    ) -> None:
        self._executor = capability_executor
        self._trace_events = trace_events
        self._depth = current_depth
        self._tool_result_callback = tool_result_callback
        self._spawn_callback = spawn_callback
        self._escalation_callback = escalation_callback
        self._taint_engine = taint_engine
        self._degradation_engine = degradation_engine
        self._reputation_enricher = reputation_enricher
        self._normalizer = IntentNormalizer()
        self._intent_sequence = 0
        self._token_totals = _TokenAccumulator()
        self._granted_escalations: dict[str, _GrantedEscalation] = {}
        self._capability_leases: dict[str, "CapabilityLease"] = {}
        self._lease_validator = LeaseValidator()
        self._escalation_count = 0
        self._flood_guard = EscalationFloodGuard()
        # DoS guards — opt-in (None = unlimited). GovernedSession sets prod defaults.
        self._max_intents_per_session = max_intents_per_session
        self._max_total_spawns = max_total_spawns
        self._value_policies = value_policies or {}
        self._consequence_overrides = consequence_overrides or {}
        # D_high partition (Corollary: stratified enforcement). Sinks the operator
        # DECLARES to have an instruction-incomplete codomain — i.e. their legitimate
        # input cannot encode an instruction, and the trusted side constrains input
        # to that codomain (constrained decoding, obligation 1). For these sinks,
        # admission flips from the X1-leaky content-derivation DENY-LIST to a sound
        # POSITIONAL ALLOW-LIST: admit only if the driving value's carrier is
        # instruction-incomplete, content-independently (closes O2 vs paraphrase).
        # Opt-in/empty by default; exec-class sinks can NEVER be declared here
        # (shell command codomain is instruction-complete by definition).
        self._positional_sinks = frozenset(positional_sinks or ())
        _illegal = {s for s in self._positional_sinks if s.lower() in _INSTRUCTION_COMPLETE_SINKS}
        if _illegal:
            raise ValueError(
                "instruction-complete sinks cannot be declared positional (D_high): "
                f"{sorted(_illegal)} — their codomain admits instructions by "
                "definition; they must stay in D_low (content-derivation)."
            )
        self._spawn_count = 0

    async def run(
        self,
        stream: AsyncIterator[ExecutorEvent],
        envelope: ExecutionEnvelope,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Process the executor stream under governance.
        Yields governed events — tool_use events are replaced by their results.
        Checks cancel_token at every event boundary — cooperative cancellation.

        On cancellation or generator-close the underlying executor stream is
        explicitly `aclose()`'d so adapter resources (HTTP connections, SDK
        streaming contexts) are released promptly.
        """
        try:
            async for event in self._run_inner(stream, envelope):
                yield event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    log.debug("executor stream aclose() raised", exc_info=True)

    async def _run_inner(
        self,
        stream: AsyncIterator[ExecutorEvent],
        envelope: ExecutionEnvelope,
    ) -> AsyncIterator[ExecutorEvent]:
        async for event in stream:
            # cooperative cancellation — check before every event
            if envelope.cancel_token.is_cancelled():
                self._record_cancellation(envelope)
                return

            match event.kind:
                case ExecutorEventKind.TOOL_USE:
                    tool_name = event.payload.get("tool", "")
                    tool_use_id = event.payload.get("tool_use_id", "")

                    # escalate_policy — mid-execution capability grant request
                    if tool_name == "escalate_policy":
                        try:
                            result = await self._handle_escalation(event, envelope)
                        except Exception as exc:
                            log.error(
                                "escalation handling failed (node=%s): %s",
                                envelope.node_id, exc, exc_info=True,
                            )
                            result = {
                                "error": "escalation_denied",
                                "reason": f"malformed escalation request: {type(exc).__name__}",
                            }
                        if self._tool_result_callback is not None:
                            await self._tool_result_callback(
                                tool_use_id, tool_name, result, True
                            )
                        else:
                            yield ExecutorEvent(
                                kind=ExecutorEventKind.TEXT,
                                payload={
                                    "tool_result": result,
                                    "approved": True,
                                },
                                node_id=envelope.node_id,
                            )
                        continue

                    # spawn_child is a special intent — not a regular tool call
                    if (
                        tool_name == "spawn_child"
                        and self._spawn_callback is not None
                    ):
                        self._intent_sequence += 1
                        spawn_intent = Intent(
                            kind=IntentKind.SPAWN_CHILD,
                            payload=event.payload,
                            node_id=envelope.node_id,
                            sequence=self._intent_sequence,
                        )
                        # Capability check BEFORE dispatching to callback —
                        # denial must be recorded in the trace regardless of
                        # whether the callback would catch the exception.
                        spawn_decision = self.resolve_spawn_intent(
                            spawn_intent, envelope
                        )
                        if spawn_decision.kind == PolicyDecisionKind.DENY:
                            self._record_denial(
                                spawn_intent, spawn_decision.reason, envelope
                            )
                            denial = _denial_result(tool_name, spawn_decision.reason)
                            if self._tool_result_callback is not None:
                                await self._tool_result_callback(
                                    tool_use_id, tool_name, denial, False
                                )
                            else:
                                yield ExecutorEvent(
                                    kind=ExecutorEventKind.TEXT,
                                    payload={
                                        "tool_result": denial,
                                        "approved": False,
                                    },
                                    node_id=envelope.node_id,
                                )
                            continue

                        # Per-value carrier/taint gate (NC2) — the capability check
                        # above only asks "may this node spawn"; it does not inspect
                        # WHAT drives the spawn. A tainted free-text task is prompt
                        # injection reaching an instruction-following sink.
                        spawn_args = event.payload.get("args", {})
                        taint_reason = self._spawn_taint_reason(spawn_args)
                        if taint_reason is not None:
                            self._record_denial(spawn_intent, taint_reason, envelope)
                            denial = _denial_result(tool_name, taint_reason)
                            if self._tool_result_callback is not None:
                                await self._tool_result_callback(
                                    tool_use_id, tool_name, denial, False
                                )
                            else:
                                yield ExecutorEvent(
                                    kind=ExecutorEventKind.TEXT,
                                    payload={"tool_result": denial, "approved": False},
                                    node_id=envelope.node_id,
                                )
                            continue

                        # Approved — count this spawn once at the dispatch site.
                        self._spawn_count += 1
                        task = event.payload.get("args", {}).get("task", "")
                        context_hint = event.payload.get("args", {}).get(
                            "context_hint", ""
                        )
                        child_result = await self._spawn_callback(
                            tool_use_id, task, context_hint
                        )

                        if self._tool_result_callback is not None:
                            await self._tool_result_callback(
                                tool_use_id, tool_name, child_result, True
                            )
                        else:
                            yield ExecutorEvent(
                                kind=ExecutorEventKind.TEXT,
                                payload={
                                    "tool_result": child_result,
                                    "approved": True,
                                },
                                node_id=envelope.node_id,
                            )
                        continue

                    resolved = await self._resolve_tool_intent(event, envelope)

                    if self._tool_result_callback is not None:
                        # adapter-driven path (e.g. ClaudeCodeExecutor + ToolResultBus)
                        # callback pushes result into adapter's bus
                        # executor reads from bus and injects into conversation itself
                        await self._tool_result_callback(
                            event.payload.get("tool_use_id", ""),
                            event.payload.get("tool", ""),
                            resolved.result,
                            resolved.approved,
                        )
                        # do NOT yield — executor manages the tool result injection
                    else:
                        # default path — yield result as TEXT event
                        # executor sees it in the stream (mock executors, tests)
                        yield ExecutorEvent(
                            kind=ExecutorEventKind.TEXT,
                            payload={
                                "tool_result": resolved.result,
                                "approved": resolved.approved,
                            },
                            node_id=envelope.node_id,
                        )

                case ExecutorEventKind.TEXT:
                    # pass through — export.py will filter later
                    yield event

                case ExecutorEventKind.STOP:
                    self._record_token_event(event, envelope)
                    yield event

                case ExecutorEventKind.ERROR:
                    yield event

    # ── Intent resolution ──────────────────────────────────────────────────────

    async def _resolve_tool_intent(
        self,
        event: ExecutorEvent,
        envelope: ExecutionEnvelope,
    ) -> ResolvedIntent:
        self._intent_sequence += 1

        tool_name = event.payload.get("tool", "")
        tool_args = event.payload.get("args", {})

        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": tool_args},
            node_id=envelope.node_id,
            sequence=self._intent_sequence,
        )

        # Session-level intent cap (DoS guard). Opt-in; None = unlimited.
        if (
            self._max_intents_per_session is not None
            and self._intent_sequence > self._max_intents_per_session
        ):
            reason = (
                f"session intent limit reached ({self._max_intents_per_session})"
            )
            self._record_denial(intent, reason, envelope)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=reason,
                result=_denial_result(tool_name, reason),
            )

        decision = self._evaluate_tool_intent(intent, envelope)

        if decision.kind == PolicyDecisionKind.DENY:
            self._record_denial(intent, decision.reason, envelope)
            denial_resp = _make_denial_response(decision.reason)
            self._record_degradation_signal(intent, denial_resp)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=decision.reason,
                result=denial_resp.to_tool_result(),
            )

        # Normalize early: the consequence gate (X5) needs the operation enum to
        # escalate a generic sink whose command is power-state-changing (e.g.
        # `bash shutdown`). Pure structural transform; reputation enrichment stays
        # telemetry-only and never gates.
        needs_normalized = (
            self._taint_engine is not None
            or self._degradation_engine is not None
            or self._reputation_enricher is not None
        )
        normalized: NormalizedIntent | None = None
        if needs_normalized:
            normalized = self._normalizer.normalize(intent)
        if self._reputation_enricher is not None and normalized is not None:
            normalized = self._reputation_enricher.enrich(normalized, intent)

        # consequence axis (TM3.1) — content-blind structural gate on the action
        # class, part of the pure `allow`. Catches consequential-action-under-
        # trusted-provenance (the OpenClaw class, X5) that the provenance axes
        # cannot see. Reads only the sink type + operation enum + policy ceiling.
        consequence_denial = self._check_consequence(
            tool_name, envelope,
            operation=normalized.operation if normalized is not None else None,
        )
        if consequence_denial is not None:
            self._record_denial(intent, consequence_denial, envelope)
            denial_resp = _make_denial_response(consequence_denial, "consequence_gate")
            self._record_degradation_signal(intent, denial_resp)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=consequence_denial,
                result=denial_resp.to_tool_result(),
            )

        # value-policy predicates (TM3.1 predicate layer) — operator-registered
        # range/enum predicates over an admissible projection of an argument
        # (e.g. transfer(amount) within bounds). Content-blind: reads the numeric/
        # enum projection, discharged by the Thm. 0 decision procedures.
        value_denial = check_value_policies(tool_name, tool_args, self._value_policies)
        if value_denial is not None:
            self._record_denial(intent, value_denial, envelope)
            denial_resp = _make_denial_response(value_denial, "value_policy")
            self._record_degradation_signal(intent, denial_resp)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=value_denial,
                result=denial_resp.to_tool_result(),
            )

        # normalize intent once — shared by taint, degradation, and reputation

        # ── allow gate (pure) — capability already checked above; here the
        # degradation-narrowed policy gate. Degradation is driven only by
        # structural facts; no probabilistic detector feeds it (ML/judge removed).
        if self._degradation_engine is not None and normalized is not None:
            # Per-value (NM3): hand the driving value's causal_root to the check
            # path so a value-keyed quarantine ("value:<src>") is matched here, not
            # only on the record_signal path. Without it derive_source_id falls back
            # to provenance/"unknown" and the narrowing silently misses.
            check_root = (
                self._taint_engine.derive_value(tool_args)
                if self._taint_engine is not None
                else None
            )
            degradation_denial = self._check_degradation_denial(
                tool_name, normalized, envelope, driving_root=check_root
            )
            if degradation_denial is not None:
                self._record_denial(intent, degradation_denial, envelope)
                denial_resp = _make_denial_response(degradation_denial)
                self._record_degradation_signal(intent, denial_resp, normalized)
                return ResolvedIntent(
                    intent=intent,
                    approved=False,
                    reason=degradation_denial,
                    result=denial_resp.to_tool_result(),
                )

        # SSRF / internal-destination gate — content-blind, always-on, and
        # independent of taint (this is a *destination* concern, not a data-flow
        # one): no agent should reach cloud metadata or the docker socket unless
        # policy explicitly allows it. Decoupling this from taint is what lets the
        # taint gate below be cleanly per-value without regressing SSRF safety.
        _INTERNAL = ("cloud_metadata", "private_network", "docker_socket")
        if normalized is not None and (
            normalized.target_kind in _INTERNAL
            or normalized.destination_kind in ("cloud_metadata", "private_network")
        ):
            reason = (
                f"ssrf gate: '{tool_name}' targets an internal destination "
                f"({normalized.target_kind}/{normalized.destination_kind}) — "
                "blocked independent of taint"
            )
            self._record_denial(intent, reason, envelope)
            denial_resp = _make_denial_response(reason, "ssrf_gate")
            self._record_degradation_signal(intent, denial_resp, normalized)
            return ResolvedIntent(
                intent=intent, approved=False, reason=reason,
                result=denial_resp.to_tool_result(),
            )

        # taint enforcement — PER-VALUE (TM2). The refactor from per-session to
        # per-value: the gate decides on the *driving argument's own* causal_root
        # (content-derivation ledger), NOT a session-wide flag. A clean-valued
        # sink passes even when other values in the session are tainted — that is
        # the per-value win the density experiment (TM3.3) measured.
        # Named gap (X1): content-derivation misses paraphrased / re-encoded
        # influence; sound over-taint of opaque-LLM output collapses to
        # session-sticky and needs an interpreter (ceded to CaMeL).
        if self._taint_engine is not None and normalized is not None:
            driving_root = self._taint_engine.derive_value(tool_args)

            # Density (TM3.3): record, per high-stakes sink firing, the per-value
            # taint (both axes) and the session-sticky shadow. The make-or-break
            # number, measured live and split integrity vs confidentiality so the
            # taint-explosion asymmetry is visible. Uses the same overrides as the
            # enforcement gate so density and enforcement agree on which sinks count.
            sink_consequence = consequence_class(
                tool_name, operation=normalized.operation,
                overrides=self._consequence_overrides,
            )
            if sink_consequence >= ConsequenceClass.REVERSIBLE:
                session_tainted, session_sensitive = (
                    self._taint_engine.session_shadow()
                    if hasattr(self._taint_engine, "session_shadow")
                    else (driving_root.is_tainted, driving_root.sensitive)
                )
                self._trace_events.append(SinkDensityEvent(
                    kind=TraceEventKind.SINK_DENSITY,
                    node_id=envelope.node_id,
                    sequence=len(self._trace_events),
                    operation=tool_name,
                    tainted=driving_root.is_tainted,
                    sensitive=driving_root.sensitive,
                    session_tainted=session_tainted,
                    session_sensitive=session_sensitive,
                ))

            # D_high POSITIONAL ADMISSION (Corollary: stratified enforcement).
            # For a sink the operator DECLARED instruction-incomplete, admission
            # flips from the X1-leaky content-derivation deny-list to a sound
            # positional allow-list: admit ONLY if the driving value's carrier is
            # instruction-incomplete (ENDORSED/CLOSED_SCHEMA), else fail-closed.
            # Crucially this does NOT consult driving_root.is_tainted — that is the
            # whole point: a paraphrase that launders the content-derivation label
            # cannot change the value's FORM, and classify_carrier is structural
            # (T0), so the induction step O2 closes against semantic derivation,
            # content-independently. classify_carrier takes the WORST carrier over
            # all argument leaves, so a single non-positional argument path nullifies
            # admission (O3 / complete mediation, local to this sink). No-upgrade
            # holds by construction: the carrier is recomputed structurally each
            # call, there is no stored positional label to launder through D_low.
            if tool_name in self._positional_sinks:
                if classify_carrier(tool_args) == Carrier.FREE_TEXT:
                    reason = (
                        f"positional gate (D_high): '{tool_name}' is a declared "
                        f"instruction-incomplete sink; its driving value is FREE_TEXT "
                        f"(non-positional) — admitted only via an instruction-incomplete "
                        f"carrier, independent of content-derivation"
                    )
                    self._record_denial(intent, reason, envelope)
                    denial_resp = _make_denial_response(reason, "positional_gate")
                    self._record_degradation_signal(intent, denial_resp, normalized)
                    return ResolvedIntent(
                        intent=intent, approved=False, reason=reason,
                        result=denial_resp.to_tool_result(),
                    )

            # Carrier / imperative-channel gate (TM1): a tainted FREE_TEXT value
            # reaching an instruction-following sink (it would be interpreted as a
            # directive — spawn a sub-agent, send a message, execute) is the
            # imperative channel. classify_carrier reads the *form*, deterministic
            # (T0). Complements per-value: catches free-text-as-directive that the
            # risky-op list below does not (e.g. spawn_child(task=<free text>)).
            if driving_root.is_tainted and _is_imperative_sink(tool_name, normalized):
                if classify_carrier(tool_args) == Carrier.FREE_TEXT:
                    reason = (
                        f"carrier gate (TM1): untrusted FREE_TEXT value into the "
                        f"instruction-following sink '{tool_name}' (imperative channel)"
                    )
                    self._record_denial(intent, reason, envelope)
                    denial_resp = _make_denial_response(reason, "carrier_gate")
                    self._record_degradation_signal(intent, denial_resp, normalized)
                    return ResolvedIntent(
                        intent=intent, approved=False, reason=reason,
                        result=denial_resp.to_tool_result(),
                    )

            exfil_destination = normalized.destination_kind in (
                "cloud_metadata", "private_network", "external_domain"
            )
            integrity_risk = driving_root.is_tainted and (
                normalized.writes_outside_workdir
                or normalized.executes_generated_code
                or exfil_destination
            )
            # Confidentiality SOUND FLOOR (TM4, 1.1b). Egress is denied while a
            # secret read is outstanding — on the FACT of the read, NOT on whether
            # THIS value's content derives as sensitive. This is the sound floor the
            # density numbers justified: per-value confidentiality (driving_root.
            # sensitive) is X1-leaky (a paraphrased secret evades content-matching),
            # so the floor gates egress coarsely and is lifted only by governance
            # endorsement of the secret. Sparse by construction — it fires only
            # after a sensitive read. The per-value sensitive check is subsumed
            # (value sensitive ⟹ a sensitive read happened ⟹ floor active).
            floor_active = (
                self._taint_engine.confidentiality_floor_active()
                if hasattr(self._taint_engine, "confidentiality_floor_active")
                else driving_root.sensitive
            )
            confidentiality_risk = exfil_destination and floor_active
            if integrity_risk or confidentiality_risk:
                axis = (
                    "confidentiality (egress under the sound floor — a secret read is "
                    "outstanding; release requires governance endorsement)"
                    if confidentiality_risk
                    else "integrity (untrusted-derived value into a high-risk operation)"
                )
                reason = (
                    f"taint enforcement (per-value): the driving argument of "
                    f"'{tool_name}' carries a tainted/sensitive value — {axis}"
                )
                self._record_denial(intent, reason, envelope)
                denial_resp = _make_denial_response(reason, "taint_enforcement")
                self._record_degradation_signal(intent, denial_resp, normalized)
                return ResolvedIntent(
                    intent=intent,
                    approved=False,
                    reason=reason,
                    result=denial_resp.to_tool_result(),
                )

        # approved or transformed — emit the appropriate trace event
        is_transform = decision.kind == PolicyDecisionKind.TRANSFORM
        self._trace_events.append(
            TraceEvent(
                kind=TraceEventKind.INTENT_TRANSFORMED if is_transform else TraceEventKind.INTENT_APPROVED,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                payload={"tool": tool_name},
            )
        )

        # approved or transformed
        effective_args = decision.transformed_payload or tool_args
        effective_intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": effective_args},
            node_id=envelope.node_id,
            sequence=self._intent_sequence,
        )

        try:
            result = await self._executor.execute(
                effective_intent, envelope.capabilities
            )
            # Register the tool output into the PER-VALUE ledger (TM2) so a later
            # sink whose argument carries this content is gated at the value level.
            # No session-taint propagation — a read taints its produced *value*,
            # not the whole session.
            self._register_value_taint(effective_intent, result, normalized)
            self._record_degradation_signal(intent, None)
            return ResolvedIntent(
                intent=intent,
                approved=True,
                reason="approved",
                result=result,
                transformed_payload=decision.transformed_payload,
            )
        except _KNOWN_TOOL_EXCEPTIONS as exc:
            reason = str(exc)
            self._record_denial(intent, reason, envelope)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=reason,
                result=_denial_result(tool_name, reason),
            )
        except Exception as exc:
            # Unexpected exception — likely a programming bug in the handler.
            # Log the full traceback so the bug isn't silent, but still convert
            # to a denial so the conversation continues. The user gets a
            # readable error; the operator gets a stack trace in logs.
            tb = traceback.format_exc()
            log.error(
                "Unhandled exception in tool '%s' (node=%s): %s\n%s",
                tool_name,
                envelope.node_id,
                exc,
                tb,
            )
            reason = f"tool execution failed: {type(exc).__name__}: {exc}"
            self._record_denial(intent, reason, envelope)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=reason,
                result=_denial_result(tool_name, reason),
            )

    def _evaluate_tool_intent(
        self,
        intent: Intent,
        envelope: ExecutionEnvelope,
    ) -> PolicyDecision:
        """
        Evaluate a tool_call intent against capabilities.

        Resolution order:
        1. Active escalation grant covers the tool?  → approve (decrement ops)
        2. Is the tool in allowed_tools?             → approve
        3. Is the tool in extra_denied?              → deny
        4. Not in capabilities at all                → deny
        """
        tool_name: str = intent.payload.get("tool", "")
        caps = envelope.capabilities
        tool_args = intent.payload.get("args", {})

        # Filesystem ceiling — applies to every path-bearing tool call regardless
        # of how it is later approved (allowed_tools, lease, or escalation grant).
        policy_paths = getattr(envelope.policy, "allowed_paths", ()) or ()
        if policy_paths:
            candidate_path = extract_path_arg(tool_args)
            if candidate_path and not path_matches_allowlist(candidate_path, policy_paths):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=(
                        f"path {candidate_path!r} is outside policy "
                        f"allowed_paths {tuple(policy_paths)!r}"
                    ),
                )

        # Check CapabilityLease first (authoritative — validates TTL + max_uses)
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

        if tool_name in caps.allowed_tools:
            return PolicyDecision(
                kind=PolicyDecisionKind.APPROVE,
                reason="tool in allowed_tools",
            )

        if tool_name in envelope.policy.tool_policy.extra_denied:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"tool '{tool_name}' is explicitly denied by policy",
            )

        return PolicyDecision(
            kind=PolicyDecisionKind.DENY,
            reason=f"tool '{tool_name}' is not in capabilities for policy '{envelope.policy.name}'",
        )

    async def _handle_escalation(
        self,
        event: ExecutorEvent,
        envelope: ExecutionEnvelope,
    ) -> dict:
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
            self._trace_events.append(
                EscalationDeniedEvent(
                    kind=TraceEventKind.ESCALATION_DENIED,
                    node_id=node_id,
                    sequence=len(self._trace_events),
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

        self._trace_events.append(
            EscalationGrantedEvent(
                kind=TraceEventKind.ESCALATION_GRANTED,
                node_id=node_id,
                sequence=len(self._trace_events),
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

    def resolve_spawn_intent(
        self,
        intent: Intent,
        envelope: ExecutionEnvelope,
    ) -> PolicyDecision:
        """
        Evaluate a spawn_child intent.
        Called by GovernedNode.wrapper when executor requests a child.
        """
        caps = envelope.capabilities
        policy = envelope.policy

        if not caps.allow_children:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"child nodes not allowed by policy '{policy.name}' (child_mode={policy.child_mode})",
            )

        if self._depth >= policy.max_child_depth:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"max child depth reached: current={self._depth}, max={policy.max_child_depth}",
            )

        # Total-spawn cap (DoS guard against wide fan-out). Opt-in; None = unlimited.
        # Read-only here — resolve_spawn_intent may run more than once per spawn
        # (gate + prepare_child), so the counter is incremented at the single
        # dispatch site in _run_inner, not here.
        if (
            self._max_total_spawns is not None
            and self._spawn_count >= self._max_total_spawns
        ):
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"session spawn limit reached ({self._max_total_spawns})",
            )

        return PolicyDecision(
            kind=PolicyDecisionKind.APPROVE,
            reason="child node approved",
        )

    # ── Token accounting ───────────────────────────────────────────────────────

    def _record_token_event(
        self,
        stop_event: ExecutorEvent,
        envelope: ExecutionEnvelope,
    ) -> None:
        usage = stop_event.payload.get("usage", {})
        self._token_totals.add(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            tool_tokens=usage.get("tool_tokens", 0),
            context_tokens=envelope.context.token_count,
            cache_creation_input_tokens=usage.get(
                "cache_creation_input_tokens", 0
            ),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )
        self._trace_events.append(
            TokensSpentEvent(
                kind=TraceEventKind.TOKENS_SPENT,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                input_tokens=self._token_totals.input,
                output_tokens=self._token_totals.output,
                tool_tokens=self._token_totals.tool,
                context_tokens=self._token_totals.context,
                cumulative=self._token_totals.total,
                cache_creation_input_tokens=self._token_totals.cache_creation,
                cache_read_input_tokens=self._token_totals.cache_read,
            )
        )

    def _spawn_taint_reason(self, spawn_args: dict) -> str | None:
        """Carrier/taint gate for spawn_child (NC2). The child's `task` is free text
        the child interprets as instructions — spawn_child is an instruction-
        following sink (it is in `_IMPERATIVE_SINKS`). A tainted FREE_TEXT task is
        the imperative channel (TM1). The regular tool path applies exactly this
        gate; the spawn branch dispatches before reaching it, so apply it here too.

        Returns a denial reason, or None to allow. (Structured/sensitive values that
        are not the imperative channel stay gated at the child's own sinks: the
        child inherits this engine's per-value ledger.)
        """
        if self._taint_engine is None:
            return None
        driving_root = self._taint_engine.derive_value(spawn_args)
        if driving_root.is_tainted and classify_carrier(spawn_args) == Carrier.FREE_TEXT:
            return (
                "carrier gate (TM1): untrusted FREE_TEXT value into spawn_child — "
                "the child task is interpreted as instructions (imperative channel)"
            )
        return None

    def _record_denial(
        self,
        intent: Intent,
        reason: str,
        envelope: ExecutionEnvelope,
    ) -> None:
        self._trace_events.append(
            IntentDeniedEvent(
                kind=TraceEventKind.INTENT_DENIED,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                intent_kind=intent.kind.value,
                reason=reason,
            )
        )

    def _register_value_taint(
        self,
        effective_intent: Intent,
        result: Any,
        normalized: NormalizedIntent | None,
    ) -> None:
        """Register a tool's output content in the per-value ledger with the
        causal_root its read introduces (TM2). Mirrors the session-taint triggers:
        external/web reads → untrusted; secret/system reads → untrusted + sensitive.
        """
        if self._taint_engine is None:
            return
        ni = normalized or self._normalizer.normalize(effective_intent)
        if ni.target_kind in ("external_url", "cloud_metadata", "docker_socket") or \
                ni.operation == "network_request":
            root = CausalRoot.external_read(TaintSource.WEB)
        elif (
            ni.target_kind in ("secret", "system_path")
            or ni.reads_secret_like_data
            or ni.writes_outside_workdir
        ):
            sensitive = ni.target_kind == "secret" or ni.reads_secret_like_data
            root = CausalRoot.external_read(TaintSource.FILE, sensitive=sensitive)
        else:
            return  # clean read — nothing to register
        self._taint_engine.register_value(result, root)

    def _check_consequence(
        self,
        tool_name: str,
        envelope: ExecutionEnvelope,
        operation: str | None = None,
    ) -> str | None:
        """Consequence axis gate (TM3.1). Return a denial reason if the sink's
        action class exceeds the policy's unattended ceiling and no governance
        gate is present, else None. Content-blind: reads only the sink type and
        the operation enum (never argument content).

        The governance gate is satisfied by an active escalation grant or
        capability lease for the tool (a human/operator-authorised path).
        """
        cls = consequence_class(
            tool_name, operation=operation, overrides=self._consequence_overrides
        )
        ceiling = getattr(
            envelope.policy, "max_unattended_consequence", ConsequenceClass.CONSEQUENTIAL
        )
        if cls <= ceiling:
            return None
        # over ceiling — admissible only through a governance gate.
        if tool_name in self._granted_escalations or tool_name in self._capability_leases:
            return None
        return (
            f"consequence gate: sink '{tool_name}' is {cls.name}, exceeding the "
            f"unattended ceiling {ceiling.name}; a governance/human gate is required"
        )

    # ── Detection layer (TM7) — out-of-band from `allow` ────────────────────────

    def _record_cancellation(self, envelope: ExecutionEnvelope) -> None:
        token = envelope.cancel_token
        self._trace_events.append(
            CancelledEvent(
                kind=TraceEventKind.CANCELLED,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                reason=token.reason.value if token.reason else "unknown",
                detail=token.detail,
                completed_intents=self._intent_sequence,
            )
        )

    # ── Degradation helpers ────────────────────────────────────────────────────

    def _record_degradation_signal(
        self,
        intent: Intent,
        denial: "DenialResponse | None",
        normalized: NormalizedIntent | None = None,
    ) -> None:
        """Call record_signal on degradation engine and flush its events into the trace."""
        if self._degradation_engine is None:
            return
        ni = normalized if normalized is not None else self._normalizer.normalize(intent)
        # Per-value: degradation keys on the driving argument's own causal_root,
        # not a session-taint flag.
        driving_root = None
        if self._taint_engine is not None:
            driving_root = self._taint_engine.derive_value(intent.payload.get("args", {}))
        self._degradation_engine.record_signal(ni, denial, driving_root=driving_root)
        for event in self._degradation_engine.drain_events():
            self._trace_events.append(event)

    def _check_degradation_denial(
        self,
        tool_name: str,
        normalized: NormalizedIntent,
        envelope: ExecutionEnvelope,
        driving_root: "CausalRoot | None" = None,
    ) -> str | None:
        """
        Return a denial reason string if degradation state forbids this tool call, else None.
        Called before the main cascade — enforces apply_to_policy narrowing.
        """
        if self._degradation_engine is None:
            return None
        source_id = self._degradation_engine.derive_source_id(normalized, driving_root)
        effective = self._degradation_engine.apply_to_policy(envelope.policy, source_id)
        # LOCKED/TERMINAL: only read + escalate allowed
        level = self._degradation_engine.state.level
        if level >= DegradationLevel.LOCKED:
            if tool_name.lower() not in _LOCKED_ALLOWED_TOOLS:
                return (
                    f"degradation enforcement: session is {level.name} — "
                    f"tool '{tool_name}' is not permitted (only read/escalate allowed)"
                )
        # RESTRICTED: write/bash/export removed for quarantined source
        elif level == DegradationLevel.RESTRICTED:
            if not effective.tool_policy.allow_bash and tool_name.lower() in {"bash", "shell", "execute", "run"}:
                return (
                    f"degradation enforcement: source '{source_id}' is quarantined — "
                    f"tool '{tool_name}' (bash/execute) is restricted"
                )
            if not effective.tool_policy.allow_write and tool_name.lower() in {"write", "edit", "multiedit"}:
                return (
                    f"degradation enforcement: source '{source_id}' is quarantined — "
                    f"tool '{tool_name}' (write) is restricted"
                )
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────


_IMPERATIVE_SINKS = frozenset({
    "spawn_child", "send", "message", "prompt", "ask", "delegate",
    "reply", "email", "slack", "post", "notify",
})

# Sinks whose codomain is instruction-COMPLETE by definition: they interpret their
# argument as a program / directive (a shell command, a child-agent task). These
# can NEVER be lifted into the D_high positional partition — a positional gate would
# either deny every legitimate call (their legit input IS free text) or, worse,
# admit a closed-schema string the sink still executes. They stay in D_low with the
# X1 residual acknowledged. Declaring one positional is a configuration error.
_INSTRUCTION_COMPLETE_SINKS = frozenset({
    "bash", "shell", "execute", "run", "exec", "execute_generated_code",
    "spawn_child", "eval", "python", "sh", "command", "system",
})


def _is_imperative_sink(tool_name: str, normalized) -> bool:
    """Instruction-following sink: it would interpret its argument as a directive
    (spawn a sub-agent, send a message, execute generated code). The imperative
    channel (TM1) — distinct from the risky-op list, which misses free-text-as-
    directive (e.g. spawn_child(task=<free text>))."""
    return (
        tool_name.lower() in _IMPERATIVE_SINKS
        or getattr(normalized, "executes_generated_code", False)
        or getattr(normalized, "operation", "") == "execute_generated_code"
    )


def _denial_result(tool_name: str, reason: str) -> dict:
    """
    Coarse denial returned to executor as tool result.

    Workers receive only category + opaque decision_id.
    Full reason is in the trace (operator-only access).
    """
    return _make_denial_response(reason).to_tool_result()


def _safe_int(value: Any, default: int) -> int:
    """Parse an int from untrusted args without raising on bad input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _make_denial_response(reason: str, category: str | None = None) -> DenialResponse:
    """Create a DenialResponse for a given reason (used by degradation signal recording)."""
    coarse_category = category if category is not None else _classify_denial(reason)
    return DenialResponse(status="denied", coarse_category=coarse_category)


def _classify_denial(reason: str) -> str:
    reason_lower = reason.lower()
    if "spawn" in reason_lower or "child" in reason_lower:
        return "spawn_denied"
    if "export" in reason_lower:
        return "export_denied"
    if (
        "anomaly" in reason_lower
        or "score" in reason_lower
        or "governance" in reason_lower
        or "auto-denied" in reason_lower
        or "suspicious" in reason_lower
    ):
        return "governance_error"
    return "tool_denied"


class _TokenAccumulator:
    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.tool = 0
        self.context = 0
        self.cache_creation = 0
        self.cache_read = 0

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        tool_tokens: int,
        context_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.input += input_tokens
        self.output += output_tokens
        self.tool += tool_tokens
        self.context += context_tokens
        self.cache_creation += cache_creation_input_tokens
        self.cache_read += cache_read_input_tokens

    @property
    def total(self) -> int:
        return self.input + self.cache_creation + self.cache_read + self.output
