from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.cancel import CancelReason
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent, IntentKind, ResolvedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.policy.consequence import consequence_class
from axor_core.policy.value_policy import check_value_policies
from axor_core.policy.gates import (
    carrier_gate,
    consequence_gate,
    driving_subset,
    integrity_superseded_by_decidable,
    positional_gate,
    ssrf_gate,
    taint_gate,
)
from axor_core.kernel.registration import (
    validate_value_policies,
    validate_egress_allowlists,
    validate_driving_arg_allowlists,
    tool_is_classified,
)
from axor_core.taint.engine import TaintEngine
from axor_core.policy.sinks import INSTRUCTION_COMPLETE_SINKS
from axor_core.policy.provenance import output_root
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import (
    CancelledEvent,
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
    extract_path_arg,
    path_matches_allowlist,
)
from axor_core.contracts.degradation import DegradationLevel
from axor_core.degradation.engine import _LOCKED_ALLOWED_TOOLS
from axor_core.policy.normalizer import IntentNormalizer
from axor_core.node.canonicalizer import IntentCanonicalizer
from axor_core.node.escalation import EscalationManager, _PendingConsumption
from axor_core.kernel.adjudicator import (
    Adjudicator,
    MemoizingAdjudicator,
    projection_hash,
)
from axor_core.federation.value import FederatedValue
from axor_core.federation.gateway import FederationError
from axor_core.taint.causal_root import CausalRoot

if TYPE_CHECKING:
    from axor_core.contracts.admission import AdmissionController
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
        budget_cap_calls: int | None = None,
        budget_cap_cost: float | None = None,
        tool_weights: "dict[str, float] | None" = None,
        default_tool_weight: float = 1.0,
        value_policies: "dict | None" = None,
        consequence_overrides: "dict | None" = None,
        positional_sinks: "frozenset[str] | set[str] | None" = None,
        adjudicator: "Adjudicator | MemoizingAdjudicator | None" = None,
        federation_gateway=None,
        egress_sinks: "frozenset[str] | set[str] | None" = None,
        untrusted_sources: "frozenset[str] | set[str] | None" = None,
        sensitive_sources: "frozenset[str] | set[str] | None" = None,
        imperative_sinks: "frozenset[str] | set[str] | None" = None,
        benign_tools: "frozenset[str] | set[str] | None" = None,
        driving_args: "dict[str, list[str]] | None" = None,
        require_egress_allowlist: bool = False,
        require_tool_roles: bool = False,
        trajectory_observers: "list | None" = None,
        invocation_recorder: "Callable[[str, dict, bool], None] | None" = None,
        admission: "AdmissionController | None" = None,
    ) -> None:
        self._executor = capability_executor
        self._trace_events = trace_events
        self._depth = current_depth
        self._tool_result_callback = tool_result_callback
        self._spawn_callback = spawn_callback
        # Escalation grants + capability leases + flood guard live in their own
        # manager; the loop only asks it to resolve/grant/cover.
        self._escalation = EscalationManager(escalation_callback=escalation_callback)
        # Default to a real engine when none is supplied: the per-value/carrier/
        # taint/floor cascade is guarded on `self._taint_engine is not None`, so a
        # None engine would silently disable the entire data-flow core (fail-open).
        # Constructing a default makes the absence of an engine impossible, so that
        # cascade always runs (fail-closed).
        self._taint_engine = taint_engine if taint_engine is not None else TaintEngine()
        self._degradation_engine = degradation_engine
        self._reputation_enricher = reputation_enricher
        self._normalizer = IntentNormalizer()
        self._intent_sequence = 0
        self._token_totals = _TokenAccumulator()
        # DoS guards — opt-in (None = unlimited). GovernedSession sets prod defaults.
        self._max_intents_per_session = max_intents_per_session
        self._max_total_spawns = max_total_spawns
        # Budget: an operator-declared ceiling on APPROVED tool calls per run
        # (spec §15). Enforced HERE at the loop boundary — the same cap the replay
        # kernel checks (axor_core.kernel.replay.evaluate_call, category="budget"),
        # so a run and its counterfactual agree on when the budget is exhausted.
        # Exhaustion is a typed denial (recorded like any gate denial and fed to
        # degradation), never a silent overrun. None = unlimited.
        self._budget_cap_calls = budget_cap_calls
        self._budget_spent_calls = 0
        # Cost budget (spec §15): a cap on the summed per-tool weights of approved
        # calls, the operator-declared deterministic cost model. Same parity as the
        # call cap — the replay kernel (KernelConfig.budget_cap_cost / weight_of)
        # applies the identical would-exceed predicate and weight table.
        self._budget_cap_cost = budget_cap_cost
        self._tool_weights = dict(tool_weights or {})
        self._default_tool_weight = default_tool_weight
        self._budget_spent_cost = 0.0
        self._value_policies = value_policies or {}
        # Registration validator: reject value policies that try to discharge a
        # field requiring rich-syntax (fuzz) checking with a simple decidable
        # predicate — false assurance must fail closed, not be silently accepted.
        _vp_errors = validate_value_policies(self._value_policies)
        if _vp_errors:
            raise ValueError(
                "invalid value_policies: " + "; ".join(_vp_errors)
            )
        self._consequence_overrides = consequence_overrides or {}
        # Positional sinks. These are sinks the operator DECLARES to have an
        # instruction-incomplete input space — i.e. their legitimate input cannot
        # encode an instruction, and the trusted side constrains input to that space
        # (e.g. via constrained decoding). For these sinks, admission flips from the
        # leaky content-derivation DENY-LIST to a sound POSITIONAL ALLOW-LIST: admit
        # only if the driving value's carrier is instruction-incomplete, independent
        # of content. Opt-in/empty by default; sinks that execute their argument can
        # NEVER be declared here (a shell command's input space admits instructions
        # by definition).
        # Advisory adjudicator. Wrap a bare Adjudicator so verdicts are memoized by
        # projection hash; a MemoizingAdjudicator is used as-is. None = off.
        self._canonicalizer = IntentCanonicalizer()
        if adjudicator is None:
            self._adjudicator = None
        elif isinstance(adjudicator, MemoizingAdjudicator):
            self._adjudicator = adjudicator
        else:
            self._adjudicator = MemoizingAdjudicator(adjudicator)
        # Federation gateway (opt-in): decides the provenance of a peer value that
        # arrives wrapped as a FederatedValue. None = federation off.
        self._federation_gateway = federation_gateway
        self._positional_sinks = frozenset(positional_sinks or ())
        # Operator tool taxonomy (complements the normalizer's heuristics): which
        # tools exfiltrate (egress_sinks) and which produce untrusted/secret data
        # (untrusted_sources / sensitive_sources). Empty by default — the normalizer
        # still classifies generic tools; a deployment declares its renamed tools.
        self._egress_sinks = frozenset(egress_sinks or ())
        self._untrusted_sources = frozenset(untrusted_sources or ())
        self._sensitive_sources = frozenset(sensitive_sources or ())
        # Operator-declared instruction-following sinks (renamed spawn/send/exec).
        # Threaded into carrier_gate so a renamed imperative sink is honoured here,
        # matching the synchronous governor (previously this path ignored it).
        self._imperative_sinks = frozenset(imperative_sinks or ())
        # Explicitly-benign reads, kept for the lazy STRICT role check below.
        self._benign_tools = frozenset(benign_tools or ())
        self._require_tool_roles = require_tool_roles
        # Per-sink driving arguments — the fields the taint decision keys on
        # (whole-args by default). Narrows over-blocking of untrusted content sent
        # to a trusted destination.
        self._driving_args = {k: frozenset(v) for k, v in (driving_args or {}).items()}
        # Stateful, domain-supplied trajectory observers (tighten-only). Shared
        # instances so their state persists across the session's runs.
        self._trajectory_observers = list(trajectory_observers or [])
        # Optional sink-side audit hook: records every resolved tool call (tool, args,
        # executed) for the session's closed-session record. Observe-only; never
        # gates execution. Default None → zero overhead when sentinel is not attached.
        self._invocation_recorder = invocation_recorder
        # Optional control-plane admission (advisory overlay, spec 12.0). None =
        # no plane attached → governance never waits on the network. Polled at
        # the intent boundary only, so pause/stop take effect between intents,
        # never mid-effect.
        self._admission = admission
        # STRICT obligation: every egress sink must carry an enum allowlist (the
        # sound, paraphrase-proof destination control). Fail closed at construction.
        if require_egress_allowlist:
            _eg_errors = validate_egress_allowlists(self._egress_sinks, self._value_policies)
            _eg_errors += validate_driving_arg_allowlists(
                self._egress_sinks, self._driving_args, self._value_policies
            )
            if _eg_errors:
                raise ValueError("strict egress allowlist: " + "; ".join(_eg_errors))
        _illegal = {s for s in self._positional_sinks if s.lower() in INSTRUCTION_COMPLETE_SINKS}
        if _illegal:
            raise ValueError(
                "instruction-complete sinks cannot be declared positional: "
                f"{sorted(_illegal)} — their codomain admits instructions by "
                "definition; they must stay on the content-derivation path."
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

            # Control-plane admission (advisory overlay). Holds while paused,
            # returns False on stop; a disconnected/absent plane always admits.
            # The plane cannot widen — it can only stop or hold — so this is
            # never part of the allow decision.
            if self._admission is not None and not await self._admission.await_admission(
                envelope.node_id
            ):
                if not envelope.cancel_token.is_cancelled():
                    envelope.cancel_token.cancel(
                        CancelReason.USER_ABORT, "stopped via control plane"
                    )
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

                        # Per-value carrier/taint gate — the capability check
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

                    # Audit hook (observe-only): record the resolved tool call for the
                    # closed-session record sentinel consumes. Wrapped so a recorder
                    # bug can never disturb the governance path.
                    if self._invocation_recorder is not None:
                        try:
                            self._invocation_recorder(
                                event.payload.get("tool", ""),
                                event.payload.get("args", {}) or {},
                                bool(resolved.approved),
                            )
                        except Exception:
                            log.debug("invocation_recorder raised", exc_info=True)

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

        decision, pending_consumption = self._evaluate_tool_intent(intent, envelope)

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

        # Budget cap (spec §15) — enforced at the loop boundary, in replay-parity
        # order (capability passed above; budget before the consequence/value/taint
        # cascade). The N+1th APPROVED call is denied: spent counts only calls that
        # cleared every gate (incremented at the execute site below), so a
        # gate-denied call burns no budget, exactly as the replay kernel folds it.
        # A budget denial is a typed fact (recorded + fed to degradation), so a run
        # that hits its ceiling stops loudly — it never silently overruns.
        budget_reason: str | None = None
        if (
            self._budget_cap_calls is not None
            and self._budget_spent_calls >= self._budget_cap_calls
        ):
            budget_reason = (
                f"budget: call cap {self._budget_cap_calls} exhausted "
                f"({self._budget_spent_calls} spent)"
            )
        elif (
            self._budget_cap_cost is not None
            and self._budget_spent_cost + self._tool_weight(tool_name)
            > self._budget_cap_cost
        ):
            budget_reason = (
                f"budget: cost cap {self._budget_cap_cost} exhausted "
                f"({self._budget_spent_cost} spent, "
                f"+{self._tool_weight(tool_name)} for '{tool_name}')"
            )
        if budget_reason is not None:
            self._record_denial(intent, budget_reason, envelope)
            denial_resp = _make_denial_response(budget_reason, "budget")
            self._record_degradation_signal(intent, denial_resp)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=budget_reason,
                result=denial_resp.to_tool_result(),
            )

        # Normalize early: the consequence gate needs the operation enum to
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
            # Opt-in: a reputation threshold-CROSSING fact may tighten degradation
            # (tightening-only, a decidable fact — not the raw score). No-op unless
            # the degradation engine was given a detection_floor. Detection never
            # returns an allow decision; it can only tighten.
            if self._degradation_engine is not None:
                self._degradation_engine.record_detection(normalized)
                for ev in self._degradation_engine.drain_events():
                    self._trace_events.append(ev)

        # STRICT role completeness (lazy, per call): an unclassified tool fails
        # closed instead of defaulting to a clean benign read. Mirrors the governor;
        # closes the fail-open-on-unknown-tool default on the streaming path too.
        if self._require_tool_roles and tool_name not in _LOCKED_ALLOWED_TOOLS \
                and not tool_is_classified(
                    tool_name,
                    untrusted_sources=self._untrusted_sources,
                    sensitive_sources=self._sensitive_sources,
                    egress_sinks=self._egress_sinks,
                    positional_sinks=self._positional_sinks,
                    benign_tools=self._benign_tools,
                    value_policies=self._value_policies,
                ):
            role_denial = (
                f"tool {tool_name!r} has no declared data-flow role; STRICT mode "
                "refuses an unclassified tool (it would default to a clean read and "
                "arm no floor)"
            )
            self._record_denial(intent, role_denial, envelope)
            denial_resp = _make_denial_response(role_denial, "unclassified_tool")
            self._record_degradation_signal(intent, denial_resp)
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=role_denial,
                result=denial_resp.to_tool_result(),
            )

        # consequence axis — content-blind structural gate on the action class, part
        # of the pure allow decision. Catches a destructive/irreversible action
        # issued under trusted provenance that the provenance axes cannot see (e.g.
        # a host shutdown or disk wipe). Reads only the sink type + operation enum +
        # policy ceiling.
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

        # value-policy predicates — operator-registered range/enum predicates over
        # an admissible projection of an argument (e.g. transfer(amount) within
        # bounds). Content-blind: reads only the numeric/enum projection, discharged
        # by decidable decision procedures.
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
            # Per-value: hand the driving value's causal_root to the check
            # path so a value-keyed quarantine ("value:<src>") is matched here, not
            # only on the record_signal path. Without it derive_source_id falls back
            # to provenance/"unknown" and the narrowing silently misses.
            check_root = (
                self._taint_engine.derive_value(
                    driving_subset(tool_args, self._driving_args.get(tool_name))
                )
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

        # Shared gate denial → ResolvedIntent (records denial + degradation signal).
        def _gate_denial(gd) -> ResolvedIntent:
            self._record_denial(intent, gd.reason, envelope)
            denial_resp = _make_denial_response(gd.reason, gd.category)
            self._record_degradation_signal(intent, denial_resp, normalized)
            return ResolvedIntent(
                intent=intent, approved=False, reason=gd.reason,
                result=denial_resp.to_tool_result(),
            )

        # SSRF / internal-destination gate — content-blind, always-on, and
        # independent of taint (this is a *destination* concern, not a data-flow
        # one): no agent should reach cloud metadata or the docker socket unless
        # policy explicitly allows it. Decoupling this from taint is what lets the
        # taint gate below be cleanly per-value without regressing SSRF safety.
        if normalized is not None:
            gd = ssrf_gate(tool_name, normalized)
            if gd is not None:
                return _gate_denial(gd)

        # taint enforcement — PER-VALUE. The gate decides on the *driving
        # argument's own* causal_root (content-derivation ledger), NOT a
        # session-wide flag. A clean-valued sink passes even when other values in
        # the session are tainted.
        # Known gap: content-derivation misses paraphrased / re-encoded influence;
        # soundly over-tainting opaque model output would collapse this back to
        # session-sticky tainting and needs a sound per-value interpreter backend.
        if normalized is not None:
            driving_root = self._taint_engine.derive_value(
                driving_subset(tool_args, self._driving_args.get(tool_name))
            )

            # Density telemetry: record, per high-stakes sink firing, the per-value
            # taint (both axes) and the session-sticky shadow, split integrity vs
            # confidentiality so the taint-explosion asymmetry is visible. Uses the
            # same overrides as the enforcement gate so density and enforcement
            # agree on which sinks count.
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

            # POSITIONAL ADMISSION. For a sink the operator DECLARED
            # instruction-incomplete, admission flips from the (leaky)
            # content-derivation deny-list to a sound positional allow-list: admit
            # ONLY if the driving value's carrier is instruction-incomplete, else
            # fail-closed. It does NOT consult driving_root.is_tainted — a paraphrase
            # that launders the content-derivation label cannot change the value's
            # FORM, and classify_carrier is structural, so admission holds against
            # semantic derivation, content-independently.
            gd = positional_gate(tool_name, tool_args, self._positional_sinks)
            if gd is not None:
                return _gate_denial(gd)

            # Carrier / imperative-channel gate: a tainted FREE_TEXT value reaching
            # an instruction-following sink (spawn a sub-agent, send a message,
            # execute) is the imperative channel. Complements per-value: catches
            # free-text-as-directive the risky-op list below does not.
            gd = carrier_gate(
                tool_name, tool_args, normalized, driving_root, self._imperative_sinks
            )
            if gd is not None:
                return _gate_denial(gd)

            # Per-value taint: integrity (untrusted-derived value into a high-risk
            # operation) + the confidentiality SOUND FLOOR (egress denied while a
            # secret read is outstanding — on the FACT of the read, content-blind, so
            # a paraphrased/re-encoded secret cannot escape; lifted only by
            # governance endorsement). `self._egress_sinks` is the operator's
            # declaration of which tools exfiltrate (complements the normalizer's
            # destination classification).
            # Contract-mandated (ValueProvenance.confidentiality_floor_active): call
            # it directly. The kernel gates confidentiality on the sound floor, never
            # on the leaky per-value `sensitive` derivation; a backend that omits it
            # fails loudly rather than silently downgrading the guarantee.
            floor_active = self._taint_engine.confidentiality_floor_active()
            # A sink whose driving args are fully guarded by satisfied decidable
            # predicates carries its integrity axis there (a stronger, content-blind
            # control), so the content-taint integrity check is superseded; the
            # confidentiality floor still applies.
            gd = taint_gate(
                tool_name, normalized, driving_root, floor_active, self._egress_sinks,
                integrity_superseded=integrity_superseded_by_decidable(
                    tool_name, tool_args, self._driving_args, self._value_policies
                ),
            )
            if gd is not None:
                return _gate_denial(gd)

        # approved or transformed — but first consult the advisory adjudicator on
        # the PROJECTION only. We are on the would-approve path: every kernel hard
        # gate has already passed, so the adjudicator can only TIGHTEN (deny), never
        # override a hard deny. Memoized by projection hash: equal projection →
        # equal verdict.
        if self._adjudicator is not None and normalized is not None:
            projection = self._canonicalizer.canonicalize(normalized, tool_args)
            if not self._adjudicator.apply(projection, kernel_allowed=True):
                reason = (
                    f"adjudicator (advisory): denied '{tool_name}' on its "
                    f"projection (hash {projection_hash(projection)})"
                )
                self._record_denial(intent, reason, envelope)
                denial_resp = _make_denial_response(reason, "adjudicator")
                self._record_degradation_signal(intent, denial_resp, normalized)
                return ResolvedIntent(
                    intent=intent, approved=False, reason=reason,
                    result=denial_resp.to_tool_result(),
                )

        # Every gate (capability, consequence, value-policy, degradation, ssrf,
        # positional, carrier, taint, adjudicator) has passed: the call is approved
        # and about to execute. NOW consume the lease use / grant op — a call denied
        # by any gate above returned earlier and burned nothing.
        if pending_consumption is not None:
            pending_consumption.commit()

        # Consume one unit of budget for this approved call, mirroring the replay
        # kernel (which counts every non-DENY TOOL_CALL). Counted here — after all
        # gates pass, before execute — so an execution error still counts (it was
        # not a gate denial), keeping runtime and counterfactual budgets identical.
        self._budget_spent_calls += 1
        self._budget_spent_cost += self._tool_weight(tool_name)

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
            # Federation ingress: a tool that delegated to a peer agent returns the
            # result wrapped with the peer's provenance receipt. Decide its trust
            # via the gateway BEFORE it is used or registered. A forged / tampered /
            # unknown-peer receipt is an attack — reject the value outright; an
            # authentic receipt either restores the peer's provenance (trusted) or
            # degrades to untrusted. Then unwrap so the agent sees the plain value.
            fed_root = None
            if self._federation_gateway is not None and isinstance(result, FederatedValue):
                try:
                    fed_root, _level = self._federation_gateway.receive(
                        result.value, result.receipt, result.peer_id
                    )
                except FederationError as exc:
                    reason = f"federation: rejected peer value — {exc}"
                    self._record_denial(intent, reason, envelope)
                    denial_resp = _make_denial_response(reason, "federation_gate")
                    return ResolvedIntent(
                        intent=intent, approved=False, reason=reason,
                        result=denial_resp.to_tool_result(),
                    )
                result = result.value
            # Register the tool output into the PER-VALUE ledger so a later
            # sink whose argument carries this content is gated at the value level.
            # No session-taint propagation — a read taints its produced *value*,
            # not the whole session.
            self._register_value_taint(
                effective_intent, result, normalized, override_root=fed_root
            )
            self._record_degradation_signal(intent, None)
            # Stateful trajectory observers see the executed (tool, args, result).
            # They may only TIGHTEN degradation (observe-only), never authorise — a
            # domain heuristic, not a structural fact.
            self._run_trajectory_observers(tool_name, effective_args, result)
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

    def _tool_weight(self, tool_name: str) -> float:
        """Operator-declared cost weight of one call to `tool_name` (spec §15).
        Mirrors KernelConfig.weight_of so runtime and replay cost agree."""
        return float(self._tool_weights.get(tool_name, self._default_tool_weight))

    def _evaluate_tool_intent(
        self,
        intent: Intent,
        envelope: ExecutionEnvelope,
    ) -> tuple[PolicyDecision, "_PendingConsumption | None"]:
        """
        Evaluate a tool_call intent against capabilities.

        Returns ``(decision, pending)``. ``pending`` is a not-yet-applied lease/grant
        consumption: the caller commits it only when the whole call is finally
        approved, so a call denied by a later data-flow gate burns no lease use /
        grant op. (Decisions that do not arise from a lease/grant carry ``None``.)

        Resolution order:
        1. Active escalation grant/lease covers the tool?  → approve (consume on commit)
        2. Is the tool in allowed_tools?                   → approve
        3. Is the tool in extra_denied?                    → deny
        4. Not in capabilities at all                      → deny
        """
        tool_name: str = intent.payload.get("tool", "")
        caps = envelope.capabilities
        tool_args = intent.payload.get("args", {})

        # Filesystem ceiling — applies to every path-bearing tool call regardless
        # of how it is later approved (allowed_tools, lease, or escalation grant).
        policy_paths = envelope.authority.allowed_paths or ()
        if policy_paths:
            candidate_path = extract_path_arg(tool_args)
            if candidate_path and not path_matches_allowlist(candidate_path, policy_paths):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=(
                        f"path {candidate_path!r} is outside policy "
                        f"allowed_paths {tuple(policy_paths)!r}"
                    ),
                ), None

        # Lease/grant resolution (authoritative — validates TTL + max_uses + path).
        escalation_decision, pending = self._escalation.evaluate(tool_name, tool_args)
        if escalation_decision is not None:
            return escalation_decision, pending

        if tool_name in caps.allowed_tools:
            return PolicyDecision(
                kind=PolicyDecisionKind.APPROVE,
                reason="tool in allowed_tools",
            ), pending

        if tool_name in envelope.authority.tool_policy.extra_denied:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"tool '{tool_name}' is explicitly denied by policy",
            ), pending

        return PolicyDecision(
            kind=PolicyDecisionKind.DENY,
            reason=f"tool '{tool_name}' is not in capabilities for policy '{envelope.authority.name}'",
        ), pending

    async def _handle_escalation(
        self,
        event: ExecutorEvent,
        envelope: ExecutionEnvelope,
    ) -> dict:
        return await self._escalation.grant_from_intent(
            event, envelope, self._trace_events
        )

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
        authority = envelope.authority

        if not caps.allow_children:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=(
                    f"child nodes not allowed by policy '{authority.name}' "
                    f"(allow_spawn={authority.child_authority.allow_spawn})"
                ),
            )

        if self._depth >= authority.child_authority.max_depth:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=(
                    f"max child depth reached: current={self._depth}, "
                    f"max={authority.child_authority.max_depth}"
                ),
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
        """Carrier gate for spawn_child, routed through the SHARED gate predicate so
        the spawn branch and the regular tool path cannot drift. spawn_child is an
        instruction-following sink (in IMPERATIVE_SINKS), so a tainted FREE_TEXT task
        is the imperative channel. The driving root is derived over the WHOLE spawn
        args (spawn is deliberately not narrowed by driving_args). Returns a denial
        reason, or None to allow — structured/sensitive values that are not the
        imperative channel stay gated at the child's own sinks (the child inherits
        this engine's per-value ledger)."""
        if self._taint_engine is None:
            return None
        driving_root = self._taint_engine.derive_value(spawn_args)
        gd = carrier_gate(
            "spawn_child", spawn_args, None, driving_root, self._imperative_sinks
        )
        return gd.reason if gd is not None else None

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
        override_root: "CausalRoot | None" = None,
    ) -> None:
        """Register a tool's output content in the per-value ledger with the
        causal_root its read introduces. Mirrors the session-taint triggers:
        external/web reads → untrusted; secret/system reads → untrusted + sensitive.

        `override_root` short-circuits the structural derivation — used when the
        federation gateway has already decided the value's provenance (a peer value
        whose receipt was restored or re-minted untrusted).
        """
        if self._taint_engine is None:
            return
        if override_root is not None:
            self._taint_engine.register_value(result, override_root)
            return
        tool_name = effective_intent.payload.get("tool", "")
        # Shared arming map (policy.provenance.output_root) — the same mapping the
        # governor registers through, so the streaming and synchronous paths cannot
        # drift on what a tool's output taints. A declared source role wins over the
        # normalizer heuristic; a clean read returns None and arms nothing.
        ni = normalized or self._normalizer.normalize(effective_intent)
        root = output_root(
            tool_name, ni,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
        )
        if root is None:
            return  # clean read — nothing to register
        self._taint_engine.register_value(result, root)

    def _check_consequence(
        self,
        tool_name: str,
        envelope: ExecutionEnvelope,
        operation: str | None = None,
    ) -> str | None:
        """Consequence axis gate. Return a denial reason if the sink's
        action class exceeds the policy's unattended ceiling and no governance
        gate is present, else None. Content-blind: reads only the sink type and
        the operation enum (never argument content).

        The governance gate is satisfied by an active escalation grant or
        capability lease for the tool (a human/operator-authorised path).
        """
        ceiling = envelope.authority.max_unattended_consequence
        # over-ceiling is admissible only through a governance gate (an active
        # escalation grant or capability lease for the tool).
        has_gate = self._escalation.covers(tool_name)
        gd = consequence_gate(
            tool_name, operation, ceiling, self._consequence_overrides,
            has_governance_gate=has_gate,
        )
        return gd.reason if gd is not None else None

    # ── Detection layer — out-of-band from the allow decision ───────────────────

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

    def _run_trajectory_observers(self, tool: str, args: dict, result: "Any") -> None:
        """Feed each stateful observer the executed (tool, args, result). A returned
        signal tightens degradation (observe-only, never authorises). Observer
        exceptions are swallowed — a buggy domain predicate must not break execution."""
        if not self._trajectory_observers or self._degradation_engine is None:
            return
        for observer in self._trajectory_observers:
            try:
                signal = observer.observe(tool, args, result)
            except Exception:
                log.debug("trajectory observer raised", exc_info=True)
                continue
            if signal is None:
                continue
            self._degradation_engine.tighten(
                signal.target_level,
                reason=signal.reason,
                trigger_intent=tool,
            )
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


def _denial_result(tool_name: str, reason: str) -> dict:
    """
    Coarse denial returned to executor as tool result.

    Workers receive only category + opaque decision_id.
    Full reason is in the trace (operator-only access).
    """
    return _make_denial_response(reason).to_tool_result()


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
