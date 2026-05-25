from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from axor_core.capability.executor import CapabilityExecutor
from axor_core.capability.flood import EscalationFloodGuard
from axor_core.contracts.anomaly import (
    AnomalyClass,
    AnomalyDetector,
    AnomalyResult,
    NormalizedIntent,
)
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent, IntentKind, ResolvedIntent
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import (
    CancelledEvent,
    EscalationDeniedEvent,
    EscalationGrantedEvent,
    IntentDeniedEvent,
    SuspiciousIntentEvent,
    TokensSpentEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.contracts.denial import DenialResponse
from axor_core.errors.exceptions import (
    ToolNotAllowedError,
    ToolNotFoundError,
)
from axor_core.capability.lease_validator import LeaseValidator, path_matches_allowlist
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.taint import TaintScope, TaintSource, TaintState
from axor_core.degradation.engine import _LOCKED_ALLOWED_TOOLS
from axor_core.node.normalizer import IntentNormalizer

if TYPE_CHECKING:
    from axor_core.contracts.lease import CapabilityLease
    from axor_core.degradation.engine import DegradationEngine
    from axor_core.taint.engine import TaintEngine

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
        anomaly_detector: AnomalyDetector | None = None,
        anomaly_window_size: int = 10,
        taint_engine: "TaintEngine | None" = None,
        degradation_engine: "DegradationEngine | None" = None,
    ) -> None:
        self._executor = capability_executor
        self._trace_events = trace_events
        self._depth = current_depth
        self._tool_result_callback = tool_result_callback
        self._spawn_callback = spawn_callback
        self._escalation_callback = escalation_callback
        self._anomaly_detector = anomaly_detector
        self._anomaly_window_size = anomaly_window_size
        self._taint_engine = taint_engine
        self._degradation_engine = degradation_engine
        self._normalizer = IntentNormalizer()
        self._intent_window: list[NormalizedIntent] = []
        self._intent_sequence = 0
        self._token_totals = _TokenAccumulator()
        self._granted_escalations: dict[str, _GrantedEscalation] = {}
        self._capability_leases: dict[str, "CapabilityLease"] = {}
        self._lease_validator = LeaseValidator()
        self._escalation_count = 0
        self._flood_guard = EscalationFloodGuard()

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
                        result = await self._handle_escalation(event, envelope)
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

        # normalize intent once — shared by anomaly detection, taint, and degradation
        needs_normalized = (
            self._anomaly_detector is not None
            or self._taint_engine is not None
            or self._degradation_engine is not None
        )
        normalized: NormalizedIntent | None = None
        if needs_normalized:
            normalized = self._normalizer.normalize(intent)

        # degradation pre-check — enforce narrowed policy from previous signals
        if self._degradation_engine is not None and normalized is not None:
            taint_state = self._taint_engine.state if self._taint_engine is not None else None
            degradation_denial = self._check_degradation_denial(
                tool_name, normalized, taint_state, envelope
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

        # anomaly detection — runs after policy approval
        if self._anomaly_detector is not None and normalized is not None:
            self._intent_window.append(normalized)
            if len(self._intent_window) > self._anomaly_window_size:
                self._intent_window.pop(0)
            try:
                anomaly = await self._anomaly_detector.score(
                    window=list(self._intent_window),
                    policy_name=envelope.policy.name,
                )
            except Exception as exc:
                reason = f"anomaly detector raised unexpectedly: {type(exc).__name__}: {exc}"
                log.error("anomaly detector error — failing closed: %s", exc, exc_info=True)
                self._record_denial(intent, reason, envelope)
                return ResolvedIntent(
                    intent=intent,
                    approved=False,
                    reason=reason,
                    result=_denial_result(tool_name, reason),
                )

            if anomaly is not None and anomaly.cls == AnomalyClass.CRITICAL:
                reason = f"anomaly detector: CRITICAL score={anomaly.score:.2f} reasons={anomaly.reasons}"
                self._record_anomaly_event(
                    tool_name=tool_name,
                    normalized=normalized,
                    anomaly=anomaly,
                    policy_action="denied",
                    envelope=envelope,
                )
                self._record_denial(intent, reason, envelope)
                return ResolvedIntent(
                    intent=intent,
                    approved=False,
                    reason=reason,
                    result=_denial_result(tool_name, reason),
                )

            if anomaly is not None and anomaly.cls == AnomalyClass.SUSPICIOUS:
                self._record_anomaly_event(
                    tool_name=tool_name,
                    normalized=normalized,
                    anomaly=anomaly,
                    policy_action="flagged",
                    envelope=envelope,
                )

        # taint enforcement — deny high-risk operations on a tainted session
        if (
            self._taint_engine is not None
            and normalized is not None
            and self._taint_engine.state.is_tainted
        ):
            risky = (
                normalized.writes_outside_workdir
                or normalized.executes_generated_code
            )
            if risky:
                reason = (
                    "taint enforcement: session is tainted by external input "
                    f"(sources={[s.value for s in self._taint_engine.state.sources]}) "
                    f"and tool '{tool_name}' performs a high-risk operation "
                    "(writes_outside_workdir or executes_generated_code)"
                )
                self._record_denial(intent, reason, envelope)
                return ResolvedIntent(
                    intent=intent,
                    approved=False,
                    reason=reason,
                    result=_denial_result(tool_name, reason),
                )

        # approved or transformed — record approved event
        self._trace_events.append(
            TraceEvent(
                kind=TraceEventKind.INTENT_APPROVED,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                payload={
                    "tool": tool_name,
                    "transformed": decision.kind
                    == PolicyDecisionKind.TRANSFORM,
                },
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
            # Propagate taint after any successful privileged read.
            if self._taint_engine is not None:
                normalized_check = normalized or self._normalizer.normalize(effective_intent)
                if normalized_check.target_kind in (
                    "external_url", "cloud_metadata", "docker_socket"
                ) or normalized_check.operation == "network_request":
                    self._taint_engine.propagate(TaintSource.WEB, TaintScope.SESSION)
                elif normalized_check.target_kind == "outside_workdir":
                    # File reads outside the trusted workspace may introduce
                    # adversarial content — taint with FILE source.
                    self._taint_engine.propagate(TaintSource.FILE, TaintScope.SESSION)
                self._taint_engine.tick_intent()
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
            tool_path = intent.payload.get("args", {}).get("path", "")
            if not self._lease_validator.check_path_allowed(lease, tool_path):
                return PolicyDecision(
                    kind=PolicyDecisionKind.DENY,
                    reason=f"lease for '{tool_name}' restricts to paths {lease.allowed_paths!r}",
                )
            lease.increment_use()

        grant = self._granted_escalations.get(tool_name)
        if grant is not None:
            if grant.paths and not lease:
                tool_path = intent.payload.get("args", {}).get("path", "")
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
            int(args.get("max_ops", 10)),
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

        self._granted_escalations[tool] = _GrantedEscalation(
            tool=tool,
            paths=paths,
            ops_remaining=max_ops,
        )
        # Also create a CapabilityLease for auditable enforcement.
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
        if lease_err is None:
            self._capability_leases[tool] = lease
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

        return PolicyDecision(
            kind=PolicyDecisionKind.APPROVE,
            reason="child node approved",
        )

    def resolve_context_expansion(
        self,
        intent: Intent,
        envelope: ExecutionEnvelope,
    ) -> PolicyDecision:
        """Evaluate an expand_context intent."""
        if not envelope.capabilities.allow_context_expansion:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                reason=f"context expansion not allowed by policy '{envelope.policy.name}' "
                f"(context_mode={envelope.policy.context_mode})",
            )
        return PolicyDecision(
            kind=PolicyDecisionKind.APPROVE, reason="context expansion approved"
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

    def _record_anomaly_event(
        self,
        tool_name: str,
        normalized: NormalizedIntent,
        anomaly: AnomalyResult,
        policy_action: str,
        envelope: ExecutionEnvelope,
    ) -> None:
        self._trace_events.append(
            SuspiciousIntentEvent(
                kind=TraceEventKind.ANOMALY_FLAGGED,
                node_id=envelope.node_id,
                sequence=len(self._trace_events),
                tool=tool_name,
                score=anomaly.score,
                anomaly_class=anomaly.cls,
                reasons=anomaly.reasons,
                policy_action=policy_action,
                provenance=normalized.provenance,
            )
        )

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
        taint_state = self._taint_engine.state if self._taint_engine is not None else TaintState()
        ni = normalized if normalized is not None else self._normalizer.normalize(intent)
        self._degradation_engine.record_signal(ni, denial, taint_state)
        for event in self._degradation_engine.drain_events():
            self._trace_events.append(event)

    def _check_degradation_denial(
        self,
        tool_name: str,
        normalized: NormalizedIntent,
        taint_state: "TaintState | None",
        envelope: ExecutionEnvelope,
    ) -> str | None:
        """
        Return a denial reason string if degradation state forbids this tool call, else None.
        Called before the main cascade — enforces apply_to_policy narrowing.
        """
        if self._degradation_engine is None:
            return None
        ts = taint_state or TaintState()
        source_id = self._degradation_engine.derive_source_id(normalized, ts)
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


def _make_denial_response(reason: str) -> DenialResponse:
    """Create a DenialResponse for a given reason (used by degradation signal recording)."""
    category = _classify_denial(reason)
    return DenialResponse(status="denied", coarse_category=category)


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
