from __future__ import annotations

import logging
import traceback
from typing import AsyncIterator, Any, Callable, Awaitable

from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent, IntentKind, ResolvedIntent
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind, ChildMode
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind, TokenUsage
from axor_core.contracts.trace import (
    TraceEventKind,
    TraceEvent,
    IntentDeniedEvent,
    TokensSpentEvent,
    CancelledEvent,
)
from axor_core.capability.executor import CapabilityExecutor
from axor_core.errors.exceptions import (
    ToolNotAllowedError,
    ToolNotFoundError,
    ChildNotAllowedError,
    MaxDepthExceededError,
    ContextExpansionDeniedError,
)

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
    ) -> None:
        self._executor = capability_executor
        self._trace_events = trace_events
        self._depth = current_depth
        self._tool_result_callback = tool_result_callback
        self._spawn_callback = spawn_callback
        self._intent_sequence = 0
        self._token_totals = _TokenAccumulator()

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
                    tool_name   = event.payload.get("tool", "")
                    tool_use_id = event.payload.get("tool_use_id", "")

                    # spawn_child is a special intent — not a regular tool call
                    if tool_name == "spawn_child" and self._spawn_callback is not None:
                        task         = event.payload.get("args", {}).get("task", "")
                        context_hint = event.payload.get("args", {}).get("context_hint", "")
                        child_result = await self._spawn_callback(tool_use_id, task, context_hint)

                        if self._tool_result_callback is not None:
                            await self._tool_result_callback(tool_use_id, tool_name, child_result, True)
                        else:
                            yield ExecutorEvent(
                                kind=ExecutorEventKind.TEXT,
                                payload={"tool_result": child_result, "approved": True},
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
                                "approved":    resolved.approved,
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
            return ResolvedIntent(
                intent=intent,
                approved=False,
                reason=decision.reason,
                result=_denial_result(tool_name, decision.reason),
            )

        # approved or transformed — record approved event
        self._trace_events.append(TraceEvent(
            kind=TraceEventKind.INTENT_APPROVED,
            node_id=envelope.node_id,
            sequence=len(self._trace_events),
            payload={"tool": tool_name, "transformed": decision.kind == PolicyDecisionKind.TRANSFORM},
        ))

        # approved or transformed
        effective_args = decision.transformed_payload or tool_args
        effective_intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": effective_args},
            node_id=envelope.node_id,
            sequence=self._intent_sequence,
        )

        try:
            result = await self._executor.execute(effective_intent, envelope.capabilities)
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
                tool_name, envelope.node_id, exc, tb,
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
        1. Is the tool in allowed_tools?            → approve
        2. Is the tool in extra_denied?             → deny
        3. Not in capabilities at all               → deny
        """
        tool_name: str = intent.payload.get("tool", "")
        caps = envelope.capabilities

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
        return PolicyDecision(kind=PolicyDecisionKind.APPROVE, reason="context expansion approved")

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
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _denial_result(tool_name: str, reason: str) -> dict:
    """
    Structured denial returned to executor as tool result.
    Executor sees this as a tool response — not a governance event.
    """
    return {
        "error": "tool_denied",
        "tool": tool_name,
        "reason": reason,
    }


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
        cache_read_input_tokens: int     = 0,
    ) -> None:
        self.input   += input_tokens
        self.output  += output_tokens
        self.tool    += tool_tokens
        self.context += context_tokens
        self.cache_creation += cache_creation_input_tokens
        self.cache_read     += cache_read_input_tokens

    @property
    def total(self) -> int:
        return self.input + self.cache_creation + self.cache_read + self.output
