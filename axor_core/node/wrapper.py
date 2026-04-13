from __future__ import annotations

import dataclasses

from axor_core.contracts.cancel import CancelToken, CancelReason, make_token
from axor_core.contracts.context import ContextView, LineageSummary, RawExecutionState
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.contracts.result import ExecutionResult, ExecutorEventKind, TokenUsage
from axor_core.contracts.trace import TraceConfig, TraceEventKind
from axor_core.contracts.extension import ExtensionBundle
from axor_core.capability.executor import CapabilityExecutor
from axor_core.context.manager import ContextManager
from axor_core.node.envelope import EnvelopeBuilder
from axor_core.node.intent_loop import IntentLoop
from axor_core.node.export import ExportFilter
from axor_core.node.spawn import ChildSpawner
from axor_core.policy.analyzer import TaskAnalyzer
from axor_core.policy.selector import PolicySelector
from axor_core.policy.composer import PolicyComposer
from axor_core.budget.policy_engine import BudgetPolicyEngine
from axor_core.trace.collector import TraceCollector
from axor_core import trace as trace_mod


class GovernedNode:
    """
    The central primitive of Axor.

    A GovernedNode wraps any raw executor and enforces governance.
    The executor never sees raw context, never self-assigns capabilities,
    and never performs privileged actions directly.

    Every execution — flat or federated — is a GovernedNode.
    Federation is just depth > 0. There is no special multi-agent mode.

    Fully wired execution flow:
        raw_state
          → TaskAnalyzer        → TaskSignal
          → PolicySelector      → ExecutionPolicy
          → PolicyComposer      → composed policy (extensions + parent)
          → ContextManager      → ContextView  (shaped, compressed, cached)
          → EnvelopeBuilder     → ExecutionEnvelope
          → BudgetPolicyEngine  → optimization check before execution
          → IntentLoop          → stream interception + intent resolution
              each tool_use     → BudgetPolicyEngine.on_intent_arrived()
              each tool_result  → BudgetPolicyEngine.on_result_arrived()
              spawn_child       → ChildSpawner → child GovernedNode
          → ExportFilter        → governed ExecutionResult
          → TraceCollector      → persist decisions to lineage
          → ContextManager.update() → update context from result
    """

    def __init__(
        self,
        executor: Invokable,
        capability_executor: CapabilityExecutor,
        analyzer: TaskAnalyzer,
        selector: PolicySelector,
        composer: PolicyComposer,
        context_manager: ContextManager | None = None,
        budget_engine: BudgetPolicyEngine | None = None,
        trace_collector: TraceCollector | None = None,
        trace_config: TraceConfig | None = None,
        current_depth: int = 0,
        child_executor: Invokable | None = None,
    ) -> None:
        self._executor         = executor
        self._child_executor   = child_executor  # None → reuse parent executor
        self._cap_executor     = capability_executor
        self._analyzer         = analyzer
        self._selector         = selector
        self._composer         = composer
        self._context_manager  = context_manager      # None → stub ContextView
        self._budget_engine    = budget_engine         # None → no budget tracking
        self._trace_collector  = trace_collector       # None → events not persisted
        self._trace_config     = trace_config or TraceConfig()
        self._depth            = current_depth

        self._envelope_builder = EnvelopeBuilder()
        self._export_filter    = ExportFilter()
        self._child_spawner    = ChildSpawner()

    async def run(
        self,
        raw_state: RawExecutionState,
        extension_bundle: ExtensionBundle | None = None,
        parent_policy: ExecutionPolicy | None = None,
        override_policy: ExecutionPolicy | None = None,
        cancel_token: CancelToken | None = None,
    ) -> ExecutionResult:
        trace_events: list = []
        cancel_token = cancel_token or make_token()

        # ── 1. Policy ──────────────────────────────────────────────────────────
        if override_policy is not None:
            policy = override_policy
        else:
            signal, signal_event = await self._analyzer.analyze(raw_state.task)
            trace_events.append(_stamp(signal_event, node_id="pending", sequence=0))
            policy = self._selector.select(signal)

        extension_fragments = list(extension_bundle.fragments) if extension_bundle else []
        policy = self._composer.compose(
            base=policy,
            extensions=extension_fragments,
            parent_policy=parent_policy,
        )

        # ── 2. Lineage ─────────────────────────────────────────────────────────
        lineage = self._build_lineage(raw_state)

        # register with trace collector
        if self._trace_collector:
            self._trace_collector.register_node(
                node_id=lineage.node_id,
                parent_id=lineage.parent_id,
                depth=lineage.depth,
                policy_name=policy.name,
            )
            # stamp pending events now that we have node_id
            trace_events = [
                _stamp(e, node_id=lineage.node_id, sequence=i)
                for i, e in enumerate(trace_events)
            ]

        # policy chosen event
        trace_events.append(trace_mod.events.policy_chosen(lineage.node_id, policy))

        # ── 3. Context ─────────────────────────────────────────────────────────
        if self._context_manager is not None:
            context = self._context_manager.build(raw_state, lineage, policy=policy)
        else:
            context = self._stub_context_view(raw_state, policy, lineage)

        # ── 4. Envelope ────────────────────────────────────────────────────────
        extension_tools = list(extension_bundle.tools) if extension_bundle else []
        envelope = self._envelope_builder.build(
            task=raw_state.task,
            context=context,
            policy=policy,
            lineage=lineage,
            extension_tools=extension_tools,
            node_id=lineage.node_id,
            parent_metadata={"session_id": raw_state.session_id},
            cancel_token=cancel_token,
        )

        # ── 5. Budget pre-check ────────────────────────────────────────────────
        if self._budget_engine:
            decision = self._budget_engine.on_intent_arrived(
                envelope=envelope,
                tool_count=len(envelope.capabilities.allowed_tools),
            )
            if cancel_token.is_cancelled():
                # budget engine fired hard stop
                return self._partial_result(envelope, "", {}, trace_events)

        # ── 6. Intent loop ─────────────────────────────────────────────────────
        # if executor supports ToolResultBus (e.g. ClaudeCodeExecutor),
        # register a callback so intent_loop pushes results into the bus
        # instead of yielding them as TEXT events
        tool_result_callback = None
        if hasattr(self._executor, "get_bus"):
            bus = self._executor.get_bus()
            if bus is not None:
                async def _push_to_bus(
                    tool_use_id: str, tool_name: str, result: object, approved: bool
                ) -> None:
                    bus.push(tool_use_id, result)
                tool_result_callback = _push_to_bus

        # spawn_callback — routes spawn_child intents to _handle_spawn_child
        # captures closure variables needed for child construction
        _ext_bundle   = extension_bundle
        _parent_pol   = policy
        _trace_events = trace_events

        async def _spawn_child_callback(
            tool_use_id: str, task: str, context_hint: str
        ) -> str:
            from axor_core.errors.exceptions import ChildNotAllowedError, MaxDepthExceededError
            child_intent = Intent(
                kind=IntentKind.SPAWN_CHILD,
                payload={"task": task, "context_hint": context_hint,
                         "tool_use_id": tool_use_id},
                node_id=envelope.node_id,
                sequence=0,
            )
            try:
                return await self._handle_spawn_child(
                    intent=child_intent,
                    envelope=envelope,
                    extension_bundle=_ext_bundle,
                    parent_policy=_parent_pol,
                    trace_events=_trace_events,
                )
            except (ChildNotAllowedError, MaxDepthExceededError) as e:
                return f"[child spawn denied: {e}]"
            except Exception as e:
                return f"[child spawn failed: {e}]"

        intent_loop = IntentLoop(
            capability_executor=self._cap_executor,
            trace_events=trace_events,
            current_depth=self._depth,
            tool_result_callback=tool_result_callback,
            spawn_callback=_spawn_child_callback,
        )

        raw_output, raw_payload = await self._collect_stream(
            intent_loop=intent_loop,
            envelope=envelope,
            extension_bundle=extension_bundle,
            parent_policy=policy,
            cancel_token=cancel_token,
        )

        # ── 7. Export filter ───────────────────────────────────────────────────
        token_usage = self._extract_token_usage(trace_events, context)
        result = self._export_filter.apply(
            raw_output=raw_output,
            raw_payload=raw_payload,
            envelope=envelope,
            token_usage=token_usage,
        )

        # ── 8. Flush trace ─────────────────────────────────────────────────────
        if self._trace_collector:
            self._trace_collector.record_many(trace_events)

        # ── 9. Update context ──────────────────────────────────────────────────
        if self._context_manager:
            self._context_manager.update(raw_output, lineage.node_id)

        return result

    # ── Stream collection ──────────────────────────────────────────────────────

    async def _collect_stream(
        self,
        intent_loop: IntentLoop,
        envelope: ExecutionEnvelope,
        extension_bundle: ExtensionBundle | None,
        parent_policy: ExecutionPolicy,
        cancel_token: CancelToken,
    ) -> tuple[str, dict]:
        output_parts: list[str] = []
        payload: dict = {}

        raw_stream = self._executor.stream(envelope)

        async for event in intent_loop.run(raw_stream, envelope):

            match event.kind:
                case ExecutorEventKind.TEXT:
                    # budget check on result arrival
                    text = event.payload.get("text", "")
                    tool_result = event.payload.get("tool_result")

                    if tool_result and self._budget_engine:
                        estimate = len(str(tool_result)) // 4
                        self._budget_engine.on_result_arrived(
                            node_id=envelope.node_id,
                            result_token_estimate=estimate,
                            policy=envelope.policy,
                        )

                    # record file reads into context manager
                    if tool_result and self._context_manager:
                        args = event.payload.get("args", {})
                        path = args.get("path", "")
                        if path and isinstance(tool_result, str):
                            self._context_manager.record_file_read(path, tool_result)

                    content = text or str(tool_result or "")
                    output_parts.append(content)

                case ExecutorEventKind.STOP:
                    payload = event.payload

        return "".join(output_parts), payload

    # ── Spawn child ────────────────────────────────────────────────────────────

    async def _handle_spawn_child(
        self,
        intent: Intent,
        envelope: ExecutionEnvelope,
        extension_bundle: ExtensionBundle | None,
        parent_policy: ExecutionPolicy,
        trace_events: list,
    ) -> str:
        """
        Handle spawn_child intent — create a child GovernedNode and run it.
        Budget check happens before spawning.
        """
        child_task = intent.payload.get("task", envelope.task)

        # budget check before spawn
        if self._budget_engine:
            decision = self._budget_engine.on_child_requested(
                parent_envelope=envelope,
                child_task=child_task,
            )
            if envelope.cancel_token.is_cancelled():
                return "[child spawn denied: budget exhausted]"

        child_task_str, child_policy, child_lineage = self._child_spawner.prepare_child(
            spawn_intent=intent,
            parent_envelope=envelope,
            intent_loop=IntentLoop(self._cap_executor, trace_events, self._depth),
            trace_events=trace_events,
        )

        child_raw_state = RawExecutionState(
            task=child_task_str,
            session_id=envelope.parent_metadata.get("session_id", ""),
            parent_export=envelope.context.working_summary,
            session_state={},
            memory_fragments=[],
            lineage=child_lineage,
        )

        child_node = GovernedNode(
            executor=self._child_executor or self._executor,
            capability_executor=self._cap_executor,
            analyzer=self._analyzer,
            selector=self._selector,
            composer=self._composer,
            context_manager=self._context_manager,
            budget_engine=self._budget_engine,
            trace_collector=self._trace_collector,
            trace_config=self._trace_config,
            current_depth=self._depth + 1,
            child_executor=self._child_executor,  # grandchildren use same child executor
        )

        child_cancel = envelope.cancel_token.child_token()
        child_result = await child_node.run(
            raw_state=child_raw_state,
            extension_bundle=extension_bundle,
            parent_policy=parent_policy,
            cancel_token=child_cancel,
        )

        # emit child_completed into parent trace
        from axor_core.contracts.trace import TraceEvent, TraceEventKind
        trace_events.append(TraceEvent(
            kind=TraceEventKind.CHILD_COMPLETED,
            node_id=envelope.node_id,
            sequence=len(trace_events),
            payload={
                "child_node_id": child_raw_state.lineage.node_id if child_raw_state.lineage else "unknown",
                "tokens": child_result.token_usage.total,
                "cancelled": child_result.metadata.get("cancelled", False),
            },
        ))

        # record child tokens in parent budget tracker so total_tokens_spent() is accurate
        if self._budget_engine:
            self._budget_engine._tracker.record(
                node_id=child_raw_state.lineage.node_id if child_raw_state.lineage else "child",
                input_tokens=child_result.token_usage.input_tokens,
                output_tokens=child_result.token_usage.output_tokens,
                tool_tokens=child_result.token_usage.tool_tokens,
                context_tokens=child_result.token_usage.context_tokens,
            )

        return child_result.output

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_lineage(self, raw_state: RawExecutionState) -> LineageSummary:
        if raw_state.lineage is not None:
            return raw_state.lineage
        from axor_core.node.envelope import _new_node_id
        return LineageSummary(
            node_id=_new_node_id(),
            parent_id=None,
            depth=self._depth,
            ancestry_ids=[],
            inherited_restrictions=[],
        )

    def _stub_context_view(
        self,
        raw_state: RawExecutionState,
        policy: ExecutionPolicy,
        lineage: LineageSummary,
    ) -> ContextView:
        """Minimal ContextView when no ContextManager is injected."""
        from axor_core.contracts.context import ContextFragment
        fragments = [
            ContextFragment(
                kind="fact",
                content=raw_state.task,
                token_estimate=len(raw_state.task) // 4,
                source="raw_task",
            )
        ]
        if raw_state.parent_export:
            fragments.append(ContextFragment(
                kind="parent_export",
                content=raw_state.parent_export,
                token_estimate=len(raw_state.parent_export) // 4,
                source="parent_node",
            ))
        total = sum(f.token_estimate for f in fragments)
        return ContextView(
            node_id=lineage.node_id,
            working_summary=raw_state.task,
            visible_fragments=fragments,
            active_constraints=[policy.context_mode.value, policy.compression_mode.value],
            lineage=lineage,
            token_count=total,
            compression_ratio=1.0,
        )

    def _extract_token_usage(
        self,
        trace_events: list,
        context: ContextView,
    ) -> TokenUsage:
        from axor_core.contracts.trace import TokensSpentEvent
        spent = next(
            (e for e in reversed(trace_events) if isinstance(e, TokensSpentEvent)),
            None,
        )
        if spent:
            return TokenUsage(
                input_tokens=spent.input_tokens,
                output_tokens=spent.output_tokens,
                tool_tokens=spent.tool_tokens,
                context_tokens=spent.context_tokens,
            )
        return TokenUsage(
            input_tokens=0, output_tokens=0,
            tool_tokens=0, context_tokens=context.token_count,
        )

    def _partial_result(
        self,
        envelope: ExecutionEnvelope,
        output: str,
        payload: dict,
        trace_events: list,
    ) -> ExecutionResult:
        """Return a partial governed result when execution is cancelled."""
        token_usage = self._extract_token_usage(trace_events, envelope.context)
        return ExecutionResult(
            node_id=envelope.node_id,
            output=output or "[cancelled]",
            export_payload={"output": output, "cancelled": True},
            token_usage=token_usage,
            metadata={
                "policy": envelope.policy.name,
                "cancelled": True,
                "cancel_reason": envelope.cancel_token.reason.value
                    if envelope.cancel_token.reason else "unknown",
            },
        )


def _stamp(event, *, node_id: str, sequence: int):
    return dataclasses.replace(event, node_id=node_id, sequence=sequence)
