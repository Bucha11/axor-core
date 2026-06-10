from __future__ import annotations

import dataclasses
import logging

from axor_core import trace as trace_mod
from axor_core.budget.policy_engine import BudgetPolicyEngine, OptimizationAction
from axor_core.capability.executor import CapabilityExecutor
from axor_core.context.manager import ContextManager
from axor_core.contracts.cancel import CancelToken, make_token
from axor_core.contracts.context import (
    ContextView,
    LineageSummary,
    RawExecutionState,
)
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.extension import ExtensionBundle
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.policy import ExecutionPolicy, ExportMode

# Export restrictiveness ordering (least → most leakage-restrictive). Used to narrow
# the export contract toward less leakage without ever widening it.
_EXPORT_RANK = {
    ExportMode.FULL: 0,
    ExportMode.SUMMARY: 1,
    ExportMode.FILTERED: 2,
    ExportMode.RESTRICTED: 3,
}


def _more_restrictive_export(a: "ExportMode | None", b: "ExportMode | None") -> "ExportMode | None":
    """Return whichever export mode leaks less (higher rank). None is treated as
    'no constraint'."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _EXPORT_RANK[a] >= _EXPORT_RANK[b] else b
from axor_core.contracts.result import (
    ExecutionResult,
    ExecutorEventKind,
    TokenUsage,
)
from axor_core.contracts.trace import TraceConfig, TraceEventKind
from axor_core.capability.locked import governance_context
from axor_core.taint.engine import TaintEngine
from axor_core.taint.causal_root import CausalRoot
from axor_core.tokens import estimate_tokens
from axor_core.degradation.engine import DegradationEngine
from axor_core.node.envelope import EnvelopeBuilder
from axor_core.node.export import ExportFilter
from axor_core.node.intent_loop import IntentLoop
from axor_core.node.spawn import ChildSpawner
from axor_core.policy.analyzer import TaskAnalyzer
from axor_core.policy.composer import PolicyComposer
from axor_core.policy.selector import PolicySelector
from axor_core.trace.collector import TraceCollector


log = logging.getLogger("axor.wrapper")


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
        escalation_callback=None,
        taint_engine: TaintEngine | None = None,
        degradation_engine: "DegradationEngine | None" = None,
        max_intents_per_session: int | None = 1000,
        max_total_spawns: int | None = 200,
        consequence_overrides: "dict | None" = None,
        value_policies: "dict | None" = None,
        positional_sinks: "frozenset[str] | set[str] | None" = None,
    ) -> None:
        self._executor = executor
        self._child_executor = child_executor  # None → reuse parent executor
        self._cap_executor = capability_executor
        self._analyzer = analyzer
        self._selector = selector
        self._composer = composer
        self._context_manager = context_manager  # None → stub ContextView
        self._budget_engine = budget_engine  # None → no budget tracking
        self._trace_collector = trace_collector  # None → events not persisted
        self._trace_config = trace_config or TraceConfig()
        self._depth = current_depth
        self._escalation_callback = escalation_callback  # None → auto-deny escalation
        self._taint_engine = taint_engine if taint_engine is not None else TaintEngine()
        self._degradation_engine = degradation_engine
        self._max_intents_per_session = max_intents_per_session
        self._max_total_spawns = max_total_spawns
        self._consequence_overrides = consequence_overrides or {}
        self._positional_sinks = frozenset(positional_sinks or ())
        self._value_policies = value_policies or {}

        self._envelope_builder = EnvelopeBuilder()
        self._export_filter = ExportFilter()
        self._child_spawner = ChildSpawner()

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
            trace_events.append(
                _stamp(signal_event, node_id="pending", sequence=0)
            )
            policy = self._selector.select(signal)

        extension_fragments = (
            list(extension_bundle.fragments) if extension_bundle else []
        )
        policy = self._composer.compose(
            base=policy,
            extensions=extension_fragments,
            parent_policy=parent_policy,
        )

        # When running as a child node, assert the derived policy actually
        # respects the parent ceiling. apply_parent_restrictions in compose()
        # enforces this, but we validate explicitly so any regression is caught
        # immediately rather than silently producing an over-privileged child.
        if parent_policy is not None and self._depth > 0:
            from axor_core.node.spawn import _validate_child_policy
            _validate_child_policy(policy, parent_policy, self._depth)

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
        trace_events.append(
            trace_mod.events.policy_chosen(lineage.node_id, policy)
        )

        # ── 3. Context ─────────────────────────────────────────────────────────
        if self._context_manager is not None:
            context = self._context_manager.build(
                raw_state, lineage, policy=policy
            )
        else:
            context = self._stub_context_view(raw_state, policy, lineage)

        # ── 4. Envelope ────────────────────────────────────────────────────────
        extension_tools = (
            list(extension_bundle.tools) if extension_bundle else []
        )
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
            if decision.action == OptimizationAction.COMPRESS_CONTEXT:
                log.warning(
                    "budget: context compression recommended (node=%s, reason=%s)",
                    envelope.node_id, decision.reason,
                )

        # ── 6. Intent loop ─────────────────────────────────────────────────────
        # if executor supports ToolResultBus (e.g. ClaudeCodeExecutor),
        # register a callback so intent_loop pushes results into the bus
        # instead of yielding them as TEXT events
        tool_result_callback = None
        if hasattr(self._executor, "get_bus"):
            bus = self._executor.get_bus()
            if bus is not None:

                async def _push_to_bus(
                    tool_use_id: str,
                    tool_name: str,
                    result: object,
                    approved: bool,
                ) -> None:
                    bus.push(tool_use_id, result)

                tool_result_callback = _push_to_bus

        # spawn_callback — routes spawn_child intents to _handle_spawn_child
        # captures closure variables needed for child construction
        _ext_bundle = extension_bundle
        _parent_pol = policy
        _trace_events = trace_events

        async def _spawn_child_callback(
            tool_use_id: str, task: str, context_hint: str
        ) -> str:
            from axor_core.errors.exceptions import (
                ChildNotAllowedError,
                MaxDepthExceededError,
            )

            child_intent = Intent(
                kind=IntentKind.SPAWN_CHILD,
                payload={
                    "task": task,
                    "context_hint": context_hint,
                    "tool_use_id": tool_use_id,
                },
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
            escalation_callback=self._escalation_callback,
            taint_engine=self._taint_engine,
            degradation_engine=self._degradation_engine,
            max_intents_per_session=self._max_intents_per_session,
            max_total_spawns=self._max_total_spawns,
            consequence_overrides=self._consequence_overrides,
            value_policies=self._value_policies,
            positional_sinks=self._positional_sinks,
        )

        raw_output, raw_payload, budget_export_mode = await self._collect_stream(
            intent_loop=intent_loop,
            envelope=envelope,
            extension_bundle=extension_bundle,
            parent_policy=policy,
            cancel_token=cancel_token,
        )

        # ── 7. Export filter ───────────────────────────────────────────────────
        # Apply any budget-imposed export narrowing to the contract before the
        # filter runs, so a crossed restrict_export threshold actually narrows what
        # leaves the node.
        export_envelope = envelope
        if budget_export_mode is not None:
            effective_mode = _more_restrictive_export(
                envelope.export_contract.mode, budget_export_mode
            )
            if effective_mode != envelope.export_contract.mode:
                narrowed_contract = dataclasses.replace(
                    envelope.export_contract, mode=effective_mode
                )
                export_envelope = dataclasses.replace(
                    envelope, export_contract=narrowed_contract
                )
        token_usage = self._extract_token_usage(trace_events, context)
        result = self._export_filter.apply(
            raw_output=raw_output,
            raw_payload=raw_payload,
            envelope=export_envelope,
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
    ) -> tuple[str, dict, "ExportMode | None"]:
        output_parts: list[str] = []
        payload: dict = {}
        budget_export_mode: "ExportMode | None" = None  # budget-imposed narrowing

        with governance_context():
            raw_stream = self._executor.stream(envelope)
            async for event in intent_loop.run(raw_stream, envelope):
                match event.kind:
                    case ExecutorEventKind.TEXT:
                        # budget check on result arrival
                        text = event.payload.get("text", "")
                        tool_result = event.payload.get("tool_result")

                        if tool_result and self._budget_engine:
                            estimate = estimate_tokens(tool_result)
                            result_decision = self._budget_engine.on_result_arrived(
                                node_id=envelope.node_id,
                                result_token_estimate=estimate,
                                policy=envelope.policy,
                            )
                            if result_decision.action in (
                                OptimizationAction.COMPRESS_CONTEXT,
                                OptimizationAction.RESTRICT_EXPORT,
                            ):
                                log.warning(
                                    "budget: %s recommended after result (node=%s, reason=%s)",
                                    result_decision.action.value, envelope.node_id, result_decision.reason,
                                )
                            # Actually ENFORCE the export restriction — narrow
                            # the export mode to the more restrictive of the contract
                            # and the budget's suggestion (never widens).
                            if result_decision.action == OptimizationAction.RESTRICT_EXPORT:
                                budget_export_mode = _more_restrictive_export(
                                    budget_export_mode, result_decision.suggested_export
                                )

                        # record file reads into context manager
                        if tool_result and self._context_manager:
                            args = event.payload.get("args", {})
                            path = args.get("path", "")
                            if path and isinstance(tool_result, str):
                                self._context_manager.record_file_read(
                                    path, tool_result
                                )

                        content = text or str(tool_result or "")
                        output_parts.append(content)

                    case ExecutorEventKind.STOP:
                        payload = event.payload

        return "".join(output_parts), payload, budget_export_mode

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
            if decision.action == OptimizationAction.DENY_CHILD:
                return f"[child spawn denied: {decision.reason}]"
            if decision.action == OptimizationAction.COMPRESS_CONTEXT:
                log.warning(
                    "budget: context compression recommended before child spawn (node=%s, reason=%s)",
                    envelope.node_id, decision.reason,
                )

        child_task_str, child_lineage = (
            self._child_spawner.prepare_child(
                spawn_intent=intent,
                parent_envelope=envelope,
                intent_loop=IntentLoop(
                    self._cap_executor, trace_events, self._depth
                ),
                trace_events=trace_events,
            )
        )

        child_raw_state = RawExecutionState(
            task=child_task_str,
            session_id=envelope.parent_metadata.get("session_id", ""),
            parent_export=envelope.context.working_summary,
            session_state={},
            memory_fragments=[],
            lineage=child_lineage,
        )

        # Build child taint engine before constructing the node so taint
        # inheritance is set atomically via the constructor, not via private field.
        child_taint = TaintEngine(node_id=child_lineage.node_id)
        # Spawn inheritance is PER-VALUE: the child inherits the parent's
        # value-ledger, NOT the coarse session-taint flag (which would re-explode
        # taint over the subtree). Subtree lock-down is via the SHARED degradation
        # engine; lateral protection across the node tree is preserved per-value.
        child_taint.inherit_value_ledger(self._taint_engine)

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
            child_executor=self._child_executor,
            escalation_callback=self._escalation_callback,
            degradation_engine=self._degradation_engine,
            max_intents_per_session=self._max_intents_per_session,
            max_total_spawns=self._max_total_spawns,
            taint_engine=child_taint,
            consequence_overrides=self._consequence_overrides,
            value_policies=self._value_policies,
            positional_sinks=self._positional_sinks,
        )

        child_cancel = envelope.cancel_token.child_token()
        child_result = await child_node.run(
            raw_state=child_raw_state,
            extension_bundle=extension_bundle,
            parent_policy=parent_policy,
            cancel_token=child_cancel,
        )

        # Cross-process re-mint: the child's returned output crosses a trust
        # boundary the parent cannot see through (the parent has no visibility into
        # the child's internal reads). Re-mint it as untrusted in the parent's
        # per-value ledger, so a parent sink later carrying child output is gated —
        # a child cannot launder a secret/web value it read by returning it through
        # its output. Forward inheritance is per-value; this closes the reverse path.
        if self._taint_engine is not None and child_result.output:
            self._taint_engine.register_value(
                child_result.output, CausalRoot.cross_process_in()
            )

        # emit child_completed into parent trace
        from axor_core.contracts.trace import TraceEvent

        trace_events.append(
            TraceEvent(
                kind=TraceEventKind.CHILD_COMPLETED,
                node_id=envelope.node_id,
                sequence=len(trace_events),
                payload={
                    "child_node_id": child_raw_state.lineage.node_id
                    if child_raw_state.lineage
                    else "unknown",
                    "tokens": child_result.token_usage.total,
                    "cancelled": child_result.metadata.get("cancelled", False),
                },
            )
        )

        # record child tokens in parent budget tracker so total_tokens_spent() is accurate
        if self._budget_engine:
            child_lineage = child_raw_state.lineage
            child_node_id = child_lineage.node_id if child_lineage else "child"
            # Register the child's lineage (parent + depth) before recording, so
            # depth_tokens()/subtree accounting see the real tree, not depth=0/parent=None.
            if child_lineage:
                self._budget_engine.register_node(
                    child_node_id, child_lineage.parent_id, child_lineage.depth
                )
            self._budget_engine.record_child_tokens(
                node_id=child_node_id,
                input_tokens=child_result.token_usage.input_tokens,
                output_tokens=child_result.token_usage.output_tokens,
                tool_tokens=child_result.token_usage.tool_tokens,
                context_tokens=child_result.token_usage.context_tokens,
                cache_creation_input_tokens=(
                    child_result.token_usage.cache_creation_input_tokens
                ),
                cache_read_input_tokens=(
                    child_result.token_usage.cache_read_input_tokens
                ),
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
                token_estimate=estimate_tokens(raw_state.task),
                source="raw_task",
            )
        ]
        if raw_state.parent_export:
            fragments.append(
                ContextFragment(
                    kind="parent_export",
                    content=raw_state.parent_export,
                    token_estimate=estimate_tokens(raw_state.parent_export),
                    source="parent_node",
                )
            )
        total = sum(f.token_estimate for f in fragments)
        return ContextView(
            node_id=lineage.node_id,
            working_summary=raw_state.task,
            visible_fragments=fragments,
            active_constraints=[
                policy.context_mode.value,
                policy.compression_mode.value,
            ],
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
            (
                e
                for e in reversed(trace_events)
                if isinstance(e, TokensSpentEvent)
            ),
            None,
        )
        if spent:
            return TokenUsage(
                input_tokens=spent.input_tokens,
                output_tokens=spent.output_tokens,
                tool_tokens=spent.tool_tokens,
                context_tokens=spent.context_tokens,
                cache_creation_input_tokens=spent.cache_creation_input_tokens,
                cache_read_input_tokens=spent.cache_read_input_tokens,
            )
        return TokenUsage(
            input_tokens=0,
            output_tokens=0,
            tool_tokens=0,
            context_tokens=context.token_count,
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
                if envelope.cancel_token.reason
                else "unknown",
            },
        )


def _stamp(event, *, node_id: str, sequence: int):
    return dataclasses.replace(event, node_id=node_id, sequence=sequence)
