from __future__ import annotations

import uuid
from typing import Any
from axor_core.contracts.cancel import make_token, CancelReason
from axor_core.contracts.context import RawExecutionState, LineageSummary
from axor_core.contracts.extension import ExtensionBundle, ExtensionLoader
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.policy import ExecutionPolicy, SignalClassifier
from axor_core.contracts.result import ExecutionResult
from axor_core.contracts.trace import TraceConfig
from axor_core.capability.executor import CapabilityExecutor
from axor_core.context.manager import ContextManager
from axor_core.node.wrapper import GovernedNode
from axor_core.policy.analyzer import TaskAnalyzer
from axor_core.policy.selector import PolicySelector
from axor_core.policy.composer import PolicyComposer
from axor_core.budget import (
    BudgetTracker,
    BudgetEstimator,
    BudgetPolicyEngine,
    BudgetThresholds,
    TokenCostRates,
    CacheSummary,
    CostSummary,
)
from axor_core.trace import TraceCollector
from axor_core.extensions.registry import ExtensionRegistry
from axor_core.extensions.sanitizer import ExtensionSanitizer
from axor_core.worker.commands import SlashCommandRouter


class GovernedSession:
    """
    Public interface for governed execution.

    Wires all subsystems together and provides a clean run() API.

    Usage:

        session = GovernedSession(
            executor=ClaudeCodeExecutor(),
            capability_executor=cap_executor,
        )
        result = await session.run("write a test for the payment endpoint")

    With optional classifier:

        session = GovernedSession(
            executor=ClaudeCodeExecutor(),
            capability_executor=cap_executor,
            classifier=LocalTrainedClassifier(),
        )

    With soft token limit:

        session = GovernedSession(
            executor=...,
            capability_executor=...,
            soft_token_limit=100_000,
        )

    With extension loaders:

        session = GovernedSession(
            executor=...,
            capability_executor=...,
            extension_loaders=[ClaudeSkillLoader(), ClaudePluginLoader()],
        )
    """

    def __init__(
        self,
        executor: Invokable,
        capability_executor: CapabilityExecutor,
        classifier: SignalClassifier | None = None,
        extension_loaders: list[ExtensionLoader] | None = None,
        trace_config: TraceConfig | None = None,
        soft_token_limit: int | None = None,
        budget_thresholds: BudgetThresholds | None = None,
        token_cost_rates: TokenCostRates | None = None,
        child_executor: Invokable | None = None,
        agent_def: "AgentDefinition | None" = None,
        memory_provider: "MemoryProvider | None" = None,
        telemetry: "Any | None" = None,
    ) -> None:
        self._session_id     = f"session_{uuid.uuid4().hex[:12]}"
        self._executor       = executor
        self._child_executor = child_executor
        self._cap_executor   = capability_executor
        self._agent_def      = agent_def
        self._memory_provider = memory_provider
        self._trace_config   = trace_config or TraceConfig()
        self._token_cost_rates = token_cost_rates
        # Duck-typed: any object exposing `ingest_trace(trace, raw_input)` and
        # optional `aclose()`. Typically axor_telemetry.TelemetryPipeline.
        # Kept Any so core does not import from telemetry packages.
        self._telemetry      = telemetry

        # derive agent domain for task analyzer
        agent_domain = "general"
        if agent_def is not None:
            agent_domain = agent_def.domain.value

        # policy subsystem
        self._analyzer  = TaskAnalyzer(
            external_classifier=classifier,
            agent_domain=agent_domain,
        )
        self._selector  = PolicySelector()
        self._composer  = PolicyComposer()

        # budget subsystem — shared across all nodes
        self._tracker       = BudgetTracker()
        self._estimator     = BudgetEstimator()
        self._budget_engine = BudgetPolicyEngine(
            tracker=self._tracker,
            estimator=self._estimator,
            soft_limit=soft_token_limit,
            thresholds=budget_thresholds,
        )

        # trace — shared across all nodes
        self._collector = TraceCollector(
            config=self._trace_config,
            session_id=self._session_id,
        )

        # extensions
        self._extension_loaders = extension_loaders or []
        self._sanitizer = ExtensionSanitizer()
        self._registry  = ExtensionRegistry()

        # context — session-scoped so symbol_table and cache persist across turns
        # policy passed per-call to build() so it always reflects current execution
        self._context_manager = ContextManager()

        # commands
        self._command_router = SlashCommandRouter(collector=self._collector)

        # active cancel token for current execution
        self._active_token = None
        self._started = False

    async def start(self) -> None:
        """Load and sanitize extensions. Auto-called on first run()."""
        if self._started:
            return
        for loader in self._extension_loaders:
            bundle = await loader.load()
            sanitized = self._sanitizer.sanitize(bundle)
            self._registry.register(sanitized)
        self._started = True

    async def run(
        self,
        task: str,
        policy: ExecutionPolicy | None = None,
        session_state: dict | None = None,
        parent_export: str | None = None,
        lineage: LineageSummary | None = None,
    ) -> ExecutionResult:
        await self.start()

        if task.strip().startswith("/"):
            return await self._handle_command(task)

        # load memory fragments if provider is configured
        memory_fragments: list[str] = []
        if self._memory_provider is not None:
            from axor_core.contracts.memory import MemoryQuery
            namespaces = ()
            if self._agent_def is not None:
                namespaces = self._agent_def.memory_namespaces
            query = MemoryQuery(namespaces=namespaces, max_results=20)
            fragments = await self._memory_provider.load(query)
            memory_fragments = [f.content for f in fragments]

        raw_state = RawExecutionState(
            task=task,
            session_id=self._session_id,
            parent_export=parent_export,
            session_state=session_state or {},
            memory_fragments=memory_fragments,
            lineage=lineage,
        )

        # inject personality as pinned context fragment (once per session)
        if not getattr(self, '_personality_injected', False):
            if self._agent_def is not None and self._agent_def.personality:
                from axor_core.contracts.context import ContextFragment
                self._context_manager.pin_fragment(ContextFragment(
                    kind="skill",
                    content=self._agent_def.personality,
                    token_estimate=len(self._agent_def.personality) // 4,
                    source=f"agent:{self._agent_def.name}:personality",
                    relevance=1.0,
                    value="pinned",
                ))
            self._personality_injected = True

        # wire ContextManager as post-execute observer once per session (idempotent)
        if not getattr(self, '_context_observer_registered', False):
            async def _context_observer(tool_name: str, args: dict, result) -> None:
                if tool_name == "read" and isinstance(result, str):
                    path = args.get("path", "")
                    if path:
                        self._context_manager.record_file_read(path, result)
                self._context_manager.cache_tool_result(tool_name, args, result)
            self._cap_executor.register_post_callback(_context_observer)
            self._context_observer_registered = True

        cancel_token = make_token()
        self._active_token = cancel_token

        node = self._make_node(self._context_manager)
        result = await node.run(
            raw_state=raw_state,
            extension_bundle=self._registry.current_bundle(),
            override_policy=policy,
            cancel_token=cancel_token,
        )
        self._active_token = None

        # record tokens in session-level tracker
        self._tracker.record(
            node_id=result.node_id,
            input_tokens=result.token_usage.input_tokens,
            output_tokens=result.token_usage.output_tokens,
            tool_tokens=result.token_usage.tool_tokens,
            context_tokens=result.token_usage.context_tokens,
            cache_creation_input_tokens=result.token_usage.cache_creation_input_tokens,
            cache_read_input_tokens=result.token_usage.cache_read_input_tokens,
        )

        # Feed telemetry pipeline, if one is attached. Failures here must
        # never propagate to the caller — the governance path is authoritative.
        if self._telemetry is not None:
            try:
                trace = self._collector.get_trace(result.node_id)
                if trace is not None:
                    await self._telemetry.ingest_trace(trace, raw_input=task)
            except Exception:
                pass

        return result

    def cancel(self, detail: str = "") -> None:
        """
        Cancel the current active execution.
        Safe to call from signal handlers or background threads.
        """
        if self._active_token:
            self._active_token.cancel(CancelReason.USER_ABORT, detail=detail)

    async def aclose(self) -> None:
        """
        Close session-scoped resources: trace JSONL file, telemetry pipeline,
        memory provider. Idempotent. Safe to call even if start() was never
        invoked.
        """
        self._collector.close()
        if self._telemetry is not None:
            close = getattr(self._telemetry, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass
        if self._memory_provider is not None:
            close = getattr(self._memory_provider, "aclose", None) or getattr(
                self._memory_provider, "close", None
            )
            if close is not None:
                res = close()
                if hasattr(res, "__await__"):
                    await res

    async def __aenter__(self) -> "GovernedSession":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ── Introspection ──────────────────────────────────────────────────────────

    def session_id(self) -> str:
        return self._session_id

    def total_tokens_spent(self) -> int:
        return self._tracker.total_tokens()

    def cache_summary(self) -> CacheSummary:
        """Aggregate prompt-cache accounting for the session."""
        return self._tracker.cache_summary()

    def cost_summary(self) -> CostSummary | None:
        """Aggregate money accounting for the session when rates are configured."""
        if self._token_cost_rates is None:
            return None
        return self._tracker.cost_summary(self._token_cost_rates)

    def all_traces(self):
        return self._collector.all_traces()

    def budget_snapshot(self) -> dict:
        return {
            nid: {
                "total": b.total,
                "input": b.total_input_tokens,
                "output": b.output_tokens,
                "depth": b.depth,
            }
            for nid, b in self._tracker.snapshot().items()
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _make_node(self, context_manager: ContextManager) -> GovernedNode:
        return GovernedNode(
            executor=self._executor,
            capability_executor=self._cap_executor,
            analyzer=self._analyzer,
            selector=self._selector,
            composer=self._composer,
            context_manager=context_manager,
            budget_engine=self._budget_engine,
            trace_collector=self._collector,
            trace_config=self._trace_config,
            child_executor=self._child_executor,
        )

    async def _handle_command(self, raw: str) -> ExecutionResult:
        from axor_core.contracts.result import ExecutionResult, TokenUsage
        result = await self._command_router.route(raw=raw, session=self)
        return ExecutionResult(
            node_id=self._session_id,
            output=str(result.output),
            export_payload={"output": str(result.output)},
            token_usage=TokenUsage(
                input_tokens=0, output_tokens=0,
                tool_tokens=0,  context_tokens=0,
            ),
            metadata={
                "command": result.command.name,
                "class": result.command_class.value,
            },
        )
