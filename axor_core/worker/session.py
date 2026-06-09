from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axor_core.contracts.agent import AgentDefinition
    from axor_core.contracts.memory import MemoryProvider
from axor_core.contracts.cancel import make_token, CancelReason
from axor_core.contracts.context import RawExecutionState, LineageSummary
from axor_core.contracts.extension import ExtensionLoader
from axor_core.contracts.drift import BehavioralDriftObserver
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import ExecutionPolicy, SignalClassifier
from axor_core.contracts.result import ExecutionResult
from axor_core.contracts.trace import TraceConfig
from axor_core.capability.executor import CapabilityExecutor
from axor_core.capability.locked import LockedExecutor
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
from axor_core.taint.engine import TaintEngine

_log = logging.getLogger("axor.session")


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
        behavioral_drift_observer: BehavioralDriftObserver | None = None,
        extension_loaders: list[ExtensionLoader] | None = None,
        trace_config: TraceConfig | None = None,
        soft_token_limit: int | None = None,
        budget_thresholds: BudgetThresholds | None = None,
        token_cost_rates: TokenCostRates | None = None,
        child_executor: Invokable | None = None,
        agent_def: "AgentDefinition | None" = None,
        memory_provider: "MemoryProvider | None" = None,
        telemetry: "Any | None" = None,
        mode: ExecutionMode = ExecutionMode.LIBRARY,
        require_isolation: bool = False,
        profile: "str | Any | None" = None,
        workspace: str | None = None,
        danger: "dict | None" = None,
    ) -> None:
        # Profile = a named bundle of existing knobs (no new mechanism); it
        # pre-fills mode / isolation / escalation / consequence-ceiling / watcher.
        self._consequence_overrides = dict(danger or {})
        _overlay_ceiling = None
        _overlay_escalation = None
        if profile is not None:
            from axor_core.profiles import resolve_profile
            prof = resolve_profile(profile)
            mode = prof.mode
            require_isolation = require_isolation or prof.require_isolation
            _overlay_ceiling = prof.consequence_ceiling
            _overlay_escalation = prof.escalation_policy
            if behavioral_drift_observer is None and prof.attach_watcher:
                from axor_core.node.drift_observer import TaintEngineDriftObserver
                behavioral_drift_observer = TaintEngineDriftObserver()
        self._overlay_allowed_paths = (workspace,) if workspace else None

        self._session_id     = f"session_{uuid.uuid4().hex[:12]}"
        self._mode           = mode
        self._behavioral_drift_observer = behavioral_drift_observer
        self._overlay_ceiling = _overlay_ceiling
        self._overlay_escalation = _overlay_escalation

        # STRICT mode is a superset of PRODUCTION.
        # Apply all STRICT-only restrictions here before anything else is wired.
        if mode == ExecutionMode.STRICT:
            # Classifier disabled in STRICT mode — policy is rule-based only.
            classifier = None
            # Force audit_required on trace config — unknown provider format or
            # trace write failure terminates session, not just denies.
            if trace_config is None:
                trace_config = TraceConfig(audit_required=True)
            elif not trace_config.audit_required:
                from dataclasses import replace as _dc_replace
                trace_config = _dc_replace(trace_config, audit_required=True)

        self._deny_on_ambiguity: bool = (mode == ExecutionMode.STRICT)
        self._strict_escalation: bool = (mode == ExecutionMode.STRICT)

        # Process-isolation gate (Phase 1). In PRODUCTION/STRICT an untrusted
        # agent should execute tools out-of-process (DaemonCapabilityClient);
        # an in-process CapabilityExecutor is not a hard boundary against a
        # compromised agent process. LockedExecutor only blocks the in-process
        # governance bypass, not native code in the agent process.
        if mode in (ExecutionMode.PRODUCTION, ExecutionMode.STRICT):
            self._enforce_isolation_policy(
                capability_executor, mode, agent_def, require_isolation
            )

        # In PRODUCTION/STRICT mode wrap executor so direct calls outside
        # the governance path raise GovernanceBypassError.
        if mode in (ExecutionMode.PRODUCTION, ExecutionMode.STRICT):
            self._executor = LockedExecutor(executor, mode)
        else:
            self._executor = executor
            # Warn when LIBRARY mode is used in a production-like environment.
            if os.environ.get("AXOR_ENV", "").lower() == "production":
                _log.warning(
                    "GovernedSession created in LIBRARY mode but AXOR_ENV=production. "
                    "LIBRARY mode does not provide strong isolation guarantees. "
                    "Use ExecutionMode.PRODUCTION for production deployments."
                )

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
        self._composer  = PolicyComposer(
            consequence_ceiling=self._overlay_ceiling,
            escalation_policy=self._overlay_escalation,
            allowed_paths=self._overlay_allowed_paths,
        )

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
        self._personality_injected = False
        self._context_observer_registered = False

        # adaptive policy: narrowest policy seen across turns (never broadens automatically)
        self._active_policy: ExecutionPolicy | None = None

        # taint engine — persists across turns so taint is sticky within a session
        self._taint_engine = TaintEngine(node_id=self._session_id)

        # degradation engine — persists across turns; level is monotonically increasing
        from axor_core.degradation.engine import DegradationEngine
        from axor_core.contracts.degradation import DegradationPolicy
        self._degradation_engine = DegradationEngine(DegradationPolicy(), node_id=self._session_id)

    @staticmethod
    def _enforce_isolation_policy(
        capability_executor: CapabilityExecutor,
        mode: ExecutionMode,
        agent_def: "AgentDefinition | None",
        require_isolation: bool,
    ) -> None:
        """Gate in-process tool execution for untrusted agents in PRODUCTION/STRICT.

        Raises IsolationRequiredError when isolation is required (explicit flag or
        AXOR_REQUIRE_ISOLATION=1) but the capability executor runs in-process.
        Otherwise warns for untrusted agents so the soft boundary is never silent.
        """
        from axor_core.contracts.agent import TrustLevel
        from axor_core.errors.exceptions import IsolationRequiredError

        if getattr(capability_executor, "is_process_isolated", False):
            return

        env_require = os.environ.get("AXOR_REQUIRE_ISOLATION", "").lower() in (
            "1", "true", "yes",
        )
        if require_isolation or env_require:
            raise IsolationRequiredError(
                f"mode={mode.value} requires a process-isolated capability executor "
                "(e.g. DaemonCapabilityClient); got an in-process executor"
            )

        trust = agent_def.trust_level if agent_def is not None else TrustLevel.STANDARD
        if trust in (TrustLevel.RESTRICTED, TrustLevel.STANDARD):
            _log.warning(
                "GovernedSession in %s mode with an untrusted agent (trust=%s) is using "
                "an in-process capability executor — tool execution is NOT isolated from "
                "a compromised agent process. Use DaemonCapabilityClient, pass "
                "require_isolation=True, or set AXOR_REQUIRE_ISOLATION=1 to enforce.",
                mode.value, trust.value,
            )

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
        # D-2 invariant: TERMINAL session raises before any intent evaluation.
        # Also check LOCKED_TTL so idle sessions that hit LOCKED eventually reach TERMINAL.
        from axor_core.contracts.degradation import DegradationLevel
        from axor_core.errors.exceptions import SessionTerminatedError
        self._degradation_engine.check_ttl()
        if self._degradation_engine.state.level == DegradationLevel.TERMINAL:
            raise SessionTerminatedError(
                f"session {self._session_id} reached TERMINAL degradation level"
            )

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
            # Re-mint on read-back (TM3.2 / TM4.1): a value persisted to memory and
            # re-read re-mints as tainted — "soft release to memory" does not launder
            # it. We do not assume in-session memory is clean (stricter than CaMeL /
            # Firewalls, which assume the environment is not poisoned).
            # Re-mint on read-back (TM3.2/TM4.1), PER-VALUE: register memory content
            # in the per-value ledger so a sink later carrying a memory-derived value
            # is gated at the value level. "Soft release to memory" does not launder.
            if memory_fragments:
                from axor_core.contracts.taint import TaintSource
                from axor_core.taint.causal_root import CausalRoot
                root = CausalRoot.external_read(TaintSource.MEMORY)
                for frag in memory_fragments:
                    self._taint_engine.register_value(frag, root)

        raw_state = RawExecutionState(
            task=task,
            session_id=self._session_id,
            parent_export=parent_export,
            session_state=session_state or {},
            memory_fragments=memory_fragments,
            lineage=lineage,
        )

        # inject personality as pinned context fragment (once per session)
        if not self._personality_injected:
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
        if not self._context_observer_registered:
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

        # Adaptive policy: re-classify each turn; capability surface can only
        # narrow automatically — broadening requires an explicit operator override.
        effective_policy = policy
        if policy is None:
            signal, _ = await self._analyzer.analyze(task)
            new_policy = self._selector.select(signal)
            if self._active_policy is None:
                self._active_policy = new_policy
            else:
                narrowed = self._composer.apply_parent_restrictions(
                    new_policy, self._active_policy
                )
                if narrowed != self._active_policy:
                    _log.info(
                        "session=%s adaptive policy narrowed: %s → %s",
                        self._session_id,
                        self._active_policy.name,
                        narrowed.name,
                    )
                self._active_policy = narrowed
            effective_policy = self._active_policy

        node = self._make_node(self._context_manager)
        result = await node.run(
            raw_state=raw_state,
            extension_bundle=self._registry.current_bundle(),
            override_policy=effective_policy,
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

    async def notify_behavioral_drift(self, agent_id: str, action: str) -> None:
        """
        Notify the session that axor-probe detected behavioral drift.

        Called by the caller who wires ProbePipeline to GovernedSession.
        Propagates taint via TaintEngineDriftObserver if one is configured.
        Failures are logged and swallowed — governance path is not interrupted.

        action: "elevated_review" | "restricted_mode"
        """
        if self._behavioral_drift_observer is None:
            return
        try:
            await self._behavioral_drift_observer.on_drift(
                session_id=self._session_id,
                agent_id=agent_id,
                action=action,
            )
        except Exception:
            _log.error(
                "behavioral_drift_observer.on_drift failed session=%s agent=%s action=%s",
                self._session_id, agent_id, action,
                exc_info=True,
            )

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

    def current_degradation_level(self) -> str:
        return self._degradation_engine.state.level.value

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
            taint_engine=self._taint_engine,
            degradation_engine=self._degradation_engine,
            consequence_overrides=self._consequence_overrides,
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
