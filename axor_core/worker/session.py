from __future__ import annotations

import logging
import os
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axor_core.contracts.agent import AgentDefinition
    from axor_core.contracts.memory import MemoryProvider
    from axor_core.contracts.session import SessionSink
    from axor_core.node.intent_loop import EscalationCallback
from axor_core.contracts.cancel import make_token, CancelReason
from axor_core.contracts.session import SessionAuditRecord, ToolInvocationRecord
from axor_core.contracts.context import RawExecutionState, LineageSummary
from axor_core.contracts.extension import ExtensionLoader
from axor_core.contracts.drift import BehavioralDriftObserver
from axor_core.contracts.observation import ContextTap
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    SignalClassifier,
)
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
from axor_core.tokens import estimate_tokens

_log = logging.getLogger("axor.session")

# Fallback ambiguity threshold, used ONLY when the analyzer does not expose
# one (custom analyzers). The stock TaskAnalyzer owns the ambiguity decision
# via its ambiguity_threshold property — the session never re-interprets
# confidence with its own number, so the two cannot diverge.
_FALLBACK_AMBIGUITY_THRESHOLD = 0.75


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
            classifier=TaskSignalClassifier(),   # axor-classifier-simple
        )

    PRODUCTION compatibility/security guard — operator-defined policy
    (classifier fully bypassed, which also disables task-aware planning;
    the target model separating authority from planning is the
    AuthorityPolicy/ExecutionPlan split) plus an operator escalation
    ceiling and approver so a too-narrow policy recovers per-tool:

        session = GovernedSession(
            executor=ClaudeCodeExecutor(),
            capability_executor=cap_executor,
            mode=ExecutionMode.PRODUCTION,
            default_policy=presets.standard(),
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=("write", "bash"),
                require_human=True,
            ),
            escalation_callback=AllowlistEscalationApprover(
                {"write": 20, "bash": 10},
                allowed_path_prefixes=("/workspace",),  # confines write grants
                unconfined_tools=("bash",),  # bash exposes no checkable path
            ),
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
        escalation_callback: "EscalationCallback | None" = None,
        escalation_policy: "EscalationPolicy | None" = None,
        default_policy: ExecutionPolicy | None = None,
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
        positional_sinks: "set[str] | frozenset[str] | None" = None,
        egress_sinks: "set[str] | frozenset[str] | None" = None,
        untrusted_sources: "set[str] | frozenset[str] | None" = None,
        sensitive_sources: "set[str] | frozenset[str] | None" = None,
        imperative_sinks: "set[str] | frozenset[str] | None" = None,
        benign_tools: "set[str] | frozenset[str] | None" = None,
        driving_args: "dict[str, list[str]] | None" = None,
        trajectory_observers: "list | None" = None,
        value_policies: "dict | None" = None,
        detection_floor: float | None = None,
        adjudicator=None,
        federation_gateway=None,
        admission=None,
        session_sink: "SessionSink | None" = None,
        context_taps: "list[ContextTap] | None" = None,
        per_node_degradation: bool = False,
    ) -> None:
        # Wall-clock the session was constructed — handed to sentinel in the
        # closed-session record (slow-and-low staging compares session start times).
        self._started_at = time.time()
        # Optional Core → Sentinel audit sink + the per-tool invocation buffer it
        # consumes at close. Default None → no record built, zero overhead.
        self._session_sink = session_sink
        self._tool_invocations: "list[ToolInvocationRecord]" = []
        self._record_emitted = False   # guard: aclose is idempotent, emit once
        # Core → Probe observation seam: taps receive a SessionContextView on the
        # governance hot path — GovernedNode fires node/context_observation on
        # each context build. Observe-only — tap failures are logged, never raised.
        self._context_taps: list[ContextTap] = list(context_taps or [])
        # Per-node degradation opt-in (spec v2 Ch.4): children get their own
        # engine seeded at max(parent level, NORMAL) instead of the shared one.
        self._per_node_degradation = per_node_degradation

        # Profile = a named bundle of existing knobs (no new mechanism); it
        # pre-fills mode / isolation / escalation / consequence-ceiling / watcher.
        self._consequence_overrides = dict(danger or {})
        self._positional_sinks = frozenset(positional_sinks or ())
        # Operator tool taxonomy — which tools exfiltrate / produce untrusted or
        # secret data. Lets the kernel govern a deployment's renamed tools that the
        # normalizer's generic heuristics do not recognise.
        self._egress_sinks = frozenset(egress_sinks or ())
        self._untrusted_sources = frozenset(untrusted_sources or ())
        self._sensitive_sources = frozenset(sensitive_sources or ())
        self._imperative_sinks = frozenset(imperative_sinks or ())
        self._driving_args = dict(driving_args or {})
        self._trajectory_observers = list(trajectory_observers or [])
        self._value_policies = dict(value_policies or {})
        self._detection_floor = detection_floor  # opt-in; None = detection observe-only
        self._adjudicator = adjudicator          # opt-in advisory layer; None = off
        self._federation_gateway = federation_gateway  # opt-in A2A trust; None = off
        self._admission = admission  # opt-in control-plane admission; None = no plane
        _overlay_ceiling = None
        _overlay_escalation = None
        if profile is not None:
            from axor_core.profiles import resolve_profile
            prof = resolve_profile(profile)
            mode = prof.mode
            require_isolation = require_isolation or prof.require_isolation
            _overlay_ceiling = prof.consequence_ceiling
            _overlay_escalation = prof.escalation_policy
            self._positional_sinks = self._positional_sinks | prof.positional_sinks
            if behavioral_drift_observer is None and prof.attach_watcher:
                from axor_core.node.drift_observer import BehavioralDriftWatcher
                behavioral_drift_observer = BehavioralDriftWatcher()
        self._overlay_allowed_paths = (workspace,) if workspace else None

        self._session_id     = f"session_{uuid.uuid4().hex[:12]}"
        self._mode           = mode
        # STRICT requires every egress sink to carry a destination allowlist (the
        # sound, paraphrase-proof control) — fail closed at session construction,
        # not deferred to the first run.
        self._require_egress_allowlist = (mode == ExecutionMode.STRICT)
        # STRICT also enforces role completeness. The session validates it at
        # construction below (it knows the registered-tool universe); the flag also
        # rides into the loop so the lazy per-call check is consistent across paths.
        self._require_tool_roles = (mode == ExecutionMode.STRICT)
        self._benign_tools = frozenset(benign_tools or ())
        if self._require_egress_allowlist:
            from axor_core.kernel.registration import (
                validate_egress_allowlists,
                validate_driving_arg_allowlists,
                validate_role_completeness,
            )
            _eg_errors = validate_egress_allowlists(self._egress_sinks, self._value_policies)
            _eg_errors += validate_driving_arg_allowlists(
                self._egress_sinks, self._driving_args, self._value_policies
            )
            if _eg_errors:
                raise ValueError("strict egress allowlist: " + "; ".join(_eg_errors))
            # STRICT role completeness: every registered tool needs an explicit
            # data-flow role — no silent clean-read default. Validated against the
            # registered handler universe (empty for the daemon path, which carries
            # its own taxonomy server-side).
            _tools = capability_executor.registered_tools()
            if _tools:
                _role_errors = validate_role_completeness(
                    _tools,
                    untrusted_sources=self._untrusted_sources,
                    sensitive_sources=self._sensitive_sources,
                    egress_sinks=self._egress_sinks,
                    positional_sinks=self._positional_sinks,
                    benign_tools=self._benign_tools,
                    value_policies=self._value_policies,
                )
                if _role_errors:
                    raise ValueError("strict role completeness: " + "; ".join(_role_errors))
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

        # Human/operator escalation gate — threaded into every node's IntentLoop
        # (children inherit it at spawn). None → require_human escalations are
        # auto-denied (fail-closed). See axor_core.capability.approvals for
        # ready-made callbacks.
        self._escalation_callback = escalation_callback
        # Operator-defined escalation ceiling (authority). Applied to every
        # classifier-selected policy: which capabilities may later be granted
        # is never derived from task text — presets carry no escalation.
        self._operator_escalation = escalation_policy
        # Session-wide explicit policy: used whenever run() gets no per-call
        # policy=. With it set the task classifier is bypassed entirely —
        # the recommended posture for PRODUCTION deployments.
        self._default_policy = default_policy
        self._classifier_policy_warned = False

        # Process-isolation gate. In PRODUCTION/STRICT an untrusted
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
        self._degradation_engine = DegradationEngine(
            DegradationPolicy(), node_id=self._session_id,
            detection_floor=self._detection_floor,
        )

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

    @classmethod
    def from_config(
        cls,
        executor: Invokable,
        capability_executor: CapabilityExecutor,
        config: "Any",
        **overrides,
    ) -> "GovernedSession":
        """Build a session from a :class:`~axor_core.config.GovernanceConfig`.

        The config supplies the declarative governance taxonomy (mode, sources,
        sinks, allowlists, consequence overrides). ``overrides`` are extra keyword
        arguments — the executor-side wiring that does not belong in a YAML file
        (``telemetry``, ``memory_provider``, ``trace_config``, ``child_executor``,
        ...) — and take precedence over the config.
        """
        kwargs = config.as_session_kwargs()
        kwargs.update(overrides)
        return cls(
            executor=executor,
            capability_executor=capability_executor,
            **kwargs,
        )

    async def run(
        self,
        task: str,
        policy: ExecutionPolicy | None = None,
        session_state: dict | None = None,
        parent_export: str | None = None,
        lineage: LineageSummary | None = None,
    ) -> ExecutionResult:
        # A session at the TERMINAL degradation level raises before any intent
        # evaluation. Check the time-to-live first so an idle session sitting at
        # LOCKED eventually advances to TERMINAL.
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
            # Re-mint on read-back, per value: a value persisted to memory and
            # re-read is re-marked as tainted — writing to memory does not launder
            # it. We do not assume in-session memory is clean. Register each memory
            # fragment in the per-value ledger so a sink later carrying a
            # memory-derived value is gated at the value level.
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
                    token_estimate=estimate_tokens(self._agent_def.personality),
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

        # Explicit policy resolution: per-call policy= wins over the session's
        # default_policy. With either set, the classifier is bypassed entirely.
        if policy is None:
            policy = self._default_policy
            if policy is None and self._mode == ExecutionMode.PRODUCTION \
                    and not self._classifier_policy_warned:
                _log.warning(
                    "session=%s PRODUCTION mode is deriving policy from task "
                    "classification (advisory, content-based). Recommended: pass "
                    "an explicit policy= per call or default_policy= at "
                    "construction to make policy operator-defined.",
                    self._session_id,
                )
                self._classifier_policy_warned = True

        # Adaptive policy: re-classify each turn; capability surface can only
        # narrow automatically — broadening requires an explicit operator override.
        # Narrowing is confidence-gated: a low-confidence re-classification must
        # not permanently strip capability from the whole session (classification
        # is advisory; its errors have to stay cheap). Recovery from an
        # over-narrow start is per-tool via escalate_policy, not re-broadening.
        effective_policy = policy
        if policy is None:
            signal, signal_event = await self._analyzer.analyze(task)
            new_policy = self._selector.select(signal)
            if self._operator_escalation is not None:
                # Escalation ceiling is operator authority, not classifier
                # output — stamp it onto whatever preset was selected.
                import dataclasses as _dc
                new_policy = _dc.replace(
                    new_policy, escalation_policy=self._operator_escalation
                )
            # Custom analyzers may return no event; treat absent confidence as
            # authoritative (legacy behaviour) — the stock TaskAnalyzer always
            # reports one. The ambiguity decision itself belongs to the
            # analyzer (single source), never re-derived here.
            confidence = getattr(signal_event, "confidence", None)
            threshold = getattr(
                self._analyzer, "ambiguity_threshold", _FALLBACK_AMBIGUITY_THRESHOLD
            )
            confident = confidence is None or confidence >= threshold
            if self._active_policy is None:
                # An AMBIGUOUS classification is applied to this turn only —
                # it must not become the session's irreversible adaptive
                # baseline. The baseline is set by the first confident
                # classification (which may be broader than an earlier
                # ambiguous guess: nothing was locked by it).
                if confident:
                    self._active_policy = new_policy
                    effective_policy = self._active_policy
                else:
                    _log.info(
                        "session=%s ambiguous first classification "
                        "(confidence %.2f < %.2f): policy %s applies to this "
                        "turn only, adaptive baseline not set",
                        self._session_id, confidence, threshold, new_policy.name,
                    )
                    effective_policy = new_policy
            elif confident:
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
            else:
                _log.info(
                    "session=%s adaptive narrowing skipped: classification "
                    "confidence %.2f < %.2f (policy stays %s)",
                    self._session_id,
                    confidence,
                    threshold,
                    self._active_policy.name,
                )
                effective_policy = self._active_policy

        node = self._make_node(self._context_manager)
        result = await node.run(
            raw_state=raw_state,
            extension_bundle=self._registry.current_bundle(),
            override_policy=effective_policy,
            cancel_token=cancel_token,
        )
        self._active_token = None

        # record tokens in session-level tracker. Register lineage first so
        # depth-/subtree-aware accounting is correct rather than warn-and-default to
        # depth=0/parent=None. The top-level node is depth 0, no parent. Idempotent.
        self._tracker.register_node(result.node_id, None, 0)
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
        Forwards the signal to a telemetry-only watcher if one is attached; it
        never mutates governance state. Failures are logged and swallowed —
        the governance path is not interrupted.

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

    def _record_invocation(self, tool: str, args: dict, executed: bool) -> None:
        """Append a resolved tool call to the session's invocation buffer.

        Wired into the intent loop only when a ``session_sink`` is attached. Audit
        only — never gates execution, and must not raise into the governance path.
        """
        try:
            self._tool_invocations.append(
                ToolInvocationRecord(tool=tool, args=dict(args), executed=executed)
            )
        except Exception:
            _log.debug("session: failed to record invocation tool=%s", tool, exc_info=True)

    def _build_session_audit_record(self) -> "SessionAuditRecord":
        """Aggregate the closed session into the record sentinel consumes.

        ``event_kinds`` and ``taint_sources`` are de-duplicated *value* strings drawn
        from the session's decision traces; ``taint_active`` is the session-wide
        taint shadow. ``source_class`` is left empty — core does not attest an
        authenticated actor class today, so sentinel keys mitigation on ``agent_id``.
        """
        event_kinds: list[str] = []
        taint_sources: list[str] = []
        seen_kinds: set[str] = set()
        seen_sources: set[str] = set()
        for trace in self.all_traces():
            for ev in getattr(trace, "events", ()):
                kind = getattr(ev.kind, "value", None) or str(ev.kind)
                if kind not in seen_kinds:
                    seen_kinds.add(kind)
                    event_kinds.append(kind)
                src = getattr(ev, "taint_source", "")
                if src and src not in seen_sources:
                    seen_sources.add(src)
                    taint_sources.append(src)
        any_tainted, _ = self._taint_engine.session_shadow()
        return SessionAuditRecord(
            session_id=self._session_id,
            agent_id=self._agent_def.name if self._agent_def is not None else "",
            started_at=self._started_at,
            taint_active=bool(any_tainted),
            taint_sources=tuple(taint_sources),
            event_kinds=tuple(event_kinds),
            tool_invocations=tuple(self._tool_invocations),
            source_class="",
        )

    async def _emit_session_record(self) -> None:
        """Hand the closed-session record to the sink. Must not raise — a failing
        observer must never disturb the governance path."""
        if self._session_sink is None or self._record_emitted:
            return
        self._record_emitted = True
        try:
            record = self._build_session_audit_record()
            await self._session_sink.on_session_closed(record)
        except Exception:
            _log.warning(
                "session: session_sink.on_session_closed failed session=%s",
                self._session_id, exc_info=True,
            )

    async def aclose(self) -> None:
        """
        Close session-scoped resources: trace JSONL file, telemetry pipeline,
        memory provider. Idempotent. Safe to call even if start() was never
        invoked.

        Emits the closed-session audit record to ``session_sink`` first (while the
        trace collector is still readable), then tears resources down.
        """
        await self._emit_session_record()
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
            escalation_callback=self._escalation_callback,
            context_manager=context_manager,
            budget_engine=self._budget_engine,
            trace_collector=self._collector,
            trace_config=self._trace_config,
            child_executor=self._child_executor,
            taint_engine=self._taint_engine,
            degradation_engine=self._degradation_engine,
            consequence_overrides=self._consequence_overrides,
            positional_sinks=self._positional_sinks,
            egress_sinks=self._egress_sinks,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
            imperative_sinks=self._imperative_sinks,
            benign_tools=self._benign_tools,
            driving_args=self._driving_args,
            trajectory_observers=self._trajectory_observers,
            invocation_recorder=(
                self._record_invocation if self._session_sink is not None else None
            ),
            require_egress_allowlist=self._require_egress_allowlist,
            require_tool_roles=self._require_tool_roles,
            value_policies=self._value_policies,
            adjudicator=self._adjudicator,
            federation_gateway=self._federation_gateway,
            admission=self._admission,
            context_taps=self._context_taps or None,
            agent_id=self._agent_def.name if self._agent_def is not None else "",
            per_node_degradation=self._per_node_degradation,
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
