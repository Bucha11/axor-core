# Architecture

## Central Primitive

Everything in Axor is a `GovernedNode` — a boundary that wraps any executor and enforces a policy envelope. Flat execution is `max_child_depth=0`. Federation is `max_child_depth=N`. There is no special case for single-agent vs multi-agent.

```
depth=0   GovernedNode(executor)              ← flat execution

depth=1   GovernedNode(executor)              ← one level of children
              └── GovernedNode(executor)

depth=N   GovernedNode(executor)              ← governed federation
              ├── GovernedNode(executor)
              └── GovernedNode(executor)
                    └── GovernedNode(executor)
```

---

## Execution Pipeline

```
raw input
  → TaskAnalyzer          → TaskSignal (complexity × nature × domain)
  → PolicySelector        → ExecutionPolicy (dynamic, per task)
  → PolicyComposer        → final policy (base + extensions + parent ceiling)
  → ContextManager        → ContextView (scoped, compressed, cached)
  → EnvelopeBuilder       → ExecutionEnvelope (the governed context object)
  → BudgetPolicyEngine    → pre-execution budget check
  → IntentLoop            → stream interception
      tool_use event      → degradation pre-check (narrow the effective policy)
                          → capability → consequence → value policies → degradation
                          → positional → carrier → per-value taint + confidentiality
                            floor → adjudicator → execute or deny
                          → register the output's provenance; record a degradation
                            signal; emit a transition event if the level changed
      spawn_child event   → _handle_spawn_child via SpawnCallback (child as a node)
      tool result         → federation ingress (peer value? gateway decides) →
                            ToolResultBus → back to executor
      text event          → ExportFilter
      cancel check        → CancelToken at every event boundary
  → ExportFilter          → governed ExecutionResult
  → TraceCollector        → decision lineage
  → ContextManager.update → result persisted into session context
```

The executor receives an `ExecutionEnvelope` — never raw context, never an unfiltered tool list.

---

## Module Tree

```
axor_core/
├── contracts/             Pure contracts — no business logic, no side effects
│   ├── mode.py            ExecutionMode (LIBRARY | PRODUCTION | STRICT)
│   ├── envelope.py        ExecutionEnvelope — governed context passed to executor
│   ├── invokable.py       Invokable — stream(envelope) → AsyncIterator[ExecutorEvent]
│   ├── cancel.py          CancelToken — cooperative cancellation, 5 reasons
│   ├── intent.py          Intent, IntentKind, ResolvedIntent
│   ├── anomaly.py         NormalizedIntent, CanonicalizedIntent
│   ├── policy.py          ExecutionPolicy, TaskSignal, SignalClassifier
│   ├── result.py          ExecutorEvent, ExecutorEventKind, ExecutionResult
│   ├── trace.py           DecisionTrace, TraceConfig, 30 typed TraceEvent kinds
│   ├── taint.py           Carrier, TaintSource, TaintState (legacy), ClearanceRecord
│   ├── provenance.py      ValueProvenance — the per-value trust-model interface
│   ├── degradation.py     DegradationLevel, DegradationPolicy, DegradationState,
│   │                      DegradationTransition, SourceRecord, GovernanceAuthority
│   ├── agent.py           AgentDefinition, AgentDomain, TrustLevel
│   ├── memory.py          MemoryProvider, MemoryFragment, FragmentValue, MemoryQuery
│   ├── context.py         ContextFragment, ContextView, LineageSummary
│   └── extension.py       ExtensionLoader, ExtensionBundle, ExtensionFragment
│
├── node/                  Governance boundary around executor
│   ├── wrapper.py         GovernedNode — central primitive, wires all subsystems
│   ├── envelope.py        EnvelopeBuilder — assembles ExecutionEnvelope
│   ├── intent_loop.py     IntentLoop — stream interception, ToolResultBus, SpawnCallback,
│   │                                   DegradationEngine pre-check + signal recording
│   ├── canonicalizer.py   IntentCanonicalizer — strips raw strings before any advisory/detection layer
│   ├── normalizer.py      IntentNormalizer — provider format → NormalizedIntent
│   ├── spawn.py           ChildSpawner — _validate_child_policy(), SpawnValidationError
│   └── export.py          ExportFilter — SummarizationMembrane, export contracts
│
├── capability/            Tool permission derivation and execution
│   ├── resolver.py        CapabilityResolver — fail-closed: unknown prefix = denied
│   ├── executor.py        CapabilityExecutor, ToolHandler
│   ├── locked.py          LockedExecutor — GovernanceBypassError if bypassed
│   ├── daemon_client.py   DaemonCapabilityClient — out-of-process tool execution
│   └── lease_validator.py LeaseValidator — TTL, max_uses, normpath path check
│
├── taint/                 Per-value provenance
│   ├── causal_root.py     CausalRoot — a value's own integrity/confidentiality labels
│   ├── ledger.py          ValueTaintLedger — refcounted content-derivation, fail-closed
│   ├── engine.py          TaintEngine — register/derive per value, confidentiality floor,
│   │                      governance release (per-value endorse / wholesale clear)
│   └── density.py         the per-value vs session-sticky measurement (observe-only)
│
├── kernel/                Decidability + the advisory layer
│   ├── decidability.py    classifies a sink field as predicate-guardable vs fuzz-required
│   ├── registration.py    validates value policies against that classification
│   └── adjudicator.py     advisory adjudicator — projection-only, memoized, tightening
│
├── federation/            Agent-to-agent trust
│   ├── signing.py         Signer/Verifier — HMAC (default) and ed25519 (optional)
│   ├── receipt.py         signed provenance receipts; mint / verify / restore
│   ├── gateway.py         the L1/L2/deny decision for an incoming peer value
│   ├── value.py           FederatedValue — the wrapped (value, receipt) transport type
│   └── transport.py       reference wire (de)serialization + in-memory peer network
│
├── degradation/           Session degradation state machine
│   └── engine.py          DegradationEngine — fact-driven, monotone level transitions,
│                          apply_to_policy narrowing, optional reputation-crossing tighten
│
├── policy/                Policy selection + consequence axis + value-policy predicates
│   ├── heuristic.py       HeuristicClassifier — rule-based, 0ms, 0 tokens
│   ├── analyzer.py        TaskAnalyzer — heuristic + domain detection + external classifier
│   ├── selector.py        PolicySelector — TaskSignal → ExecutionPolicy (7-policy matrix)
│   ├── composer.py        PolicyComposer — parent restrictions always applied to child
│   ├── consequence.py     consequence axis — content-blind action-class classification
│   ├── value_policy.py    value-policy predicates (numeric range / enum) on arguments
│   ├── presets.py         readonly, sandboxed, standard, federated, research, support, analysis
│   ├── topics.py          WORD_TOPICS, SYNONYM_MAP, TOPIC_IMPLICATIONS, DESTRUCTIVE_TOKENS
│   └── keyword_relevance.py Tool relevance scoring — extract_query_keywords, score_tool_relevance
│
├── context/               Context management — session-scoped, persists across turns
│   ├── manager.py         ContextManager — build / pin_fragment / add_knowledge
│   ├── cache.py           ContextCache — content hash cache, tool result memoization (TTL)
│   ├── compressor.py      ContextCompressor — FragmentValue-aware, 11 waste categories
│   ├── selector.py        ContextSelector — relevance scoring, working set management
│   ├── invalidator.py     ContextInvalidator — git TTL, symbol drift detection
│   ├── symbol_table.py    SymbolTable — live symbol registry, rename detection, TODOs
│   └── lineage.py         LineageManager — child context slice derivation
│
├── budget/                Token optimization
│   ├── tracker.py         BudgetTracker — full spawn tree accounting, thread-safe
│   ├── estimator.py       BudgetEstimator — cost estimates, slice sufficiency checks
│   └── policy_engine.py   BudgetPolicyEngine — 60/80/90/95% thresholds
│
├── trace/                 Governance decision recording
│   ├── collector.py       TraceCollector — lineage-aware, thread-safe, privacy controls
│   └── events.py          Typed event constructors for all TraceEvent kinds
│
├── extensions/            Extension loading and sanitization
│   ├── sanitizer.py       ExtensionSanitizer — size cap, reserved command protection
│   └── registry.py        ExtensionRegistry — session-scoped active extensions
│
├── worker/                Entry layer
│   ├── session.py         GovernedSession — STRICT enforcement on init, personality injection
│   ├── commands.py        SlashCommandRouter — GOVERNANCE | CONTEXT | PASSTHROUGH
│   └── dispatcher.py      Dispatcher — routes input to node flow
│
└── errors/                Explicit error hierarchy rooted at AxorError
    └── exceptions.py      GovernanceBypassError, TaintClearanceError, SpawnValidationError,
                           DegradationClearanceError, SessionTerminatedError,
                           NormalizerError, UnknownProviderFormatError, BudgetExceededError
```

---

## Trust Rings

The subsystems are grouped into three rings by how load-bearing they are for a
correct decision. The test for each: *can a bug here cause a wrong ALLOW?*

| Ring | Role | Subsystems |
|---|---|---|
| **0 — kernel** | the trusted computing base: gate logic + the data it reasons over | `contracts`, `errors`, `taint`, `security`, `policy` (incl. `gates`, `normalizer`), `kernel`, `degradation` |
| **1 — runtime** | wires the kernel to an executor: capability boundary, node/worker orchestration, opt-in federation | `capability`, `node`, `worker`, `federation` |
| **2 — platform** | quality / cost / observability — a bug wastes resources, not safety | `budget`, `context`, `trace`, `extensions` |

The one guarantee that matters and is **machine-enforced**: the kernel must not
depend on the runtime or the platform, so a platform bug cannot reach the
decision. This is checked in CI by `import-linter` (the `kernel-purity` contract
in `.importlinter`); a kernel module importing a runtime/platform module fails the
build. The reverse edge is allowed — the runtime legitimately orchestrates
platform services (e.g. `node` uses `budget`/`context`/`trace`).

The same six stateless gates run on both enforcement paths: the streaming
`IntentLoop` and the synchronous `ToolCallGovernor` both delegate to
`policy/gates.py`, so the decision logic exists once and cannot drift between them.

**Kernel-only bypass.** The package uses lazy imports (PEP 562), so you pay only
for the rings you touch. `from axor_core import ToolCallGovernor` loads the Ring-0
kernel and nothing from the runtime or platform — a caller that owns its own agent
loop gets the gate engine without the orchestration, budget, context, or trace
machinery. `from axor_core import GovernedSession` pulls the full stack on demand.
The guarantee is regression-tested (`tests/test_kernel_only_import.py`).

---

## Design Invariants

**Everything is a GovernedNode.** Flat execution is `depth=0`. No special cases for single-agent vs multi-agent.

**Core never imports providers.** Zero provider SDK imports in `axor_core/`. Verified by static analysis in CI.

**Policy meaning belongs to core.** Adapters translate envelopes — they never define governance semantics.

**Executors never self-assign capabilities.** Always derived from policy by `CapabilityResolver`. Fail-closed: unknown tool prefix = denied.

**A structural deny is final.** Gates run in a fixed order and a deny short-circuits the rest; an advisory adjudicator is consulted only on the would-approve path and can only add a deny.

**Child policy cannot exceed parent ceiling.** `_validate_child_policy()` and `PolicyComposer` both enforce this. No path through the codebase allows a child to exceed parent capabilities.

**Workers cannot clear taint.** `TaintEngine.clear_by_governance()` requires governance authority. Worker-path clear attempt raises `TaintClearanceError`.

**Degradation level is monotonically increasing.** `DegradationEngine` level never decreases without a `GovernanceAuthority` object passed to `clear_by_governance`. Worker-path clear attempt raises `DegradationClearanceError`. A `TERMINAL` session raises `SessionTerminatedError` before any intent is evaluated.

**Degradation is source-aware.** A quarantined source loses write/bash/export capability; clean sources in the same session are unaffected until session-level thresholds are crossed. `apply_to_policy` for a quarantined source always returns `export_mode=RESTRICTED`.

**Child nodes share the parent's DegradationEngine instance.** Children cannot start below the parent's current `DegradationLevel` — the shared instance acts as the floor.

**Canonical intent contains no raw strings.** `IntentCanonicalizer` strips all raw strings before they reach any advisory or detection layer. Verified by schema-injection adversarial tests.

**Core does not decompose tasks.** Agents decide when to spawn children. Core governs each spawn and provides a minimum sufficient context slice.

**spawn_child is an intent, not a tool.** `IntentLoop` intercepts `spawn_child` tool_use events before they reach `CapabilityExecutor`. The routing is `SpawnCallback → _handle_spawn_child` — internal to core. Adapters never see federation.

**Denied spawns never crash.** `SpawnValidationError` is caught in the spawn callback and returned as a structured denial string. The executor sees a tool result, not an exception.

**Child tokens belong to parent budget.** `session.total_tokens_spent()` always includes the full spawn tree.

**Pinned fragments bypass all compression and selection.** `ContextManager._pinned_fragments` is prepended to `ContextView` after the full compress → select → scope pipeline. Compressor and selector never touch them. They are always first.

**Personality is governance-injected, not adapter-injected.** `AgentDefinition.personality` is injected as a pinned fragment by `GovernedSession` — once per session, deduplicated by source.

**Waste elimination always runs.** Compression mode controls aggressiveness, not whether optimization happens. `LIGHT` mode still deduplicates, collapses repeated errors, and normalizes paths.

**Context policy is per-turn.** `ContextManager.build(raw_state, lineage, policy=policy)` receives the actual policy selected for each task. A `rewrite repo` task gets `BROAD` context with `LIGHT` compression. A `write test` task gets `MINIMAL` context with `BALANCED` compression.

**Privacy by default.** `TraceConfig(local_only=True, persist_inputs=False)`. Nothing leaves the machine without explicit `training_opt_in=True`.

---

## Defense System

### Per-Intent Gate Sequence (Core)

Every tool call that reaches `IntentLoop` passes through a fixed sequence of structural gates. A deny at any gate is final — subsequent gates do not run.

```
IntentLoop receives tool_use event
  │
  ├─ degradation pre-check — narrow the effective policy for a degraded session
  │    (locked → read/escalate only; quarantined source → no write/bash/export)
  │
  ├─ 1. capability       — is the tool allowed by the active policy at all?
  ├─ 2. consequence      — action-class irreversibility (content-blind; shutdown/wipe)
  ├─ 3. value policies   — decidable predicates on arguments (amount in range, enum)
  ├─ 4. degradation      — refuse a call a quarantined source would drive
  ├─ 5. positional       — declared sinks: admit only instruction-incomplete forms
  ├─ 6. carrier          — untrusted free text into an instruction-following sink?
  ├─ 7. per-value taint  — integrity (untrusted → high-risk) + confidentiality floor
  ├─ 8. adjudicator      — optional advisory second opinion (tightening-only)
  │
  ├─ execute → register the output's provenance (federation ingress decides the
  │            provenance of a value returned from a peer agent)
  │
  └─ record a degradation signal — fact-driven, monotone:
       NORMAL → CAUTIOUS → RESTRICTED → LOCKED → TERMINAL, cleared only by governance
```

Detection (reputation, drift) is **not** a gate — it is observe-only and may, opt-in, only *tighten* degradation. See [governance-model.md](governance-model.md).

`DegradationEngine` is source-aware. A quarantined source (malicious document) is restricted; clean sources in the same session continue at reduced but functional capability. Only when session-level thresholds are crossed does the whole session degrade.

---

### Cross-Session Reputation (Sentinel)

`axor-sentinel` solves a problem the per-session gates cannot: slow-and-low staging attacks, where exfiltration is distributed across many individually normal sessions over days or weeks.

**Background audit cycle — `SentinelCycle`** (runs hourly, not on the hot path):

```
sessions (axor-core) → SentinelCycle
  ├─ apply time decay           score halves every 30 days
  ├─ apply hot weights          READ=0.4 / READ_SUMMARIZE=0.6 /
  │                             EXPORT_ADJACENT=0.8 / EXPORT_FAILED=1.0
  ├─ apply caution weights      adjacent resources: BASE_CAUTION=0.3 × topology_factor
  ├─ fanout detection           z-score vs agent historical baseline; fires at z > 2.5
  └─ snapshot swap              atomic write: symlink rename / os.replace
```

Score accumulation uses logarithmic diminishing returns — each signal contributes proportionally to remaining headroom, bounding scores to `[0, 1]`:

```
new_score = current + new_weight × (1 − current)
```

**Hot-path enrichment — `SnapshotIntentEnricher`** (pure dict lookup, no Neo4j, no I/O):

```
ReputationSnapshot (loaded at startup)
  │
  ▼
SnapshotIntentEnricher.enrich(intent)
  └─ populates NormalizedIntent.target_resource_reputation
       │
       ▼
  IntentLoop Phase 1 check:
    score ≥ 0.8 AND after_external_read → deterministic deny
    decision recorded before any LLM inference
```

The snapshot is swapped atomically in the background. The hot path reads a flat dict — no lock contention, no graph queries per intent.

---

### Behavioral Drift Detection (Probe)

`axor-probe` detects context-induced reasoning drift during a running session by comparing an out-of-band snapshot probe instance against an isolated shadow baseline. The live session is never touched.

**Pipeline:**

```
Live session (axor-core)
  │
  │  read-only snapshot at trigger (context-growth / external-content / manual)
  ▼
StateSnapshot               session_id, timestamp, bounded context_window,
                            system_prompt_hash, CanonicalizedContextSummary
                            — no memory write handles, no tool executor refs
  │
  ├───────────────────────────────────────┐
  ▼                                       ▼
ProbeExecutor                       ShadowInstance
snapshot probe instance             receives CanonicalizedContextSummary only
(bounded context window)            (no raw live-session strings)
  │                                       │
  └────────────────┬───────────────────────┘
                   ▼
              Comparator
              BINARY (default): deterministic field comparison
                decision / disclosed_data / policy_cited / reasoning
              TRIANGULATED (on ambiguity): adds contextless shadow baseline
                snapshot ≠ shadow ≠ baseline → drift signal
                snapshot ≈ shadow, baseline differs → context explains behavior
                   │
                   ▼
              DriftSignal + longitudinal accumulation (DriftAccumulator)
              DriftAction: LOG_ONLY / ELEVATED_REVIEW / RESTRICTED_MODE*
              * RESTRICTED_MODE only when calibration_status == "CALIBRATED"
                   │
                   ▼
              PayloadRedactor → ProbeAuditStore
              reasoning field scrubbed; unknown fields redacted by default
```

**Trigger coalescing — `ProbeController`:**

Multiple triggers inside `cooldown_window_seconds` coalesce into one dispatch. Fixed-interval probing is intentionally avoided. `max_probes_per_session` hard cap enforced.

**Integration direction is one-way:**

```
axor-probe → axor-core      (notify_core — ELEVATED_REVIEW or calibrated RESTRICTED_MODE)
axor-probe → axor-sentinel  (emit_to_sentinel — contributes to reputation graph)
axor-core  -X→ axor-probe   (core does not depend on probe)
```

`LOG_ONLY` signals remain local telemetry. Core and sentinel only receive significant signals.

**Key design constraint:** the shadow instance receives a `CanonicalizedContextSummary` (structural metadata only), never raw live-session content. This eliminates cross-contamination but introduces unavoidable context asymmetry — legitimate contextual reasoning can look like drift. Triangulation exists to reduce this false-positive class.

---

### Defense Layer Composition

```
                      ┌─────────────────────────────┐
                      │        Live Session          │
                      └──────────────┬──────────────┘
                                     │
                   ┌─────────────────┼──────────────────┐
                   ▼                 ▼                   ▼
        ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐
        │   Core gates     │  │   Sentinel   │  │     Probe     │
        │  (per-intent)    │  │  (cross-     │  │  (behavioral  │
        │                  │  │   session)   │  │   drift)      │
        │  L1 rule         │  │              │  │               │
        │  L2 anomaly      │◄─┤ reputation   │  │ shadow compare│
        │  L3 verifier     │  │ enrichment   │  │ drift signal  │
        │  DegradationEng  │  │              │  │               │
        └────────┬─────────┘  └──────┬───────┘  └──────┬────────┘
                 │                   │                  │
                 └───────────────────┴──────────────────┘
                                     │
                            independent failure
                            no single-point bypass
```

Each layer operates independently across a different time horizon:

| Layer | Scope | Latency | Signal source |
|-------|-------|---------|---------------|
| Core gates | per intent | synchronous | tool calls, spawn events, export |
| Sentinel | cross-session | hourly background | resource access patterns over time |
| Probe | per session | out-of-band | behavioral consistency under policy pressure |

A bypass of probe does not disable the enforcement gates. A gap in sentinel's reputation coverage does not disable per-intent enforcement — detection only ever observes (or, opt-in, tightens degradation); it cannot allow.

---

## Implementing an Adapter

Three components to implement:

```python
# 1. Invokable — translate ExecutionEnvelope → provider calls
from axor_core import Invokable
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

class MyProviderExecutor(Invokable):
    async def stream(self, envelope: ExecutionEnvelope):
        # envelope.task          — the task string
        # envelope.context       — ContextView (scoped, never raw)
        # envelope.capabilities  — allowed tools
        # envelope.cancel_token  — check before each yield
        # envelope.policy        — compression mode, export mode, etc.
        async for chunk in self._client.stream(
            prompt=envelope.task,
            tools=self._translate_tools(envelope.capabilities.allowed_tools),
        ):
            if envelope.cancel_token.is_cancelled():
                return
            yield self._translate_event(chunk, envelope.node_id)


# 2. ToolHandler — one per tool
from axor_core import ToolHandler

class BashHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "bash"

    async def execute(self, args: dict) -> str:
        import subprocess
        result = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr


# 3. IntentNormalizer — translate provider format → NormalizedIntent
from axor_core.node.normalizer import IntentNormalizer
from axor_core.contracts.anomaly import NormalizedIntent

class MyNormalizer(IntentNormalizer):
    def normalize(self, raw_event: dict) -> NormalizedIntent:
        ...
```

For adapters with multi-turn tool loops (like Claude), implement `get_bus()` to receive a `ToolResultBus` and push tool results back to the executor after `IntentLoop` executes them.

---

## Context Compression

`ContextCompressor` eliminates eleven categories of waste before context reaches the executor. Waste elimination always runs regardless of compression mode — mode controls aggressiveness, not whether it happens.

| Waste category | Mechanism |
|----------------|-----------|
| Verbose old assistant prose | Key decision extraction, verbose text discarded |
| Oversized command output | Smart truncation: head + tail |
| Stale git / branch history | Git TTL-based cache invalidation |
| Repeated file reads | Content hash cache — file never re-read if unchanged |
| Symbol drift | Deprecated symbols get relevance penalty |
| File rediscovery | `cached_paths()` registry prevents re-discovery |
| Unnecessary rereads | Post-execute callback → auto-cache on every read |
| Turn accumulation | Rolling summary after N turns (LIGHT=20, BALANCED=6, AGGRESSIVE=3) |
| Error repetition | Collapse repeated errors to single entry with count |
| Working set drift | Inactive files penalized by turn distance |
| Path explosion | Absolute paths normalized to relative |

`FragmentValue` controls per-fragment compression behavior:

| Value | Compressor behavior |
|-------|-------------------|
| `PINNED` | Never touched — bypasses all compression and selection |
| `KNOWLEDGE` | Dedup + collapse only — no truncation |
| `WORKING` | Normal pipeline — default |
| `EPHEMERAL` | Aggressive compression regardless of mode |

Eviction priority: `EPHEMERAL` → `WORKING` → `KNOWLEDGE` → `PINNED` (never evicted).
