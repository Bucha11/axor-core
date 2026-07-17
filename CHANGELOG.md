# Changelog

## Unreleased

### Added — authority/plan separation (RFC)

Task classification may optimize execution, but no longer determines agent
authority.

- **`AuthorityPolicy` / `ExecutionPlan`** (`contracts/authority.py`,
  `contracts/planning.py`): the trusted and advisory halves of the legacy
  `ExecutionPolicy`, with `split_legacy_policy` / `merge_to_legacy_policy`
  round-tripping every shipped preset byte-identically.
- **`ExecutionEnvelope.authority` / `.plan`**: enforcement consumers read
  authority; context/budget consumers read the plan. Two import-linter
  contracts pin the boundary.
- **`axor_core.planning`**: `ExecutionPlanner` protocol,
  `HeuristicExecutionPlanner`, plan presets (`local`/`component`/
  `repository`/`neutral`), `PlanComposer` (budget-bounded, non-monotonic).
- **Session API**: `GovernedSession(authority=, default_plan=, planner=)`
  and `run(authority=, plan=)`; legacy `policy=` conflicts fail fast;
  PRODUCTION warns once when authority is classifier-derived.
- **Dynamic replanning**: `request_plan_expansion` intent — plan widening
  within `ResourceBudget`, never a capability escalation. New trace kinds
  `EXECUTION_PLAN_CHANGED`, `PLAN_CONSTRAINED_BY_BUDGET`.

### Deprecated

- Classifier-selected `ExecutionPolicy` (the legacy path in
  `PolicySelector`) — authority should come from `authority=`;
  removal in the next major release.

## 0.9.2 — 2026-07-13

The multi-agent runtime layer (spec v2): labels ride in message envelopes,
boundary gates run locally on both edge ends, and inter-federation trust is a
ladder — discount, never label authority.

### Added

- **Kernel messaging (`kernel/messaging.py`, Ch.4).** New event kinds
  `NODE_SPAWNED` / `MESSAGE_SENT` / `MESSAGE_RECEIVED`; the pure sender-side
  gate `evaluate_message_send` (LOCKED admits no sends; undeclared peer edges
  fail closed) and `fold_carried_root` (a message that lost its labels re-mints
  untrusted). `replay()` folds `MESSAGE_RECEIVED` with carried labels intact and
  monotonically on cycles (A→B→A cannot launder); `replay_tree()` folds a
  multi-node trace per node — no shared governance state; size-1 degenerates to
  exactly `replay()`.
- **Runtime messaging (`node/messaging.py`).** `MessageEnvelope` (labels travel
  with the value; federation-key signable, tamper → rejected) and
  `InMemoryMessageBus` emitting kernel events on both edge ends. Peer-edge
  delivery re-derives through the trust ladder; the foreign root is kept as an
  opaque forensic ref beside the minted local root.
- **Inter-federation trust ladder (`federation/ladder.py`, Ch.1).**
  `PeerDeclaration` (undeclared = L0), `receive_foreign` (L0 full taint / L1
  attribution-only / L2 bounded discount never-to-clean, floor never
  peer-negotiable; forged assertion falls to L0 evidenced),
  `effective_root_for_sink` (critical sinks ignore discounts entirely),
  `establish_channel` (MCP-as-A2A pinned L0/L1; governance attestation —
  signed kernel+config-hash — verified at establishment, failures evidenced).
  The gateway restore path is documented as intra-keyset-only.
- **Causal subgraph (`kernel/subgraph.py`, Ch.3).** Backward provenance walk
  from an anchored claim/denial: minimal causes, roles
  (origin/conduit/container/anchor), `fault_origin`, `contained_at`,
  `federation_scope`; message hops become subgraph edges with carried labels.
- **Spawn & death (Ch.4).** `node/spawn.inherit_degradation` — opt-in per-node
  degradation (`per_node_degradation` on `GovernedSession`/`GovernedNode`):
  a child starts at max(parent level, NORMAL), narrow-or-preserve. A crashed
  child leaves a `CHILD_STALE` trace event and tightens the parent to CAUTIOUS
  (bridged to a kernel `node_stale` FACT) — absence is a fact, never a clean
  return.
- **Bridge.** `CHILD_SPAWNED` → `NODE_SPAWNED`, `CHILD_COMPLETED` →
  `MESSAGE_RECEIVED` (delegation), `CHILD_STALE` → FACT `node_stale`.

### Compatibility

- Additive only: a single node is a tree with no edges — the v0.13 single-agent
  paths are unchanged and the 0.9.1 suite passes unmodified (the platform's
  size-1 golden gate pins byte-identical behavior end-to-end).

## 0.9.1 — 2026-07-10

The kernel/platform split and one-implementation gate engine.

### Added

- **`contracts/observation.py` — the Core → Probe observation seam.**
  `SessionContextView` (structural context snapshot, incl. `taint_canaries`
  drawn from `ContextFragment.taint_mark`) and the `ContextTap` protocol
  axor-probe's `CoreContextTap` matches structurally (P-34). `GovernedSession`
  accepts `context_taps=[...]`; `GovernedNode` fires
  `node/context_observation.emit_context_view` on every context build (the
  governance hot path), so a tap sees exactly the context the agent runs
  against. Observe-only: a pure `ContextView → SessionContextView` mapper
  derives all fields from the built view, and tap failures are logged per tap,
  never raised.
- **`context/excision.py` — authority-gated context repair.** `apply_excision`
  applies an axor-probe `RepairProposal` (fragment ids cross as plain strings —
  no import edge): an `automated_policy` authority may remove only the clean
  pure-tainted `auto_excise` set; the `escalate` set (collateral/diffuse)
  requires `human_operator`/`trusted_boundary` and is otherwise deferred —
  recorded, not removed. Completes the context-healing loop: probe
  `repair.localize` → `RepairProposal` → core `apply_excision`.
- **Trust rings, machine-enforced.** Subsystems grouped into Ring 0 (kernel — the
  TCB), Ring 1 (runtime), Ring 2 (platform). An `import-linter` `kernel-purity`
  contract (`.importlinter`, run in CI) forbids the kernel from importing the
  runtime or platform, so a platform bug cannot reach a decision.
- **`policy/gates.py` — one shared gate engine.** The six stateless gates
  (consequence, value policies, SSRF, positional, carrier, per-value taint +
  confidentiality floor) are pure functions. The streaming `IntentLoop` and the
  synchronous `ToolCallGovernor` both delegate to them — the decision logic exists
  once and cannot drift.
- **Operator tool taxonomy on the session path.** `egress_sinks`,
  `untrusted_sources`, `sensitive_sources` are threaded through `GovernedSession →
  GovernedNode → IntentLoop` (previously only the standalone governor had them), so
  the main path governs a deployment's renamed tools. New integration test proves a
  declared-tool exfiltration is blocked end-to-end.
- **Kernel-only import bypass.** Lazy package `__init__` (PEP 562): `from axor_core
  import ToolCallGovernor` loads Ring 0 and nothing from the runtime or platform;
  `GovernedSession` lazy-loads the full stack. Regression-tested.
- **Robust identifier tokenisation** in the taint ledger: case-fold + structural
  delimiter splitting + Unicode NFKC/zero-width normalisation, so an attacker
  address written `mailto:x@y.z` / `'x@y.z'` / `cc=x@y.z;` / fullwidth / split by an
  invisible char still matches the clean form a model extracts. Exhaustive
  evasion-surface tests; documented residuals (encoding, sub-12-char shredding,
  cross-script homoglyphs, semantic paraphrase) marked strict-xfail.
- **Declarative `GovernanceConfig` (YAML)** with fail-closed parsing and key
  material by reference only (`*_env` / `*_file`), `profiles` presets, and the
  synchronous `ToolCallGovernor` for framework-owned agent loops.
- **Federation A2A**: signed provenance receipts (HMAC default, optional ed25519),
  the L1/L2 trust ladder, replay defence (per-(peer, nonce) cache, pruned by
  expiry, receiver-side TTL clamp, legacy-receipt reject), and value-hash binding.
- **STRICT mode obligations**: every egress sink needs a destination allowlist,
  every tool needs a declared data-flow role, and an egress sink that narrows its
  taint check to `driving_args` must carry its allowlist on the driving arg.
- **Extension points**: tightening-only `TrajectoryObserver` and the advisory,
  projection-only adjudicator.

### Changed

- `IntentNormalizer` moved `node/normalizer.py → policy/normalizer.py` (Ring-0
  structural classification; it never belonged in the runtime ring).
- The role→provenance output mapping is now a single shared `policy/provenance.py`
  (`output_root`), used by both enforcement paths instead of being duplicated.
- `ValueProvenance` now includes `confidentiality_floor_active`: the kernel gates
  confidentiality on the sound floor via the contract, not a silent fallback.
- Escalation grants, capability leases, and the flood guard moved into
  `node/escalation.py` (`EscalationManager`); the intent loop delegates instead of
  implementing them inline.
- spawn's carrier check routes through the shared `policy.gates.carrier_gate`
  instead of an ad-hoc reimplementation.
- Removed the unused `AnomalyDetector` / `LLMVerifier` protocols (their only
  implementer is retired); the sentinel-facing detection surface is unchanged.

### Security

- `restore_root` degrades an unknown federated source label to an untrusted
  re-mint instead of silently restoring a clean root (under-taint fix).
- Trace filenames are derived from a sanitised session-id stem (path-injection
  guard).
- Lease use / grant op is consumed only when a call is finally approved, so a call
  denied by a later data-flow gate no longer burns a use.
- The confidentiality-floor map is bounded: a flood of distinct secret reads flips
  a sticky fail-closed flag instead of growing memory without bound.

## 0.8.0 — 2026-06-10

The v4.12 governance pass — per-value provenance becomes the enforcement spine, with
a stratified soundness story across the integrity and confidentiality axes.

### Added

- **Per-value enforcement (TM2).** Sinks decide on the driving argument's own
  `CausalRoot` via the `ValueProvenance` contract, not a session-wide flag. The
  content-derivation `ValueTaintLedger` is refcounted (no endorsement over-release)
  and fails closed on flood saturation (no silent fragment drop).
- **Confidentiality sound floor (TM4, 1.1b).** Egress is gated on the *fact* of a
  sensitive read, independent of the egress value's content (paraphrase-proof);
  released only by governance endorsement.
- **D_high positional admission (Corollary / 1.1c).** Operator-declared
  instruction-incomplete sinks admit only via a positional carrier check
  (content-independent), closing the O2/paraphrase gap on that partition. Exec-class
  sinks cannot be lifted (lift ban).
- **Consequence axis (TM3.1) wired through the gate**, incl. power-state shell
  command detection (X5 OpenClaw).
- **Thm. 0 decidability classifier** (`kernel/decidability.py`) wired into value-policy
  registration; K5/T4 fuzz-floor for the path and carrier classifiers.
- **Persistence / cross-process re-mint (TM3.2 / TM4.1):** memory read-back and a
  child's returned output are re-minted untrusted.
- **Advisory adjudicator (TM3.4):** projection-only, memoized by π-hash,
  tightening-only.
- **Opt-in detection→degradation (TM7.1):** a decidable reputation
  threshold-crossing may tighten degradation; per-tenant isolated.
- **Federation L1/L2 (TM4.2):** authenticated peers with HMAC provenance receipts;
  L2 restores provenance, incompatible kernels/domains degrade to L1, forged
  receipts are denied.

### Fixed

- SSRF host classification across all obfuscated notations; extension-override and
  deployment-overlay escalations; spawn taint gate; degradation clearance now resets
  quarantine; budget `register_node` wiring and `restrict_export` enforcement;
  carrier classifier no longer admits paths/URLs/`Infinity`/`NaN`; null-byte paths
  fail closed; sanitizer reserved-prefix match is case-insensitive; symbol
  deprecation match is whole-word.

### Known gaps

- **X1** in-process-LLM implicit flow on the D_low integrity partition remains open
  (documented as strict `xfail`); closed only by a CaMeL trust-model backend.

## 0.5.0 — 2026-05-25

### Added

- **`DegradationEngine`** (`axor_core/degradation/`) — source-aware session degradation
  state machine. Converts accumulated taint and denial signals into a monotonically
  increasing `DegradationLevel` that progressively narrows the capability surface.
  Unlike a simple global counter, degradation is per-source: one malicious document
  quarantines its origin while clean sources remain at full capability until session-level
  thresholds are crossed.

  Levels: `NORMAL → CAUTIOUS → RESTRICTED → LOCKED → TERMINAL`.

  Key behaviours:
  - Any cross-origin export denial (`destination_kind=external_domain/private_network`)
    escalates to `LOCKED` immediately.
  - `source.tool_pressure_count >= 2` or `source.instruction_pressure_count >= 1`
    quarantines the source and escalates to `RESTRICTED`.
  - `session_deny_count >= 5` escalates to `LOCKED`.
  - `LOCKED` for `LOCKED_TTL` seconds (default 300s) without human clearance
    auto-escalates to `TERMINAL`.

- **`DegradationPolicy`** — configurable thresholds dataclass
  (`SOURCE_TOOL_PRESSURE_THRESHOLD`, `SOURCE_INSTR_PRESSURE_THRESHOLD`,
  `SESSION_DENY_THRESHOLD`, `LOCKED_TTL`).

- **`GovernanceAuthority`** — authority object required for `clear_by_governance`.
  Worker-path clear raises `DegradationClearanceError`.

- **`SessionTerminatedError`**, **`DegradationClearanceError`** — new exception types
  in `axor_core.errors.exceptions`.

- **`DegradationTransitionEvent`**, **`SourceQuarantinedEvent`** — two new
  `TraceEventKind` values (`DEGRADATION_TRANSITION`, `SOURCE_QUARANTINED`) with
  typed event dataclasses. Emitted into `DecisionTrace` on every level change
  or quarantine event.

- **`IntentLoop`** integration — `degradation_engine` optional parameter. Pre-cascade
  degradation check enforces `apply_to_policy` narrowing before Layer 1.
  `record_signal` is called after every cascade outcome (pass or deny) to update
  engine state for the next intent.

- **`GovernedSession`** — constructs `DegradationEngine` on init; checks
  `TERMINAL` level at the top of `run()` and raises `SessionTerminatedError`.

- **`GovernedNode`** — passes `degradation_engine` to child nodes (shared instance,
  enforcing the D-6 child floor invariant).

- **Security invariants D-1 through D-7** added to
  `tests/invariants/test_security_invariants.py` (14 new tests).

- **Adversarial test file** `tests/adversarial/test_degradation.py` — 14 tests
  covering monotonicity, source isolation, cross-origin export, LOCKED_TTL
  auto-terminal, and child floor inheritance.

### Tests

- 468 tests pass (was 440 at 0.4.0). +28 new tests.

## 0.4.0 — 2026-04-29

### Added
- `axor_core.policy.keyword_relevance` — provider-agnostic relevance
  scorer for tool selection. Adapters use it to gate tools by domain
  alignment with the latest user task without any hand-curated allowlist.
  Public exports: `extract_query_keywords`, `compute_topic_strength`,
  `expand_with_synonyms`, `tool_topics`, `name_has_destructive_token`,
  `score_tool_relevance`.
- `axor_core.policy.topics` — domain vocabulary (topics, stopwords,
  destructive tokens, one-hop topic implications). Pure data, no runtime
  dependencies. Powers `keyword_relevance`.
- `TokenCostRates` re-exported from `axor_core` so adapters can pass
  per-model pricing into `GovernedSession`. Defaults to Anthropic-style
  cache multipliers (cache write `1.25 × input`, cache read `0.1 × input`).

### Changed
- **Federation invariant strengthened in `PolicyComposer`.** Children
  are now forced to be most-restrictive across `child_mode`,
  `context_mode`, `compression_mode`, `child_context_fraction`,
  `allowed_passthrough_commands`, and `allow_model_switch` — not only
  `child_mode=DENIED`. A parent with `SHALLOW` correctly demotes a
  child requesting `ALLOWED`.
- Extension override `allow_search` now requires the base policy to also
  permit search (parity with `allow_bash` / `allow_write`).
- `HeuristicClassifier` no longer raises at import time when
  `heuristic_coefficients.json` is missing or malformed — falls back to
  empty patterns so the rest of the package stays usable.
- Per-pattern source-length cap (`_MAX_REGEX_SOURCE = 1024`) to keep
  classifier import predictable and pre-empt ReDoS-amplification.

### Tests
- New: `tests/policy/test_keyword_relevance.py`, `tests/budget/test_budget.py`.
- 166 tests pass (was 160 at 0.3.0).

## 0.3.0 — 2026-04-23

### Added
- Phase-0 telemetry instrumentation.
- `SignalChosenEvent.scores` carries the full classifier distribution
  (not just the winner).
- `axor_core/policy/heuristic_coefficients.json` — externalised regex
  patterns and weights, refreshable from anonymized telemetry.
- `TraceCollector` streams JSONL per session and flushes on
  `record_many`.
- `GovernedSession.aclose()` closes the trace file and the memory
  provider.
- Whitelist-based event serialization for anonymized records
  (fail-closed for new fields).
- `AnonymizedTraceRecord.input_embedding` made optional plus
  `fingerprint_kind`.

### Changed
- `AnonymizedTraceRecord` field order changed (keyword-only callers
  unaffected).

## 0.2.0 — 2026-04-23

### Added
- `ContextFragment.turn` field for per-turn provenance.
- `BudgetPolicyEngine.record_child_tokens()` — public API for parent
  accounting of child node usage (replaces direct access to
  `BudgetTracker._tracker`).

### Changed
- `ContextManager` / `ContextCompressor` / `ContextSelector` aligned
  with the new contracts.
- Node wrapper and memory contract adjustments.

## 0.1.0 — 2026-04-14

Initial release.

### Added
- Core governance kernel.
- Dynamic policy selection (7-policy matrix: complexity × nature).
- `ContextManager` foundation.
- `ToolResultBus` for the async tool loop.
- Federation via `spawn_child` with `child_executor`.
- `CancelToken` cooperative cancellation (5 reasons).
- `BudgetPolicyEngine` (60/80/90/95% thresholds).
- `TraceCollector` with lineage (17 event kinds).
