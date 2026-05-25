# Changelog

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
