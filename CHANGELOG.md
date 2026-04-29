# Changelog

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
