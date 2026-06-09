# Reverse Osmosis: Defense Architecture

## Philosophy

Classic injection defense tries to detect malicious content at the input boundary —
sanitizing what comes in. This doesn't work reliably:

- Injections are trivially encoded, rephrased, split across turns
- Tool return values (MCP, web search, file reads) arrive after the policy check
- Filtering content means understanding semantics — a hard problem at ingestion time that doesn't scale to arbitrary encodings and phrasings

**Reverse osmosis flips the model.**

Don't filter what the agent reads. Govern what the agent *does*.
A compromised agent operating under enforced execution boundaries — where exfiltration
is blocked, child spawning is policy-governed, and writes are path-restricted — is
contained regardless of what it was told.

The behavioral signal matters, not the content signal.

---

## Trust Model

```
External World
      │
      │  tool returns (MCP, web, files, APIs)
      ▼
┌─────────────┐
│    Agent    │  ← may be infected via prompt injection
│  (executor) │    treated as untrusted after external reads
└──────┬──────┘
       │  intents (tool_call, spawn_child, export, escalate_policy)
       ▼
┌─────────────┐
│  IntentLoop │  ← the membrane
│  (axor-core)│    governs every privileged action
└──────┬──────┘
       │  NormalizedIntent → CanonicalizedIntent (no raw content)
       ▼
┌─────────────┐
│   Defense   │  ← separate process/context
│   Cascade   │    never sees raw tool returns or conversation
└─────────────┘
```

**The verifier must be isolated from the worker.**
If the worker is compromised via prompt injection, asking it to verify itself is
unreliable. The verifier receives only a canonical intent schema — the same injection
vector that reached the worker cannot trivially reach the verifier.

---

## Intent Schema

The defense cascade never sees raw content. It operates on two derived schemas.

**`NormalizedIntent`** — used by Layer 1 and Layer 2. Structural shape of the intent, no values:

```python
@dataclass(frozen=True)
class NormalizedIntent:
    # Identity
    tool: str                         # "write", "bash", "spawn_child", ...
    sequence: int                     # position in the intent stream

    # Structural args — shape only, never values
    args_schema: dict[str, str]       # {"path": "str", "content": "str"} — no content
    target_path: str | None           # normalized path if applicable

    # Causal signals — derived from intent history, not content
    preceded_by_external_read: bool   # True if recent tool return came from outside
    executes_generated_code: bool     # True if write→execute sequence detected
    cross_origin_export: bool         # True if exporting data read from external source

    # Context
    depth: int                        # node depth in federation tree
    escalation_active: bool           # True if an escalation grant is in effect
    intents_since_last_external: int  # turns since last external tool return
```

`preceded_by_external_read` is the key signal. It captures causality without
inspecting content: the agent read something external, and now it wants to act.
That pattern is the attack surface.

**`CanonicalizedIntent`** — used by Layer 3 only. All raw strings stripped by `IntentCanonicalizer`:

```python
CanonicalizedIntent(
    tool_name="read",
    path_extension=".py",         # extension only — never the full path
    arg_count=1,
    taint_state=TaintState(sources={WEB}, sticky=True, intent_age=3),
    escalation_count=2,
    # raw task text:  gone
    # raw file path:  gone
    # raw tool args:  gone
)
```

**The isolation boundary between the two schemas is intentional:**

`NormalizedIntent` is evaluated within the trusted runtime — Layers 1 and 2 operate inside the governance boundary and have full structural access.

`CanonicalizedIntent` is the schema passed *across* the isolation boundary to Layer 3. The verifier operates in a separate context with no shared conversation state. The current architecture is designed to prevent raw strings from crossing this boundary — a prompt injection embedded in a file path, tool argument, or task text is intended not to reach the verifier.

---

## Defense Cascade

Three defense layers plus a session-level behavioral state machine (`DegradationEngine`).
Each layer is independently packaged and optional (except Layer 1 and the degradation engine,
which always run). A deny at any layer is final — the current implementation is designed so
that later layers do not override an earlier deny.

```
Intent arrives
      │
      ▼
┌─────────────────────────────────────────────┐
│  DegradationEngine pre-check                │
│  (apply_to_policy — runs before Layer 1)    │
│                                             │
│  TERMINAL           → SessionTerminatedError│
│  LOCKED             → tools frozen;         │
│                       only read/escalate    │
│  quarantined source → write/bash/export off │
└──────────────────────┬──────────────────────┘
                       │ effective policy
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Rules  (axor-core, always)        │
│                                             │
│  Hard denies. Zero cost. Zero deps.         │
│  Synchronous, unconditional, first.         │
│  • write to path outside allowed set        │
│  • bash after external read without grant   │
│  • export when export_mode=RESTRICTED       │
│  • spawn_child when child_mode=DENIED       │
│  • spawn depth ≥ max_child_depth            │
│  • tool not in capabilities.allowed_tools   │
│  • lease expired, exhausted, or path OOB    │
└──────────────────────┬──────────────────────┘
                       │ passed rules
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 2: ML Scorer  (axor-classifier-simple│
│           optional)                         │
│                                             │
│  GradientBoosting on NormalizedIntent       │
│  feature window. Scores 0.0–1.0.            │
│  Trained on synthetic + anonymized traces.  │
│                                             │
│  score < low_threshold  → allow             │
│  score > high_threshold → deny              │
│  gray zone              → Layer 3           │
└──────────────────────┬──────────────────────┘
                       │ gray zone only
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 3: LLM Verifier  (axor-classifier-llm│
│           optional)                         │
│                                             │
│  Separate API call, clean context.          │
│  Receives CanonicalizedIntent only —        │
│  no raw args, no conversation history.      │
│  Static system prompt, not user-modifiable. │
│                                             │
│  Returns: allow | deny | ask_human          │
└──────────────────────┬──────────────────────┘
                       │ final outcome (allow or deny)
                       ▼
┌─────────────────────────────────────────────┐
│  DegradationEngine.record_signal()          │
│  (post-cascade — runs after every outcome)  │
│                                             │
│  Updates session degradation state.         │
│  Emits DegradationTransitionEvent /         │
│  SourceQuarantinedEvent on level change.    │
└─────────────────────────────────────────────┘
```

### Layer 1 — Rules

Always present. Implemented in `axor_core/node/intent_loop.py` and `axor_core/capability/`.

The rules are hard constraints derived from `ExecutionPolicy`. They fire before any ML inference. A Layer 1 deny is not heuristic — it is structural.

| Signal | Deny condition |
|--------|---------------|
| `write` | path outside `allowed_paths` after `normpath()` |
| `bash` | `preceded_by_external_read=True` and no escalation grant |
| `export` | `export_mode=RESTRICTED` |
| `spawn_child` | `child_mode=DENIED` or depth ≥ `max_child_depth` |
| any tool | tool not in `capabilities.allowed_tools` |
| any tool | lease expired, `max_uses` exhausted, or path outside `allowed_paths` |

On deny: `DenialResponse(status, coarse_category, opaque_decision_id)` — no policy state, no trace internals, no raw capability details exposed to the executor.

### Layer 2 — ML Scorer

Package: [`axor-classifier-simple`](https://github.com/Bucha11/axor-classifier-simple). No LLM dependency. Inference target: **< 1ms**.

**Features** (all from `NormalizedIntent`, no content):

```
tool_is_write, tool_is_bash, tool_is_export, tool_is_spawn
preceded_by_external_read
executes_generated_code
cross_origin_export
depth
escalation_active
intents_since_last_external
sequence_position
```

**Output**: anomaly score `0.0–1.0`. Default thresholds:

| Score | Action |
|-------|--------|
| `< 0.3` | allow |
| `> 0.7` | deny |
| `0.3–0.7` | escalate to Layer 3 if available, else deny |

Thresholds are configurable per deployment. In `ExecutionMode.STRICT`, `deny_on_ambiguity=True` — the gray zone auto-denies without reaching Layer 3.

### Layer 3 — LLM Verifier

Package: [`axor-classifier-llm`](https://github.com/Bucha11/axor-classifier-llm). Requires LLM API access.

**Isolation properties (current implementation):**
- Separate API call, separate context window
- Receives only `CanonicalizedIntent` — no raw args, no conversation history
- System prompt is static and not user-modifiable
- Intended not to share context with the worker agent — worker conversation content is not forwarded to the verifier
- `IntentCanonicalizer` is designed to strip control characters from path extensions before canonicalization

**Output**: structured verdict:

```json
{
  "score": 0.82,
  "class": "critical",
  "reasons": ["external read followed by immediate file write", "taint propagated to write target"]
}
```

`ask_human` surfaces via the `escalation_callback` mechanism, same path as `EscalationPolicy.require_human=True`.

### DegradationEngine — Session-Level Behavioral State Machine

`DegradationEngine` is not a layer in the cascade — it is a parallel state machine that
observes cascade outcomes and progressively narrows the capability surface as adverse signals
accumulate. It runs before the cascade (to apply the current level) and after it (to record
the outcome).

Unlike a simple "N denials → restrict" global counter, degradation is **source-aware**: a
malicious document quarantines its origin; clean sources in the same session continue at
reduced but functional capability until session-level thresholds are crossed.

| Level | Effect |
|-------|--------|
| `NORMAL` | Baseline — full policy in effect |
| `CAUTIOUS` | Triggering document soft-blocked from context |
| `RESTRICTED` | Quarantined source blocked; write/bash/export removed for tainted calls |
| `LOCKED` | All tools frozen; only `read` + `escalate` permitted; export forced `RESTRICTED` |
| `TERMINAL` | Session stopped; `SessionTerminatedError` raised on next intent |

**Transition rules:**

| Trigger | New level |
|---------|-----------|
| Cross-origin export denied (`destination_kind=external_domain/private_network`) | `LOCKED` immediately |
| `source.tool_pressure_count ≥ 2` | quarantine source + `RESTRICTED` |
| `source.instruction_pressure_count ≥ 1` | quarantine source + `RESTRICTED` |
| `session_deny_count ≥ 5` | `LOCKED` |
| `LOCKED` for `LOCKED_TTL` seconds (default 300 s) without governance clearance | `TERMINAL` |

Degradation level is **monotonically increasing** — it never decreases without a
`GovernanceAuthority` object passed to `clear_by_governance`. Worker-path clear raises
`DegradationClearanceError` (invariant D-1).

Child nodes share the parent's `DegradationEngine` instance — a child cannot start below
the parent's current `DegradationLevel` (invariant D-6).

---

## Taint: Per-Value Provenance (TM2)

When a session processes external content — a web fetch, an MCP result, an untrusted file read — the **value** it produced is registered with its causal root. Enforcement is **per-value**: a sink decides on the causal root of its driving argument (by content derivation), not on a session-wide flag. A value's provenance does not decay on its own; only a governance boundary can release it.

```python
# External content enters the session — register the produced VALUE
engine.register_value(web_text, CausalRoot.external_read(TaintSource.WEB))

# A sink derives the causal root of its driving argument. A value carrying the
# registered fragment is flagged; an argument that does not carry it is clean —
# the per-value win over session-sticky taint.
assert engine.derive_value(f"... {web_text} ...").is_tainted is True
assert engine.derive_value("unrelated text").is_tainted is False

# Child agents inherit the parent's per-value ledger (cannot launder provenance)
child_engine.inherit_value_ledger(parent_engine)

# Workers cannot release taint — raises TaintClearanceError. Only governance can,
# per value (endorse_value) or wholesale (clear_by_governance).
engine.endorse_value(value, authority=..., authority_type=..., reason_code=...)
engine.clear_by_governance(authority=..., authority_type=..., reason_code=...)
```

A value's causal root carries its source set and the confidentiality (`sensitive`) label.
The `DegradationEngine` is fed by the per-value *driving root* at a denied sink: the source
identifier derived from that root tracks per-source behavioral pressure counts, driving the
quarantine-and-restrict transitions when a single origin accumulates tool or instruction
pressure signals.

### Taint sources

A registered value records which `TaintSource`(s) it derived from in its causal root.

| Source | `TaintSource` | Typically sensitive |
|--------|--------------|---------------------|
| Web fetch / search | `WEB` | no |
| MCP result | `MCP` | no |
| File read (untrusted workspace) | `FILE` | depends (secret reads → yes) |
| External API | `API` | no |
| Child agent output | `CHILD_AGENT` | inherited |
| Memory (external session) | `MEMORY` | inherited |

Child agents inherit the parent's per-value ledger (`inherit_value_ledger`). The worker path cannot release inherited provenance — child agents are not designed to launder a parent's tainted values.

---

## Execution Modes

| Mode | Isolation | Classifier | On gray zone |
|------|-----------|------------|--------------|
| `LIBRARY` | None — same process | Enabled | Escalate |
| `PRODUCTION` | `LockedExecutor` — bypass raises `GovernanceBypassError` | Enabled | Escalate |
| `STRICT` | `LockedExecutor` + audit-required trace | Task classifier disabled for policy selection | Deny |

`STRICT` is a superset of `PRODUCTION`. In `STRICT`, there are no content-based policy decisions — policy is set by the operator, not derived from task text. Gray-zone intents are denied without reaching Layer 3 (`deny_on_ambiguity=True`).

---

## Policy Ceiling Invariant

Child agents cannot exceed the parent's capability surface. Enforced by `_validate_child_policy()` in `spawn.py`:

```python
if child_tp.allow_bash and not parent_tp.allow_bash:
    raise SpawnValidationError("child requests allow_bash but parent forbids it")
```

And by `PolicyComposer`, which applies parent restrictions to any child policy before the child node is created:

```python
def compose(self, policy, parent_policy) -> ExecutionPolicy:
    # child.allow_write = child.allow_write AND parent.allow_write
    ...
```

The current implementation is designed so that no path allows a child to exceed parent capabilities, enforced at two points: `_validate_child_policy()` on spawn and `PolicyComposer` on policy construction.

---

## Trace Access Control

`TraceCollector` is not passed to workers. Workers receive a `DenialResponse` — not trace internals.

`TraceAccessGuard` uses constant-time token comparison to mitigate timing attacks on the access check. Direct access to the trace requires a valid token.

---

## Escalation Flood Guard

`EscalationFloodGuard` protects against DoS via repeated escalation requests:

- Identical requests are deduplicated — a request seen before does not consume capacity
- Rate limiting enforces a maximum escalation rate per time window
- Auto-deny activates when the flood threshold is exceeded — all subsequent requests in the window are denied without evaluation

---

## Postmortem Mode

The cascade is not only a pre-execution filter. Intent sequences are recorded in `DecisionTrace` for every session. This enables:

- **Why was this denied?** — full intent chain with scores at each layer
- **Pattern analysis** — which intent sequences correlate with injection attempts
- **Threshold tuning** — adjust ML thresholds based on false positive/negative rates
- **Dataset curation** — confirmed traces may be curated into future training datasets for Layer 2

The trace is the primary debugging surface. When the system gets something wrong, the postmortem is in the trace.

```python
for trace in session.all_traces():
    for event in trace.events:
        print(f"{event.kind.value}: {event.payload}")

# intent_approved          {tool: "read", layer1: "pass", layer2: 0.12}
# intent_denied            {tool: "bash", layer1: "deny", reason: "preceded_by_external_read"}
# taint_propagated         {source: "WEB", scope: "SESSION"}
# source_quarantined       {source_id: "web-abc123", reason: "tool_pressure_count >= 2"}
# degradation_transition   {previous_level: "NORMAL", new_level: "RESTRICTED",
#                           trigger_source_id: "web-abc123", reason: "source_quarantined"}
# degradation_transition   {previous_level: "RESTRICTED", new_level: "LOCKED",
#                           trigger_source_id: "", reason: "cross_origin_export_denied"}
```

---

## Adversarial Test Suite

Security properties are tested by a structured regression suite in `tests/adversarial/`. Minimum variant counts per category are enforced by CI — dropping below minimums fails the build.

| Category | Min variants | What it tests |
|----------|-------------|---------------|
| Burstfire | 3 | Taint persists through N=1, N=10, N=50 benign intents after external read |
| Warm-up | 3 | Taint persists through same-task delay, unrelated-task delay, child-agent delay |
| Schema injection | 5 | Newline, unicode control chars, null byte, path traversal, prompt-like filename — none reach canonical fields |
| Escalation flood | 3 | Dedup, rate limiting, auto-deny under flood |
| Federation lateral | 4 | Tainted parent → child inherits; child cannot exceed parent ceiling; export bounded by parent |
| Trace side-channel | 3 | Direct trace read blocked; denial response has no sensitive fields; coarse categories only |
| Lease ceiling | 6 | Tool expansion rejected; path restriction enforced; `../` traversal blocked; expired/exhausted leases rejected |
| Layer override | 2 | ML approve + Layer 1 deny = deny; LLM approve + Layer 1 deny = deny |
| Degradation | 14 | Monotonicity (level never decreases without governance), source isolation (clean source unaffected by quarantined peer), cross-origin export → LOCKED, LOCKED_TTL auto-TERMINAL, child floor inheritance (D-1..D-7) |

### Bypass patterns found by this suite

Two bypass patterns were discovered during development and fixed before release. Both have regression tests that reproduce the exact bypass and confirm it is blocked. Details are in the [Implementation Notes](#implementation-notes) appendix.

---

## Security Invariants

30 invariants are codified in `tests/invariants/test_security_invariants.py` and tested against known bypass patterns. Each has a positive test and an adversarial pair.

| Invariant | How enforced in current implementation |
|-----------|---------------------------------------|
| Taint is sticky — worker-path clearing is blocked | `TaintClearanceError` raised on worker-path clear attempt |
| Child policy ceiling bounded by parent | `SpawnValidationError` in `_validate_child_policy()` |
| Layer 1 runs first and is not bypassed by later layers | `_evaluate_tool_intent()` is synchronous; scorer not called on Layer 1 deny |
| `CanonicalizedIntent` is designed to contain no raw strings | Verified by schema-injection adversarial tests |
| Denial response is designed to expose only coarse fields | `DenialResponse`: `status`, `coarse_category`, `opaque_decision_id` |
| Trace access uses constant-time token comparison | `TraceAccessGuard` implementation |
| Executor requires active governance context in PRODUCTION/STRICT | `LockedExecutor` → `GovernanceBypassError` |
| STRICT disables task classifier for policy selection | Enforced at `GovernedSession.__init__()` |
| **D-1** Degradation level is monotonically increasing — worker-path clear blocked | `DegradationClearanceError` raised on worker-path clear attempt |
| **D-2** TERMINAL session raises before any intent is evaluated | `GovernedSession.run()` checks `TERMINAL` at the top; `SessionTerminatedError` raised |
| **D-3** Cross-origin export denial → LOCKED immediately | `DegradationEngine.record_signal()` detects `destination_kind ∈ {external_domain, private_network}` |
| **D-4** Quarantined source loses write/bash/export; clean sources unaffected | `apply_to_policy(source_id=quarantined)` narrows; `apply_to_policy(source_id=clean)` unchanged |
| **D-5** `apply_to_policy` for quarantined source always returns `export_mode=RESTRICTED` | Enforced unconditionally in `DegradationEngine.apply_to_policy()` |
| **D-6** Child nodes cannot start below parent's current `DegradationLevel` | Child nodes receive the same `DegradationEngine` instance as the parent |
| **D-7** Governance clearance requires `GovernanceAuthority` object | `clear_by_governance()` type-checks authority; `attempt_clear_by_worker()` always raises |

---

## Package Structure

```
axor-core                        always present
  └─ Layer 1 (rules)             zero deps, zero latency

axor-classifier-simple           pip install "axor-classifier-simple[ml]"
  └─ Layer 2 (ML scorer)         scikit-learn, joblib

axor-classifier-llm              pip install "axor-classifier-llm[llm]"
  └─ Layer 3 (LLM verifier)      anthropic SDK
```

Layers are injected at `GovernedSession` construction time. Core never imports them. They implement protocols defined in `axor-core`:

```python
class AnomalyDetector(Protocol):
    async def score(self, intent: NormalizedIntent) -> AnomalyResult: ...

class LLMVerifier(Protocol):
    async def verify(self, window: list[CanonicalizedIntent]) -> AnomalyResult: ...
```

Session construction:

```python
session = GovernedSession(
    executor=...,
    capability_executor=...,
    anomaly_detector=MLAnomalyDetector(),    # Layer 2, optional
    llm_verifier=LLMAnomalyVerifier(),       # Layer 3, optional
)
```

If neither is passed — Layer 1 only. If only ML — cascade stops at Layer 2. If both — full three-layer cascade.

---

## Known Limitations

Axor governs execution boundaries under specific, tested conditions. The following are explicitly out of scope:

- **Full prompt injection prevention.** The adversarial suite tests specific known patterns. Novel injection vectors are not covered.
- **Covert channel elimination.** Timing channels, side channels through token counts, and other covert channels are not in scope.
- **Semantic malware detection.** Content analysis (detecting harmful intent in text) is not Axor's defense boundary. Behavior-based detection catches behavioral patterns, not semantic content.
- **All provider-specific edge cases.** Only tested providers and formats are covered.
- **Future attack families.** Zero-day attacks against the governance layer are not predicted by current tests.

Governance is a defense layer, not a guarantee of safety. Operators are responsible for policy design and deployment context.

---

## Why Not Filter Inputs?

| Approach | Problem |
|----------|---------|
| Sanitize tool returns | Requires semantic understanding — brittle at scale |
| Blocklist injection phrases | Trivially bypassed by encoding, rephrasing |
| Isolate external reads | Breaks legitimate research/analysis workflows |
| **Govern intent sequence** | Attack surface is behavior, not content ✓ |

An agent that read a malicious README is designed to be prevented from exfiltrating your `.env` when:
- `export` is `RESTRICTED`
- `bash` is denied when `preceded_by_external_read=True`
- `write` is path-restricted

The content of the README is irrelevant. The behavioral constraint is the defense.

---

## Implementation Notes

Two bypass patterns were discovered during development of the adversarial test suite and fixed before release. Both have regression tests that reproduce the exact bypass and confirm it is blocked.

### Newline injection in path canonicalization

`os.path.splitext("legit.py\nignore")` returns `(".py\nignore", "")` — the newline is not treated as a separator. A crafted filename with a trailing newline could cause the extension field in `CanonicalizedIntent` to carry an injected suffix.

**Fix:** `_path_extension()` in `axor_core/node/canonicalizer.py` now truncates the input at the first control character (`ord(c) < 32` or `\x7f`) before calling `os.path.splitext()`.

### Path traversal bypass in lease validation

`"/safe/path/../etc/passwd".startswith("/safe/path/")` evaluates to `True`. A crafted path with `../` components could pass the prefix check while resolving outside the allowed set.

**Fix:** `check_path_allowed()` in `axor_core/capability/lease_validator.py` now applies `os.path.normpath()` to both the candidate path and each allowed prefix before comparison.
