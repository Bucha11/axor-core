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

**`NormalizedIntent`** — the structural shape of an intent that the gates read, no values:

```python
@dataclass(frozen=True)
class NormalizedIntent:
    # Identity
    tool: str                         # "write", "bash", "spawn_child", ...
    operation: str                    # file_read | file_write | network_request | ...

    # Structural target / destination — kind only, never values
    target_kind: str                  # workdir | system_path | secret | external_url
    destination_kind: str             # none | localhost | external_domain | private_network
    provenance: str                   # user | repo | official_docs | external_web | unknown

    # Causal signals — derived from intent history, not content
    reads_secret_like_data: bool      # target looks like a credential/secret read
    writes_outside_workdir: bool      # write escapes the working directory
    executes_generated_code: bool     # write→execute sequence detected
    after_external_read: bool         # a recent tool return came from outside
    after_secret_access: bool         # a secret was read earlier this session
    data_flow: str                    # none | local_to_local | local_to_external | external_to_shell

    # Observe-only enrichment (telemetry; never gates a decision)
    target_resource_reputation: float = 0.0
    target_container_reputation: float = 0.0
```

`after_external_read` is the key causal signal. It captures causality without
inspecting content: the agent read something external, and now it wants to act.
That pattern is the attack surface. (Reputation fields are observe-only — they
feed degradation signals, never a gate; see governance-model.md §8.)

**`CanonicalizedIntent`** — the content-free projection the adjudicator and detection see. All raw strings stripped by `IntentCanonicalizer`:

```python
CanonicalizedIntent(
    tool_category=ToolCategory.READ,
    path_extension=".py",            # extension only — never the full path
    path_hash="…",                   # sha256[:16] of path, for dedup not content
    argument_shape="path",           # sorted arg key names, no values
    argument_length_bucket=1,        # 0=empty 1=<256 2=medium 3=>4096
    taint_state_summary="tainted:web",   # per-value provenance summary
    after_external_read=True,
    # raw task text:  gone
    # raw file path:  gone
    # raw tool args:  gone
)
```

**The isolation boundary between the two schemas is intentional:**

`NormalizedIntent` is evaluated within the trusted runtime — the structural gates operate inside the governance boundary and have full structural access.

`CanonicalizedIntent` is the projection passed to any advisory adjudicator or detection layer. It carries no raw strings — a prompt injection embedded in a file path, tool argument, or task text cannot reach a layer that only sees the projection, so it cannot steer the decision.

---

## The Gate Sequence

Every tool call is converted into an intent and run through a fixed sequence of
structural gates plus a session-level degradation state machine. A deny at any gate
is final — nothing downstream turns it back into an allow.

```
Intent arrives
      │
      ▼
  Degradation pre-check  ── narrows the effective policy for a degraded session
      │                     (locked → read/escalate only; quarantined source → no write/bash/export)
      ▼
  1. Capability          ── is the tool allowed by the active policy at all?
  2. Consequence         ── how irreversible is the ACTION class? (content-blind;
                            a shutdown/wipe exceeds the unattended ceiling)
  3. Value policies      ── decidable predicates on arguments (amount in range, enum)
  4. Degradation         ── refuse a call a quarantined source would drive
  5. Positional          ── declared sinks: admit only instruction-incomplete FORMS
  6. Carrier             ── untrusted free text into an instruction-following sink?
  7. Per-value taint     ── integrity (untrusted → high-risk op) + confidentiality floor
  8. Adjudicator         ── optional advisory second opinion (tightening-only)
      │
      ▼
  Execute, then register the output's provenance for later calls
```

Gates 1–4 read the call and the policy; 5–6 read the *form* of the value (never its
content); 7 reads the value's own provenance. Detection (reputation, anomaly, drift)
is **not** on this path — it is observe-only and may, opt-in, only *tighten*
degradation. See [governance-model.md](governance-model.md) for the full statement.

### Why structure beats content at gates 5–6

Content tracking is sound but incomplete: an in-process model can paraphrase an
untrusted value so a content match misses it. The carrier and positional gates read
the value's **form** instead — a closed schema / enum / number cannot encode an
instruction, regardless of its content — so paraphrase is irrelevant. A sink whose
legitimate input genuinely needs free text (a shell command) is instruction-complete
by definition and **cannot** be declared positional; it stays on content tracking
with the acknowledged residual below.

### Degradation — a one-way tightening

A session-level state machine that narrows the surface as adverse *facts* accumulate.
It applies the current level before the gates and records the outcome after.

| Level | Effect |
|-------|--------|
| `NORMAL` | Baseline — full policy in effect |
| `CAUTIOUS` | Triggering source soft-blocked from context |
| `RESTRICTED` | Quarantined source blocked; write/bash/export removed for its calls |
| `LOCKED` | All tools frozen; only `read` + `escalate` permitted; export forced `RESTRICTED` |
| `TERMINAL` | Session stopped; `SessionTerminatedError` raised on next intent |

Transitions are driven by **decidable structural facts**, not counters or scores: a
cross-origin export of an untrusted-rooted value goes straight to `LOCKED`; a denied
dangerous call quarantines its source and moves to `RESTRICTED`; a further dangerous
untrusted-rooted fact while `LOCKED` goes `TERMINAL`. Pressure counters are kept as
telemetry only — they no longer drive a transition.

Degradation is **monotone** — it never decreases without a `GovernanceAuthority`
passed to `clear_by_governance`; a worker-path clear raises `DegradationClearanceError`.
Clearing below `RESTRICTED` releases the quarantine and resets the pressure, so the
session truly returns to a clean state. Child nodes share the parent's degradation
state and cannot start below the parent's current level.


---

## Taint: Per-Value Provenance

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
# per value (endorse_value) or wholesale (clear_by_governance). The clearance
# capability is the GovernanceAuthority value object (its authority_type / reason_code
# are fields of that object), not loose kwargs.
authority = GovernanceAuthority(authority_id=..., authority_type=..., reason_code=...)
engine.endorse_value(content, authority)
engine.clear_by_governance(authority)
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

`STRICT` is a superset of `PRODUCTION`. In `STRICT`, there are no content-based policy decisions — policy is set by the operator, not derived from task text. Ambiguous policy selection fails closed rather than escalating (`deny_on_ambiguity=True`).

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

- **Why was this denied?** — the full intent chain with the deciding gate and reason at each step
- **Pattern analysis** — which intent sequences correlate with injection attempts
- **Threshold tuning** — adjust the opt-in detection thresholds (reputation / drift) from observed false positive/negative rates; the enforcement gates are structural booleans with nothing to tune
- **Dataset curation** — confirmed traces may be curated into future detection datasets

The trace is the primary debugging surface. When the system gets something wrong, the postmortem is in the trace.

```python
for trace in session.all_traces():
    for event in trace.events:
        print(f"{event.kind.value}: {event.payload}")

# intent_approved          {tool: "read"}
# intent_denied            {tool: "bash", reason: "consequence gate: catastrophic action class"}
# sink_density             {operation: "curl", tainted: true, sensitive: true}
# source_quarantined       {source_id: "web-abc123", reason: "tool_pressure_threshold:curl"}
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
| Advisory override | 2 | an adjudicator ALLOW never overrides a structural deny; it can only add a deny |
| Degradation | 14 | Monotonicity (level never decreases without governance), source isolation (clean source unaffected by quarantined peer), cross-origin export → LOCKED, LOCKED_TTL auto-TERMINAL, child floor inheritance |

### Bypass patterns found by this suite

Two bypass patterns were discovered during development and fixed before release. Both have regression tests that reproduce the exact bypass and confirm it is blocked. Details are in the [Implementation Notes](#implementation-notes) appendix.

---

## Security Invariants

30 invariants are codified in `tests/invariants/test_security_invariants.py` and tested against known bypass patterns. Each has a positive test and an adversarial pair.

| Invariant | How enforced in current implementation |
|-----------|---------------------------------------|
| Taint is sticky — worker-path clearing is blocked | `TaintClearanceError` raised on worker-path clear attempt |
| Child policy ceiling bounded by parent | `SpawnValidationError` in `_validate_child_policy()` |
| A structural deny is final and not bypassed by an advisory layer | the adjudicator is consulted only on the would-approve path; it can only add a deny |
| `CanonicalizedIntent` is designed to contain no raw strings | Verified by schema-injection adversarial tests |
| Denial response is designed to expose only coarse fields | `DenialResponse`: `status`, `coarse_category`, `opaque_decision_id` |
| Trace access uses constant-time token comparison | `TraceAccessGuard` implementation |
| Executor requires active governance context in PRODUCTION/STRICT | `LockedExecutor` → `GovernanceBypassError` |
| STRICT disables task classifier for policy selection | Enforced at `GovernedSession.__init__()` |
| Degradation level is monotonically increasing — worker-path clear blocked | `DegradationClearanceError` raised on worker-path clear attempt |
| TERMINAL session raises before any intent is evaluated | `GovernedSession.run()` checks `TERMINAL` at the top; `SessionTerminatedError` raised |
| Cross-origin export denial → LOCKED immediately | `DegradationEngine.record_signal()` detects `destination_kind ∈ {external_domain, private_network}` |
| Quarantined source loses write/bash/export; clean sources unaffected | `apply_to_policy(source_id=quarantined)` narrows; `apply_to_policy(source_id=clean)` unchanged |
| `apply_to_policy` for quarantined source always returns `export_mode=RESTRICTED` | Enforced unconditionally in `DegradationEngine.apply_to_policy()` |
| Child nodes cannot start below parent's current `DegradationLevel` | Child nodes receive the same `DegradationEngine` instance as the parent |
| Governance clearance requires `GovernanceAuthority` object | `clear_by_governance()` type-checks authority; `attempt_clear_by_worker()` always raises |

---

## Package Structure

```
axor-core              the kernel: gates, per-value provenance, degradation,
                       federation — zero required dependencies

axor-sentinel          cross-session resource reputation (observe-only)
axor-probe             behavioural drift (observe-only, out-of-band)
axor-classifier-*      richer task-signal classification (policy selection only)
```

Everything pluggable is injected at `GovernedSession` construction; core never
imports an implementation. The enforcement-relevant knobs are all opt-in:

```python
session = GovernedSession(
    executor=..., capability_executor=...,
    # enforcement
    positional_sinks={"publish"},                 # declare instruction-incomplete sinks
    value_policies={"transfer": [numeric_range("amount", 0, 1000)]},
    federation_gateway=gateway,                    # agent-to-agent trust
    adjudicator=my_oracle,                         # advisory second opinion
    # detection (observe-only; may opt-in tighten degradation)
    detection_floor=0.3,                           # reputation threshold-crossing
    behavioral_drift_observer=probe_observer,
)
```

A detection layer (reputation, drift) only ever observes — it cannot return an
allow/deny and is not on the gate path. The one exception is `detection_floor`: a
reputation reading crossing it is a decidable fact that may *tighten* degradation.

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
- the confidentiality floor is armed by the secret read, so egress is denied until governance endorses it
- a value whose driving root carries the README's content (`after_external_read`) cannot reach an instruction-following or code-executing sink (carrier / taint gates)
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
