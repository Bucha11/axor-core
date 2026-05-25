# Axor Threat Model

## Overview

Axor is a post-compromise containment layer for AI agents.  The core
assumption is that the worker agent **may become compromised** after reading
external content.  Policy enforcement must remain valid even under compromise.

---

## Trusted Components

| Component | Trust Level | Notes |
|-----------|-------------|-------|
| Operator configuration | Trusted | Source of all policy authority |
| `axor-core` governance process | Trusted | Policy evaluation, taint engine |
| `PolicyComposer` / `PolicySelector` | Trusted | Compile and select execution policy |
| `ToolInterceptor` / `IntentLoop` | Trusted | Intercepts all tool calls |
| `ProviderNormalizer` | Trusted | Converts raw provider events |
| `TaintEngine` | Trusted | Tracks taint; workers cannot clear |
| `DegradationEngine` | Trusted | Monotonically degrades session level; workers cannot clear |
| `LeaseValidator` | Trusted | Validates CapabilityLease claims |
| `TraceCollector` | Trusted | Out-of-band audit trail |
| `IntentCanonicalizer` | Trusted | Strips raw content before Layer 3 |
| `LockedExecutor` | Trusted | Blocks direct executor bypass |
| Layer 2 ML scorer | Conditionally trusted | Advisory; cannot expand capability |
| Layer 3 LLM verifier | Conditionally trusted | Advisory; cannot override Layer 1 |
| Provider adapters post-normalization | Conditionally trusted | After NormalizedIntent is produced |

---

## Untrusted Components

- Worker agent (may be compromised by prompt injection)
- Executor runtime (not governance-aware)
- Provider model output (raw)
- Child agents (inherit taint, cannot exceed parent policy ceiling)
- External tools (return values treated as hostile)
- Raw provider tool calls (must be normalized before policy evaluation)
- Raw tool outputs (may contain prompt injection payloads)

---

## Attacker-Controlled Inputs

All of the following are treated as hostile or potentially hostile:

- User prompt
- External tool return value
- MCP result
- Web search result
- File read result (from untrusted workspace)
- API response
- Child-agent output
- Retrieved memory (from external or historical session state)
- Provider model output
- Serialized tool call fields (name, args, tool_use_id)
- File paths
- URLs
- Branch names
- Filenames
- Tool arguments
- Metadata generated from untrusted execution

---

## Security Boundaries

### Process Boundary

`GovernedSession` owns the executor lifecycle.  In PRODUCTION/STRICT mode,
the executor is wrapped in `LockedExecutor`.  Calling `executor.stream()`
outside the governance context raises `GovernanceBypassError`.

### Normalization Boundary

All tool calls must pass through `ProviderNormalizer` before reaching
`ToolInterceptor`.  Raw provider formats are not allowed to skip this step.
Unknown formats raise `UnknownProviderFormatError` → execution denied.

### Taint Boundary

The `TaintEngine` tracks whether any external input has been processed.
Taint is sticky (persists until governance clears it).  Workers cannot clear
taint.  Child nodes inherit parent taint.

### Degradation Boundary

The `DegradationEngine` observes accumulated denial signals and taint metadata
to progressively narrow the capability surface.  Level transitions are
monotonically increasing — the level never decreases without a
`GovernanceAuthority` object passed to `clear_by_governance`.
Workers cannot lower the degradation level; `attempt_clear_by_worker()` always
raises `DegradationClearanceError`.  At `TERMINAL` level,
`SessionTerminatedError` is raised before any intent is evaluated.
Child nodes share the parent's `DegradationEngine` instance — a child cannot
start below the parent's current `DegradationLevel`.

### Canonicalization Boundary (Layer 3)

`IntentCanonicalizer` strips all raw content from `NormalizedIntent` before
producing `CanonicalizedIntent` for the LLM verifier.  File paths, tool
arguments, and web content never reach the verifier prompt.

### Denial Boundary

`DenialResponse` exposes only: `status`, `coarse_category`, `opaque_decision_id`.
Workers never receive: denial reasons, scores, taint history, or thresholds.

### Trace Boundary

`DecisionTrace` is accessible only via `operator_read(auth_token)`.
`read_all()` always raises `PermissionError`.  Worker execution context never
holds a `TraceCollector` reference.

---

## Privileged Actions

Axor enforces policy before any privileged action executes:

- Tool execution (any registered tool)
- Shell / subprocess execution
- File writes (inside or outside workspace)
- File reads outside allowed scope
- Child-agent spawning
- External network requests
- Memory writes
- Export to parent or external system

---

## Security Invariants

See `axor-core/tests/invariants/test_security_invariants.py` for the full list
of 30 invariants and their executable tests.

Key invariants:

1. Layer 1 hard deny cannot be overridden by ML or LLM layers.
2. Worker cannot read `DecisionTrace` without operator token.
3. External taint persists until governance clears it.
4. Worker cannot initiate taint clearance.
5. Child nodes inherit parent taint and cannot exceed parent policy ceiling.
6. Runtime denial responses are coarse — no sensitive detail exposed to worker.
7. (D-1) Degradation level is monotonically increasing; worker-path clear raises `DegradationClearanceError`.
8. (D-2) A `TERMINAL` session raises `SessionTerminatedError` before any intent is evaluated.
9. (D-3) A cross-origin export denial escalates directly to `LOCKED`.
10. (D-6) Child nodes share the parent's `DegradationEngine` instance and cannot start below the parent's current level.

---

## Non-Goals

Axor does **not** guarantee:

- Full prevention of prompt injection in user-visible output (only in policy/verifier path)
- Covert channel elimination (side channels beyond governance scope)
- Semantic malware detection (content analysis is not a security boundary)
- Prevention of all provider-specific edge cases
- Protection when operator configuration is itself compromised

---

## Known Limitations

- LIBRARY mode does not provide strong process isolation — only PRODUCTION/STRICT mode does.
- Layer 2 and Layer 3 are advisory; they cannot expand capabilities but they also
  cannot guarantee detection of all adversarial patterns.
- Taint clearance by governance is auditable but depends on operator judgment.
- Cross-provider parity is tested against three normalizers; unlisted providers
  require their own normalizer and parity test.
