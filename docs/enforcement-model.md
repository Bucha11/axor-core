# Axor Runtime Enforcement Model

## Overview

Every privileged action must pass through the following call flow before
execution.  The flow is identical across all providers — providers execute,
axor-core governs.

---

## Required Call Flow

```
1. Agent emits tool call or privileged intent
2. Provider adapter accumulates full raw tool call (if streamed)
3. ProviderNormalizer     →  NormalizedIntent
4. IntentCanonicalizer    →  CanonicalizedIntent (the content-free projection an
                             advisory adjudicator / detection layer sees)
5. DegradationEngine.apply_to_policy()  →  effective policy for this call
   (narrows ExecutionPolicy if the session level is RESTRICTED/LOCKED/TERMINAL)
6. IntentLoop runs the structural gates, in order — a deny at any gate is final:
     capability → consequence → value policies → degradation → positional →
     carrier → per-value taint (integrity) + confidentiality floor → adjudicator
7. DegradationEngine.record_signal(denial_or_none) → state updated for next intent
   (emits DegradationTransitionEvent / SourceQuarantinedEvent if level changed)
8. If allowed → executor runs the underlying tool; its output's provenance is
   registered (federation ingress decides the provenance of a peer-returned value)
9. If denied  → executor receives a coarse DenialResponse
10. DecisionTrace written out-of-band (operator channel only)
```

A deny at any gate is final; nothing downstream loosens it. The optional advisory
adjudicator is consulted only on the would-approve path, so it can add a deny but
never override one. Detection (reputation, drift) is observe-only and is not on this
path; it may, opt-in, only tighten degradation. Step 5 (degradation pre-check) runs
first — a quarantined or locked session denies the tool before the gates start.

**One gate implementation, two callers.** The six stateless gates in step 6
(consequence, value policies, SSRF, positional, carrier, per-value taint +
confidentiality floor) live as pure functions in `policy/gates.py`. The streaming
`IntentLoop` above is one caller; the synchronous `ToolCallGovernor` — for
frameworks that own their own agent loop — is the other. Both delegate to the same
functions, so the decision logic exists once and cannot drift. The IntentLoop adds
the stateful wrapper (capability, degradation, leases, escalation, execution); the
governor adds only the per-value ledger. Capability, degradation, and the
adjudicator are IntentLoop-only and not part of the shared core.

---

## Provider-Specific Call Flows

### Claude (Anthropic SDK)

```
AnthropicClient.stream()
  → StreamNormalizer.process(sdk_event)    # accumulate partial chunks
  → ClaudeNormalizer.normalize()           # ToolUseBlock → NormalizedIntent
  → IntentLoop._resolve_tool_intent()      # the structural gate sequence
  → CapabilityExecutor.execute()          # if approved
  → ToolResultBus.push(result)            # tool result back to executor
```

### LangChain

**Wrapper mode (enforcement)**:
```
LangChain agent emits tool_call
  → AxorToolWrapper._run() / _arun()      # intercepts before original tool
  → LangChainNormalizer.normalize()       # BaseTool format → NormalizedIntent
  → policy_fn(normalized_intent)          # deny or allow
  → wrapped_tool._run() if approved
```

**Callback mode (observability only)**:
```
LangChain agent emits tool_call
  → AxorCallbackHandler.on_tool_start()   # observability only — no enforcement
  → WARNING logged: "use AxorToolWrapper for enforcement"
  → tool runs unconditionally
```

### Mock OpenAI / OpenRouter

```
OpenAI-format tool call dict
  → MockOpenAINormalizer.normalize()     # function call dict → NormalizedIntent
  → IntentLoop (same as Claude path)
```

```
OpenRouter tool call (OpenAI-compat)
  → MockOpenRouterNormalizer.normalize() # accumulated streaming → NormalizedIntent
  → IntentLoop (same as Claude path)
```

### CLI

```
User types tool invocation
  → GovernedSession.run(task)
  → GovernedNode._classify_and_run()
  → IntentLoop.run(stream, envelope)
  → (standard flow above)
```

---

## ToolInterceptor Requirements

`IntentLoop` acts as the ToolInterceptor.  It must:

- Accept `ExecutorEvent(TOOL_USE)` from the executor stream
- Convert to `Intent` via `IntentNormalizer`
- Evaluate against `ExecutionPolicy` (the structural gates)
- Consult an advisory adjudicator if configured (projection only; can only add a deny)
- Apply per-value taint rules (`TaintEngine.register_value()` on external reads; `derive_value()` at sinks)
- Apply lease rules (`LeaseValidator.check_tool_allowed()`)
- Apply the degradation pre-check (`DegradationEngine.apply_to_policy()`) before the gates
- Record `DegradationEngine.record_signal()` after every outcome (pass or deny)
- Emit `DegradationTransitionEvent` / `SourceQuarantinedEvent` when level changes
- Fail closed: any exception → `DenialResponse`
- Record `IntentDeniedEvent` / `SuspiciousIntentEvent` in `DecisionTrace`
- Never expose trace to worker

---

## Executor Bypass Prevention

In PRODUCTION and STRICT mode:

- `GovernedSession` wraps the executor in `LockedExecutor`
- `LockedExecutor.stream()` raises `GovernanceBypassError` if called outside
  `governance_context()`
- `GovernedNode` activates `governance_context()` before delegating to executor
- No public API exposes the raw executor reference
- `ExecutionEnvelope` contains no `TraceCollector` reference

In LIBRARY mode, direct executor calls are allowed but a warning is emitted
when `AXOR_ENV=production` is set.

### What the process boundary does and does not cover

`LockedExecutor` is an **in-process soft boundary**: it stops the *worker code* from
calling the executor outside `governance_context()`, but a worker process that runs
attacker-controlled native code can bypass it. The real boundary is out-of-process
execution via `DaemonCapabilityClient` → `axor-daemon` (set `require_isolation=True`
or `AXOR_REQUIRE_ISOLATION=1` to force it).

It is important to be precise about what crosses to the daemon. The daemon
enforces, server-side and independent of the worker:

- the **capability ceiling** — `operator_policy` set at daemon startup (the worker
  cannot widen it), plus the session ceiling from a signed grant;
- the **filesystem sandbox** — path arguments are normalised against the sandbox root;
- **tool implementations** — they live only in the daemon, so a compromised worker
  cannot touch the filesystem/network directly; it must go through the socket.

The daemon does **not** currently re-run the per-value data-flow gates (taint,
confidentiality floor, consequence, carrier, positional). Those run only in the
worker's `IntentLoop`. The consequence, for the two threat models:

- **Prompt injection** (the model is manipulated, worker code intact) — fully
  covered: the model can only emit tool calls, and the in-process gates intercept
  them before they reach the daemon. This is the threat axor targets.
- **Code-level worker compromise** (RCE / malicious native code in the worker) — the
  daemon confines it to the capability ceiling and the path sandbox, but within
  those allowed tools the data-flow gates are bypassed, so e.g. exfiltration through
  an allowed egress tool is not stopped out-of-process.

Closing that residual is cheap given the shared gate engine: the daemon can host a
`ToolCallGovernor` (the same `policy/gates.py` functions, run synchronously with a
per-session taint ledger) and run the full sequence server-side. Until it does, the
sound data-flow mechanisms are an in-process guarantee, not a cross-process one.

---

## Fail-Closed Semantics

| Failure | Outcome |
|---------|---------|
| Interceptor exception | Deny intent, log error |
| Normalizer exception (`NormalizerError`) | Deny intent |
| Unknown provider format (`UnknownProviderFormatError`) | Deny intent (STRICT: terminate session) |
| Malformed tool call | Deny intent |
| Budget tracker failure | Log; record continues best-effort |
| Lease validation failure | Deny intent |
| Taint engine failure | Deny privileged intent |
| Degradation engine: `TERMINAL` level | `SessionTerminatedError` raised before intent evaluation |
| Degradation engine: `LOCKED` level | Tools frozen; only `read`/`escalate` permitted |
| Trace writer failure (`audit_required=True`) | Raise `PermissionError` |
| Telemetry sink failure | Continue; governance path not affected |
