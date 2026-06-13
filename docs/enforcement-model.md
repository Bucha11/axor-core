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
     capability → consequence → value policies → degradation → ssrf/internal-dest →
     positional → carrier → per-value taint (integrity) + confidentiality floor →
     adjudicator
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

The daemon also re-runs the **per-value data-flow gates** server-side. It hosts a
per-session `ToolCallGovernor` (the same `policy/gates.py` functions, with its own
taint ledger and confidentiality floor) and runs `evaluate` before touching a
handler and `register_output` after — so taint, the confidentiality floor,
consequence, carrier, and positional are enforced on the daemon's side of the
boundary, not only in the worker. The data-flow taxonomy
(`untrusted_sources` / `sensitive_sources` / `egress_sinks`) is **operator-set at
daemon startup**, never taken from the client, so a compromised worker cannot
disable the gates by declaring an empty taxonomy. The consequence, for the two
threat models:

- **Prompt injection** (the model is manipulated, worker code intact) — covered at
  both layers: the in-process gates intercept the tool call, and the daemon would
  catch it again if the worker path were skipped.
- **Code-level worker compromise** (RCE / malicious native code in the worker) — the
  worker can send raw tool calls straight to the socket, bypassing its own
  `IntentLoop`, but the daemon still enforces the capability ceiling, the path
  sandbox, **and** the data-flow gates against its own ledger. An exfiltration
  through an allowed egress tool, driven by a value the daemon saw arrive from an
  untrusted read, is denied server-side
  (`axor-daemon/tests/test_dataflow_enforcement.py`).

Residual: the gates are only as complete as the operator's declared taxonomy and
the content-derivation ledger's documented limits (§ governance-model §7); and the
daemon's ledger is its own, so it tracks provenance across the calls it actually
sees. With no taxonomy declared, the daemon still applies the normalizer's generic
heuristics, the same as the in-process path.

### The two ledgers are redundant, not split

In the daemon topology there are two per-value ledgers — the worker's (inside its
`IntentLoop`) and the daemon's (inside its per-session governor). They are **not**
split: `GovernedSession` has a single capability executor, so in daemon mode *every*
approved tool call is forwarded to the daemon, and the worker registers the
daemon-returned output into its own ledger. Both ledgers therefore see the same full
call stream and stay consistent; the daemon's is simply the authoritative copy that
survives a compromised worker. Their roles are complementary:

- the **worker** ledger drives the in-process gates plus the trace and the
  degradation ladder — but a code-compromised worker can bypass it;
- the **daemon** ledger drives the per-call gates server-side and cannot be bypassed
  by the worker, but it does not run degradation or write the trace.

Two genuine residuals remain, neither of which is a ledger divergence:

1. **Executor-internal tools.** A tool the LLM executor runs *without* yielding a
   `tool_use` event (a provider-native search, a built-in code interpreter) reaches
   neither the worker gates nor the daemon. This is the executor-contract concern —
   the executor must surface every tool call — not a provenance gap between ledgers.
2. **Degradation is worker-only.** The daemon enforces the per-call gates but not the
   session degradation ladder, so a compromised worker can bypass *degradation*
   (the slow tightening) while every individual call still hits the daemon's
   per-value gates (the hard floor).

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
