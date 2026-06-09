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
4. IntentCanonicalizer    →  CanonicalizedIntent  (Layer 3 only)
5. DegradationEngine.apply_to_policy()  →  effective policy for this call
   (narrows ExecutionPolicy if session level is RESTRICTED/LOCKED/TERMINAL)
6. ToolInterceptor (IntentLoop) evaluates:
     Layer 1: rule-based policy  →  DENY | APPROVE | ASK_HUMAN
     Layer 2: ML anomaly detect  →  NORMAL | SUSPICIOUS | CRITICAL
     Layer 3: LLM verifier       →  gray-zone verification
7. DegradationEngine.record_signal(denial_or_none) → state updated for next intent
   (emits DegradationTransitionEvent / SourceQuarantinedEvent if level changed)
8. If allowed → executor runs underlying tool
9. If denied  → executor receives coarse DenialResponse
10. DecisionTrace written out-of-band (operator channel only)
```

Layers 2 and 3 run only if Layer 1 returns APPROVE.
No ML or LLM output can override a Layer 1 hard deny.
Step 5 (degradation pre-check) runs before Layer 1 — a quarantined or locked session
denies the tool before the policy cascade starts.

---

## Provider-Specific Call Flows

### Claude (Anthropic SDK)

```
AnthropicClient.stream()
  → StreamNormalizer.process(sdk_event)    # accumulate partial chunks
  → ClaudeNormalizer.normalize()           # ToolUseBlock → NormalizedIntent
  → IntentLoop._evaluate_tool_intent()     # Layer 1 policy
  → anomaly_detector.score()              # Layer 2 ML (if configured)
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
- Evaluate against `ExecutionPolicy` (Layer 1)
- Run anomaly detection (Layer 2) if configured
- Run LLM verifier (Layer 3) if anomaly score is in gray zone
- Apply per-value taint rules (`TaintEngine.register_value()` on external reads; `derive_value()` at sinks)
- Apply lease rules (`LeaseValidator.check_tool_allowed()`)
- Apply degradation pre-check (`DegradationEngine.apply_to_policy()` before Layer 1)
- Record `DegradationEngine.record_signal()` after every cascade outcome (pass or deny)
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
