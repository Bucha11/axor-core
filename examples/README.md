# Examples

Runnable examples that exercise the kernel. These are intentionally minimal and
dependency-free — they are not production adapters (the full Claude adapter
lives in `axor-claude`).

## `live_governance_demo.py` — live end-to-end governance of a real model

Runs an actual Anthropic model (`claude-haiku-4-5`) inside a `GovernedSession`
and shows the kernel governing the model's real tool calls.

The scenario is two innocent-looking steps:

1. the model reads a config file (`/app/config/db.env`) to inspect settings —
   this arms the **confidentiality floor**;
2. the model sends a benign health ping (`{"service":"api","status":"ok"}`, no
   secret in the body) to a monitoring endpoint.

The kernel connects them: the egress in step 2 is **denied on the fact that a
secret was read**, not on whether the outbound bytes contain the secret. The
model never sees a reason to refuse either step, so the *kernel* — not the
model's own judgement — is what stops the exfiltration. The demo asserts that a
denial was actually recorded, so a model that declines on its own is reported as
inconclusive rather than a false pass.

```
export ANTHROPIC_API_KEY=sk-ant-...
python -m examples.live_governance_demo
```

Expected output ends with `PASS` and a recorded
`IntentDeniedEvent: taint enforcement (per-value): ... confidentiality (egress
under the sound floor ...)`.

## `anthropic_executor.py` — a minimal real `Invokable`

A small Claude executor that drives the native tool-use loop and feeds every
tool call through the kernel. It implements the `Invokable.stream()` contract
plus the `get_bus()` result-bus hook the node wrapper uses to hand governed tool
results back to the model. The API call is a raw `urllib` POST run in a worker
thread — no SDK — which keeps the example aligned with the core's zero-dependency
design. Reuse it as the starting point for your own provider adapter.

## `run_agentdojo.py` + `agentdojo_adapter.py` — the AgentDojo benchmark

Runs the [AgentDojo](https://github.com/ethz-spylab/agentdojo) prompt-injection
benchmark against a real Claude model, comparing an undefended pipeline to one
where every tool call passes an axor `ToolCallGovernor` first
(`GovernedToolsExecutor`). `agentdojo_adapter.py` also provides `RawAnthropicLLM`,
a raw-urllib AgentDojo LLM element (the SDK cannot connect from this sandbox).

```
pip install agentdojo
export ANTHROPIC_API_KEY=sk-ant-...
python -m examples.run_agentdojo
```

See **`agentdojo_results.md`** for a real run and an honest read of what it does
and does not show — including that `claude-haiku-4-5` already resists these
injections (no ASR headroom) and that naive content-taint over-blocks legitimate
transfers when a valid recipient co-occurs with the injection channel. The
governor itself is a first-class kernel API (`axor_core.governor.ToolCallGovernor`)
usable by any tool-wrapping framework, not just AgentDojo.
