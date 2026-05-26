# axor-core

[![CI](https://github.com/Bucha11/axor-core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-core/actions/workflows/ci.yml)
[![Security](https://github.com/Bucha11/axor-core/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-core/actions/workflows/security.yml)
[![PyPI](https://img.shields.io/pypi/v/axor-core?cacheSeconds=300)](https://pypi.org/project/axor-core/)
[![Python](https://img.shields.io/pypi/pyversions/axor-core?cacheSeconds=300)](https://pypi.org/project/axor-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Execution governance kernel for AI agents.**

Axor places a policy membrane between agent intent and tool execution.

Instead of trying to detect malicious prompts, Axor governs what actions are allowed to cross the execution boundary.

```
same agent · same model · same prompt
different policy → different execution behavior
```

---

## The Problem

Raw agent execution has no defense layer.

A production LLM agent can:

- call shell tools it was never meant to have
- read files far outside the task scope
- spawn uncontrolled child agents down arbitrary call trees
- carry tainted external content through N subsequent intents without any signal
- export intermediate reasoning that should never leave the execution boundary

None of this is a model problem. None of this is a prompt problem.

**It's an execution governance problem.**

Prompting an agent to behave safely is not a defense. It's a suggestion.

---

## Not Another Agent Framework

| Tool | What it does |
|------|-------------|
| LangGraph | orchestration |
| CrewAI | coordination |
| AutoGen | conversation |
| **Axor** | **execution governance** |

Axor doesn't build agents. Axor governs how agents execute.

Governance is not an adapter. It's a layer.

---

## The Reverse Osmosis Model

Raw agent execution is high-entropy. Agents trend toward unrestricted execution: accumulate context, expand scope, request more tools, spawn additional agents, export excessive state.

Axor applies policy membranes between agent intent and actual execution. Only execution that passes policy constraints survives.

```
Raw Agent Intent
       ↓
  ToolInterceptor      ← rule-based, runs first, unconditional
       ↓
  ReputationEnricher   ← cross-session resource reputation (axor-sentinel)
       ↓
  Phase 1 Rule         ← reputation ≥ 0.8 + external read → deterministic deny
       ↓
  AnomalyDetector      ← behavioral sequence scoring, pluggable
       ↓
  Isolated Verifier    ← canonical features only, no raw strings
       ↓
 Approved Execution
```

Minimum sufficient permeability: the smallest capability surface required for task completion.

Axor does not maximize agent freedom. It maximizes useful execution per unit of allowed capability.

---

## 30-Second Demo

```python
from axor_core import GovernedSession
from axor_core.contracts.mode import ExecutionMode
from axor_claude import ClaudeCodeExecutor

session = GovernedSession(
    executor=ClaudeCodeExecutor(api_key="sk-ant-..."),
    mode=ExecutionMode.PRODUCTION,
)

result = await session.run("Write tests for auth middleware")
```

Governance enforcement trace — same agent, no prompt changes:

```
tool_use: bash("rm -rf ./dist")
→ DENIED   ToolInterceptor   bash not in capabilities(focused_generative)

spawn_child(depth=4)
→ DENIED   TopologyPolicy    max_child_depth=1 for moderate task

context_expand(scope="repo")
→ REDUCED  ContextView(3 files, 1 840 tokens)  from 24 files

tool_use: read("../../../etc/passwd")
→ DENIED   LeaseValidator    path outside allowed_paths after normalization

export(include_reasoning=True)
→ FILTERED ExportFilter      mode=summary_only

task: completed within governed execution envelope
```

Denied capabilities are intercepted before reaching the executor. Context is scoped before the executor receives it. Path-based access is validated against normalized allowed paths before execution.

---

## How Enforcement Works

Axor enforces policy at three independent layers. A deny at any layer is final — later layers cannot override it.

**Layer 1 — Rule-Based Interception.** Every tool call is evaluated against the active policy before execution. Synchronous, unconditional. A Layer 1 deny cannot be overridden by Layer 2 or 3.

**Layer 2 — Behavioral Anomaly Scoring.** After Layer 1 approves, the execution *sequence* is evaluated — not just the individual call. Pluggable via `axor-classifier-simple` (scikit-learn, zero tokens) or `axor-classifier-llm`.

**Layer 3 — Isolated Intent Verification.** Gray-zone intents reach an LLM verifier — but only as canonical behavioral features. Raw strings (task text, file paths, tool arguments) are stripped by `IntentCanonicalizer` before the verifier sees anything. A prompt injection in a file path cannot influence a governance decision.

**DegradationEngine — Source-Aware Session Degradation.** Running in parallel with the cascade, the `DegradationEngine` converts accumulated denial signals and taint metadata into a session-level `DegradationLevel` that progressively narrows the capability surface. Unlike a simple "N denials → restrict" counter, degradation is **source-aware**: a malicious document quarantines its origin; clean sources in the same session continue at reduced but functional capability. Only when session-level thresholds are crossed does the whole session degrade.

| Level | Effect |
|-------|--------|
| `NORMAL` | Baseline — full policy in effect |
| `CAUTIOUS` | Triggering document soft-blocked from context |
| `RESTRICTED` | Quarantined source blocked; write/bash/export removed for tainted calls |
| `LOCKED` | All tools frozen; only `read` + `escalate` permitted; export forced `RESTRICTED` |
| `TERMINAL` | Session stopped; `SessionTerminatedError` raised on next intent |

Degradation level is monotonically increasing — it never decreases without explicit `GovernanceAuthority` clearance (same authority as `TaintEngine.clear_by_governance`).

See [docs/reverse-osmosis.md](docs/reverse-osmosis.md) for the full enforcement model, taint propagation semantics, and adversarial test coverage.

---

## Execution Modes

| Mode | Isolation | Classifier | On ambiguity |
|------|-----------|------------|--------------|
| `LIBRARY` | None — same process | Enabled | Escalate |
| `PRODUCTION` | LockedExecutor — bypass raises `GovernanceBypassError` | Enabled | Escalate |
| `STRICT` | LockedExecutor + audit-required trace | Task classifier disabled for policy selection | Deny |

`STRICT` is a superset of `PRODUCTION`. In `STRICT`, there are no content-based policy decisions — policy is set by the operator, not derived from task text.

---

## Task-Aware Policies

Policy may be selected dynamically from task signals, or fixed explicitly by the operator.

| Task | Policy | Context | Compression | Children |
|------|--------|---------|-------------|----------|
| Fix a typo | focused_mutative | minimal | aggressive | denied |
| Write tests | focused_generative | minimal | balanced | denied |
| Refactor module | moderate_mutative | moderate | balanced | shallow (d=1) |
| Rewrite repo | expansive | broad | light | allowed (d=3) |
| Audit security | focused_readonly | minimal | aggressive | denied |

The heuristic classifier ships in core: rule-based, zero tokens, zero latency. Plug in `axor-classifier-simple` or `axor-classifier-llm` for higher accuracy on ambiguous tasks.

Policies derive minimum sufficient execution conditions — not static caps. A "rewrite repo" task gets broad context because the task requires it, not because limits were relaxed.

---

## Provider Independence

The same governance semantics apply regardless of provider.

```python
# Swap the executor. Policy stays identical.
session = GovernedSession(executor=ClaudeCodeExecutor(), policy=policy)
session = GovernedSession(executor=OpenRouterExecutor(), policy=policy)
session = GovernedSession(executor=LangChainExecutor(), policy=policy)
```

Providers execute. Policies govern.

Cross-provider parity is tested: Claude, mock-OpenAI, and mock-OpenRouter normalizers produce identical `NormalizedIntent` for the same semantic intent.

---

## Federation

Every child agent is a `GovernedNode` — not a raw sub-agent.

```
executor yields spawn_child
  → IntentLoop intercepts before CapabilityExecutor
  → child policy validated against parent ceiling
  → child inherits parent taint state
  → child context = scoped slice of parent context
  → child runs under its own governed envelope
```

Child capabilities cannot exceed the parent's. Child taint cannot be laundered. Child tokens roll up to the parent budget. `session.total_tokens_spent()` always reflects the full spawn tree. Child nodes share the parent's `DegradationEngine` instance — a child cannot start below the parent's current `DegradationLevel` (invariant D-6).

```
depth=1   GovernedNode
              └── GovernedNode    ← taint inherited, policy bounded by parent

depth=3   GovernedNode
              ├── GovernedNode
              └── GovernedNode
                    └── GovernedNode  ← cannot exceed grandparent ceiling
```

---

## Architecture

```
axor_core/
├── contracts/      Pure contracts — no business logic, no side effects
├── node/           Governance boundary: interception, canonicalization, export
├── capability/     Tool permission derivation, execution, lease validation
├── taint/          TaintEngine — sticky propagation, source tracking, clearance audit
├── degradation/    DegradationEngine — source-aware session degradation state machine
├── policy/         Dynamic policy selection — heuristic + external classifier
├── context/        Session-scoped context: compression, cache, selection
├── budget/         Token accounting across full spawn tree
├── trace/          Decision trace collection and access control
├── worker/         GovernedSession, SlashCommandRouter, Dispatcher
└── errors/         GovernanceBypassError, TaintClearanceError, DegradationClearanceError,
                    SessionTerminatedError, SpawnValidationError
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full module tree and design invariants.

---

## Benchmarks

Governance is measurable. Methodology and reproduction commands are documented in [BENCHMARKS.md](../BENCHMARKS.md).

| Metric | Result |
|--------|--------|
| Token reduction vs raw execution | 30.8% average |
| Topology containment (child spawn denial) | 100% of policy-blocked attempts |
| Context reduction under scoped policy | up to 61% fewer tokens injected |
| Export leak rate | 0 leaks observed across benchmark runs |

What benchmarks do **not** prove: full prompt injection prevention, covert channel elimination, semantic malware detection, all provider edge cases, or future attack families not yet in the regression suite.

---

## Ecosystem

| Package | Role |
|---------|------|
| [`axor-core`](https://github.com/Bucha11/axor-core) | Governance kernel — this package |
| [`axor-sentinel`](../axor-sentinel/README.md) | Cross-session behavioral analysis — slow-and-low staging detection |
| [`axor-daemon`](https://github.com/Bucha11/axor-daemon) | Process-isolated capability executor — enforcement outside the agent process |
| [`axor-cli`](https://github.com/Bucha11/axor-cli) | Governed terminal runtime |
| [`axor-claude`](https://github.com/Bucha11/axor-claude) | Claude / Claude Code adapter |
| [`axor-langchain`](https://github.com/Bucha11/axor-langchain) | LangChain governance middleware |
| [`axor-classifier-simple`](https://github.com/Bucha11/axor-classifier-simple) | ML task signal derivation (zero tokens) |
| [`axor-classifier-llm`](https://github.com/Bucha11/axor-classifier-llm) | LLM verifier for gray-zone escalation |
| [`axor-memory-sqlite`](https://github.com/Bucha11/axor-memory-sqlite) | Cross-session memory (SQLite) |
| [`axor-telemetry`](https://github.com/Bucha11/axor-telemetry) | Privacy-preserving governance feedback |
| [`axor-benchmarks`](https://github.com/Bucha11/axor-benchmarks) | Governance proof layer |

---

## Installation

```bash
pip install axor-core
```

Zero required dependencies by design. Provider adapters are separate packages:

```bash
pip install axor-claude          # Claude / Claude Code
pip install axor-langchain       # LangChain governance middleware
pip install axor-memory-sqlite   # Cross-session persistence (SQLite)
```

Memory providers are optional — install only if you need cross-session persistence:

```bash
pip install axor-memory-sqlite   # SQLite, local, zero extra dependencies
```

Classifiers are optional — install only if you need ML-based classification or anomaly detection:

```bash
pip install axor-classifier-simple[ml]  # ML task signal classifier + anomaly detector
pip install axor-classifier-llm[llm]    # LLM gray-zone verifier (requires Anthropic SDK)
```

`axor-core` has **zero required dependencies** by design.

cd axor-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
```

The CI security gate enforces minimum adversarial variant counts per category. Dropping below minimums fails the build.

---

## Contributing

Issues, PRs, and governance policy proposals welcome.

If you're building multi-agent systems and hitting containment problems — open an issue. That is what this is for.

When submitting a security fix: include a test that reproduces the bypass before the fix, and confirms it is blocked after.

---

## License

MIT

---

> Agents should not self-govern execution.
