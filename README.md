# axor-core

[![CI](https://github.com/Bucha11/axor-core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-core/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/axor-core?cacheSeconds=300)](https://pypi.org/project/axor-core/)
[![Python](https://img.shields.io/pypi/pyversions/axor-core?cacheSeconds=300)](https://pypi.org/project/axor-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


**Provider-agnostic governance kernel for agent systems.**

Raw agents execute freely. Axor turns them into governed nodes.

---

## What is Axor

Axor is not an agent framework. It is not an orchestration layer. It is not a provider wrapper.

Axor is a **governance kernel** — it takes arbitrary executors and runs them under governed execution conditions.

```
same agent + same provider + different governance envelope = different execution behavior
```

Most frameworks ask: *how do agents cooperate.*
Axor asks: *under what governed conditions are they allowed to act.*

---

## The Problem

Raw agents operating without governance:

- See too much context — entire conversation history, all files ever read
- Accumulate uncontrolled state — no cache, no deduplication, no compression
- Call tools without explicit permission — no policy, no audit trail
- Spawn sub-agents without hard boundaries — no depth limits, no context slicing
- Export intermediate reasoning freely — no export contracts

This leads to unstable cost, unstable scope, weak reproducibility, and poor topology discipline.

---

## Core Concept

The central primitive is a **GovernedNode** — a boundary that wraps any raw executor and enforces execution rules.

Every execution in Axor — whether a single agent or a tree of federated agents — is a `GovernedNode`. Flat execution is simply `max_child_depth=0`. Federation is not a special mode. It is the default architecture with depth controlled by governance.

```
depth=0   GovernedNode(executor)          # flat — single agent
depth=1   GovernedNode(executor)          # one level of children
              └── GovernedNode(executor)
depth=N   GovernedNode(executor)          # arbitrary federation
              ├── GovernedNode(executor)
              └── GovernedNode(executor)
                    └── GovernedNode(executor)
```

---

## Installation

```bash
pip install axor-core
```

Provider adapters are separate packages with zero coupling to core:

```bash
pip install axor-claude    # Claude / Claude Code
pip install axor-openai    # OpenAI (coming soon)
```

Memory providers are optional — install only if you need cross-session persistence:

```bash
pip install axor-memory-sqlite   # SQLite, local, zero extra dependencies
```

`axor-core` has **zero required dependencies** by design.

---

## Quick Start

```python
import asyncio
from axor_core import GovernedSession, Invokable, CapabilityExecutor, ToolHandler
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from typing import AsyncIterator


# 1. Wrap any executor
class MyExecutor(Invokable):
    async def stream(self, envelope: ExecutionEnvelope) -> AsyncIterator[ExecutorEvent]:
        # envelope contains: task, context, policy, capabilities, cancel_token
        # executor only sees what governance allows
        yield ExecutorEvent(
            kind=ExecutorEventKind.TEXT,
            payload={"text": "result"},
            node_id=envelope.node_id,
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": {"input_tokens": 100, "output_tokens": 50, "tool_tokens": 0}},
            node_id=envelope.node_id,
        )


# 2. Register tool handlers
class ReadHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args: dict) -> str:
        return open(args["path"]).read()


# 3. Create session and run
async def main():
    cap_executor = CapabilityExecutor()
    cap_executor.register(ReadHandler())

    session = GovernedSession(
        executor=MyExecutor(),
        capability_executor=cap_executor,
    )

    result = await session.run("write a test for the payment endpoint")
    print(result.output)
    print(f"policy: {result.metadata['policy']}")   # focused_generative
    print(f"tokens: {result.token_usage.total}")

asyncio.run(main())
```

With Claude (requires `pip install axor-claude`):

```python
import axor_claude

session = axor_claude.make_session(
    api_key="sk-ant-...",           # or set ANTHROPIC_API_KEY
    soft_token_limit=100_000,
)
result = await session.run("refactor the auth module")
```

With agent definition and memory:

```python
from axor_core import AgentDefinition, AgentDomain
from axor_memory_sqlite import SQLiteMemoryProvider

session = axor_claude.make_session(
    api_key="sk-ant-...",
    agent_def=AgentDefinition(
        name="research-assistant",
        domain=AgentDomain.RESEARCH,
        personality="You are a meticulous research assistant. Always cite sources.",
        memory_namespaces=("research",),
    ),
    memory_provider=SQLiteMemoryProvider("~/.axor/memory.db"),
)
result = await session.run("summarize recent papers on transformer attention")
```

---

## How It Works

### Execution Pipeline

```
raw input
  → TaskAnalyzer          → TaskSignal (complexity × nature)
  → PolicySelector        → ExecutionPolicy (dynamic, per task)
  → PolicyComposer        → final policy (extensions + parent restrictions)
  → ContextManager        → ContextView (shaped, compressed, cached)
  → EnvelopeBuilder       → ExecutionEnvelope (central execution object)
  → BudgetPolicyEngine    → pre-execution optimization check
  → IntentLoop            → stream interception
      tool_use event      → Intent → resolve → execute or deny
      tool result         → pushed to ToolResultBus → back to executor
      text event          → passes through export filter
      cancel check        → cooperative cancellation at every boundary
  → ExportFilter          → governed ExecutionResult
  → TraceCollector        → decision lineage recorded
  → ContextManager.update → result persisted into session context
```

The executor receives an `ExecutionEnvelope` — never raw context, never unfiltered tool lists.

### Dynamic Policy

Policy is selected dynamically from the task signal — not configured statically:

| Task | Complexity | Nature | Policy | Context | Children | Tools |
|------|-----------|--------|--------|---------|----------|-------|
| Write a test for endpoint | FOCUSED | GENERATIVE | focused_generative | minimal | denied | read, write |
| Explain why this is slow | FOCUSED | READONLY | focused_readonly | minimal | denied | read, search |
| Fix auth bug | FOCUSED | MUTATIVE | focused_mutative | minimal | denied | read, write, bash |
| Refactor auth module | MODERATE | MUTATIVE | moderate_mutative | moderate | shallow (d=1) | all |
| Rewrite repo to Go | EXPANSIVE | any | expansive | broad | allowed (d=3) | all |

**Principle: minimum sufficient for quality — not a hard cap.**

"Rewrite repo to Go" gets broad context and all tools not because limits are overridden — because the task genuinely requires it. A focused task gets minimal context not because we restrict it — because that is sufficient for quality.

### Intent Model

All privileged actions surface as **Intents**. The executor never performs them directly.

```
executor requests bash tool
  → tool_use event intercepted by IntentLoop
  → Intent(kind=TOOL_CALL, payload={tool: "bash", args: {...}})
  → policy check: is "bash" in capabilities.allowed_tools?
  → approved → CapabilityExecutor.execute() → result → ToolResultBus → executor
  → denied   → structured denial → ToolResultBus → executor
              executor sees {"error": "tool_denied", "reason": "..."}
              executor never knows governance intercepted it
```

### ToolResultBus — Multi-turn Tool Loop

For adapters like `axor-claude` that drive their own tool loop, the `ToolResultBus` provides a clean async handoff between `IntentLoop` (which executes tools) and the executor (which needs results to continue the conversation):

```python
# In ClaudeCodeExecutor.stream():
self._bus.expect(1)
yield ExecutorEvent(kind=TOOL_USE, ...)   # executor suspends here
                                           # IntentLoop: resolve → execute → bus.push()
results = await self._bus.drain()          # result already in queue
# continues with next Claude API call using the result
```

Two modes in `IntentLoop`:
- **Default** (no bus): yields tool result as TEXT event — works for mock executors and tests
- **Bus mode**: calls callback, no yield — adapter manages result injection into conversation

### Context Optimization

The `ContextManager` is session-scoped (persists across turns) and eliminates eleven categories of waste:

| Waste | Mechanism |
|-------|-----------|
| Verbose old assistant prose | `compressor.py` — extracts key decisions, discards verbose text |
| Oversized command outputs | Smart truncation: head + tail, not naive cut |
| Stale git/branch history | `invalidator.py` — git TTL-based cache invalidation |
| Repeated file reads | `cache.py` — content hash cache, file never re-read if unchanged |
| Symbol drift | `symbol_table.py` — deprecated symbols get relevance penalty |
| File rediscovery | `cache.py` — `cached_paths()` registry prevents re-discovery |
| Unnecessary rereads | Post-execute callback → auto-cache on every read |
| Turn accumulation | Rolling summary after N turns (threshold: LIGHT=20, BALANCED=6, AGGRESSIVE=3) |
| Error repetition | Collapse repeated errors to single entry with count |
| Working set drift | Inactive files penalised by turn distance |
| Path explosion | Absolute paths normalized to relative |

**Critical design:** waste elimination always runs regardless of compression mode. Mode controls aggressiveness (token limits, thresholds) — not whether optimization happens. `LIGHT` mode still deduplicates, collapses errors, and normalizes paths.

**Policy per turn:** `ContextManager.build(raw_state, lineage, policy=policy)` receives the actual policy selected for each task. A `rewrite repo` task gets `BROAD` context with `LIGHT` compression. A `write test` task gets `MINIMAL` context with `BALANCED` compression. The session-scoped manager remembers files and symbols across turns regardless.

---

## Federation

Child agents are not raw sub-agents. Every child is a `GovernedNode` with its own derived governance envelope — a context slice from the parent, a restricted policy, and an independent lineage.

### How spawn works

The executor requests a child by emitting a `spawn_child` tool use event. `IntentLoop` intercepts it — before the tool executor — and routes it to `_handle_spawn_child` via `SpawnCallback`:

```
executor yields spawn_child tool_use
  → IntentLoop detects tool_name == "spawn_child"
  → SpawnCallback fires (registered by wrapper.py)
  → policy check: child_mode == ALLOWED? depth < max_child_depth?
  → approved → ChildSpawner.prepare_child()
      → derive child lineage (depth+1, ancestry)
      → derive child policy (parent restrictions always applied)
      → build child context slice (fraction of parent context)
      → create child GovernedNode
      → run child
  → denied → structured denial string → ToolResultBus → executor
```

Denied spawns never crash — the executor receives a structured result and can handle the denial gracefully.

### Separate child executor

By default children reuse the parent executor. For federated tasks where children should behave differently, pass a `child_executor`:

```python
import axor_claude
from axor_core import GovernedSession, CapabilityExecutor, presets

# parent: full Claude with all tools
parent_executor = axor_claude.ClaudeCodeExecutor(api_key="sk-ant-...")

# children: same model, lighter — no write/bash
child_executor  = axor_claude.ClaudeCodeExecutor(api_key="sk-ant-...")

cap = CapabilityExecutor()
# ... register handlers ...

session = GovernedSession(
    executor=parent_executor,
    capability_executor=cap,
    child_executor=child_executor,   # children use this
)

result = await session.run("analyze security, performance, maintainability",
                           policy=presets.get("federated"))
```

Child executors propagate to grandchildren — the `child_executor` is inherited down the full spawn tree.

### Token accounting

Child tokens are recorded in the parent budget tracker. `session.total_tokens_spent()` always reflects the full tree:

```python
result = await session.run("task", policy=presets.get("federated"))

# parent tokens only
print(result.token_usage.total)

# parent + all children + grandchildren
print(session.total_tokens_spent())
```

### Trace lineage

Every spawn is recorded:

```python
for trace in session.all_traces():
    for event in trace.events:
        if event.kind.value == "child_spawned":
            print(f"child {event.child_node_id} at depth {event.child_depth}")
        if event.kind.value == "child_completed":
            print(f"child tokens: {event.payload['tokens']}")
```

### Policies that allow children

| Policy | child_mode | max_depth |
|--------|-----------|-----------|
| focused_* | DENIED | 0 |
| moderate_* | SHALLOW | 1 |
| expansive | ALLOWED | 3 |
| preset:federated | ALLOWED | 3 |

---



### Budget Tracking

```python
session = GovernedSession(
    executor=MyExecutor(),
    capability_executor=cap_executor,
    soft_token_limit=100_000,    # advisory — triggers optimization signals
)
# Thresholds (% of soft_limit):
# 60% → suggest context compression
# 80% → deny new child nodes
# 90% → restrict export mode
# 95% → hard stop via CancelToken
```

Budget is an optimizer, not a hard cap. A task that needs 120k tokens to complete correctly will not be artificially truncated — budget signals suggest compression and deny new children to preserve quality of the current execution.

### Preset Policy

```python
from axor_core import presets

result = await session.run(
    "analyze this code",
    policy=presets.get("readonly"),  # skip task analysis
)

# available: readonly, sandboxed, standard, federated
```

### External Classifier

The heuristic classifier ships in core (rule-based, ~65-70% accuracy, zero tokens, zero latency). For higher accuracy on ambiguous tasks, plug in an external classifier:

```python
from axor_core import GovernedSession, SignalClassifier
from axor_core.contracts.policy import TaskSignal

class MyTrainedClassifier(SignalClassifier):
    async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
        # your ML model — returns (signal, confidence)
        ...

session = GovernedSession(
    executor=MyExecutor(),
    capability_executor=cap_executor,
    classifier=MyTrainedClassifier(),
    # escalates to classifier only when heuristic confidence < 0.75
)
```

Classifier packages:
- `axor-classifier-local` — trained on your own traces (coming soon)
- `axor-classifier-cloud` — trained on anonymized traces from all users (coming soon)

### Extensions

```python
from axor_core.contracts.extension import ExtensionLoader, ExtensionBundle, ExtensionFragment

class MySkillLoader(ExtensionLoader):
    async def load(self) -> ExtensionBundle:
        return ExtensionBundle(
            fragments=(
                ExtensionFragment(
                    name="domain_context",
                    context_fragment="This project uses FastAPI with async SQLAlchemy.",
                    required_tools=("read",),
                    policy_overrides={},
                    source="my_loader",
                ),
            ),
        )

session = GovernedSession(
    executor=MyExecutor(),
    capability_executor=cap_executor,
    extension_loaders=[MySkillLoader()],
)
```

Extensions are sanitized before use — fragment size is capped, reserved governance commands cannot be overridden.

### Cancellation

```python
import signal

session = GovernedSession(executor=..., capability_executor=...)

# cancel from signal handler (safe for threads)
signal.signal(signal.SIGINT, lambda s, f: session.cancel("user interrupt"))

# cancel from coroutine
async def watchdog():
    await asyncio.sleep(30.0)
    session.cancel("timeout")

# partial governed result always returned
result = await session.run("long running task")
if result.metadata.get("cancelled"):
    print(f"cancelled: {result.metadata['cancel_reason']}")
```

Cancellation is **cooperative** — the intent loop checks the `CancelToken` at every event boundary. Five cancel reasons: `user_abort`, `budget_exhausted`, `parent_cancelled`, `policy_denied`, `timeout`.

Child nodes receive independent tokens — parent cancellation does not automatically cancel children:

```python
child_token = parent_token.child_token()
# explicit propagation if needed:
child_token.cancel(CancelReason.PARENT_CANCELLED)
```

---

## Agent Definition

`AgentDefinition` is a first-class entity — a declaration that governance reads at session construction time to set domain defaults, inject personality, and determine what memory to load.

Executors never see `AgentDefinition` directly. It is governance input, not execution input.

```python
from axor_core import AgentDefinition, AgentDomain, TrustLevel

agent = AgentDefinition(
    name="research-assistant",
    domain=AgentDomain.RESEARCH,       # adjusts policy defaults + domain detection
    trust_level=TrustLevel.STANDARD,   # affects capability derivation
    personality="You are a meticulous research assistant. Always cite sources.",
    default_tools=("read", "search"),  # hint, not a hard override
    memory_namespaces=("research", "shared"),  # loaded at session start
)

session = GovernedSession(
    executor=MyExecutor(),
    capability_executor=cap,
    agent_def=agent,
)
```

### Domains

| Domain | Policy defaults | Context | Compression |
|--------|----------------|---------|-------------|
| `CODING` | standard coding defaults | moderate | balanced |
| `RESEARCH` | `preset:research` defaults | **broad** | **light** — knowledge must not be over-compressed |
| `SUPPORT` | `preset:support` defaults | minimal | aggressive — keep turns lean |
| `ANALYSIS` | `preset:analysis` defaults | moderate | balanced |
| `GENERAL` | task-signal only | varies | varies |

Domain detection also runs on raw input text — agent domain takes priority over task-level detection:

```python
# agent domain = RESEARCH → all tasks default to research policy
# regardless of whether the specific task says "write a test"
session = GovernedSession(agent_def=AgentDefinition(domain=AgentDomain.RESEARCH), ...)
```

### Personality

Personality is injected as a **pinned context fragment** — it survives all compression and is always first in `ContextView`. Injected once per session, never duplicated across turns.

```python
agent = AgentDefinition(
    name="security-auditor",
    personality="""
You are a security-focused code reviewer.
Always check for: SQL injection, XSS, hardcoded secrets, insecure defaults.
Never suggest accepting user input without validation.
""",
)
```

### Trust Levels

`TrustLevel` adjusts what `CapabilityResolver` is willing to grant when policy allows a range:

| Level | Effect |
|-------|--------|
| `RESTRICTED` | Read-only, no bash, no spawn — regardless of policy |
| `STANDARD` | Default — capabilities as defined by policy |
| `ELEVATED` | May use extended tools if policy permits |
| `FULL` | No trust-based restrictions (policy still governs) |

Child agents inherit at most `STANDARD` trust, never higher than parent:

```python
child_def = agent.child_def(name="sub-agent")
# child_def.trust_level <= STANDARD
```

---

## Memory

Memory is pluggable. Core defines `MemoryProvider`. Implementations live in separate packages.

```bash
pip install axor-memory-sqlite   # SQLite, local, zero extra dependencies
```

### FragmentValue — lifecycle semantics

Every context fragment has a `value` that controls how the compressor treats it:

| Value | Compressor behavior | Use for |
|-------|--------------------|---------| 
| `PINNED` | Never touched — survives all turns | Personality, system rules, critical facts |
| `KNOWLEDGE` | Dedup + collapse only — no truncation | RAG docs, domain references, API specs |
| `WORKING` | Normal pipeline — default | Tool results, file reads, intermediate findings |
| `EPHEMERAL` | Aggressive compression regardless of mode | Debug output, scratch, one-turn observations |

Eviction priority: `EPHEMERAL` first → `WORKING` → `KNOWLEDGE` → `PINNED` (never).

### Basic usage

```python
from axor_memory_sqlite import SQLiteMemoryProvider
from axor_core import AgentDefinition, FragmentValue, MemoryFragment

provider = SQLiteMemoryProvider("~/.axor/memory.db")

session = GovernedSession(
    executor=...,
    capability_executor=...,
    agent_def=AgentDefinition(
        name="my-agent",
        memory_namespaces=("my-agent",),   # loaded at session start
    ),
    memory_provider=provider,
)

# save memory after a task
await provider.save([
    MemoryFragment(
        namespace="my-agent",
        key="project_context",
        content="This project uses FastAPI with async SQLAlchemy.",
        value=FragmentValue.KNOWLEDGE,    # retained but may be summarized
    ),
    MemoryFragment(
        namespace="my-agent",
        key="user_preference",
        content="User prefers type annotations everywhere.",
        value=FragmentValue.PINNED,       # survives forever
    ),
])
```

### Pin and knowledge helpers

`ContextManager` exposes two direct methods:

```python
from axor_core.contracts.context import ContextFragment

# pin — survives all compression, always first in view
session._context_manager.pin_fragment(ContextFragment(
    kind="skill",
    content="Never modify test files.",
    token_estimate=10,
    source="rule:no_test_modification",
    value="pinned",
))

# add_knowledge — compressed carefully, retained across turns
session._context_manager.add_knowledge(
    content="The payments module uses Stripe SDK v5.",
    source="knowledge:payments",
)
```

### Memory provider interface

```python
from axor_core import MemoryProvider, MemoryFragment, MemoryQuery, FragmentValue

class MyVectorProvider(MemoryProvider):
    async def load(self, query: MemoryQuery) -> list[MemoryFragment]: ...
    async def save(self, fragments: list[MemoryFragment]) -> None: ...
    async def delete(self, namespace: str, keys: list[str]) -> int: ...
    async def evict(self, namespace: str, values=(), max_age_seconds=None) -> int: ...
    async def namespaces(self) -> list[str]: ...

    # optional — override for semantic search
    async def search(self, query_text: str, namespace=None, max_results=10): ...
```

---

## Slash Commands

Slash commands are first-class governance objects — intercepted before reaching the executor:

| Class | Commands | Behavior |
|-------|----------|----------|
| **GOVERNANCE** | `/tools` `/policy` `/cost` `/status` | Answered from envelope/trace — executor never sees them |
| **CONTEXT** | `/compact` `/clear` `/memory` | Routed to context subsystem |
| **PASSTHROUGH** | everything else | Forwarded if policy allows, always logged in trace |

```python
result = await session.run("/cost")
# → "Tokens spent this session: 4,200"

result = await session.run("/compact")
# → context compressor runs before next execution

result = await session.run("/status")
# → Session: session_abc123 | Tokens: 4,200 | Nodes: 3 (1 children)
```

To include money estimates, pass model pricing explicitly. Prices are provider
and model specific, so core does not hardcode them:

```python
from axor_core import GovernedSession, TokenCostRates

session = GovernedSession(
    executor=executor,
    capability_executor=cap_executor,
    token_cost_rates=TokenCostRates(input_per_m=3.00, output_per_m=15.00),
)

result = await session.run("/cost")
# includes estimated cost plus input/cache-write/cache-read/output breakdown
```

Prompt-cache rates default to Anthropic-style multipliers when not supplied:
cache writes use `1.25 * input_per_m`; cache reads use `0.1 * input_per_m`.
Override `cache_creation_input_per_m` and `cache_read_input_per_m` when a
provider or model uses different pricing.

---

## Implementing an Adapter

Three things to implement:

```python
# 1. Invokable — translate ExecutionEnvelope → provider calls
from axor_core import Invokable
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

class MyProviderExecutor(Invokable):
    async def stream(self, envelope: ExecutionEnvelope):
        # envelope.task               — the task string
        # envelope.context            — ContextView (shaped, never raw)
        # envelope.capabilities       — what tools are allowed
        # envelope.cancel_token       — check before each yield
        # envelope.policy             — compression mode, export mode, etc.

        async for chunk in self._client.stream(
            prompt=envelope.task,
            tools=self._translate_tools(envelope.capabilities.allowed_tools),
        ):
            if envelope.cancel_token.is_cancelled():
                return
            yield self._translate_event(chunk, envelope.node_id)


# 2. ToolHandler — one per tool
from axor_core import ToolHandler

class BashHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "bash"

    async def execute(self, args: dict) -> str:
        import subprocess
        result = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr


# 3. ExtensionLoader — optional, for provider-specific context loading
from axor_core import ExtensionLoader
from axor_core.contracts.extension import ExtensionBundle, ExtensionFragment

class MyContextLoader(ExtensionLoader):
    async def load(self) -> ExtensionBundle:
        # read config files, project context, etc.
        return ExtensionBundle(fragments=(...))
```

For adapters with multi-turn tool loops (like Claude), implement `get_bus()` to receive a `ToolResultBus` and push tool results back to the executor after `IntentLoop` executes them.

---

## Trace & Observability

Every governance decision is recorded in the lineage trace:

```python
result = await session.run("refactor auth module")

for trace in session.all_traces():
    print(f"node: {trace.node_id}  policy: {trace.policy_name}")
    for event in trace.events:
        print(f"  {event.kind.value}")

# signal_chosen      {classifier: "heuristic", confidence: 0.72}
# policy_chosen      {context_mode: "moderate", child_mode: "shallow"}
# intent_approved    {tool: "read"}
# intent_denied      {tool: "bash", reason: "not in capabilities for focused_generative"}
# context_compressed {before: 4200, after: 1800, ratio: 0.43}
# tokens_spent       {input: 800, output: 400, cumulative: 1200}
```

Privacy by default:

```python
TraceConfig(
    local_only=True,        # traces never leave the machine
    persist_inputs=False,   # raw inputs never written to disk
    training_opt_in=False,  # anonymized records not sent for cloud training
)
```

---

## Architecture

```
axor-core/
└── axor_core/             (72 Python files)
    ├── contracts/         pure contracts — no business logic, no side effects
    │   ├── invokable.py   Invokable — stream(envelope) → AsyncIterator[ExecutorEvent]
    │   ├── cancel.py      CancelToken — cooperative cancellation, 5 reasons
    │   ├── envelope.py    ExecutionEnvelope — central execution object
    │   ├── policy.py      ExecutionPolicy, TaskSignal (+ domain), SignalClassifier
    │   ├── intent.py      Intent, IntentKind, ResolvedIntent
    │   ├── extension.py   ExtensionLoader, ExtensionBundle, ExtensionFragment
    │   ├── trace.py       DecisionTrace, 17 typed TraceEvent kinds
    │   ├── agent.py       AgentDefinition, AgentDomain, TrustLevel
    │   ├── memory.py      MemoryFragment, FragmentValue, MemoryProvider, MemoryQuery
    │   └── context.py     ContextFragment (+ value field), ContextView, LineageSummary
    │
    ├── policy/            dynamic policy selection
    │   ├── heuristic.py   HeuristicClassifier — rule-based, 0ms, 0 tokens
    │   ├── analyzer.py    TaskAnalyzer — heuristic + domain detection + external classifier
    │   ├── selector.py    PolicySelector — TaskSignal → ExecutionPolicy (7-policy matrix)
    │   ├── composer.py    PolicyComposer — merge: base + extensions + parent restrictions
    │   └── presets.py     readonly, sandboxed, standard, federated, research, support, analysis
    │
    ├── capability/        tool permission derivation and execution
    │   ├── resolver.py    CapabilityResolver — policy + trust_level → Capabilities (fail-closed)
    │   └── executor.py    CapabilityExecutor — executes approved intents, post-callbacks
    │
    ├── node/              governance boundary around executor
    │   ├── wrapper.py     GovernedNode — central primitive, wires all subsystems
    │   │                  child_executor param — separate executor for child nodes
    │   ├── envelope.py    EnvelopeBuilder — assembles ExecutionEnvelope
    │   ├── intent_loop.py IntentLoop — stream interception, ToolResultBus + SpawnCallback
    │   ├── export.py      ExportFilter — applies export contract to output
    │   └── spawn.py       ChildSpawner — governed child-node creation, lineage derivation
    │
    ├── budget/            token optimization
    │   ├── tracker.py     BudgetTracker — token accounting across node tree, thread-safe
    │   ├── estimator.py   BudgetEstimator — cost estimates, slice sufficiency checks
    │   └── policy_engine.py BudgetPolicyEngine — real-time optimizer (60/80/90/95% thresholds)
    │
    ├── context/           context management — session-scoped, persists across turns
    │   ├── manager.py     ContextManager — build/pin_fragment/add_knowledge, pinned always first
    │   ├── cache.py       ContextCache — file hash cache + tool result memoization (TTL)
    │   ├── compressor.py  ContextCompressor — FragmentValue-aware: pinned→untouched, ephemeral→aggressive
    │   ├── selector.py    ContextSelector — relevance scoring, working set management
    │   ├── invalidator.py ContextInvalidator — stale detection: git TTL, symbol drift
    │   ├── symbol_table.py SymbolTable — live symbol registry, rename detection, TODOs
    │   └── lineage.py     LineageManager — child context slice derivation
    │
    ├── trace/             governance decision recording
    │   ├── collector.py   TraceCollector — lineage-aware, thread-safe, privacy controls
    │   └── events.py      typed event constructors for all 17 event kinds
    │
    ├── extensions/        extension loading and sanitization
    │   ├── sanitizer.py   ExtensionSanitizer — size cap, reserved command protection
    │   └── registry.py    ExtensionRegistry — session-scoped active extensions
    │
    ├── worker/            entry layer
    │   ├── session.py     GovernedSession — agent_def + memory_provider + personality injection
    │   ├── commands.py    SlashCommandRouter — GOVERNANCE | CONTEXT | PASSTHROUGH
    │   └── dispatcher.py  Dispatcher — routes input to node flow
    │
    └── errors/            explicit error hierarchy rooted at AxorError
```

### Design Invariants

**Everything is a GovernedNode.** Flat execution is `depth=0`. No special cases for single-agent vs multi-agent.

**Core never imports providers.** Zero provider SDK imports in `axor_core/`. Verified by static analysis.

**Policy meaning belongs to core.** Adapters translate envelopes — they never define governance semantics.

**Executors never self-assign capabilities.** Always derived from policy by `CapabilityResolver` (fail-closed: unknown tool prefix = denied).

**Core does not decompose tasks.** Agents decide when to spawn children. Core governs each spawn and provides a minimum sufficient context slice.

**spawn_child is an intent, not a tool.** `IntentLoop` intercepts `spawn_child` tool_use events before they reach `CapabilityExecutor`. The routing is `SpawnCallback → _handle_spawn_child` — internal to core. Adapters never see federation.

**Denied spawns never crash.** `ChildNotAllowedError` and `MaxDepthExceededError` are caught in the spawn callback and returned as structured denial strings. The executor sees a tool result, not an exception.

**Child tokens belong to parent budget.** `session.total_tokens_spent()` always includes the full spawn tree. Budget thresholds apply to the cumulative total.

**Pinned fragments bypass all compression and selection.** `ContextManager._pinned_fragments` is prepended to `ContextView` after the full compress → select → scope pipeline. Compressor never sees them. Selector never scores them. They are always first.

**Personality is governance-injected, not adapter-injected.** `AgentDefinition.personality` is injected as a pinned fragment by `GovernedSession` — once per session, deduplicated by source. Adapters and executors never see `AgentDefinition` directly.

**Memory providers are injected, never imported by core.** `MemoryProvider` is a protocol defined in contracts. Core never imports implementations. `NullMemoryProvider` is the default — no-op, zero storage.

**Domain is a hint, not a constraint.** `AgentDomain` adjusts policy defaults and context selection — it does not restrict what tools an agent can use or what tasks it can perform.

**Waste elimination always runs.** Compression mode controls aggressiveness, not whether optimization happens. `LIGHT` mode still deduplicates, collapses repeated errors, and normalizes paths.

**Context policy is per-turn.** `ContextManager.build(raw_state, lineage, policy=policy)` receives the actual policy selected for each task. A `rewrite repo` task gets `BROAD` context with `LIGHT` compression. A `write test` task gets `MINIMAL` context with `BALANCED` compression. The session-scoped manager remembers files and symbols across turns regardless.

**Privacy by default.** `TraceConfig(local_only=True, persist_inputs=False)`. Nothing leaves the machine without explicit `training_opt_in=True`.

---

## Requirements

- Python 3.11+
- No required dependencies

### Development

```bash
git clone https://github.com/your-org/axor-core
cd axor-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
