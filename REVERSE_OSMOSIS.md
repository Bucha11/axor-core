# Reverse Osmosis: Defense Architecture

## Philosophy

Classic injection defense tries to detect malicious content at the input boundary —
sanitizing what comes in. This doesn't work reliably:

- Injections are trivially encoded, rephrased, split across turns
- Tool return values (MCP, web search, file reads) arrive after the policy check
- Filtering content means understanding semantics — an unsolvable problem at ingestion time

**Reverse osmosis flips the model.**

Don't filter what the agent reads. Govern what the agent *does*.
A compromised agent that cannot exfiltrate data, cannot spawn unauthorized children,
and cannot write outside its allowed paths is contained — regardless of what it was told.

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
       │  NormalizedIntent (no raw content)
       ▼
┌─────────────┐
│   Defense   │  ← separate process/context
│   Cascade   │    never sees raw tool returns or conversation
└─────────────┘
```

**The verifier must be isolated from the worker.**
If the worker is infected via prompt injection, asking it to verify itself is meaningless.
The verifier receives only a normalized intent schema — it is impossible to inject
through the same vector that infected the worker.

---

## NormalizedIntent Schema

The defense cascade never sees raw content. It receives a structured schema derived
from each intent:

```python
@dataclass(frozen=True)
class NormalizedIntent:
    # Identity
    tool: str                       # "write", "bash", "spawn_child", ...
    sequence: int                   # position in the intent stream

    # Structural args — shape only, never values
    args_schema: dict[str, str]     # {"path": "str", "content": "str"} — no content
    target_path: str | None         # normalized path if applicable

    # Causal signals — derived from intent history, not content
    preceded_by_external_read: bool # True if recent tool return came from outside
    executes_generated_code: bool   # True if write→execute sequence detected
    cross_origin_export: bool       # True if exporting data read from external source

    # Context
    depth: int                      # node depth in federation tree
    escalation_active: bool         # True if an escalation grant is in effect
    intents_since_last_external: int # turns since last external tool return
```

`preceded_by_external_read` is the key signal. It captures causality without
inspecting content: the agent read something external, and now it wants to act.
That pattern is the attack surface.

---

## Defense Cascade

Three layers. Each is optional and independently packaged.
Core always runs Layer 1. Layers 2 and 3 are injected at session construction.

```
Intent arrives
      │
      ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Rules  (axor-core, always)        │
│                                             │
│  Hard denies. Zero cost. Zero deps.         │
│  • write to path outside allowed set        │
│  • bash after external read without grant   │
│  • export when export_mode=RESTRICTED       │
│  • spawn_child when child_mode=DENIED       │
│  • escalation beyond EscalationPolicy       │
└──────────────────────┬──────────────────────┘
                       │ passed rules
                       ▼
┌─────────────────────────────────────────────┐
│  Layer 2: ML Scorer  (axor-classifier-simple│
│           optional)                         │
│                                             │
│  GradientBoosting on NormalizedIntent       │
│  features. Scores intent chain 0.0–1.0.     │
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
│  Separate model, clean context.             │
│  Receives NormalizedIntent only.            │
│  Returns: allow | deny | ask_human          │
│                                             │
│  Cannot be infected through the same        │
│  vector as the worker — different process,  │
│  no shared conversation state.              │
└─────────────────────────────────────────────┘
```

### Layer 1 — Rules

Always present. Implemented in `axor_core/node/intent_loop.py`.

The rules are not heuristics — they are hard constraints derived from `ExecutionPolicy`.
No configuration needed. They fire before any ML inference.

Key rules:

| Signal | Deny condition |
|---|---|
| `write` | path outside `ToolPolicy.allowed_paths` |
| `bash` | `preceded_by_external_read=True` and no escalation grant |
| `export` | `export_mode=RESTRICTED` |
| `spawn_child` | `child_mode=DENIED` or depth ≥ `max_child_depth` |
| `escalate_policy` | tool not in `EscalationPolicy.grantable_tools` |

### Layer 2 — ML Scorer

Package: `axor-classifier-simple`. No LLM dependency.

**Features used** (all from NormalizedIntent, no content):

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

**Model**: GradientBoosting. Trained on:
1. Synthetic data — generated injection scenarios and benign traces
2. Anonymized user traces (opt-in, client-side embeddings, no raw content)

**Output**: anomaly score 0.0–1.0.

Thresholds are configurable per deployment. Default:
- `< 0.3` → allow (pass to policy)
- `> 0.7` → deny
- `0.3–0.7` → escalate to Layer 3 if available, else deny

### Layer 3 — LLM Verifier

Package: `axor-classifier-llm`. Requires LLM API access.

**Isolation guarantees:**
- Separate API call, separate context window
- Receives only serialized `NormalizedIntent` — no raw args, no conversation history
- System prompt is static and not user-modifiable
- Cannot be reached by content from the infected agent's context

**Output**: structured verdict — `allow`, `deny`, or `ask_human`.

`ask_human` surfaces to the operator via the `escalation_callback` mechanism,
same path as `EscalationPolicy.require_human=True`.

---

## Postmortem Mode

The cascade is not only a pre-execution filter. Intent sequences are recorded in
`DecisionTrace` for every session. This enables:

- **Why was this denied?** — full intent chain with scores at each layer
- **Pattern analysis** — which intent sequences correlate with injection attempts
- **Threshold tuning** — adjust ML thresholds based on false positive/negative rates
- **Training data** — confirmed attacks become synthetic training examples for Layer 2

The trace is the primary debugging surface. When the system gets something wrong,
the postmortem is in the trace, not in logs.

---

## Package Structure

```
axor-core                     always
  └─ Layer 1 (rules)          zero deps

axor-classifier-simple        pip install axor-classifier-simple
  └─ Layer 2 (ML scorer)      sklearn, numpy

axor-classifier-llm           pip install axor-classifier-llm
  └─ Layer 3 (LLM verifier)   anthropic / openai sdk
```

Layers are injected at `GovernedSession` construction time.
Core never imports them. They implement `AnomalyScorer` protocol:

```python
class AnomalyScorer(Protocol):
    async def score(self, intent: NormalizedIntent) -> float: ...
```

The cascade is assembled in `GovernedSession.__init__` based on what's passed in:

```python
session = GovernedSession(
    executor=...,
    capability_executor=...,
    anomaly_scorer=MLScorer(),        # Layer 2, optional
    anomaly_verifier=LLMVerifier(),   # Layer 3, optional
)
```

If neither is passed — Layer 1 only. If only ML — cascade stops at Layer 2.
If both — full three-layer cascade.

---

## Why Not Filter Inputs?

| Approach | Problem |
|---|---|
| Sanitize tool returns | Must parse semantics — unsolvable at scale |
| Blocklist injection phrases | Trivially bypassed by encoding, rephrasing |
| Isolate external reads | Breaks legitimate research/analysis workflows |
| **Govern intent sequence** | Attack surface is behavior, not content ✓ |

An agent that read a malicious README cannot exfiltrate your `.env` if:
- `export` is `RESTRICTED`
- `bash` is denied when `preceded_by_external_read=True`
- `write` is path-restricted

The content of the README is irrelevant. The behavioral constraint is the defense.
