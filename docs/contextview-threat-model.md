# ContextView Threat Model

## Problem

The claim "agents only see what policy permits" requires a concrete selection model.

Context slicing can itself introduce untrusted content into the agent context.
A malicious file that is *relevant* may be selected by a relevance-based
algorithm, even if the operator did not intend it to be visible.

---

## How Context Is Selected

`ContextManager` builds a `ContextView` from fragments.  Selection modes:

| Mode | Behavior | Trust implications |
|------|----------|-------------------|
| `relevance` | Select by embedding similarity to task | **Malicious-but-relevant content may be included** |
| `recency` | Select by last-modified or last-read time | Recent external reads included if not excluded by pattern |
| `explicit` | Only operator-pinned fragments | Highest trust — content must be explicitly added |
| `pinned` | Fragments locked in by `pin_fragment()` | Operator authority required to pin |
| `policy-only` | Only fragments matching policy source rules | Recommended for high-security deployments |

**Relevance-based selection is not a trust decision.**
If a malicious file is relevant to the task, it may be selected.

---

## What Can Be Excluded

Operators may configure exclusion patterns:

- Path prefixes (`/tmp/`, `/var/`, external workspace paths)
- Source tags (e.g., exclude `source=external_web`)
- Fragment kinds (e.g., exclude `kind=mcp_result`)
- Taint-infected fragments (when `exclude_tainted=True`)

Exclusion happens before `ContextView` is built — excluded fragments never
reach the executor.

---

## What Happens When External Content Is Selected

When a file read, web fetch, MCP result, or API response enters `ContextView`:

1. The produced **value** is registered with its causal root:
   `engine.register_value(content, CausalRoot.external_read(TaintSource.WEB | MCP | FILE | API))`.
2. Enforcement is **per-value**: a sink decides on the causal root of its driving
   argument (by content derivation), so only actions that actually *carry* a
   tainted/sensitive value are constrained — not every subsequent action.
3. The per-value *driving root* at a denied sink feeds `DegradationEngine` to track
   per-source behavioral pressure — accumulating denials from the same origin may
   quarantine that source and escalate the session's `DegradationLevel`.
4. Child agents inherit the parent's per-value ledger via `inherit_value_ledger`.

A value's provenance does not decay on its own; it is released only by governance —
per value via `endorse_value()` or wholesale via `clear_by_governance()`.

---

## How Taint Is Recorded

A registered value records which `TaintSource`(s) it derived from in its causal root.

| Source | TaintSource |
|--------|-------------|
| Web fetch / search | `WEB` |
| MCP result | `MCP` |
| File read (untrusted workspace) | `FILE` |
| External API | `API` |
| Child agent output | `CHILD_AGENT` |
| Memory (external session) | `MEMORY` |
| Provider tool output | `PROVIDER_TOOL` |
| Unknown external | `UNKNOWN_EXTERNAL` |

---

## Why Selection Is Not Treated as Trust

Relevance scoring and recency scoring are **content-neutral**.  They do not
evaluate whether content is trusted.

Taint provides the trust signal:

- Selection controls **visibility** (what the agent sees).
- Per-value provenance controls **privilege** (what may be done with a *specific
  value* derived from untrusted content).

A value carrying untrusted/sensitive provenance cannot be driven into a risky sink
(an integrity write, an egress) that would normally be allowed — while actions that
do not carry that value remain unconstrained (the per-value win).

This separation means that even if a malicious file enters `ContextView`, the
resulting execution is still governed.  The agent may be instructed to do
something harmful by the content, but the `ToolInterceptor` (Layer 1 policy)
will evaluate the resulting tool calls independently of the content.

---

## Tainted Visibility in Child Agents

When a child node is spawned from a tainted parent:

1. `GovernedNode._handle_spawn_child()` calls `child_node._taint_engine.inherit_value_ledger(parent_engine)`.
2. The child's per-value gate now flags any value the parent marked tainted/sensitive.
3. The child cannot exceed the parent's policy ceiling (enforced by `_validate_child_policy()`).
4. The child cannot release inherited provenance via the worker path.

A child agent cannot be used to launder a parent's tainted values.

---

## Required `ContextViewPolicy` Fields

When implementing `ContextViewPolicy`, operators must specify:

- `selection_mode`: one of `relevance | recency | explicit | pinned | policy-only`
- `max_files`: maximum number of file fragments
- `max_tokens`: token budget for context
- `exclude_patterns`: path/source patterns to exclude
- `include_patterns`: path/source patterns to force-include
- `source_trust_rules`: per-source trust level overrides
- `external_source_handling`: `include_with_taint | exclude | quarantine`
- `taint_integration`: `propagate | log_only | disabled`
