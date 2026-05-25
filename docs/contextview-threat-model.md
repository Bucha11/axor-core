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

1. `TaintEngine.propagate(TaintSource.WEB | MCP | FILE | API, scope=SESSION)` is called.
2. The session (or node) is marked tainted for the remainder of execution.
3. Subsequent privileged actions are subject to stricter policy evaluation.
4. `TaintPropagatedEvent` is written to the `DecisionTrace`.
5. Child agents spawned after taint inherits the parent's taint state.
6. The taint source identifier is used by `DegradationEngine` to track per-source
   behavioral pressure — accumulating denials from the same origin may quarantine
   that source and escalate the session's `DegradationLevel`.

Taint is sticky — it persists until governance explicitly clears it via
`TaintEngine.clear_by_governance()`.

---

## How Taint Is Updated

| Source | TaintSource | Default Scope |
|--------|-------------|---------------|
| Web fetch / search | `WEB` | SESSION |
| MCP result | `MCP` | SESSION |
| File read (untrusted workspace) | `FILE` | NODE or SESSION |
| External API | `API` | SESSION |
| Child agent output | `CHILD_AGENT` | SUBTREE |
| Memory (external session) | `MEMORY` | NODE |
| Provider tool output | `PROVIDER_TOOL` | INTENT or NODE |
| Unknown external | `UNKNOWN_EXTERNAL` | SESSION |

---

## Why Selection Is Not Treated as Trust

Relevance scoring and recency scoring are **content-neutral**.  They do not
evaluate whether content is trusted.

Taint provides the trust signal:

- Selection controls **visibility** (what the agent sees).
- Taint controls **privilege** (what the agent can do after seeing it).

A tainted session cannot expand its capability surface via tool calls that
would normally be allowed — it remains under the same policy ceiling, but
with additional escalation requirements.

This separation means that even if a malicious file enters `ContextView`, the
resulting execution is still governed.  The agent may be instructed to do
something harmful by the content, but the `ToolInterceptor` (Layer 1 policy)
will evaluate the resulting tool calls independently of the content.

---

## Tainted Visibility in Child Agents

When a child node is spawned from a tainted parent:

1. `GovernedNode._handle_spawn_child()` calls `child_node._taint_engine.inherit_from_parent(parent.state)`.
2. The child starts with `parent_inherited=True` in its `TaintState`.
3. The child cannot exceed the parent's policy ceiling (enforced by `_validate_child_policy()`).
4. The child cannot clear inherited taint via the worker path.

A child agent cannot be used to launder parent taint.

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
