# Migration: ExecutionPolicy → AuthorityPolicy + ExecutionPlan

The legacy `ExecutionPolicy` mixed two concerns in one object: **authority**
(what the agent may effect — trusted, operator-defined) and **execution
planning** (how to run efficiently — advisory, classifier-shaped). Because
the task classifier selected that object, untrusted task text influenced the
capability surface. The split removes that channel:

```
before:  untrusted task text → classifier → ExecutionPolicy → tools/paths/spawn
after:   untrusted task text → classifier → ExecutionPlan   → context/budget only
         operator            → AuthorityPolicy              → tools/paths/spawn
```

## New API

```python
from axor_core import (
    GovernedSession, AuthorityPolicy, ChildAuthorityPolicy,
    HeuristicExecutionPlanner, plan_presets,
)

session = GovernedSession(
    executor=..., capability_executor=...,
    authority=AuthorityPolicy(                 # trusted: what MAY happen
        name="standard_workspace",
        tool_policy=ToolPolicy(allow_read=True, allow_write=True, allow_bash=True),
        allowed_paths=("/workspace",),
    ),
    planner=HeuristicExecutionPlanner(),       # advisory: how to run
    # default_plan=plan_presets.neutral(),     # optional: skip classification
)

result = await session.run(task)               # classifier shapes only the plan
result = await session.run(task, plan=plan_presets.repository())   # explicit plan
result = await session.run(task, authority=stricter_authority)     # per-run override
```

Resolution order — authority: `run(authority=)` → session `authority=` →
legacy classifier path (deprecated). Plan: `run(plan=)` → session
`default_plan=` → classifier + planner → `NEUTRAL_PLAN`.

## Mapping legacy fields

| legacy `ExecutionPolicy` field | new home |
|---|---|
| `tool_policy`, `allowed_paths`, `max_unattended_consequence`, `escalation_policy`, `allowed_passthrough_commands`, `allow_model_switch` | `AuthorityPolicy` |
| `child_mode`/`max_child_depth` (permission part) | `AuthorityPolicy.child_authority` (`allow_spawn`, `max_depth`) |
| `export_mode` (ceiling part) | `AuthorityPolicy.export_policy.max_mode` |
| `context_mode`, `compression_mode`, `child_context_fraction` | `ExecutionPlan` |
| `child_mode` (shape part) | `ExecutionPlan.decomposition` |
| `derived_from` | `ExecutionPlan.expected_scope` (telemetry) |

`split_legacy_policy(policy)` / `merge_to_legacy_policy(authority, plan)`
round-trip every shipped preset byte-identically. One normalization:
`SHALLOW` with `max_child_depth > 1` (contradictory legacy config) becomes
`ALLOWED` at the same depth.

## Mid-run adjustment channels

- **`request_plan_expansion`** — more context / deeper decomposition hint /
  token reservation. Bounded by `ResourceBudget`; adjusts only the plan.
- **`escalate_policy`** — a missing tool or path. Operator-gated
  (`EscalationPolicy`, approval callback, flood guard, TTL leases).

An agent that needs *information* asks for plan expansion; an agent that
needs *permission* escalates. The two channels never substitute for each
other.

## Deprecation timeline

- **Now (minor):** both APIs work; legacy `policy=` conflicts with the new
  API fail fast; PRODUCTION warns when authority is classifier-derived.
- **Next major:** legacy `ExecutionPolicy` path, classifier-selected
  authority, `run(policy=...)` and the authority-producing `PolicySelector`
  are removed.
