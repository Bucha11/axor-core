# Stateful trajectory observers

Some risks are not a property of a single call or a single value, but of the
*session's trajectory* — and they need to read tool **results**, not just calls:

- a stove that has been on too long with no `turn_off`;
- a patient whose metric is not improving after a treatment;
- an agent that keeps retrying a failing action.

These cannot be a content-blind per-call gate (they need state across calls) or a
`tool -> class` configuration table (they read world-state observations). They are
the one risk class the declarative config deliberately does **not** cover.

## The extension point

A `TrajectoryObserver` (`axor_core.contracts.trajectory`) is a stateful, domain-
supplied object fed every executed `(tool, args, result)`:

```python
class TrajectoryObserver(Protocol):
    def observe(self, tool: str, args: dict, result) -> TrajectorySignal | None: ...
```

When its accumulated state crosses a domain threshold it returns a
`TrajectorySignal(target_level, reason)` that **tightens** the session's degradation
level. Register one (or several) per session:

```python
session = GovernedSession(..., trajectory_observers=[StoveOnTooLongObserver()])
```

## Two deliberate constraints (the honest part)

1. **It can only tighten, never allow.** A trajectory predicate reads world-state
   observations — a domain heuristic, not a sound structural fact. So it rides the
   same rule as all detection: it may raise degradation (narrow the surface, force a
   human gate on the next risky action); it can never authorise one. The kernel does
   not put a heuristic on the enforcement path.

2. **It is owned by the domain/agent developer** — who alone knows what "stove" or
   "patient metric" mean. It is *code with state*, not config, which is exactly why
   this risk class is an extension point and not a YAML knob.

A tightened-to-LOCKED session leaves only `read` + `escalate`, so the next step goes
through a human/operator who can authorise a remediation or intervene. The observer
does not pretend to tell a genuine emergency from a false alarm — it raises pressure
and hands the decision to a human. (A maintenance "stove on" during cooking is the
same false-positive shape as a benign admin restart: resolved by the gate, not by
the kernel guessing.)

This is **not** what `axor-sentinel` does. Sentinel is *cross-session* resource
reputation (slow-and-low staging across many sessions). A trajectory observer is
*within-session* state over this session's calls and results. Different layer,
different problem.

## Run it

```bash
python examples/trajectory/demo.py
```

The agent checks the stove (on for 45 min), the observer tightens to LOCKED, and the
subsequent `start_oven` is frozen.

## Files

- `stove.py` — `StoveOnTooLongObserver`, a result-reading stateful observer.
- `demo.py` — a runnable GovernedSession showing the surface freeze.
