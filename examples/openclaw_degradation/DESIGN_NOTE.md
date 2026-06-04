# Design note — domain restriction predicates as a core extension point

> Status: **design note, not implemented.** This proposes a small core change
> motivated by the OpenClaw degradation test (`README.md`). No code in
> `axor_core/` has been modified. `privileged_shutdown` in `harness.py` remains
> an illustration overlay.

## What the test established

The generic `DegradationEngine` is **deny-driven**: it escalates on *denied*
intents (denial used as a proxy for danger), plus pressure heuristics. The
OpenClaw trace exposed two blind spots that follow directly from that design:

- **Pure failures** (`denial is None`) accumulate nothing — `record_signal`
  returns early at its `denial is None` guard in
  `axor_core/degradation/engine.py`, before touching any state.
- **Un-denied privileged operations** (`restart_gateway` → `shutdown`) are
  semantically catastrophic but, since they are neither denied nor in any
  pressure set, register as *nothing*. The engine plateaus at RESTRICTED.

The harm in OpenClaw lived precisely where the deny-based proxy has no receptor:
in a *successful* privileged operation. This is `pressure ≠ harm` on a real
trace.

Config B showed that a single domain predicate (`privileged_shutdown`) closes
the gap — but as an overlay *on top of* the engine, not inside it. This note
sketches what it takes to make that a first-class, adapter-supplied capability.

## Proposal: `DomainDegradationPredicate` protocol

This fits the existing axor-core pattern exactly: **core defines a protocol,
the adapter implements it, core never imports the implementation** — the same
shape as `AnomalyDetector`, `LLMVerifier`, and `SignalClassifier` (which live in
`axor-classifier-*`). Domain predicates are just one more injected layer.

```python
# axor_core/contracts/degradation.py  (sketch)

@dataclass(frozen=True)
class DomainSignal:
    """A domain layer's verdict on a single intent."""
    target_level: DegradationLevel   # level this intent should force
    reason: str                      # audit string, e.g. "privileged_shutdown"

@runtime_checkable
class DomainDegradationPredicate(Protocol):
    """Adapter-supplied, intent-level risk classifier.

    Receives the same NormalizedIntent the generic engine sees — never raw
    tool output. Returns a DomainSignal to raise the level, or None to abstain.
    Core defines this protocol; adapters implement it. Core never imports an
    implementation.
    """
    def evaluate(self, intent: NormalizedIntent) -> DomainSignal | None: ...
```

## Where it injects (the one consequential change)

The whole point of the OpenClaw finding is the early return at the
`denial is None` guard in `record_signal`. Domain predicates must be
evaluated **before / independent of** that gate — otherwise they only ever see
the denied stream, and the blindness to failures and to the un-denied shutdown
survives unchanged.

Minimal wiring:

1. `DegradationEngine.__init__` gains
   `domain_predicates: tuple[DomainDegradationPredicate, ...] = ()`.
2. A new entrypoint `record_intent(intent, taint_state)` runs the predicates on
   **every** intent (denied or not). On a fired predicate it raises the level
   through the existing private `_transition_to(...)`.
3. `record_signal` keeps its current deny-driven behavior; it (or the session)
   calls `record_intent` first so the two layers compose.

Because escalation flows through `_transition_to`, the domain layer inherits the
existing guarantees for free: **monotonicity** (never decreases), **observe-mode
shadowing**, and **trace emission** (`DegradationTransitionEvent`). No new
state-machine, no duplicated invariants — this is exactly the "one predicate,
not a domain engine" scope from the spec.

Estimated size: the protocol + `DomainSignal` (~15 lines of contract) and the
`record_intent` path (~20–30 lines in the engine), plus adapter-side predicates.
Low risk: it hangs beside the existing mechanism rather than rewriting it.

## The real design question (for the group)

Adding `record_intent` changes the engine's **contract**, not just its code.
Today the engine is purely *deny-driven* — it only wakes on a governance
denial. The moment it evaluates predicates on every intent, it becomes
*intent-observing*: it looks at each action regardless of the generic verdict.

That is the right shift to catch OpenClaw-style harm, but it is a shift, and it
raises the questions this artifact is meant to hand back:

- **Where is the generic/domain boundary?** Generic stays denial- and
  pressure-based (domain-agnostic, ships in core). Domain predicates carry the
  "this specific operation is load-bearing" knowledge. The boundary is: *generic
  reacts to denied danger; domain reacts to consequential semantics.*
- **Who authors and owns domain predicates?**
  - The **agent/adapter developer** who knows `restart_gateway` is load-bearing?
  - A **policy author** declaring high-stakes operations out of band?
  - Something **learned** from traces (and if so, how does it stay honest /
    auditable)?
- **How do predicates stay honest as the system evolves** — what stops the
  domain layer from drifting into a second, ad-hoc policy engine?

The corpus (`README.md` → *Corpus*) makes the last question concrete. The
`benign_admin_restart` control is a *legitimate* restart, yet `privileged_shutdown`
fires on it → a **false positive**, because the predicate keys off the tool name
alone and cannot tell a malicious restart from a maintenance one. A first-class
`DomainDegradationPredicate` does not remove this risk — it relocates it. Whoever
owns the predicate owns its false-positive surface, and the protocol should make
that ownership explicit (e.g. the `reason` string on `DomainSignal` is an audit
hook for exactly this). The mechanism is cheap; keeping its predicates honest and
accountable is the part that needs an owner.

Config B demonstrates the mechanism is cheap. The expensive, collaborative part
is deciding who owns that layer and how its predicates are governed.
