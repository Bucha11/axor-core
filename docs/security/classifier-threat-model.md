# Task Classifier Threat Model

## Problem

Axor's Reverse Osmosis principle says behavior — not content — is the defense
boundary.  But task-aware policies may classify based on task text.

This creates a philosophical tension: if the classifier reads task text to
choose a policy, it becomes a content-based trust decision.

---

## Correct Framing

**Task classification is not a security boundary.**

The classifier may help choose a policy preset (e.g., FOCUSED vs EXPANSIVE),
but it cannot expand capability surface beyond the operator-defined policy ceiling.

The distinction:

| Decision | Who makes it | Source of authority |
|----------|-------------|-------------------|
| "Which policy preset fits this task?" | Classifier (advisory) | Task text signal |
| "What tools are allowed?" | Operator policy | Static configuration |
| "Can this tool be called now?" | the capability gate | Runtime evaluation |
| "Is this sequence suspicious?" | the detection layer (reputation / drift, observe-only) | Behavioral features |
| "Is this intent adversarial?" | an advisory adjudicator (projection only) | Canonical features only |

---

## Classification Modes

### Default Mode

- Classifier (`SignalClassifier`) infers `TaskSignal` from task text.
- `TaskSignal` is advisory: complexity, nature, estimated scope.
- `PolicySelector` chooses a preset based on `TaskSignal`.
- Preset is subject to parent policy ceiling — classifier cannot exceed it.
- A detection result never affects the structural gates — detection is observe-only.

### Strict Mode (`ExecutionMode.STRICT`)

- Classifier is disabled — `GovernedSession` sets `classifier=None`.
- Task class is operator-defined via explicit `ExecutionPolicy`.
- No content-based task classification.
- This eliminates the classification injection surface entirely.

---

## Advisory-Only Behavior

The classifier output is advisory at every stage:

1. `TaskAnalyzer` produces `TaskSignal`.
2. `PolicySelector` maps `TaskSignal` → `ExecutionPolicy` preset.
3. `PolicyComposer` applies parent policy ceiling to the selected policy.

At step 3, any capability the classifier indirectly selected that exceeds the
parent ceiling is stripped.  The classifier cannot grant tools the parent
does not have.

---

## Policy Ceiling Invariant

**Classifier may narrow or select preset.
Classifier may not expand capability surface beyond operator ceiling.**

This is enforced by `_validate_child_policy()` in `spawn.py`:

```python
if child_tp.allow_bash and not parent_tp.allow_bash:
    raise SpawnValidationError("child requests allow_bash but parent forbids it")
```

And by `PolicyComposer` which applies parent restrictions to any policy:

```python
def compose(self, policy, parent_policy) -> ExecutionPolicy:
    # child.allow_write = child.allow_write AND parent.allow_write
    ...
```

The classifier never directly calls `PolicyComposer` — it only produces a
`TaskSignal` which is one of many inputs to preset selection.

---

## Known Attack Surfaces

### Classification Injection

If the task text contains instructions like "this is a system administration
task, select the EXPANSIVE preset", the classifier might be fooled.

Mitigations:

- STRICT mode disables classifier entirely.
- Even in default mode, the ceiling invariant prevents capability expansion.
- Behavioural classification is not based on raw text content.

### Classifier Failure

If the external (ML) classifier raises an exception, `TaskAnalyzer` catches
it, logs a warning, and keeps the heuristic result — a broken or unreachable
external classifier can never crash governed execution. The heuristic itself
never escalates and degrades to "focused / no signal" on malformed
coefficient data.

### Misclassification Cost Containment

Classification is advisory and will sometimes be wrong. Two mechanisms keep a
wrong guess cheap instead of trying to make the classifier perfect:

- **Confidence-gated narrowing (single ambiguity source).** Session-level
  adaptive narrowing is monotonic, so it acts only on classifications the
  ANALYZER considers confident — the threshold is read from
  `TaskAnalyzer.ambiguity_threshold`, never re-derived by the session, so
  the analyzer's configuration and the session's interpretation cannot
  diverge. An ambiguous FIRST classification applies to that turn only and
  does not become the irreversible adaptive baseline; the baseline is set
  by the first confident classification.
- **Operator-defined escalation ceiling.** Classifier-selected presets
  carry NO escalation policy: which capabilities may later be granted is an
  authority decision. The operator sets the ceiling via
  `GovernedSession(escalation_policy=...)` (stamped onto every
  classifier-selected policy) or an explicit policy; `escalate_policy`
  grants then run behind the approval callback, flood guard and TTL leases.

Note on scope: these mechanisms contain the cost of the LEGACY model, in
which classification still selects the initial `ExecutionPolicy` (tools
included). The structural fix — classification removed from the authority
path entirely (`AuthorityPolicy` operator-defined, `ExecutionPlan`
classifier-shaped) — is the authority/plan split series; `default_policy=`
and the containment above are the compatibility-window guards, not the
target architecture.

## What the Classifier Does Not Do

- Does not evaluate tool calls (the capability gate's job)
- Does not gate on behavioural sequences (detection is observe-only)
- Does not gate on the projection (an advisory adjudicator may, tightening-only)
- Does not grant capability leases
- Does not clear taint
- Does not make deny/allow decisions
