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
| "Can this tool be called now?" | ToolInterceptor (Layer 1) | Runtime evaluation |
| "Is this sequence suspicious?" | ML scorer (Layer 2) | Behavioral features |
| "Is this intent adversarial?" | LLM verifier (Layer 3) | Canonical features only |

---

## Classification Modes

### Default Mode

- Classifier (`SignalClassifier`) infers `TaskSignal` from task text.
- `TaskSignal` is advisory: complexity, nature, estimated scope.
- `PolicySelector` chooses a preset based on `TaskSignal`.
- Preset is subject to parent policy ceiling — classifier cannot exceed it.
- Classifier result does not affect Layer 1 rule-based checks.

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
- Classify-by-behavior (Layer 2 ML) is not based on text content.

### Classifier Failure

If the classifier raises an exception, `TaskAnalyzer` falls back to the
DEFAULT policy preset.  The DEFAULT preset is the most restrictive preset —
fail-closed.

### Adversarial Task Text

Task text is attacker-controlled when the user is untrusted or when the
session is processing external content.  In STRICT mode, no classification
happens from task text, so this attack surface is eliminated.

---

## What the Classifier Does Not Do

- Does not evaluate tool calls (that is Layer 1's job)
- Does not evaluate behavioral sequences (that is Layer 2's job)
- Does not verify canonical intent features (that is Layer 3's job)
- Does not grant capability leases
- Does not clear taint
- Does not make deny/allow decisions
