# Effect specification — a first sketch (one sink)

Status: **draft, evidence only.** This is a specification in authority/effect terms
for one configuration, a differential test of `IntentLoop` against it, and a
pen-and-paper outline of the invariant argument. It does not prove the
implementation correct.

Code: `tests/spec/effect_spec.py` (the spec; imports nothing from `axor_core`),
`tests/spec/test_loop_refines_effect_spec.py` (the harness).

## 1. What is specified

Policy `P` (operator declaration, effect vocabulary only):

| Component | This configuration |
|---|---|
| callable tools | `read_doc`, `read_secret`, and `send_money` in one variant |
| needs approval | `send_money`: every execution |
| grantable | `send_money` |
| egress | `send_money` |
| secret sources | `read_secret` |
| recipient allowlist | `{acme, bob}`: every named recipient, if present |
| amount bounds | a real number in `[0, 1000]`, if present |
| budget | at most `N` executed calls of any tool |

Spec state `s = (spent, secret_delivered, call_auth, class_auth)`:

- `call_auth`: the multiset of unspent call-scoped approvals, keyed by (tool, args)
  up to effect identity (`100 == 100.0`, never `True == 1`).
- `class_auth`: the remaining uses of each class-scoped approval.

Events, all observed from outside the kernel:

| Event | Observed where |
|---|---|
| `Approve(scope, tool, args, max_ops)` | the trusted approver said yes |
| `Exec(tool, args)` | a handler was dispatched with these args (the effect) |
| `Deliver(tool)` | the tool's output was returned to the agent |

### Allowed / Update

```
Approve(call, t, a):     always; if t ∈ grantable: call_auth += {(t,a)}
Approve(class, t, _, k): always; if t ∈ grantable ∧ k>0: class_auth += [(t,k)]
Deliver(t):              always; if t ∈ secret_sources: secret_delivered := true
Exec(t, a) allowed iff
    t ∈ callable ∪ grantable
  ∧ (budget = ⊥ ∨ spent < budget)
  ∧ (t ∈ egress ⟹ ¬secret_delivered ∧ recipients(a) ⊆ allow ∧ amount(a) ∈ bounds)
  ∧ (t ∈ needs_approval ∨ t ∉ callable ⟹ (t,a) ∈ call_auth ∨ ∃ (t,k>0) ∈ class_auth)
  Update: spent += 1; consume one matching call approval OR one class use
```

The consumption choice is nondeterministic. The checker tracks the set of
consistent spec states, and a trace is accepted iff that set never becomes empty.

**Deliberately absent: provenance.** The Alice/Bob trace (an injection picks
`bob`) is Allowed (`test_spec_admits_the_alice_bob_case`). Taint, carrier,
degradation and the adjudicator can only refuse more, and trace inclusion permits
that.

## 2. Two modelling decisions the test forced

1. **`Exec` is the dispatch, not the completion.** The first falsifying example was
   a harness artifact. The recorder logged the effect *after* an await, so a
   transfer dispatched before the secret arrived appeared after `Deliver`. The
   arguments are frozen before dispatch, so they cannot contain a secret delivered
   later.
2. **The secret floor arms on `Deliver`, not on the read's `Exec`.** A call issued
   while a secret read is still in flight cannot carry that secret. The
   implementation arms at registration, which comes before `Deliver`, so it is
   stricter here.

Both decisions are choices about the spec, not observations about the
implementation. Each is a place where a reader could reasonably disagree, and each
moves an obligation outside this argument rather than discharging it:

- **Dispatch ≠ external effect.** The guarantee is about which handler is entered
  with which frozen arguments. That the handler's external effect corresponds to
  those arguments (it does not re-read state, re-derive a recipient, or act later
  on changed data) is a separate per-handler obligation. Nothing here checks it.
- **Why `Deliver` is the right arming point.** The argument is: a model can only
  copy into a call what it has been shown, and it is shown a tool's output only at
  `Deliver`. That premise holds for one agent context. It fails if another channel
  carries the secret into the context earlier, such as a shared memory a concurrent
  sub-agent writes, or a streaming handler that shows partial output before it
  returns. In such a configuration the arming point would have to move to the
  earliest such channel.

## 3. Simulation relation and the per-transition argument (outline)

Relation `R(impl, s)`:

- `impl._budget_spent_calls = s.spent`
- `impl.floor_active ⇐ s.secret_delivered`
- unclaimed call evidence in `EscalationManager` ⊆ `s.call_auth`
- each class grant's `ops_remaining` ≤ the matching `class_auth` entry

Obligation: every implementation step that emits an observed event maps to an
Allowed spec step, and `R` holds afterwards.

| Spec transition | Implementation (`axor_core/node/intent_loop.py`) | What keeps R / Allowed |
|---|---|---|
| `Exec`: `t` callable or granted | capability decision `:584`; executor re-check at `:1044` | the tool is in A₀, or a grant or evidence covers it (`:967`) |
| `Exec`: budget | `:620-640` before gates; re-check after the approval await `:948` | spent is incremented at `:1002`, with no await between it and dispatch |
| `Exec`: recipients / amount | `check_value_policies` `:740` | stateless; the predicate is the spec clause |
| `Exec`: ¬secret_delivered | `taint_gate(floor_active)` `:876-883`; re-check `:948` → `:1497` | **stateful**: needs a re-check after every await between the check and dispatch |
| `Exec`: approval consumed | consequence deferral `:724`; JIT `grant_for_call` `:921`; `covers` `:967`; `claim_call` `:975`; `commit` `:996` | check-and-consume with no await in between (`:967`→`:1044`) |
| `Approve` | `EscalationManager.grant_for_call` / `grant_from_intent` | impl evidence ⊆ approved (a grant may fail to be stored, which is stricter) |
| `Deliver(secret)` | `_register_value_taint` `:1070`, synchronous after execute | the floor arms before the result is returned |

**Where the argument has content beyond "each gate checks its clause":**

- The stateless clauses (allowlist, range, tool membership) reduce to gate
  properties.
- The stateful clauses (budget, floor, approval consumption) do not. They hold only
  because of a property of the loop: between the last check of mutable state and
  dispatch there is either no await, or a re-check after the only await (the
  approval). No gate states this, and before this test no check enforced it
  end to end.

## 3a. Lemma: check–dispatch atomicity

The stateful rows of §3 all rest on one property of the loop rather than of any
gate. This section states it with its premises.

**Definitions.**

- A *suspension point* is a point where the running coroutine returns control to
  the event loop. A textual `await` is a suspension point only if the awaited
  coroutine actually yields.
- *Shared mutable state* Σ is:
  - the budget counters;
  - the taint engine (secret floor, ledger, context root);
  - the escalation tables (grants, leases, call evidence, in-flight digests,
    counters);
  - the degradation engine;
  - the cancel token.
- The *last check* of call `c` depends on the path:
  - if `c` does not wait for an approver: the capability/lease evaluation `:584`;
  - if `c` waits: the post-approval re-check `:948`.
- The *critical segment* of `c` runs from its last check to the entry of `c`'s
  handler.

**Premises.**

| | Premise | Status |
|---|---|---|
| P1 | Σ is mutated only by code on the event-loop thread | **assumed**. Nothing in core enforces it. `CancelToken` has no lock. The console approver runs `input` in a thread (`capability/approvals.py:246`) but only returns a bool. A host that shares an engine with another thread, or with the synchronous `ToolCallGovernor` from another thread, breaks it |
| P2 | no suspension point inside a critical segment | **tested** (`test_check_dispatch_atomicity.py`, ticker instrument plus a non-vacuity test). Static reading: <ul><li>the only textual awaits in `_resolve_tool_intent` are `:922` (approver) and `:1044` (dispatch);</li><li>`CapabilityExecutor.execute` has no await before the handler call (`capability/executor.py:106`);</li><li>`grant_for_call` has no await between the callback's return and storing the evidence (`node/escalation.py:613`→`:646`)</li></ul> |
| P3 | synchronous code in a critical segment mutates Σ only by `c`'s own consumption (`claim_call`, `commit`, `spent += 1`) and by trace/degradation records | **assumed** for host-supplied code inside the segment. This covers the adjudicator in particular: core treats it as a pure consumer, but a host can pass any object |
| P4 | a time-dependent predicate (lease TTL, `lease.is_valid`) holds at dispatch if it held at the check | **assumed** (ε-tolerance). The spec is untimed. A timed spec has to state this tolerance explicitly |
| P5 | `call_digest(t,a) = call_digest(t,a′) ⇒` the spec's `canon(a) = canon(a′)` | partly checked: the digest distinguishes `100`/`100.0`, `True`/`1`, list/tuple, `0.0`/`-0.0`, `2⁵³`/`2⁵³+1`, NFC/NFD, so it is finer than `canon`. Beyond that, collision resistance is assumed |

**Lemma.** Under P1–P3, every component of Σ has the same value at `c`'s handler
entry as at `c`'s last check, except for `c`'s own consumption. So any predicate
over Σ that held at the last check holds at dispatch, evaluated before
consumption.

*Proof sketch.* With one thread (P1), another coroutine can change Σ only while it
runs. It runs only when the running coroutine suspends. By P2 there is no
suspension in the segment, and by P3 nothing in the segment changes Σ other than
the listed consumption.

**What the lemma does not cover on the waiting path.** It covers only what is
re-checked after `:922`:
- `:948`: cancel, budget, degradation, floor and integrity taint;
- `:967`: `covers`;
- `:975`: `claim_call`.

The checks done only before `:922` are capability (`:584`), value predicates
(`:740`), SSRF, carrier, positional and the adjudicator. Most of these read only
the frozen arguments and immutable configuration, so the wait cannot change them.
The exception is capability that came from a grant or lease, which is mutable.

**Sub-lemma** (static, not separately tested): on the waiting path the capability
did not come from a grant or lease, i.e. `pending_consumption is None`.

- `_evaluate_tool_intent` (`:584`) returns a pending consumption only when a grant
  or a valid lease exists for the tool.
- The consequence gate's `covers()` (`:1531`) reads the same tables in the same
  segment, so it then reports a governance gate.
- So no consequence denial is deferred and no approver is awaited.

**What remains after the lemma** is preservation of the relation R for each event
(§3), with the lemma supplying "the checked predicate still holds at dispatch":

- **Floor:** armed synchronously at `:1070`, before the result is returned. No
  endorsement in this configuration.
- **Budget:** `spent` is incremented inside the critical segment (`:1002`) and
  counts every dispatch, including a handler that raises.
- **Authorisation:** evidence is created only after an approver's yes
  (`escalation.py:646` for the just-in-time path, `:477` for a requested
  escalation).
  - On the just-in-time path, existing unspent evidence is never overwritten
    (`:590`, in-flight guard `:592`).
  - A requested escalation (`grant_from_intent`) does overwrite evidence with the
    same digest. Two approvals can then leave one usable authorisation. That
    direction is stricter than the spec, so inclusion is preserved, but the
    implementation can lose an approval. A class grant's uses are bounded by the `max_ops` the approver was
  shown.

This is a pen-and-paper argument for this configuration. It is not mechanised.

## 4. What the test shows (evidence, not proof)

The test draws an arbitrary agent with Hypothesis:

- calls with well-formed and malformed arguments: NaN, ±inf, `"100"`, `True`,
  lists of recipients, Unicode look-alikes;
- class- and call-scoped escalation requests;
- concurrent batches;
- attacker-controlled read content;
- approver answers;
- scheduling delays.

It uses a real `TaintEngine` and `DegradationEngine`.

- Unmodified code: 3000 examples with no counterexample. About 13% of runs execute
  at least one transfer; 30% deliver a secret; 13% exhaust the budget.
- Mutation check: each mechanism was disabled in turn.

| Mutant | Result |
|---|---|
| value policy off | killed |
| confidentiality floor off | killed |
| post-approval re-check off | killed (only after adding scheduling delays) |
| budget ignored | killed |
| JIT approval granted without asking | killed |
| class-grant uses never consumed | killed |
| spent call approval reusable (`claim_call` only) | **survived** (see below) |
| spent call approval reusable (`claim_call` + `covers`) | killed |

The survivor is **redundant in the explored execution model**: one event-loop
thread, and the configuration and paths the harness drives. In that model
`covers()` already rejects claimed evidence, and no suspension separates `covers`
(`:967`) from `claim_call` (`:975`) (premise P2 below), so the `False` branch is not
reached. This is not a claim of global equivalence. The branch becomes
load-bearing exactly when P1 or P2 fails: another thread touches the tables, or a
suspension is introduced between the two calls.

Under the mutant, `test_two_concurrent_calls_cannot_both_spend_one_authorisation`
still passes. That test is not wrong: it checks the behaviour it states (no double
spend), which in the current code is delivered by `covers`. It does not show that
`claim_call` is necessary. Only `test_claim_is_single_use_and_reports_which_case`,
which calls `claim_call` directly, fails.

## 5. Not covered

- **Configuration:**
  - one egress sink;
  - default (non-STRICT) mode;
  - `require_human=True` (auto-approve is a different spec: authority comes from
    the policy, not an approver);
  - no governance endorsement (the floor never lifts);
  - no leases issued outside `grant_from_intent`;
  - no child nodes or federation;
  - no `TRANSFORM`.
- **Harness:** it calls `_resolve_tool_intent` and `grant_from_intent` directly, not
  the streaming `run()` dispatcher.
- **Argument:** §3 is an outline. It covers only the transitions this configuration
  exercises and is not mechanised.
