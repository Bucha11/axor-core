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
implementation. Each is a place where a reader could reasonably disagree.

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
| spent call approval reusable (`claim_call` only) | **survived: equivalent mutant** |
| spent call approval reusable (`claim_call` + `covers`) | killed |

The survivor is informative. `covers()` already rejects claimed evidence, and no
await separates `covers` (`:967`) from `claim_call` (`:975`). So in the current loop
the `False` branch of `claim_call` is unreachable defense in depth. Under that
mutant the existing unit test
`test_two_concurrent_calls_cannot_both_spend_one_authorisation` also passes; only
`test_claim_is_single_use_and_reports_which_case`, which calls `claim_call`
directly, fails. One-shot behaviour actually rests on `covers` plus the absence of
an await between check and claim.

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
