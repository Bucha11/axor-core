# Guarantees and premises — one table

The claim is about execution. For an arbitrary model behaviour M (any sequence of
proposed calls):

```
Execute(s, c)  ⟹  ∧_{j ∈ J(p)}  P_j(s, c)
```

where each `P_j` is a constraint stated independently of the others by the
operator's policy `p`. The table lists each `P_j`, the mechanism that enforces it,
the conditions under which that mechanism establishes it, the premises the kernel
cannot check, and the evidence. "Evidence" distinguishes three kinds, and only those
three: a **unit/regression** test, a **property test** (Hypothesis — evidence over the
drawn examples, not a proof), and a **measured negative control** (the same
scenario with the condition removed, showing the effect it prevents). Nothing here
is machine-checked.

Untrusted bytes can change a verdict — an out-of-range amount is denied because of
what the model wrote, and the gate must see it to deny it. What they must never be
is an unchecked ground for *widening* authority; [kernel-theorem §3b](kernel-theorem.md)
lists every permitting ground (rendered from `axor_core/kernel/grounds.py`).

| `P_j` | Mechanism | Holds when | Premise (not checkable in core) | Evidence |
|---|---|---|---|---|
| Only a permitted tool executes | capability check in the loop and again in the executor; locked executor | every effect goes through the sink ring (A-3) | in-process, A-3 is a soft boundary; the daemon is the hard one | `tests/invariants/test_security_invariants.py::test_inv15_adversarial_denied_tool_never_reaches_executor` (unit) |
| The authority a call is checked against cannot be rewritten by the agent (K5) | ceilings, grants and leases read from live state; grants validated against `G` | operator-supplied policy (STRICT refuses otherwise) | the operator's ceiling is the intended one | `tests/invariants/test_authority_transitions.py` (property) |
| The destination is in the operator's allowlist | `enum` / `tuple_enum` / keyed key, decided on the prepared call | the field has a **consumption contract** (`identifier[_list]`) | handler conformance to its contract; for an attested sink, conformance itself | `tests/adversarial/test_check_vs_consumer.py` (negative controls: alias, list-in-scalar); `tests/integration/test_end_to_end_chain.py::test_no_proposal_sequence_executes_outside_the_policy` (property) |
| A number is within its range | `numeric_range` on the prepared value | the field is a contracted `number` (required, or with a declared default) | as above | `test_check_vs_consumer.py` (negative control: absent amount → sweep); same property test |
| No declared egress while the confidentiality floor is up | outstanding sensitive-read map; lifted only by `GovernanceAuthority` | sensitive sources are declared (STRICT role completeness) | the governance authority is not exposed to the agent | `tests/adversarial/test_confidentiality_floor.py`; `test_end_to_end_chain.py::test_floor_holds_when_the_integrity_labeler_fails` (fault injection: arbitrary labeler output) |
| No unattended call above the consequence ceiling | consequence gate; call-bound approval re-checks state and is spent once | every tool has a declared class (STRICT) | **the approval callback is not model-backed** | `tests/node/test_call_bound_authority.py`; `test_end_to_end_chain.py::test_a_model_backed_approver_is_a_ground_the_model_controls` (negative control) |
| An untrusted-derived driving value does not reach a high-risk sink | per-value taint (explicit flows), `integrity_default="context"` | STRICT (context default, declared driving args) | O2 covers explicit flows only; implicit flows are out of scope | `tests/adversarial/test_context_default_integrity.py`; `test_end_to_end_chain.py::test_a_model_carried_mark_is_not_a_ground` |
| — relaxed by integrity supersession | membership in a closed operator set stands in for the origin ground | every driving arg present and covered by a closed codomain | the operator's set is the intended one (the attacker can still pick a member) | `test_end_to_end_chain.py::test_supersession_relaxes_integrity_only`; `tests/adversarial/test_decidable_supersession.py` |
| The call executed is the call checked and approved | consumption preparation before the first gate; frozen argument set; digest over every prepared argument; volatile re-check; atomic claim | always (both wrapping paths; replay re-prepares) | — | `test_end_to_end_chain.py::test_the_call_that_runs_is_the_call_checked_and_approved`; `tests/kernel/test_consumption.py` (replay parity, idempotence property) |
| Path / interpreted / free-text fields stay in bounds | path ceiling, positional/carrier gates | — | **no decidable guarantee**: a contract refuses a decidable predicate on these fields | `tests/kernel/test_decidability.py`, `tests/adversarial/test_property_fuzz.py` (fuzz — evidence, not a discharge) |

What the table does not claim: that any constraint holds in a configuration other
than the one named in "Holds when"; that a handler honours its contract; that an
approval is correct; that the operator's policy is the right one. Those are the
premises, and they are stated rather than implied.
