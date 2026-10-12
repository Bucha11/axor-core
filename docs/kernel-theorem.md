# The Kernel Theorem — Perimeter Non-Interference

This document states the single load-bearing claim of the kernel, its preconditions,
and — for every precondition — the regression or CI gate that fails the build if the
precondition is violated. The other docs describe *how the kernel decides*
([governance-model](governance-model.md), [enforcement-model](enforcement-model.md),
[reverse-osmosis](reverse-osmosis.md)); this one states *what is guaranteed, under
what conditions, and where each condition is pinned*.

It is written to stand on its own. Nothing here introduces a new mechanism — every
property below is already implemented and tested; this document names the theorem
those mechanisms exist to support, so the claim is anchored rather than implied.

---

## 1. The claim (K4)

Let `x` be a raw artifact an agent wants to push across an effect boundary, `π(x)`
its structural projection, and `allow` the decision function.

> **Perimeter Non-Interference.** For any sink invocation:
> 1. **no effect is reachable bypassing `allow`** (complete mediation), and
> 2. **`allow` inspects only admissible projections** (§4) — *provided each such
>    projection has discharged its faithfulness obligation (T4) and was produced by a
>    non-interpreting process (T0)*.
>
> Given that proviso, the decision cannot be steered by raw bytes except through an
> admissible projection whose codomain is non-instruction-complete relative to its
> consumer — **regardless of trust model**.

Operationally, the safety content is the non-interference invariant:

```
∀ x₁, x₂ :  π(x₁) = π(x₂)  ⟹  allow(π(x₁), p) = allow(π(x₂), p)
```

Raw input influences the decision *only* through `π`, whose codomain admits no
instruction. Modulo explicit declassification (governance endorsement), two inputs
the consumer would treat differently cannot share a projection.

### The conditionality is the point

K4 is **not** unconditional, and pretending it is would be the central overclaim.
The *factorization* (clauses via O1 and O3 below) is structural and unconditional.
The *safety content* of clause (2) rests on two per-projection obligations:

- **T0** — the projection-producing process is non-interpreting (deterministic /
  structural, never a model reading the governed content).
- **T4** — the projection's *effective* codomain equals its *nominal* one (no hidden
  residual channel / weird machine).

The honest footing, sharpened by the decidability split (§4):

> **For the defined predicate language (enum / numeric / relational) over fields
> whose consumption is fixed by a registered consumption contract, T4 is decided
> by a procedure. Every other consumer — rich-syntax fields (path / carrier /
> string-subfield), and guarded fields whose handler conformance is only attested
> — needs a separate argument; fuzzing there is evidence, not a discharge.**

The weird-machine class did not disappear; it is *localized* to the rich-syntax
codomains. Both real bugs ever found in this layer (newline injection in path
extraction, `../` traversal in lease validation) were T4-class and lived in path
handling — the fuzzing region, exactly where the split predicts bugs must live.

---

## 1b. The second claim (K5) — authority containment

K4 is about the *first* argument of the decision: raw input reaches it only through
an admissible projection. It says nothing about the *second* — where the policy
came from. That is a separate claim, and the one the "untrusted data cannot create
authority" thesis actually needs:

> **Authority containment.** For every step of a session, on every axis,
> `A_t ⊆ B`, and the only move that changes `B` is a trusted event.

With three sets kept apart (`axor_core/kernel/authority.py`):

```
A₀   what the policy makes callable now        (CapabilityResolver.resolve)
G    what the operator declared grantable      (escalation_policy.grantable_tools)
B    the trusted ceiling for this turn         (A₀ ∪ G, after parent + overlay composition)
A_t  what is in force right now                (A₀ plus live authority, minus what the
                                                degradation level refuses)
```

**Why it is not "the composition contains no union."** That phrasing cannot
describe a system with lawful recovery: an escalation legitimately *extends* what
is in force. The content is a containment invariant plus a closed, typed set of
moves — `NARROW`, `RESTORE_CLASS`, `RESTORE_CALL`, `SPEND`, `SET_CEILING` — each
with its own legality condition. A transition is typed by its **cause**, declared
by the caller and checked against the two states (`authority.check`), because the
cause is not always visible in the states: a trusted re-assertion of the turn's
policy is a `SET_CEILING` even when the ceiling value is unchanged. Where no cause
is available, `authority.classify` infers one and reports an unexplained widening —
that inference path is the silent-widening detector, not the primary check.

**K5 splits in two, and only the first half is about authority composition:**

```
  A_t ⊆ B(t)                       authority composition — this claim
  execute(s, e) ⟹ e ∈ A_t          policy enforcement — the gate sequence + O3
```

Neither half implies the other. A correct lattice says nothing about a labeller or
a normalizer admitting an effect outside what is in force; complete mediation says
nothing about where the ceiling came from.

**Status.** The containment half is property-tested (Hypothesis) over random
interleavings of the real engines — evidence over the drawn interleavings, not a
machine-checked proof (not a model of them: every view is read back through
`axor_core/node/authority_probe.py` from the policy, the degradation engine, the
capability resolver and the escalation manager's own tables). The enforcement half
is O1/O3 above. What is *not* claimed: that the authority a transition restores is
*appropriate* — only that it stays inside a ceiling the untrusted side cannot
move; and that the policy's description of an effect is precise enough, which is
`validate_effect_coverage` and explicitly advisory.

**Two carve-outs, declared rather than assumed:**

- `escalate` / `escalate_policy` are intent kinds the loop handles itself, never
  governed sinks, and the degradation engine adds them at LOCKED ("everything
  frozen except read + escalate"). The containment check excludes them —
  `KERNEL_INTERNAL_INTENTS` — and the state machine asserts that nothing *else*
  ever appears there, so a real tool riding the exemption fails the build.
- Authority does not outlive the turn it was granted in: the `IntentLoop`, and
  with it the `EscalationManager`, is constructed inside `GovernedNode.run`. That
  is what keeps `A_t ⊆ B` across a ceiling that drops. If authority is ever hoisted
  to session scope, lowering a ceiling will have to retract authority above it; the
  state machine already detects that case
  (`test_a_ceiling_that_drops_under_live_authority_is_caught`).

| Obligation | What it requires | Pinned by |
|---|---|---|
| **K5-1** containment | `A_t ⊆ B` on tools, paths and the unattended consequence ceiling, at every step | `tests/invariants/test_authority_transitions.py` (`AuthorityMachine`, invariant after every rule); `axor_core/kernel/authority.py::violations` |
| **K5-2** typed transitions | every move is one of five kinds and satisfies that kind's condition; an undeclared widening is reported | `authority.check` / `authority.classify`; `tests/invariants/test_authority_transitions.py` (hand-built illegal pairs: restoration of a non-grantable tool, spend that widens, authority with no record) |
| **K5-3** one step never widens its trusted operand | each composition entry point (parent restriction, deployment overlay, extension overrides) returns no authority wider than its trusted operand, over **every** declared authority field | `tests/invariants/test_authority_ceiling_sweep.py` (field set derived from `AuthorityPolicy`, so a new field without a declared order fails the build) |
| **K5-4** delegation ≠ availability | `G` is not callable before a grant; a grant is validated against `A₀ ∪ G`; an authorised call is dispatched with `A₀ ∪ {that tool}` and the tool is out of reach again once spent | `tests/capability/test_lease_validator.py` (`delegation_ceiling`); `tests/node/test_call_bound_authority.py::test_a_grantable_tool_is_unreachable_until_granted_and_after_spending` |
| **K5-5** authority stays current | a call authorisation is bound to one call, re-checked against mutable state after the approval await, spent atomically, and discarded if it cannot be used now | `tests/node/test_call_bound_authority.py` (floor armed / cancelled during the wait, two concurrent claimants, replaced argument) |
| **K5** scope | the model is read from the live objects, never maintained alongside them | `axor_core/node/authority_probe.py`; `EscalationManager.authority_records`; `DegradationEngine.tool_denial` (one source for what a level permits, read by both the enforcement path and the probe) |

---

## 2. The three structural obligations (O1 / O2 / O3)

The theorem decomposes into three obligations. Each is a property of the
implementation, not a runtime check the agent can influence.

### O1 — the decision factors through the projection

`allow` is a pure function of `(projection, policy)`; there is no `allow'` in the
trusted path that reads raw `x`. Reputation and trajectory history are **not**
arguments to `allow` — admitting them would reintroduce a hidden raw channel.

- The six stateless gates live once as pure functions in
  `axor_core/policy/gates.py`; both callers (the streaming `IntentLoop` in
  `axor_core/node/intent_loop.py` and the synchronous `ToolCallGovernor` in
  `axor_core/governor.py`) delegate to them, so the decision cannot drift.
- The advisory adjudicator sees only the projection and is memoized by its hash
  (`axor_core/kernel/adjudicator.py`), so even the optional oracle is a
  deterministic function of `π`.
- The projection itself is stripped of raw strings by the canonicalizer
  (`axor_core/node/canonicalizer.py`) before any advisory/detection layer sees it.

### O2 — label soundness (explicit flows only)

The per-value provenance never *under*-estimates the untrusted sources that
**explicitly** influenced a value:

```
causal_root(v)  ⊇  the untrusted sources that explicitly influenced v
```

Proved by induction over the closed constructor set (`constant`, `external_read`,
`mint` = join over inputs, `parse` = passthrough, `cross_process_in` = re-mint).
Within that set the failure direction is deliberate over-taint (`⊇`, never silent).

- Constructors and the join live in `axor_core/taint/causal_root.py`; the
  refcounted, fail-closed ledger is `axor_core/taint/ledger.py`; per-value
  register/derive, the confidentiality floor, and governed release are
  `axor_core/taint/engine.py`.

**Scope boundary (part of the claim, not a hidden gap):** O2 is **explicit-flow
soundness only**. An implicit (control-flow) leak — a deterministic interior branch
on content that influences an output with `causal_root = ∅` — is **not** covered.
CaMeL's STRICT-mode dependency augmentation (adding the conditional test or loop
iterable as a dependency of everything assigned in the block,
[arXiv:2503.18813](https://arxiv.org/abs/2503.18813) §5) is the known partial
technique and is not yet adopted. See §6 and
[governance-model §7](governance-model.md).

**Do not claim this boundary is shared with FIDES — on *this* axis it is not.**
FIDES ([arXiv:2505.23643](https://arxiv.org/abs/2505.23643), Proposition 1)
guarantees *non-interference for the integrity of tool calls and data*, and only
*explicit secrecy* for confidentiality. Their integrity lattice's non-interference
explicitly "includes consequential control flow decisions", and the mechanism is that
policy P-T checks the **tool label** — whether the call was generated in a trusted
context — not merely the arguments. So the axis on which FIDES accepts the
implicit-flow weakening is **confidentiality**, which is the axis where *our* control
is strongest (the floor is content-blind and whole-call, §5), and the axis where we
take the weakening is **integrity**, which is the one FIDES closes. The profiles are
mirror images, not the same scope choice. The construct that would move us onto their
integrity profile is the one already specified:
`integrity_default="context"` labels a value by the *context it was generated in*
rather than by what it contains
([rfc-integrity-context-default](rfc-integrity-context-default.md)) — a value-level
analogue of their trusted-context tool label.

**The model edge is not a constructor (current gap, not a scope choice).** A value
the model writes into a tool call is produced by none of the constructors above. The
kernel reconstructs that edge by content derivation: `ValueTaintLedger.derive`
labels the value with the sources whose registered fragments it *contains*, and
returns `constant()` when it contains none. So O2 holds for a model-generated value
only when the value carries a verbatim (normalised) fragment of what the model read.
A re-encoded copy of an untrusted identifier — compacted, re-cased, re-separated,
split across fields, base64'd — derives clean, and so does a verbatim copy of an
identifier too short in every block to be segmented (a space-grouped IBAN). That is
an **explicit** flow under-tainted silently, i.e. O2 does not hold on this edge today.
It is pinned by `tests/adversarial/test_context_default_integrity.py` (sound
behaviour asserted `xfail(strict=True)`), bounded where an `enum` allowlist guards
the driving arg (T4, content-blind), and does not touch confidentiality (the floor is
content-blind). The fix — model-generated values carry the context root unless they
provably originate from a trusted source — is
[rfc-integrity-context-default](rfc-integrity-context-default.md):
`integrity_default="context"` (`axor_core/taint/engine.py`,
`axor_core/taint/trusted.py`), the default under Strict and opt-in otherwise. Under it the same regressions pass and O2 holds on
this edge with a new constructor, `generate(ctx) = mint(root(c) for c in ctx)`,
and one declassification rule, whole-value equality with a trusted value
(`tests/adversarial/test_context_default_property.py`: the mode never loosens the
legacy label, is inert before untrusted data, taints every generated re-encoding of
an attacker identifier, and keeps trusted spellings clean).

### O3 — complete mediation (A-3)

No real-world effect is reachable except through a guarded coercion that calls
`allow`. Because the sink ring is finite (K2), this is statically surveyable.

- **In-process (soft) boundary:** `axor_core/capability/locked.py` raises
  `GovernanceBypassError` if the executor is reached outside the governance context.
- **Out-of-process (hard) boundary:** the daemon re-derives the capability ceiling
  and re-runs the per-value data-flow gates server-side, so a code-compromised
  worker cannot bypass them (see [enforcement-model](enforcement-model.md), "What
  the process boundary does and does not cover").
- An unregistered sink fails closed under the high-assurance ceiling.

---

## 3. T0 — the producing process must be non-interpreting

A projection produced by a model that reads the governed content is in the untrusted
domain even when its output type-checks: its *faithfulness* is steerable (the
Firewalls input-firewall LLM-projector is the public counterexample; for a *measured*
one, built and broken on purpose, see
[paper-model-carried-labels](paper-model-carried-labels.md) — a projection with a
two-element, instruction-free codomain consumed as a case-split, i.e. impeccable under
T4, whose producer is the model, and which an adversarial search defeats in one round
without ever touching the projection itself. That instance is what separates T0 from T4:
a flawless codomain does not rescue a steerable producer). Such a process
may feed the detection layer only, never the trusted path.

In this codebase every trusted-path projection is produced by a deterministic,
structural function — there is no model in the decision loop:

- carrier classification — `axor_core/security/carrier.py`
- consequence (action-class) classification — `axor_core/policy/consequence.py`
- structural normalization — `axor_core/policy/normalizer.py`
- raw-stripped canonical intent — `axor_core/node/canonicalizer.py`
- provenance label (`driving_root`) — `axor_core/taint/ledger.py` over the lattice
  in `axor_core/taint/causal_root.py`

**Assurance status.** T0 is pinned by a dedicated gate in two halves:

- **Structural half** — the `t0-producers-non-interpreting` contract in
  `.importlinter` forbids the six producer modules from importing the in-core
  advisory/model surface (`axor_core/kernel/adjudicator.py`). Run by `lint-imports`
  in CI; a violating import fails the build.
- **External half** — `tests/invariants/test_t0_producers_non_interpreting.py`
  statically scans each producer's imports for a model SDK / network / subprocess
  surface (which import-linter cannot see, as it graphs only the root package) and
  asserts the pure classifiers (`classify_carrier`, `consequence_class`) and ledger
  derivation are deterministic.

The adjudicator is a *consumer* of the projection (advisory, projection-only,
tightening-only) and may itself be a model — T0 constrains the producer, not the
consumer, which is why the rule is scoped to the producer modules.

---

## 3b. Every input that can PERMIT a call, and what must back it

T0 (§3) says no model produces a trusted-path projection. That is necessary and
not sufficient for the claim the kernel actually makes — that the model is outside
the trusted base of the guarantee. A model that pre-manufactured a trusted label,
chose the permitting policy, or answered an approval is already part of the ground
on which the call was admitted, whatever `allow` itself calls.

So the chain is audited end to end: for every input that can turn a call from
denied into allowed, what is the admissible ground, and what actually produces it.

The audit is kept as data — `axor_core/kernel/grounds.py` (`GROUNDS`) — and
the table below is rendered from it (`render_markdown()`); a test fails when the two
differ, when a named mechanism does not import, or when a named pinning test does
not exist (`tests/invariants/test_permitting_grounds.py`). For each ground it states
who may issue it, what it permits, how it is checked, what happens when it is
absent or fails, and what it does **not** override.

<!-- grounds:begin -->
| Permitting input | Issued by | Permits | Checked by core | Absent / fails | Does not override | Premise (not checked by core) | Status |
|---|---|---|---|---|---|---|---|
| Operator policy (A₀: tools, paths, consequence ceiling) | operator configuration (policy= / default_policy=) | the capability surface and the unattended consequence ceiling | STRICT refuses to run without an operator-supplied policy | STRICT: PolicyError, nothing runs; other modes: classifier-selected, not covered | every per-call gate | the operator's policy is the intended one | enforced-under-strict |
| Delegation ceiling (G) | operator configuration (escalation_policy.grantable_tools) | which tools a grant or approval may ever reach | grants and leases validated against available ∪ grantable | no escalation surface: an over-ceiling tool stays unreachable | per-call gates; a grant only adds the tool to the capability set | the operator's ceiling is the intended one | enforced |
| Action class (consequence axis) | built-in table + operator consequence_overrides, keyed on the sink name | an unattended call whose class is within the ceiling | exact-name lookup; STRICT: unclassified ⇒ CATASTROPHIC | STRICT: refused at construction / treated as catastrophic | value, taint, floor, carrier, SSRF gates | the declared class matches the tool's real irreversibility | enforced-under-strict |
| Sink roles (egress / integrity / imperative / positional / benign) | operator declaration | which gates a tool is subject to; a positional sink admits only structured carriers | STRICT role completeness; an instruction-complete sink cannot be declared positional | STRICT: an unclassified tool is refused | the floor and the value gates | the declared role matches what the tool does (a mis-declared sensitive source arms no floor) | enforced-under-strict |
| Consumption contract (T4 for a guarded field) | operator declaration per sink field (field_contracts) | a value predicate to count as faithful for the field it guards | predicate role vs field kind at registration; every call prepared into the contract before the first gate; the prepared set is what executes | see the attestation row; outside STRICT, nothing — reported 'unverified' | every gate after preparation | the handler consumes each field as its contract states (its defaults, aliases and parsing are the ones declared) | enforced |
| Consumption attestation | operator statement per sink (consumption_attested) | a guarded sink with no contract to construct under STRICT | none beyond presence: core does not compare the predicates with any field description and prepares nothing | STRICT: construction refused for a guarded sink with neither contract nor attestation | every gate — but the gates decide the raw proposal, which is also what executes | the whole of T4 for that sink: the handler consumes the guarded fields exactly as the predicates assume | premise |
| Integrity supersession | operator, by declaring a closed allowlist on every driving arg | skipping the origin (integrity-taint) check: membership in the pre-authorised set satisfies that obligation instead | every driving arg present and covered by a closed codomain (enum / tuple_enum / keyed key) | origin check applies; numeric_range never supersedes | the confidentiality floor, consequence, value ranges, capability | the operator's set is the intended one; T4 for the covered arg rests on that sink's contract, or on its attestation | enforced |
| Confidentiality floor down | no sensitive read outstanding; lifted only by GovernanceAuthority endorsement | egress | outstanding-read map; endorsement verifies the authority token, per value fingerprint | egress denied while a sensitive read is outstanding | integrity, consequence, value gates | the governance authority is not reachable by the agent | enforced |
| Trusted value (TrustedOrigin) | TASK: the principal's task at depth 0 · OPERATOR: allowlist members · TOOL: a declared benign tool · ENDORSED: governance | a model-emitted value to escape the context root at an integrity sink | register_trusted from kernel-side code only; TOOL requires a benign declaration under STRICT | the value carries the context root (STRICT) and is denied at an integrity sink | the floor, consequence, value gates | a declared benign tool's output is in fact not attacker-writable | enforced-under-strict |
| Call-bound approval | the operator-supplied EscalationCallback | one call over the consequence ceiling | call digest over every prepared argument; volatile state re-checked after the answer; claimed atomically, one use | the consequence denial stands | every other gate (asked only after all of them pass) | the callback is a trusted party or a checkable rule — not a model | premise |
| Projection features (carrier, path class, provenance label, normalized intent) | the T0 producer modules | the per-value gates' inputs | import contract (no model surface) + determinism tests | — | — | — | enforced |
| Peer provenance label | a peer, by signed receipt | restoring a peer value's trust level | signature, nonce, binding to the value; across keysets only a bounded discount | the value is untrusted | every gate | the peer's key is held by the peer | enforced |
| Absence of a model-carried mark | the model (it must reproduce the mark onto the argument) | nothing — inadmissible as a positive ground | — | admissible only in the deny direction, as detection | — | — | inadmissible |
<!-- grounds:end -->

**What the table is, and what it is not.** It is a specification: each row is held
to the code (the mechanism imports, the pinning tests exist, the document matches
the data). "Checked by core" and "Premise" are separate columns so that nothing a
ground merely *rests on* reads as something core verified — in particular a
consumption **attestation** is its own row, status `premise`: core prepares
nothing for an attested sink and compares its predicates with no field
description, so the gates decide the raw proposal and the operator's sentence is
the whole of T4 there.

The tests do **not** show the table is complete. Completeness — every input that
can turn a denial into an allow has a row — is argued by enumerating what the allow
path reads. In `IntentLoop._resolve_tool_intent` (and the same predicates in
`ToolCallGovernor.evaluate` and `kernel/replay.evaluate_call`) a call is executed
only if every one of these admits it, and each admission reads only the listed
inputs:

| Step on the allow path | Inputs it reads that can admit | Row |
|---|---|---|
| consumption preparation | the sink's contract (or none) | contract / attestation |
| capability (`_evaluate_tool_intent`, executor re-check) | policy `A₀`, leases and grants within `G` | operator policy, delegation ceiling, call-bound approval |
| STRICT role check | role declarations | sink roles |
| consequence | class table + overrides, ceiling; deferred approval | action class, call-bound approval |
| value policies | operator predicates over the prepared call | contract / attestation (T4) |
| degradation, SSRF, positional, carrier | structural projections of the call | projection features, sink roles |
| per-value taint | driving root (ledger, context root, trusted index), floor state, supersession | trusted value, floor, supersession, peer label |
| adjudicator, budget, admission | tighten only — cannot admit | — |

The argument is only as good as this enumeration, and nothing mechanical keeps
the enumeration current: a new admitting input added to the allow path needs a
row here and a line in this table, and no test will notice if it gets neither.

Two readings of the table matter for the claim:

- **Determinism is not correctness.** A deterministic producer can still issue the
  wrong ground; the "Checked by core" column is the check, and the "Status" column says
  where the check cannot be made in core (`premise`). The negative controls in
  `tests/integration/test_end_to_end_chain.py` build two such grounds on purpose —
  an approver whose codomain is {yes, no} but which follows the call's arguments,
  and a datamark whose absence would admit a re-typed value — and show the first
  executing an unattended consequential call while every other constraint still
  holds, and the second refused by the context-root integrity ground.
- **Supersession is a substitution, not a security-free relaxation.** The attacker
  *can* choose a member of the operator's allowlist (a statement can name the
  approved payee). Supersession means: the operator pre-authorised that set, and
  membership satisfies the integrity obligation in place of an origin ground. The
  floor, consequence, value ranges and capability remain in force
  (`test_supersession_relaxes_integrity_only`).

**The result of the audit is a mode statement, not a global one.** The three
inputs whose ground is a heuristic or a default rather than a declaration —
policy selection, trusted-tool-output, and the integrity default — are exactly
the three that STRICT closes. So:

> The chain "producer → projection → gate → execution" excludes the model from
> the trusted base **under STRICT, with an operator-supplied policy and an
> approval callback that is not model-backed**. Any other configuration is not
> the one the theorem covers, and the kernel now refuses the specific case where
> it would otherwise look covered (STRICT without an operator policy) instead of
> relying on a warning.

**Two premises remain premises, and neither is closable from inside core:**

- an approval is only as good as what answers it — core can bound its *scope*
  (one call, one use, re-checked, atomically spent) but cannot inspect the
  callable;
- outside STRICT, `TrustedOrigin.TOOL` and `integrity_default="clean"` are
  declaration-free, so the exclusion claim is not available there. This is a
  configuration boundary, stated as one.

**What was closed to make the chain reach execution at all:** a `TRANSFORM`
decision rewrites the call, and the gates used to run on the arguments *before*
the rewrite while the executor received the arguments *after* it. Nothing in core
emits `TRANSFORM`, so no deployment was affected — but the premise "the gates
verified the call that runs" did not hold for the contract as written. The
transform is now resolved before normalization, so one argument set is
normalized, gated, bound to the approval, re-checked after the await, and
executed (`tests/node/test_transform_is_gated.py`; the suite fails without it).

---

## 4. T4 and the decidability split (Theorem 0)

T4 (effective = nominal) is the obligation where the theorem's safety content lives.
Whether it is *decidable* depends on the codomain **and** the consumer, and it splits
cleanly. The split itself is a classification (`axor_core/kernel/decidability.py`);
what makes it say something about execution is that the consumer is *registered*
per field and every call is put into that registration before it is checked
(`axor_core/kernel/consumption.py`, below). A claim of novelty rests on that
mechanism and on what it measurably prevents, not on the statement of the split.

**Admissible codomains (the closed whitelist, Def. 3b).** `allow` reads only
projections drawn from a closed set of *kinds*:

```
{ enum, bounded-numeric-range, origin-class, path-class, provenance-label }
```

The set of *projection kinds* is closed (by the rule, not by taste — a kind that read
semantics would be instruction-complete and is inadmissible). The set of *predicates*
over an admissible projection (membership, range, prefix, equality) is open, and that
is fine: every predicate consumes an already-admissible projection, so none opens a
content channel. A value policy such as `transfer(amount in 0..1000)` is a predicate
over the `bounded-numeric` projection, **not** a fifth axis.

**Decidable — discharged by a decision procedure, not fuzzing.** For a **finite enum**
consumed as a case-split, and a **bounded numeric** range consumed *numerically*
(comparison / arithmetic / range only), T4 holds **by construction**: two inputs with
equal projections cannot be split by the consumer into distinct effects, so no
residual channel can exist. The procedure (`verify_enum`, `verify_bounded_numeric`)
is two finite checks: codomain membership, and that the registered consumption mode
is case-split / numeric (not re-parsing).

This is decidable **only because the consumption mode is a registered property of a
finite, surveyable sink (K2)** — otherwise it would be whole-program analysis (Rice),
undecidable. And "equal projections cannot be split into distinct effects" is a
statement about the *consumer*, not about the predicate: an exact `enum` check says
nothing about a handler that defaults an absent argument, reads an alias, formats a
list where it expects a scalar, or resolves the checked string against a
filesystem. So the consumption mode is now registered **per field**, not inferred
from the predicate's kind:

- **Consumption contracts** (`axor_core/kernel/consumption.py`). Per sink, the
  operator names every field the handler reads, its kind (`number`, `flag`,
  `identifier`, `identifier_list`, `path`, `interpreted`, `text`) and what the
  handler does when it is absent (refuse, a declared default, not consumed).
- **Registration** (`validate_consumption_contracts`) checks each predicate against
  the contract of the field it guards: an `enum` on a `path` field is refused with
  the classifier's FUZZ_REQUIRED verdict (the branch that used to be latent is now
  reached through a real field), a range on an identifier is refused, a predicate on
  a field the contract does not name is refused. `validate_value_policies` (the
  predicate-kind check) still runs, and still fails an unknown kind closed.
- **Preparation** (`prepare_args`) runs before the first gate on both wrapping paths
  and in replay: closed field set, declared defaults materialised, the declared
  numeric conversion applied once. The prepared set is the one every gate reads,
  the call-bound approval digests, and the handler receives (`ToolCallGovernor`
  returns it as `decision.args`).
- **Without a contract** the handler's conformance is a premise. Under STRICT it
  must be stated per sink (`consumption_attested`); `conformance_obligations()`
  reports, per guarded argument, whether it rests on a contract, an attestation, or
  nothing.

What the contract does **not** establish: that the handler honours it. The kernel
makes the check and the dispatch agree with the contract; whether the handler's code
agrees with the contract is the operator's statement. The measured cases — a real
handler behind the real loop, with and without a contract — are in
`tests/adversarial/test_check_vs_consumer.py`: an absent amount the handler turns
into a balance sweep, an alias field that redirects the transfer, a list booked as
an account string, and an allowlisted path that is a symlink out of the root all
execute without a contract and are refused with one; an exponent string and a
comma-joined recipient are refused either way (controls — not gaps today).

**Fuzzing only — undecidable in general.** For **path** (a filesystem resolver),
**string subfields** (a shell / SQL / URL / template interpreter), and **carrier over
free text**, the consumer *is* a rich-syntax interpreter; effective can exceed
nominal, and T4 stays a **fuzzing obligation** — fuzzing produces evidence over the
inputs it drew, it does not discharge the obligation. The two historical bugs live
exactly here:

- newline injection in path extraction — fixed in `axor_core/node/canonicalizer.py`
- `../` traversal in lease validation — fixed in
  `axor_core/capability/lease_validator.py` (now `os.path.normpath` before compare);
  the sound matcher is `axor_core/security/paths.py`.

---

## 5. Obligation → enforcing test / CI (the crosswalk)

The theorem is "fixed" precisely when every premise is tied to a regression that
fails if the premise is violated. Test paths are relative to the repo root.

| Obligation | What it requires | Pinned by |
|---|---|---|
| **O1** factorization | decision factors through `π`; no hidden raw channel; equal projection → equal decision | `tests/invariants/test_pure_allow.py` (`test_gate_decision_is_stable`, `test_no_probabilistic_component_in_the_loop`); adjudicator memoized by `projection_hash` (`axor_core/kernel/adjudicator.py`); `tests/invariants/test_security_invariants.py` inv13 (no raw content in normalized intent); `tests/adversarial/test_schema_injection.py` |
| **O1** kernel/trust-model factorization | kernel does not depend on runtime/platform — the decision survives platform replacement | `lint-imports` (`.importlinter` `kernel-purity` contract), run in CI; `tests/test_kernel_only_import.py` |
| **O2** label soundness | `causal_root ⊇` explicit untrusted influence; over-taint; worker cannot clear | `tests/taint/test_value_ledger.py`; `tests/adversarial/test_ledger_soundness.py` (saturation fails closed, endorse does not under-taint a shared fragment); `tests/invariants/test_security_invariants.py` inv04/inv05 |
| **O2** model edge (gap) | a re-encoded / unsegmented untrusted identifier the model emits is *not* labelled — recorded, not silently missed | `tests/adversarial/test_context_default_integrity.py` (controls pass; the attacker encodings assert the sound behaviour `xfail(strict=True)`); fix in [rfc-integrity-context-default](rfc-integrity-context-default.md) |
| **O2** scope boundary | implicit/control-flow leaks are *out of scope* — claimed, not silently missed | `tests/adversarial/test_implicit_flow_gap.py` (sound behaviour asserted `xfail(strict=True)`: the suite trips the moment a sound backend closes the gap) |
| **O3** complete mediation | no effect bypasses `allow`; unknown sink fails closed; ceilings cannot be widened | `axor_core/capability/locked.py` (`GovernanceBypassError`); `tests/invariants/test_security_invariants.py` inv15 (denied tool never reaches executor); `tests/adversarial/test_unknown_sink_posture.py`; `tests/adversarial/test_critical_bypasses.py`; `tests/adversarial/test_e2e_gate.py` (lethal-trifecta egress denied) |
| **T0** non-interpreting producer | no model / network / subprocess produces a trusted-path projection | `.importlinter` `t0-producers-non-interpreting` contract (run by `lint-imports`); `tests/invariants/test_t0_producers_non_interpreting.py` (import scan + determinism); `tests/invariants/test_pure_allow.py` (`test_no_probabilistic_component_in_the_loop`) |
| **T4 decidable** enum/numeric | decided for the predicate language over fields with a registered consumption contract; a predicate is checked against its field's contract (an enum on a path field fails with FUZZ_REQUIRED); every call is prepared into the contract before the first gate; without a contract STRICT requires an explicit attestation | `axor_core/kernel/decidability.py` (`verify_enum`, `verify_bounded_numeric`); `axor_core/kernel/registration.py` (`validate_value_policies`); `axor_core/kernel/consumption.py` (`validate_consumption_contracts`, `prepare_args`); `tests/kernel/test_decidability.py`, `tests/adversarial/test_projection_validators.py`, `tests/kernel/test_consumption.py`, `tests/adversarial/test_check_vs_consumer.py` |
| **Permitting grounds** | every input that can permit a call has a stated issuer, check, absence behaviour and non-overridden obligations | `axor_core/kernel/grounds.py`; `tests/invariants/test_permitting_grounds.py` |
| **End-to-end chain** | ground → projection → consumption → dispatch → observed effect, with an arbitrary and an adaptive proposer, and with a failed integrity labeler | `tests/integration/test_end_to_end_chain.py` (property-tested, not proved) |
| **T4 fuzzing** path/carrier/string | effective codomain stays in-bounds over the fuzzed inputs (evidence, not a discharge) | `tests/kernel/test_decidability.py` (`test_path_normalizer_fuzz_floor_no_escape`, floor = 8); `tests/adversarial/test_property_fuzz.py` (allowlist never false-accepts; carrier never admits a dangerous bare string) |
| **"Any trust model"** | the decision factors through the trust model's projection — swap the model, K4 still holds | `tests/contracts/test_value_provenance.py` (`test_noninterference_through_alt_trust_model`, `test_complete_mediation_through_alt_trust_model`): a second trust model with a *different* tainting rule re-proves non-interference + mediation; interface is `axor_core/contracts/provenance.py` |
| **Non-interference caveats** | adjudicator memoization is "T1 modulo cache state"; detection stays out of `allow` | `axor_core/kernel/adjudicator.py`; `tests/adversarial/test_detection_degradation.py`; [governance-model §8](governance-model.md) |

CI runs the three structural checks plus the suite on every push: the
no-provider-imports static analysis, `lint-imports` (rings), `python tools/check_docs.py`
(no doc drift), and `pytest tests/`.

---

## 6. What the theorem does *not* claim (scope)

Stated as boundaries of the claim, not as incidental gaps:

- **Implicit information flow** — O2 is explicit-flow soundness only (§2). A
  control-flow side channel with `causal_root = ∅` is not covered. Bounded by
  `tests/adversarial/test_implicit_flow_gap.py`; the standard partial fix
  (CaMeL STRICT-mode dependency augmentation) is not yet adopted. This boundary is
  **ours, not a shared one**: FIDES closes it on the integrity axis by checking the
  trusted context of the call (§2).
- **The integrity model-edge gap** — a value the model generates that contains no
  registered fragment derives `constant()` (trusted). This covers paraphrase, and
  also re-encoded or unsegmented *copies* of an untrusted identifier, so it is not a
  paraphrase-only residual: it is an O2 gap on the LLM edge (§2). It is closed where
  an `enum` allowlist guards the driving arg (Strict obliges one on egress sinks,
  not on outside-workdir writes or generated-code execution). The confidentiality
  floor and the positional/carrier gates are unaffected on *their* partition; the gap
  is integrity-only. See [governance-model §7](governance-model.md) and
  [rfc-integrity-context-default](rfc-integrity-context-default.md).
- **A-3 is a premise, not a theorem** — O3 holds *iff* the finite sink ring is the
  only path to an effect. In-process this is a soft boundary; the hard boundary is the
  daemon.
- **Availability is outside the safety perimeter** — K4 guarantees integrity and
  confidentiality, not uptime. The opt-in detection→degradation path can cost
  availability under a miscalibrated threshold (reputation-poisoning-upward); that is
  a named non-goal, not a counterexample to K4.
- **"Any trust model" — demonstrated on two instances, not proved for all** — the
  kernel/trust-model factorization is enforced at the *import* level (`.importlinter`)
  and a substitution test (§5) re-proves non-interference and complete mediation under
  a second trust model with a different tainting rule. That raises the claim from
  argued-from-the-interface to demonstrated-on-≥2-instances; it is not a mechanized
  proof over *all* conforming backends, which would be the inductive content (O2) in a
  proof assistant.

---

## 7. Status

- **Open gap, pinned:** O2 on the model edge (§2) — tracked by a strict-xfail
  regression, fix specified in
  [rfc-integrity-context-default](rfc-integrity-context-default.md).
- **Stated and pinned:** O1, O2 (explicit-flow scope, kernel constructors), O3, T0 (import contract +
  scan/determinism test, §3; the permitting-ground specification, §3b), T4 (the
  decidable branch through consumption contracts; the fuzz branch as tested
  evidence), the non-interference invariant and its caveats — each with a named
  regression in §5.
- **Premises, stated as such:** handler conformance to its consumption contract;
  an approval callback that is not model-backed.
- **Demonstrated on two instances:** the trust-model-agnostic claim — a substitution
  test re-proves non-interference + mediation under a second trust model (§5/§6).
- **Argued, not mechanized:** universality over *all* conforming trust models; the
  obligations are a framing with checkable premises, not a machine-checked proof (O2
  is the inductive content, a Coq/Lean target).

This document is the narrative anchor for the theorem. When a mechanism below it
changes, the corresponding crosswalk row is the contract that must still hold.
