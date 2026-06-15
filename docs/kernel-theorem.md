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

> **K4 holds outright for the enum/numeric fraction of the perimeter, and
> conditionally — discharged by fuzzing — for the rich-syntax (path / carrier /
> string-subfield) fraction.**

The weird-machine class did not disappear; it is *localized* to the rich-syntax
codomains. Both real bugs ever found in this layer (newline injection in path
extraction, `../` traversal in lease validation) were T4-class and lived in path
handling — the fuzzing region, exactly where the split predicts bugs must live.

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
The failure direction is deliberate over-taint (`⊇`, never silent).

- Constructors and the join live in `axor_core/taint/causal_root.py`; the
  refcounted, fail-closed ledger is `axor_core/taint/ledger.py`; per-value
  register/derive, the confidentiality floor, and governed release are
  `axor_core/taint/engine.py`.

**Scope boundary (part of the claim, not a hidden gap):** O2 is **explicit-flow
soundness only**. An implicit (control-flow) leak — a deterministic interior branch
on content that influences an output with `causal_root = ∅` — is **not** covered.
This is shared with FIDES by design; CaMeL's STRICT-mode dependency augmentation is
the known partial technique and is not yet adopted. See §6 and
[governance-model §7](governance-model.md).

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
Firewalls input-firewall LLM-projector is the public counterexample). Such a process
may feed the detection layer only, never the trusted path.

In this codebase every trusted-path projection is produced by a deterministic,
structural function — there is no model in the decision loop:

- carrier classification — `axor_core/security/carrier.py`
- consequence (action-class) classification — `axor_core/policy/consequence.py`
- structural normalization — `axor_core/policy/normalizer.py`

**Honest assurance status.** T0 is enforced *structurally* (there is no model to
plug into a gate) and is partially pinned by the no-probabilistic-component test
(§5). There is **no single named CI gate** that asserts "no projection-producing
function reads a model"; today that property holds by construction and by the
absence of any such hook, not by a dedicated check. Closing this with an explicit
T0 lint is the cleanest remaining assurance item for the theorem.

---

## 4. T4 and the decidability split (Theorem 0)

T4 (effective = nominal) is the obligation where the theorem's safety content lives.
Whether it is *decidable* depends on the codomain **and** the consumer, and it splits
cleanly. This split is the kernel's one genuinely new (if modest) result; it is
implemented in `axor_core/kernel/decidability.py`.

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
undecidable. So: *T4 is a theorem for enum/bounded-numeric, conditional on K2.*
`axor_core/kernel/registration.py` makes this load-bearing — it **rejects** a
configuration that tries to guard a fuzz-required field with a decidable predicate
(the silent misconfiguration that would give false assurance).

**Fuzzing only — undecidable in general.** For **path** (a filesystem resolver),
**string subfields** (a shell / SQL / URL / template interpreter), and **carrier over
free text**, the consumer *is* a rich-syntax interpreter; effective can exceed
nominal, and T4 stays a **fuzzing obligation**. The two historical bugs live exactly
here:

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
| **O2** scope boundary | implicit/control-flow leaks are *out of scope* — claimed, not silently missed | `tests/adversarial/test_implicit_flow_gap.py` (sound behaviour asserted `xfail(strict=True)`: the suite trips the moment a sound backend closes the gap) |
| **O3** complete mediation | no effect bypasses `allow`; unknown sink fails closed; ceilings cannot be widened | `axor_core/capability/locked.py` (`GovernanceBypassError`); `tests/invariants/test_security_invariants.py` inv15 (denied tool never reaches executor); `tests/adversarial/test_unknown_sink_posture.py`; `tests/adversarial/test_critical_bypasses.py`; `tests/adversarial/test_e2e_gate.py` (lethal-trifecta egress denied) |
| **T0** non-interpreting producer | no model produces a trusted-path projection | structural (deterministic classifiers in `axor_core/security/carrier.py`, `axor_core/policy/consequence.py`); partially `tests/invariants/test_pure_allow.py` (`test_no_probabilistic_component_in_the_loop`). **No dedicated CI gate — see §3.** |
| **T4 decidable** enum/numeric | faithfulness by construction; config that mis-guards a fuzz field is rejected | `axor_core/kernel/decidability.py` (`verify_enum`, `verify_bounded_numeric`); `axor_core/kernel/registration.py` (`validate_value_policies`); `tests/kernel/test_decidability.py` |
| **T4 fuzzing** path/carrier/string | effective codomain stays in-bounds under fuzzing | `tests/kernel/test_decidability.py` (`test_path_normalizer_fuzz_floor_no_escape`, floor = 8); `tests/adversarial/test_property_fuzz.py` (allowlist never false-accepts; carrier never admits a dangerous bare string) |
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
  (CaMeL STRICT-mode dependency augmentation) is not yet adopted.
- **The integrity paraphrase residual** — on the non-liftable partition (a generic
  write/exec that cannot be made positional), an in-process model that paraphrases an
  untrusted value before a sink is not caught by content derivation. The
  confidentiality floor and the positional/carrier gates are paraphrase-proof on
  *their* partition; this residual is integrity-only. See
  [governance-model §7](governance-model.md).
- **T0 has no dedicated CI gate** (§3) — held by construction, not by a check.
- **A-3 is a premise, not a theorem** — O3 holds *iff* the finite sink ring is the
  only path to an effect. In-process this is a soft boundary; the hard boundary is the
  daemon.
- **Availability is outside the safety perimeter** — K4 guarantees integrity and
  confidentiality, not uptime. The opt-in detection→degradation path can cost
  availability under a miscalibrated threshold (reputation-poisoning-upward); that is
  a named non-goal, not a counterexample to K4.
- **"Any trust model" is argued, not regression-tested** — the kernel/trust-model
  factorization is enforced at the *import* level (`.importlinter`) and the per-value
  trust-model interface is `axor_core/contracts/provenance.py`, but there is no test
  that swaps the reference trust model for another and re-derives K4. The claim rests
  on the factorization (O1) plus the trust-model lemma (O2 over guarded coercions),
  not on a mechanized substitution proof.

---

## 7. Status

- **Stated and pinned:** O1, O2 (explicit-flow scope), O3, T4 (both branches),
  the non-interference invariant and its caveats — each with a named regression in §5.
- **Held structurally, not by a dedicated gate:** T0 (§3).
- **Argued, not mechanized:** the trust-model-agnostic claim (§6); the obligations
  are a framing with three checkable premises, not a machine-checked proof (O2 is the
  inductive content, a Coq/Lean target).

This document is the narrative anchor for the theorem. When a mechanism below it
changes, the corresponding crosswalk row is the contract that must still hold.
