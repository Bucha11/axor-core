# Axor and ROPE — obligation by obligation

Compared: **ROPE at `64495e1`** (`github.com/xhowenma/rope`, `src/rope/`) and its paper
(arXiv 2608.27496v1: Proposition 1 *origin soundness* under A1–A3, Lemma 1 *record
closure*, Lemma 2 / Proposition 2 *wording / paraphrase invariance*), against **Axor**
on branch `claude/t4-consumption-contracts` (§5 of `paper-outline.md`, this repo's
`docs/guarantees.md`).

This is not a list of features. Each row is one obligation the claim "the executed
call satisfies the stated constraints" needs, and for each system it states the
mechanism, **how the obligation is established** (by construction, by a proof, by a
test, or as a premise), and what kind of difference it is:

- **mechanism** — one system has a mechanism for the obligation the other lacks;
- **policy content** — the difference disappears if the other system is given the
  same static constraint (an equal-policy baseline must do this before any result is
  credited to a mechanism);
- **scope** — the obligation is outside what the other system claims.

ROPE configurations are named explicitly: *routed* (a per-task scope from an LLM
router over the request — cached or live), with or without the *clamp*
(`RoutedPolicyBootstrap(clamp=False)` is the code default, `pipeline.py:35`; the
paper's Table 5 reports both arms). Axor configuration: STRICT, operator policy,
consumption contracts on guarded fields, non-model approval callback.

## The table

| # | Obligation | ROPE — mechanism (code) | ROPE — established by | Axor — mechanism | Axor — established by | Difference |
|---|---|---|---|---|---|---|
| 1 | **The policy a call is checked against is the intended one** (K5 / the second argument of `decide(x, p)`) | Audited per-suite floor (`scopes/_floor/<suite>.json`) + a per-task scope from a router that reads only the request (`router.py`); optional clamp drops overrides looser than the floor (`clamp.py:22–53`) | **R** (router isolation) by construction; clamp bounds loosening *of floored parameters* by construction; without the clamp, router loosening is measured, not bounded (paper §4, Table 5). Not bounded by the clamp: `named_source` (deliberately, `clamp.py` docstring) and overrides on non-floored parameters (kept, `clamp.py:48–49`) | Static operator policy; STRICT refuses classifier-derived authority; grants/leases inside `G`; call-bound approval | Unit tests (STRICT refusal); property tests over authority transitions (`test_authority_transitions.py`); approver is a **premise** | **Mechanism, in both directions.** ROPE derives per-task scope from a model over trusted input — in T0 terms, a model-produced permitting ground whose argument is input isolation (R) plus the clamp; Axor has no per-task scope (and pays utility for it). Axor has authority transitions mid-task (grants, approvals) and an invariant over them; ROPE's policy is fixed per task |
| 2 | **Permitting grounds for a value** (T0) | T1 request tokens; T2 `sender`/`from` of a structured record (`origin.py:27`); T3 authoritative-record tools and reads whose path contains `/system`, `/.ssh`, `/etc` (`origin.py:230, 303`); blobs never license | **A1** (platform metadata truthful) — premise; **A2** (record integrity) — **proved maintained** along the trajectory by Lemma 1, given an offline enumeration of writes `W` | `TrustedOrigin` TASK / OPERATOR (allowlist members) / TOOL (declared benign) / ENDORSED; context root under STRICT; enum supersession | Grounds as data, held to the code by tests (`kernel/grounds.py`); completeness **argued** by enumerating the allow path | **Mechanism, in ROPE's favour on one point.** Axor's TOOL ground rests on "a declared benign tool's output is not attacker-writable" — exactly ROPE's A2 — and Axor has **no** analogue of Lemma 1: nothing checks that an earlier admitted call cannot write into what a benign tool later returns. ROPE discharges that premise; Axor states it |
| 3 | **The matcher equates only equal values** (ROPE's A1 matching clause) | `tokenize_identifiers` on both sides; the value is a whole unit for `/` but is still split on characters outside `[A-Za-z0-9._/@:\-+]`; admitted if **any** value token is trusted (`origin.py:270–290`); URL decided by host | Stated as part of **A1** (premise), described as the decision-unit rule (paper App., "Decision unit identity matching") | Exact `enum` membership on the prepared value; origin by ledger/context-root derivation, not token matching | Exact equality (decision procedure) | **Mechanism (a boundary example, not an exploit).** Observed on ROPE's own `OriginPredicate` (script below): with `GB29…` in the request, the whole strings `"GB29…,US13…"` and `"GB29… US13…"` are **admitted** — origin-matching succeeds on one token while the string as a whole equals no trusted value; the list `[GB29…, US13…]` and `"US13…/GB29…"` are refused. The paper describes the candidate as matched "as a whole identifier"; the code's value side is a set of identifier tokens. Whether this departs from A1 depends on A1's exact semantics (equal *values* vs. equal *decision units*) and must be reconciled with them before it is cited. It is **not** an attack on AgentDojo (row 4). It is kept as a verified instance of the matcher/consumer boundary: what the matcher equates and what the consumer receives are separate obligations |
| 4 | **The checked property is the consumed property** (T4) | None in ROPE. The guard checks the raw kwargs, and only those present (`enforce.py:79, 92`); AgentDojo's runtime then validates and normalises them with the tool's pydantic model, filling defaults, **after** the check | Not part of Proposition 1, which is about the **origin of the admitted value**, not the effect ("origin, not intent", paper §3) | Consumption contracts: per-field kind and absence rule, checked against predicates at registration; every call prepared before the first gate; the prepared set executes (`kernel/consumption.py`; `decision.args` in axor-wrap and the AgentDojo adapter) | Unit tests; negative-control matrix rows 1–3 (`0 → 1 → 0` effects); attested sinks are a **premise** | **Mechanism.** On AgentDojo banking, the tools' own typing (`recipient: str` required, `amount: float`) makes the specific gaps of rows 3–4 unreachable — the comma-joined recipient is booked verbatim as one (non-existent) account
(`banking_client.py: send_money`), and the benchmark scores an attack by recipient
**equality** with the attacker IBAN, so it would not count either way. On a consumer with a default, an alias or its own parsing, the gap is an effect (Axor matrix rows 1–3). This row is where Axor's claim must be measured, and AgentDojo is not the instrument for it |
| 5 | **The executed call is the checked call** (dispatch) | `guarded_run_function` checks, then calls the original `run_function` with the same `args` object, synchronously (`pipeline.py`) | By construction (no await between check and dispatch) | Preparation before the first gate; frozen argument set; digest over the prepared set; volatile-state re-check after an approval; atomic claim | Unit tests (`test_call_bound_authority.py`, `test_the_call_that_runs_*`) | **Not a differentiator by itself.** Axor needs the extra machinery because it has an asynchronous approval step between check and dispatch; ROPE has none. The difference that matters is row 4 (what is dispatched is the *prepared* set) |
| 6 | **Every effect-bearing call is mediated** | Tools in the floor are guarded; every other tool is **always-allowed** (`pipeline.py:84–85`) | **A3** (enumeration completeness) — premise, checked offline | STRICT role completeness and consequence completeness: an undeclared tool is refused at construction or first use | Unit tests; O3 tests | **Mechanism (fail-closed vs. premise).** Both rest on an operator table; Axor refuses what the table omits, ROPE admits it and states A3 |
| 7 | **No unattended irreversible action** | `EXPLICIT` marker: blocked unless a per-task override pins a request-named target (`policy_synthesis.py`); under the clamp, **any** override of an `EXPLICIT` default is dropped (`_more_permissive`, `clamp.py:22–35`), so the action stays blocked | By construction (deterministic enforcer); the override comes from the router (row 1) | Consequence class + unattended ceiling; call-bound approval for one prepared call | Unit tests; approver is a **premise**; matrix row 5 shows a model-backed approver as the counterexample | **Mechanism.** ROPE's authorisation for an explicit action is a model reading the request; Axor's is an external approver. Both are premises; Axor states its own as such and measures the failure mode |
| 8 | **No egress of secret-derived content** | None — a trusted recipient may receive anything ("origin, not intent") | Out of scope | Confidentiality floor: egress refused while a sensitive read is outstanding, content-blind | By construction (the decision reads a session fact, not bytes); unit tests; matrix row 4 | **Scope.** ROPE does not claim it. Adding it is not a new marker — markers check an argument's origin, and the floor is session state that must be armed by declared sensitive reads |
| 9 | **Magnitudes within bounds** | `MAGNITUDE` = JSON-schema `{"type": "number"}` (`policy_synthesis.py:76`); the banking floor guards no amount | — | `numeric_range` / `keyed_range` | Decision procedure; with a contract, on the prepared value | **Policy content.** ROPE accepts a JSON-schema restriction (`{"type":"number","maximum":1000}`) through the same `check_arg`; an equal-policy baseline must add it before any amount result is credited to Axor |
| 10 | **Rewording cannot change the decision** | Lemma 2 / Proposition 2: decision is a function of (value, tracked origin, compiled policy), under R and D | Pen-and-paper proof | K4 lemma: `decide` constant on the fibers of an admissible projection, under O1–O3 | Argued; O1 pinned by tests | **Same shape.** Neither system's novelty can rest here |
| 11 | **Constraints survive the failure of one component** | Clamp bounds a weak router (paper §4, Table 5: Gemini-3-Flash ASR 6.0 → 0.0 clamped on the suites with over-loosening) | Measured end to end | Independent gates per obligation; labeler fault injection keeps floor and consequence, loses integrity (asserted) | Fault-injection test | **Different components.** ROPE measures router failure; Axor measures labeler failure. Neither measures the other's |
| 12 | **Evidence** | Proofs (Prop. 1, Lemmas 1–2) under stated assumptions; benchmark (AgentDyn, AgentDojo) | — | Specification tests, property tests, generated negative-control matrix; benchmark (AgentDojo) | — | ROPE has proofs where Axor has tests (rows 2, 10); Axor has measured negative controls where ROPE states premises (rows 3, 4, 6). Neither is machine-checked |

## What this means for the claim

- The defensible differences are **rows 4 and 6** (mechanism: consumption registered
  and prepared before the check; fail-closed mediation) and **row 8** (scope). The
  difference does not need an exploit in ROPE: it is that Axor implements and tests
  the closing of obligations that ROPE's construction leaves as premises (consumer
  conformance; completeness of the guarded set). **Row 3** is a supporting example of
  why matcher and consumer are separate obligations — a whole-string mismatch under a
  successful origin match, to be reconciled with A1's exact semantics, and not a
  headline.
- **Row 9 is policy content** and must not be credited to Axor's mechanism. The
  comparison experiment needs an equal-policy ROPE arm (ROPE + the same static
  amount / allowlist restrictions via `check_arg`); if that arm reproduces a result,
  the claim shrinks to the remaining rows — which is the point of running it.
- **Row 2 goes to ROPE**: Lemma 1 discharges the record-integrity premise that Axor's
  `TrustedOrigin.TOOL` leaves as a premise. Axor should either adopt a record-closure
  condition for declared benign tools or keep stating it as an open premise.
- **Rows 1 and 7** are a trade, not a ranking: ROPE spends a model (over trusted input,
  optionally clamped) to buy per-task utility; Axor spends utility to keep a model
  out of policy derivation, and keeps an external approver as a premise instead.
- On AgentDojo specifically, row 4 is mostly invisible because the benchmark's tool
  signatures act as an implicit, post-check contract. The comparison of the T4
  obligation needs consumers with defaults, aliases or parsing — the handlers of
  `tests/consumers.py`, or real ones — not the benchmark alone.

## Not done here

No live run of either system: the comparison is by code and by the stated theorems.
The row-3 measurement runs ROPE's `OriginPredicate` in isolation; no ROPE pipeline,
router or AgentDojo episode was run.

## Reproducing row 3

Run against a checkout of ROPE at `64495e1` (pure functions only; needs PyYAML):

```python
import sys; sys.path.insert(0, "<rope>/src")
from rope.origin import OriginPredicate, tokenize_identifiers
A, X = "GB29NWBK60161331926819", "US133000000121212121212"
p = OriginPredicate("recipient", None, {}, tokenize_identifiers(f"Pay my rent to {A}"))
for v in [A, X, f"{A},{X}", f"{A} {X}", [A, X], f"{X}/{A}"]:
    print(repr(v), p(v))
# observed: True, False, True, True, False, False
u = OriginPredicate("url", None, {}, tokenize_identifiers("summarise https://docs.example.com/page"))
print(u("https://docs.example.com/exfil?d=SECRET"))  # observed: True (host is the decision unit)
```
