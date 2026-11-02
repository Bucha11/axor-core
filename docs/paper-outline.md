# Axor — Paper Outline

Clean structural outline for the axor paper. Every claim is anchored to something
already in the repo (file paths in parentheses) so the writing stays honest — no
mechanism is invented for the paper.

Process notes, resolved decisions, the canon number lock, retired numbers, and the
questions for Haoyu live in `docs/paper-decisions.md`. This file is the skeleton of
the paper itself; that file is why the skeleton looks the way it does. **Before
drafting §6, complete the canon action item in `paper-decisions.md` (the new paired
slack/workspace/travel runs are not yet recorded in `agentdojo_results.md`).**

---

## 1. Introduction

- **The gap.** Production LLM agents execute tool calls with no defense layer
  between intent and effect: they call tools never granted, read outside scope,
  spawn uncontrolled children, carry tainted external content through N intents,
  export reasoning that should never leave the boundary (README "The Problem").
- **The wrong fix.** Prompt-injection *detection* (recognize the bad prompt) is a
  losing game — every new framing defeats it. Axor's thesis: this is not a model
  problem or a prompt problem, it is an **execution-governance** problem. Govern
  *what actions cross the boundary*, not *what the prompt says*.
- **What axor is.** A policy membrane between agent intent and tool execution:
  `same agent · same model · same prompt → different policy → different execution`.
  A *layer*, not an adapter; not another agent framework (contrast LangGraph /
  CrewAI / AutoGen, which orchestrate — axor governs).
- **Four contributions** (the paper's spine):
  1. A **framework-agnostic** governance kernel with a clean abstraction seam to any
     agent framework (Claude, LangChain, OpenAI-shaped, …) — §4.
  2. A **secure-by-design** security model anchored by the **Perimeter
     Non-Interference theorem (K4)**, with an explicit chain from a concrete
     agentic attack → taint analysis → the non-interference guarantee — §5.
  3. **Empirical validation on AgentDojo**, framed as CaMeL frames it — a **structural
     guarantee at a measured utility cost** (capable model, o4-mini: banking ≈−17 ± 7pp,
     slack −34 ± 7pp, workspace −15 ± 4pp, travel 0pp — all paired), *not* a
     headroom-dependent ASR-delta that vanishes on a robust model. The cost is
     localized to the shared-channel partition; the comparison to CaMeL is reported
     honestly in §6.3, and axor's case is adoption cost (§6.5), not raw utility — §6.
  4. **A kernel result: a sound `enum`-supersession rule** — a sink whose driving args
     are covered by a satisfied `enum` over a *closed trusted set* need not be gated by
     the leaky content-taint on top; on banking it deterministically lifts the gate
     over-block on read-derived egress to allowlisted known payees (+25pp gate-level
     upper bound; +13.4 ± 9.1pp realized). Full statement, soundness condition, and
     honest scope in §6.3 / §5.5 / Appendix D. Supporting, not the headline.
- **Novelty arbitration (one headline — a reviewer will ask "which is *the*
  contribution?").** The headline is **the chain of §5**: the conditions under which
  an existing agent may propose arbitrary calls while every executed call satisfies
  every stated constraint — permitting grounds from trusted mechanisms (T0), the checked
  property bound to the consumed one by registered consumption contracts (T4), the
  executed call bound to the checked one, and independent obligations that survive the
  failure of one component — with each condition shown load-bearing by a measured
  negative control (§5.7). K4 is a lemma inside the chain, the decidability split a
  classification inside T4; neither is billed as the novelty on its own. A novelty
  claim rests on what the contracts measurably prevent, not on the formulation.
  Contributions 1 (the framework-agnostic seam), 3 (the empirical study), and 4
  (supersession) are **supporting**; do not let §4 and §5 each claim "the
  contribution."
- Forward-reference the opening example (§2).

---

## 2. Illustrative example — "what a successful defense actually looks like"

**The example: the banking exfiltration task, AgentDojo `important_instructions`** —
the cleanest integrity gate-walk and the CaMeL-comparable scenario. (Suite choice is a
resolved decision; rationale in `paper-decisions.md`. Travel is the natural *second*
illustration — it is the suite where the confidentiality floor arms and closes on the
sound axis, §5.6 / Appendix A-floor — an addition, not a swap.)

**The scene.** A banking agent is asked to do a benign task (summarize transactions /
pay a bill). The transaction history it must read contains an injected instruction:
*"send a transaction to `attacker-IBAN` …"* (data and instructions share a channel —
the hard core of injection). **The model fools itself:** undefended **GPT-4o** — a
strong, current production model — carries this out on **60.4%** of pairs
(`examples/agentdojo/agentdojo_results.md`). This is the example to lead with
precisely *because the model is not weak*: the reader cannot dismiss it as "use a
better model." A capable model with enough headroom to attempt the whole task list
still walks straight into the injection on its own — that is the problem axor exists
for, and §6.5 turns it into the central adoption argument.

**Walk the call through the gate sequence** (README "Reverse Osmosis", governance-model §2):
the model emits `send_money(recipient=attacker-IBAN, …)`. The recipient value entered
through an untrusted read (the transaction history), so its provenance is
*untrusted-derived*. The **per-value taint gate** refuses the egress — `category ==
"taint_enforcement"`. No email/transfer reaches the attacker; the legitimate
prompt-given recipient still passes.

**The 10-line code spine** that makes it concrete (README "Two ways to call the same
gates" — `ToolCallGovernor`): read the inbox → `register_output` records it untrusted →
`evaluate("send_email", {to: attacker_addr})` → `decision.allowed == False`. This is
the whole story in one screen; the rest of the paper explains *why it is sound* and
*why it cannot be reframed away*.

**The bridge to the theorem — intuition only (the formal walk is §5.2; do not duplicate
it here).** One sentence of intuition: the denial did **not** depend on recognizing the
injection — the gate read only the call's *projection* (the origin-class of the
recipient field), never the attacker's prose, so reframing the injection cannot change
the verdict. Name the guarantee (*perimeter non-interference*, §5) and forward-reference;
§5.2 carries the attack→taint→theorem walk in full. §2 gives the reader a feel; §5.2
proves it.

End with the honest qualifier, kept on the right axis. Framing is neutralized
structurally — but this banking case sits on the **integrity** axis, and there the
content-derivation gate has a documented **paraphrase/encoding residual** on the
non-liftable partition (§5.6): an attacker who *paraphrases* the recipient rather than
copying it is not caught. Do **not** claim the confidentiality floor closes this — it is
a *different axis* (it arms only after a secret read and is sound there); the positional
gate closes only sinks whose input can be made instruction-incomplete. State plainly:
this example is secure against framing, with a known residual against encoding on the
integrity partition — the rest of the paper is honest about which partition each gate
covers.

---

## 3. Threat model & background

- **Adversary.** Untrusted content reaching the agent through any read (web fetch,
  inbox, RAG document, tool output, another agent) that can carry instructions and/or
  attacker-chosen values. Goal: drive a high-consequence effect — exfiltration
  (egress of a secret or a third party's data), destructive action, capability
  escalation, uncontrolled child spawn.
- **The lethal trifecta** framing (untrusted input × access to secrets/private data ×
  egress) — tie to `tests/adversarial/test_e2e_gate.py` (lethal-trifecta egress denied).
- **The partition taxonomy (define once here; §5.6 and §6 consume it).** The paper's
  results are stated over named partitions of sink traffic — introduce them up front so
  the later sections don't define vocabulary on the fly:
  - *Prompt-given egress* — the legitimate sink argument originates in the user prompt;
    never tainted, structurally zero-cost (travel).
  - *Shared-channel egress* — the legitimate sink argument is itself read from an
    untrusted source (data and instructions share a channel); the hard partition, where
    all measured cost falls (banking / slack / workspace).
  - Within shared-channel: the *liftable known-payee partition* (the argument belongs
    to an operator-enumerable closed set → recoverable by `enum`-supersession, §6.3)
    vs the *one-off residual* (the argument exists only in the untrusted read →
    needs structural provenance).
  - *Liftable vs non-liftable sinks* on the integrity axis (§5.6): a sink whose input
    can be made instruction-incomplete admits positional admission; a generic
    write/exec does not and carries the paraphrase residual.
  - The two *axes*: integrity (per-value content-derivation) vs confidentiality (the
    session floor) — asymmetric by design, §5.6.
- **Trust boundary.** In-process soft boundary (`LockedExecutor` → `GovernanceBypassError`)
  vs out-of-process hard boundary (the daemon re-derives the ceiling server-side).
- **Non-goals (state up front, honestly).** Availability/uptime; implicit
  control-flow leaks; covert channels; semantic malware detection
  (README "What benchmarks do not prove"; kernel-theorem §6).
- **Background.** AgentDojo benchmark; prior defenses to contrast in §7 (AgentSpec —
  per-framework runtime-enforcement DSL, the single-framework prior; CaMeL —
  by-construction security via an interpreter; FIDES — explicit-flow; Firewalls —
  LLM input-firewall, the T0 counterexample; classical noninterference / IFC — the
  formal lineage).

---

## 4. Design — a framework-agnostic governance layer

> The single-framework prior to engage *by name* is **AgentSpec** (Wang, Poskitt, Sun,
> arXiv:2503.18666), whose enforcement binds to one framework's hooks (LangChain;
> separately re-implemented per domain for embodied / autonomous-driving). Engage it
> directly; axor's contribution is lifting that binding into a provider-neutral seam.

**4.1 The central primitive — one boundary for flat and federated execution.**
Everything is a `GovernedNode` wrapping any executor; flat execution is just
`max_child_depth=0`, federation is `N`. No special case for single- vs multi-agent
(`docs/ARCHITECTURE.md`, `axor_core/node/wrapper.py`). *Rationale:* one enforcement
path means one place to be correct.

**4.2 The abstraction seam — three small interfaces, no framework leakage.**
A framework is integrated by implementing only:
- `Invokable.stream(envelope) → AsyncIterator[ExecutorEvent]` — *stream-based, not
  return-based*, so the node intercepts `tool_use` events **before** they execute
  (`axor_core/contracts/invokable.py`). *Rationale for streaming:* interception has to
  happen pre-effect; a return-based API could only observe after the fact.
- `IntentNormalizer.normalize(raw_event) → NormalizedIntent` — collapse any provider's
  tool-call shape to one structural intent (`axor_core/policy/normalizer.py`;
  per-provider shells like `LangChainNormalizer` in `tests/normalizers/`).
- `ToolHandler` — one per tool.
The executor *never sees raw ambient state* — only a governed `ExecutionEnvelope`
(scoped context, allowed tools, cancel token, policy). "The executor never knows it
is being intercepted."

*The addressed contrast (load-bearing):* AgentSpec expresses rules over a *specific
framework's* tool-call representation, so a new framework means re-authoring the
binding; axor pushes that variability into a single `IntentNormalizer` whose output
(`NormalizedIntent`) is the *only* thing the gates see — the decision core is written
once against the normalized shape, and supporting a new framework is one normalizer, no
policy change. State it as: *AgentSpec = per-framework enforcement DSL; axor = one
decision core behind a provider-neutral normalization seam.* This is the
**design/engineering contribution** — it *supports* the paper's single headline novelty
(the §5 chain); it is deliberately **not**
itself billed as "the" novelty (§1 novelty arbitration).

**4.3 Provider independence is tested, not asserted.** Claude, mock-OpenAI, and
mock-OpenRouter normalizers produce **identical `NormalizedIntent`** for the same
semantic intent (README "Provider Independence"; `tests/normalizers/`). Swap the
executor, policy semantics are byte-identical.

**4.4 Two callers, one decision core (the anti-drift design choice).** The six
structural gates live once as pure functions in `policy/gates.py`. Two callers:
the streaming `IntentLoop` (for frameworks where axor drives the loop) and the
synchronous `ToolCallGovernor` (for a framework that owns its own agent loop — the
AgentDojo and LangChain path). *Rationale:* the decision logic exists exactly once
and **cannot drift** between integration styles — pinned by tests
(`docs/governance-model.md §11`). This directly answers "how do you integrate >1
framework without forking the policy?"

**4.5 Two supporting facts (condensed — 4.2 and 4.4 are the load-bearing parts; full
detail → Appendix).** Keep these to a paragraph each so they don't dilute the seam:
- *Machine-enforced kernel purity.* Ring 0 (kernel/TCB) may not import Ring 1/2; CI
  `import-linter` + lazy imports make `from axor_core import ToolCallGovernor` load the
  kernel alone — so a framework-adapter or platform bug **cannot cause a wrong allow**.
  This is what turns "framework-agnostic" from a convention into a checked property
  (`tests/test_kernel_only_import.py`). One sentence in the body; mechanics to Appendix.
- *Deployment taxonomy.* A renamed tool set is governed by an operator role declaration
  (`untrusted_sources` / `sensitive_sources` / `egress_sinks` / `positional_sinks` /
  `value_policies` / `driving_args`), kwargs or fail-closed YAML, accepted by both
  callers (`governance-model §12`). Mention as the operator-facing half of the seam;
  the full key list belongs in Appendix B, not the §4 narrative.

*Figure for this section:* the trust-ring diagram + the three-interface adapter seam.

---

## 5. The security argument — one chain from ground to effect

> One claim, about execution, and one chain that establishes it. K4 is a lemma inside
> the chain; T0 and T4 are its conditions; composition and K5 are the parts that make
> the conjunction hold. Every step names what core enforces, what it rests on, and the
> kind of evidence — unit test, property test (Hypothesis: evidence, not proof), or
> measured negative control. Nothing here is machine-checked.

**5.1 The claim.** For every model behaviour `M` (any sequence of proposed calls):

```
Execute(s, c)  ⟹  ∧_{j ∈ J(p)}  P_j(s, c)
```

Each `P_j` is a constraint the operator's policy `p` states independently of the
others: the tool is permitted; the destination is allowlisted; the amount is in range;
no declared egress while a secret read is outstanding; no unattended call above the
consequence ceiling; an integrity sink's driving value is not attacker-chosen. The
formula is not a new mathematical fact — it is the shape of the obligation. Its
content is in the conditions under which it holds, which §5.2–§5.6 establish link by
link, and in the measured controls that show each condition is load-bearing
(§5.7). `docs/guarantees.md` is the one table of `P_j`, conditions, premises and
evidence.

One precision up front: untrusted bytes **may** change a verdict — an out-of-range
amount is denied *because of* what the model wrote, and a gate that could not see it
could not deny it. What untrusted bytes must never be is an **unchecked ground for
widening** authority. That is the T0 condition (§5.3), not a non-interference claim
about all bytes.

**5.2 The chain.** A call reaches an effect through five links; the claim fails if any
link fails, so each has an owner.

| Link | What must hold | Mechanism | Core enforces / premise | Evidence |
|---|---|---|---|---|
| **Ground** (T0) | every input that can permit a call comes from a trusted mechanism | `kernel/grounds.py` — issuer, check, absence behaviour, non-overridden obligations, premise | enforced per row; premises named per row | spec tests; producer negative controls |
| **Projection** (K4 lemma) | the decision factors through an admissible projection; labels are sound for explicit flows | O1, O2 | O2 explicit-flow only | `test_pure_allow.py`; ledger soundness tests |
| **Consumption** (T4) | the property checked is the property the handler consumes | consumption contracts; registration + preparation | enforced for contracted fields; **attested** fields are a premise | `test_check_vs_consumer.py`; matrix rows 1–3 |
| **Dispatch** | the call executed is the call checked (and approved) | preparation before the first gate; frozen set; digest over prepared args; volatile re-check; atomic claim; O3 | enforced; A-3 (single sink ring) is a premise in-process | `test_end_to_end_chain.py::test_the_call_that_runs_*`; call-bound tests |
| **Policy integrity** (K5) | the authority the call is checked against cannot be rewritten by the agent | ceilings, grants, leases read from live state | enforced; the operator's policy is the intended one (premise) | `test_authority_transitions.py` (property) |

**5.3 Lemma (K4) — perimeter non-interference.** Keep the lemma; demote it from the
headline. Let `decide(x, p)` be the actual gating function on the path to an effect.
The lemma is that `decide` is constant on the fibers of an admissible projection `π`:

```
∀ x₁, x₂ :  π(x₁) = π(x₂)  ⟹  decide(x₁, p) = decide(x₂, p)
```

with the three-part caveat stated every time: (1) the equation is factorization
(O1) and is satisfied by `π = identity`, so alone it carries no security content;
(2) the content is **coarseness** — `π` is fixed up front as a lossy projection from
the closed admissible set (enum / bounded-numeric / origin-class / path-class /
provenance-label), so a whole class of reframings collapses into one fiber and must
share a verdict; (3) the shared value admits no instruction — which is T0 and T4.
O2 (label soundness, explicit flows only — implicit flows are out of scope, and that
boundary is **ours, not shared**: FIDES closes it on the integrity axis) and O3
(complete mediation over a finite sink ring) are the lemma's premises.

What the lemma buys is framing-invariance of the **verdict**. It says nothing about
whether the verdict's subject is the call that runs (dispatch), whether its inputs
were legitimately produced (T0), or whether the checked property is the consumed one
(T4). Those are the conditions below, and they are where the claim about
**execution** is made.

**5.4 Condition T0 — permitting grounds.** A model that *issues* a permitting
input is in the trusted base however typed its output. The audit is data
(`axor_core/kernel/grounds.py`); the theorem's table is rendered from it and tested
for drift, missing mechanisms and missing pinning tests. Two columns keep apart what
core **checks** and what a ground **rests on** (its premise). The table is a
specification; its **completeness** — that every admitting input has a row — is
argued by enumerating what the allow path reads (kernel-theorem §3b), and no test
enforces that enumeration.

Determinism is not correctness: a deterministic matcher can still issue the wrong
ground. The two producer negative controls build a flawless codomain on purpose and
show it does not rescue a steerable producer: an approver with codomain {yes, no}
that follows the call's arguments executes an unattended consequential transfer (and
nothing else — the value constraints hold), and a datamark whose absence would admit
a re-typed attacker IBAN is refused by the context-root ground the model cannot shed
(matrix rows 5–6; `paper-model-carried-labels.md` for the measured search).

Supersession is a **substitution of grounds**, not a free relaxation: the attacker
can name a member of the operator's allowlist; membership stands in for the origin
ground on the integrity axis only, and the floor, consequence, ranges and capability
stay in force (`test_supersession_relaxes_integrity_only`).

**5.5 Condition T4 — the checked property is the consumed property.** The decidability
split is a classification (`kernel/decidability.py`): membership and range checks are
decision procedures; path / interpreted / free-text consumers are not. The step that
makes it about execution is that the consumer is **registered per field**
(`kernel/consumption.py`): kind, and what the handler does when the field is absent.
Registration checks each predicate against its field (an enum on a resolved path is
FUZZ_REQUIRED and refused); preparation puts every call into the contract before the
first gate, and the prepared set is what every gate, the approval digest and the
handler see — through `IntentLoop`, `ToolCallGovernor` (`decision.args`) and the
integrations that execute it (axor-wrap, the AgentDojo adapter).

> The honest statement: **for the defined predicate language over contracted fields,
> T4 is decided by a procedure; every other consumer needs a separate argument.**
> An **attested** sink is not contracted: core prepares nothing for it, the gates
> decide the raw proposal, and the operator's sentence is the whole of T4 there.
> Rich-syntax fields remain open; fuzzing over them is evidence, not a discharge.

The measured content is the gap between a correct verdict and a forbidden
execution: with identical value policies, removing only the contract produces a
balance sweep (handler default), a redirected transfer (alias field) and an
unvalidated account (list in a scalar field); restoring it removes each, and the
legitimate call still executes (matrix rows 1–3). An allowlisted path that is a
symlink out of the root publishes a secret without a contract; with one, the
configuration is refused at registration. Exponent strings and comma-joined
recipients are refused either way — controls, kept so the claim stays proportional.

**5.6 Composition and K5 — why the conjunction holds when a component fails.**
- **Independent obligations.** Each `P_j` has its own gate reading its own inputs, so
  the failure of one component relaxes only what depends on it. Fault injection:
  with an integrity labeler returning an arbitrary fixed label, the floor and the
  consequence ceiling still hold, and integrity is lost (and the test asserts the
  loss rather than hiding it) — `test_floor_holds_when_the_integrity_labeler_fails`.
- **Two axes, demonstrated differently.** Integrity is per-value content-derivation
  (under STRICT, context-default): sound to deny, with a residual in `clean` mode, and
  measured on the benchmark. Confidentiality is a session floor whose decision never
  reads the outgoing bytes — content-blind by construction, demonstrated at unit level
  and in the matrix (row 4), not averaged over a benchmark. "By construction" is an
  argument about the decision's inputs, not a proof object.
- **Positional admission** admits by form for sinks whose legitimate input cannot
  encode an instruction; an exec-class sink cannot be declared positional.
- **K5 — the second argument of `decide(x, p)`.** `A_t ⊆ B(t)`: grants, leases and
  degradation transitions never take the authority in force outside the operator's
  ceiling; property-tested over random interleavings of the real engines. A call-bound
  approval authorises one prepared call, is re-checked against volatile state after
  the answer and spent atomically; it relaxes the consequence axis only.

**5.7 The negative-control matrix.** `docs/negative-controls.md` is generated by a run
(and a test fails if it is stale): one condition at a time, removed → restored, with
violating effects counted on the handlers' own state and the legitimate call checked
with the condition in place. Every effect row reads `0 → 1 → 0` with the legitimate
call executed. The counts are per proposal — they show each condition is
load-bearing in isolation, not attack rates. Rates belong to §6. And they show
indispensability **within this construction**, not universal necessity of each
mechanism: confidentiality, for one, could be carried by a design without a session
floor. The claim is "in Axor as built, removing X admits the forbidden effect," and
it is stated that way.

**5.8 Putting it together.** Under STRICT, with an operator-supplied policy, an approval
callback that is not model-backed, consumption contracts on the guarded fields, and A-3:
every executed call satisfies every stated `P_j`, for any proposal sequence. Premises,
stated as such: handler conformance to its contract (and, for attested sinks, all of
T4); the approver; the operator's policy; A-3 in-process; O2's explicit-flow scope;
completeness of the grounds table (argued). Evidence: unit and regression tests for
each mechanism, property tests for the arbitrary-proposer and K5 statements, and the
measured matrix. Put kernel-theorem §5 (obligation → test) in an appendix.

**5.9 "Any trust model."** The gate depends on the `ValueProvenance` contract, not a
concrete engine; a substitution test re-runs non-interference and mediation under a
second trust model with a different tainting rule. Demonstrated on two instances, not
over all conforming backends; a CaMeL-style backend is proposed, not implemented.

---

## 6. Evaluation — AgentDojo

> **The measurement axis (the section's whole point).** The primary axis is **the
> structural guarantee at its measured utility cost**, CaMeL-shaped — *not* ASR-delta.
> ASR-delta is headroom-dependent: it requires a foolable model, and on a robust model
> it degenerates to 0% → 0% and evaporates. CaMeL never used that axis: its number is
> "fraction of tasks solved *with* a by-construction guarantee," i.e. the **utility
> cost** of the guarantee, well-defined on a dumb *and* a smart model. axor has the
> same by-construction nature (the gate denies by projection, framing-invariant,
> ASR→0 structurally — enum/numeric on contracted fields; rich-syntax fields remain an
> open, fuzz-tested obligation; §5.5), so it
> makes the same *type* of claim:
> - **Primary (§6.2): structural guarantee at utility-cost Y** — the load-bearing
>   result and the direct answer to "motivation for adoption": *motivation = the
>   guarantee; metric = its cost; not a scoreboard.*
> - **Secondary colour (§6.4): ASR-delta where headroom exists** (GPT-4o 60.4→0, Qwen
>   slack 76.2→0) — shows the threat the guarantee neutralizes is live. It degenerates
>   to 0→0 on robust models *and that is fine*, because it was never the load-bearing
>   axis. (Smart models make a by-construction defense look *better* — CaMeL v2's
>   utility rose on smarter models at the same guarantee.)
>
> This is the same honesty as §5.5 — a guarantee with a named, bounded conditionality —
> applied to the empirics. Canon: every number in §6/§2/§1 traces to
> `examples/agentdojo/agentdojo_results.md` (see the canon lock in `paper-decisions.md`).

**6.1 Setup.** Four AgentDojo suites (banking, slack, workspace, travel); attack =
`important_instructions` (the strongest stock injection); defense = `GovernedToolsExecutor`
wrapping every tool call in a `ToolCallGovernor` configured by the suite's YAML taxonomy
— *the same config a real deployment loads*, not one tuned for the benchmark.
**Primary model: o4-mini** — a CaMeL-v2 defended backbone, chosen precisely so the cost
comparison is **model-matched** (§6.3); banking/slack/workspace are 7-pass paired
studies, travel is a 2-pass paired study. Supporting models, each for a specific
purpose, never the cost axis: **GPT-4o** for the illustrative self-injection example
(§2, 60.4% undefended ASR) and the ASR-delta colour (§6.4) — CaMeL v2 has no GPT-4o
backbone, so it is axor-only; **Qwen-2.5-72b** susceptible-model supplement;
**claude-haiku-4-5** robust-model contrast (`examples/agentdojo/agentdojo_results.md`).

CaMeL's figures are the **version-pinned v2 reference** (arXiv:2503.18813v2), never
re-run on our harness, and always the **o4-mini-high row of Table 2 (Difference,
no-attack)** — model-matched to our o4-mini run, never a cross-model average:
banking **+18.8 ± 1.8**, slack **−23.8 ± 7.6**, workspace **−7.5 ± 2.9**, travel
**+10.0 ± 1.9** (overall −3.1 ± 0.4). (Version facts, per-model spread, and the list
of CaMeL numbers we must *not* cite are pinned in `paper-decisions.md`.)

**6.2 The guarantee and its measured cost — the PRIMARY result (lead §6 with this).**
State the guarantee first, in §5's vocabulary: axor's gate denies by *projection*, so on
every declared egress sink the exfiltration path is refused **structurally** —
framing-invariant, ASR→0 by construction (enum/numeric on contracted fields; rich-syntax
fields an open, fuzz-tested obligation, §5.5), independent of whether the model would have fallen. The empirical
question is then CaMeL's question, not a scoreboard: **what does the guarantee cost in
benign utility?** Measured per suite on the *benign* task list:

Measured on **o4-mini** (a capable CaMeL-v2 backbone), full CAMEL_MODE. ASR is 0%
throughout on this model — undefended, because o4-mini resists the stock injection on
its own (the headroom-dark case the cost axis exists for); governed, structurally.
**All rows are paired-run means** (undefended vs governed within each pass; undefended
is noisy, so the paired mean is the defensible number — see caveat):

| suite | benign undef → gov | cost | denials | mechanism |
|---|---|---|---|---|
| **banking** (paired, n=7) | 67.9% → 50.9% | **≈ −17 ± 7pp** | 3–5/pass | shared channel — payee read from an untrusted source |
| **slack** (paired, n=7) | 84.8% → 50.6% | **−34.1 ± 7.2pp** | 13–19/pass | shared channel — post derived from channel reads |
| **workspace** (paired, n=7) | 84.3% → 68.8% | **−15.5 ± 3.8pp** | ~15/pass | shared channel (`send_email`/`share_file` taint) **+** a `delete`=catastrophic consequence gate |
| **travel** (paired, n=2) | 62.5% → 70.0% | **0 (structural)** | **0/pass** | egress recipient comes from the **prompt**, not a read |

*Workspace's cost is **not purely shared-channel**: of ~104 benign denials over 7 passes,
**92 are `send_email`/`share_file` taint (shared channel)** and **12 are `delete_file`
consequence-gate** denials — the taxonomy raised `delete` to *catastrophic*, so benign
delete-after-acting tasks are gated too. That ~12% is an operator-taxonomy choice
(consequence axis), not the injection defense per se; state the split rather than
attributing all −15.5pp to shared channel.*

*Travel's cost is 0 by construction — **0 denials every pass** (the gate never fires,
because the legitimate egress recipient is prompt-given). The governed rate reads 70.0
vs undefended 62.5, but that **+7.5pp is sampling noise, not a gain**: with zero
denials, governed and undefended are the same trajectories up to the model's run-to-run
variance on n=2, and a gate that denies nothing cannot add utility. Report travel as
structural 0.*

*Caveat on the banking row — carry the paired number, not the dramatic single pass.*
o4-mini's **undefended** banking rate is noisy (single passes 56–87.5%; governed is the
stable signal, 50.9%, σ=5.2), so we carry the **per-pass paired cost −17.0 ± 7.2pp**
(within-pass, n=7), not the −37.5pp from one high-undefended (87.5%) pass — that is the
worst pass, not the headline. (The choice doesn't change the §6.3 conclusion either way.)

The ≈ −17pp is the **generic, pre-supersession** cost (the "denials" column is a per-pass
count, not a structural total). §6.3 splits this cost into a recoverable known-payee
partition and a one-off shared-channel residual, and reports what a deployment allowlist
recovers; Appendix D has the task-level breakdown.

**The load-bearing, honest finding: the cost is *localized to the shared-channel
partition* (§3 taxonomy), not universal.** axor pays a real cost where the legitimate
egress argument is read from an untrusted source — banking ≈ −17 ± 7pp, slack
−34.1 ± 7.2pp, workspace −15.5 ± 3.8pp (of which ~12% is the consequence gate, not
shared channel), all paired — and **exactly 0 (zero denials) where the legitimate
recipient is prompt-given (travel)**. The cost falls on the intersection: tasks the
model can complete *and* whose egress is derived from an untrusted read.

The *partition* is taxonomy-fixed, but the *cost on it scales with model capability*:
on Qwen, which fails the shared-channel slack tasks anyway, the same block cost ≈0;
on o4-mini, which completes them undefended (84.8%), it costs −34pp. Only **travel's 0
is structural** (prompt-given egress) and robust across models. (The retired
"slack = zero cost" claim and its provenance are logged in `paper-decisions.md`.)

The banking cost is exactly where a content ledger is weaker than CaMeL's structural
provenance, and the gap *widens* on a capable model (§6.3 / Appendix D). The
**confidentiality (sound floor)** axis is deliberately not a row here — it is a structural
property (Appendix A-floor, §5.6), and the stock travel suite does not exercise it (0 floor
denials in the o4-mini run; the floor never armed because benign travel tasks don't
read-secret-then-egress and o4-mini resisted the overt injection).

**6.3 CaMeL comparison — the *same kind* of claim; on utility, CaMeL is ahead.** axor
and CaMeL report the **same type of result** — a *structural* guarantee plus its *measured
utility cost* — so they are comparable in *kind* (not in number: different harness /
subset / version). This is the one place the utility comparison is stated in full; state
it plainly once here and let the rest of the paper reference it.

> **Paper-ready methods statement (use verbatim).** "We compare each defense's per-suite
> utility cost against its own undefended baseline; absolute utilities are not comparable
> across harnesses, so we report cost profiles rather than a head-to-head. CaMeL's figures
> are taken from its v2 tables and read from the o4-mini-high row, the defended backbone
> shared with our runs, so the cost comparison is model-matched. They are not re-run on
> our harness: a faithful re-implementation of CaMeL's interpreter is a project in
> itself, and an imperfect one would mislead more than a version-pinned reported
> reference."

The cost profiles, own-baseline deltas, all model-matched on o4-mini(-high):
- **banking**: axor **≈ −17 ± 7pp** (paired, n=7) vs CaMeL **+18.8pp** → CaMeL clearly
  ahead (CaMeL *gains* utility on banking; the gap is ~36pp on own-baseline deltas).
  A deployment `enum` allowlist recovers *part* of this over-block (below); it does not
  close the gap.
- **slack** (multi-step): axor **−34.1 ± 7.2pp** (paired, n=7) vs CaMeL **−23.8pp** →
  CaMeL ahead by ~10pp (both pay a real shared-channel cost; CaMeL pays less).
- **workspace** (email/calendar/files): axor **−15.5 ± 3.8pp** (paired, n=7) vs CaMeL
  **−7.5pp** → CaMeL ahead by ~8pp. Note axor's cost here is ~88% shared-channel
  taint + ~12% a `delete`=catastrophic consequence gate (a taxonomy choice, §6.2), so
  the shared-channel-only gap to CaMeL is a little narrower still.
- **travel** (prompt-driven egress): axor **0pp** (0 denials) vs CaMeL **+10.0pp** →
  neither defense loses utility (axor costs nothing, CaMeL *gains* +10pp).

**Hard caveat — compare deltas vs each system's own baseline, never absolutes** (axor
slack-undefended 84.8 ≠ CaMeL native-slack 95.2); verify every CaMeL delta is
benign/no-attack (Table 2, not Table 3). And o4-mini's banking undefended is unstable
(87.5 / 68.8 / 62.5 across runs — pure sampling variance); use the *paired* run and read
*per-task* losses, not absolute rates.

**A sound `enum`-supersession rule recovers *part* of the banking gap — deterministically,
but it does not close it (contribution #4, *supporting*; full mechanics in Appendix D +
`agentdojo_results.md`).** The generic banking over-block is a content-ledger *false
positive*: a legitimate recipient that is read-derived but is an operator-*known* payee gets
tainted, where CaMeL's structural provenance would not. When every driving arg of a sink is
covered by a satisfied `enum` over a closed, operator-declared trusted set, that enum carries
the integrity axis and the content-taint is skipped on that arg — soundly: attacker
destinations stay denied (not in the set), the confidentiality floor is not superseded, and
the codomain restriction (T4, §5.5: `enum` qualifies, `numeric_range` does not — the kernel
refuses the latter) plus a static, attacker-inaccessible allowlist keep the trusted set
uncontaminable. The lift covers the 4-task known-payee read-derived partition — the strict
value-coincidence subset ({3, 4, 6}: the prompt-given value coincides with a read-derived
one) plus the read-only known-payee case (task 15). On banking this is a **+25pp
deterministic gate upper bound**, of which **+13.4 ± 9.1pp is realized** on o4-mini (7 paired
passes; a wide interval, positive in every pass; below the bound because the model converts
only part of the lifted set). That recovers part of the −17pp cost, but **CaMeL stays ahead**
(+18.8pp): the one-off shared-channel residual (a payee that exists *only* in an untrusted
read) needs structural provenance and is unrecoverable by config. So the supersession
sharpens *where* the content ledger is weaker than provenance — it does not make axor win on
utility. Axor's case is adoption cost, not utility (§6.5).

**6.4 ASR-delta — secondary colour, where headroom exists (NOT the load-bearing axis).**
On a *foolable* model the undefended attack lands and the structural guarantee neutralizes
it — useful to *show the threat is live*, not to carry the result:

| model · suite | undefended ASR | governed ASR | note |
|---|---|---|---|
| GPT-4o · banking | **60.4%** | **0.0%** | a strong model carries out the injection unaided |
| Qwen · slack (full) | **76.2%** | **0.0%** | mass exfiltration neutralized |
| Qwen · banking-PII | 66.7% | **0.0%** | **ASR-only — degenerate utility (0 in both), never a zero-cost win** |
| claude-haiku · banking | 0% | 0% | **degenerates — robust model, no headroom; expected, not a failure** |

Say the quiet part: on a robust model this axis collapses to 0 → 0, and **that is not a
failure of the defense** — the guarantee and its cost (§6.2) are untouched; only the
delta-metric goes dark. The guarantee would fire just the same had the model failed.
(Footnote, principled: a 9-pair slice — *pre-registered* as the pairs where the
undefended injection derailed the task into failure — shows utility 33.3% → 88.9%;
report only with that definition + the n=9 / wide-CI caveat, as a mechanism
illustration, never a bare "governance raises utility.")

**6.5 The adoption motivation — and the head-on "why not just use CaMeL?" answer.**
The utility verdict is §6.3's and is not re-argued here: CaMeL is ahead, and the
supersession lift narrows but does not close the gap. Axor's case is **lower adoption
cost at comparable-or-somewhat-lower utility**:

- **axor is a gate in front of an *unmodified* agent loop** — you keep your existing
  agent (any provider, any framework) and drop a `ToolCallGovernor` in front of its tool
  calls. **CaMeL requires re-architecting the agent into an interpreter-mediated plan
  emitter** (the model emits a restricted-Python program a custom interpreter executes);
  their own repo flags it as a research artifact that "likely contains bugs ... might
  crash." Most production agents will not be rewritten that way.
- **axor works on any model**, including weak ones; CaMeL needs a model capable of
  writing valid plans (its utility drops on weaker models).
- **axor is framework-agnostic** (the §4 seam); CaMeL's interpreter approach is a
  heavier, more invasive integration.
- **Make the adoption claim measurable, not asserted (planned artifact — Appendix E):**
  a small integration-cost table — per framework (Claude-native, LangChain,
  OpenAI-shaped/AgentDojo), the size of the one normalizer implemented
  (`tests/normalizers/`, the AgentDojo `GovernedToolsExecutor` shim), agent-loop changes
  required (0 lines), policy changes required (0 — same YAML). This is the §4 seam
  quantified, and it is what "drop-in" means in numbers.

**The strongest answer to "why not just CaMeL?" is not "instead" — it is "on top of."**
axor's enforcement is defined against a *swappable* per-value trust-model contract,
**`ValueProvenance`** (`axor_core/contracts/provenance.py`), whose own docstring admits "a
content-derivation taint tracker, **a data-flow interpreter**, or a label-propagating tracker"
as conforming backends, and whose substitutability is a *tested invariant* (§5.9,
`tests/contracts/test_value_provenance.py` re-proves non-interference + complete mediation
under a second trust model). **A CaMeL-style structural provenance is exactly that "data-flow
interpreter" backend:** implement `derive_value` from CaMeL's capability/provenance graph in
place of the content-derivation ledger, and axor's gate sequence + complete mediation + the K4
guarantee run **unchanged on top** — now closing the very one-off shared-channel residual (§6.3)
the content ledger cannot, because the backend *knows* the read-derived payee is prompt-bound.
That reframes CaMeL from a rival into a **pluggable provenance backend under axor's
framework-agnostic gate**: you would get CaMeL's shared-channel coverage *and* axor's drop-in,
any-framework, any-model gate with its structural guarantee — the gate generalizes the
provenance line, it does not compete with it. **Honest scope (do not overclaim):** the contract
admits such a backend and a *custom* trust model passes it in tests, but **we have not wired
real CaMeL provenance in** — this is a *proposed integration / future direction* (§8), not a
measured result.

So the honest pitch: *if you can adopt CaMeL (rewrite your agent, accept the interpreter)
and want maximum utility-at-security today, use CaMeL. But axor is not the either/or rival
it first looks like: it is the **framework-agnostic gate** for the common case where you
will not rewrite your agent, and — by the `ValueProvenance` seam — a **host** that could
run CaMeL-style provenance underneath. The frontier is not "axor vs CaMeL on utility" but
a different axis: drop-in, any-framework, any-model structural guarantee at a measured
cost (§6.2), with a path to fold provenance in rather than choosing between them.*
**Do not claim parity on utility; claim a different — and composable — point on the
cost/assurance/integration frontier.**

The motivation, restated on the right axis: **the value is the guarantee + low adoption
cost; the metric for the guarantee is its utility cost** (§6.2), not a scoreboard that
vanishes on a good model. Two supporting points (support, not headline — §1 novelty):

1. **Model choice is not a *general* defense — so the guarantee is what you're buying.**
   A *strong* model still fools itself (undefended GPT-4o, 60.4%, `agentdojo_results.md`),
   and a model that *passes this bench* can still fall to a stronger framing — the
   established **indirect / nested prompt-injection** class (Greshake et al., AISec 2023,
   "Not what you've signed up for," and the nested/recursive-injection follow-ups —
   **cite the published prior art, pin the exact follow-up**). Model selection only ever
   covers attacks a model was already hardened against; a by-construction gate covers the
   whole projection-equivalence class regardless — with the §5.5/§5.6 conditionality
   (never state this as "all attacks"). That is *why* you want the guarantee.
2. **NNSI is illustrative only, never load-bearing — it is *our own* attack.** A result
   built on NNSI is **self-refereed** ("we built X and beat X"); a reviewer discounts it
   to zero, and the 62%-undefended figure is self-reported on a non-standard attack that
   would itself need independent validation. Use NNSI *only* as an illustrative stressor
   for "standard benchmarks underestimate the attack surface," and *only* anchored to the
   published class in point 1 — never as a bare number, never as the reason to adopt. The
   reason to adopt is §6.2 (cost of a real guarantee) + §5 (the guarantee), both of which
   stand without NNSI.

**6.6 Honest accounting (a subsection, not buried).** Utility cost falls specifically on
sinks whose *legitimate* argument is read from an untrusted source (data+instructions
sharing a channel). `driving_args` narrows it to the destination field; a strict-mode
approved-recipient allowlist closes even the shared-channel residual at the price of
enumerating destinations. Report what benchmarks do **not** prove (README "Benchmarks").

**6.7 Containment benchmarks (secondary, security-relevant).** Topology containment —
100% of policy-blocked child spawns denied; export leak rate — 0 leaks observed across
runs (README "Benchmarks"). These are structural-containment numbers, not the
utility/ASR axis; frame as corroborating the gate sequence beyond the injection setting.
(Token-economy figures are deliberately omitted — they are not a security result and do
not belong in this paper.)

---

## 7. Related work

> **ROPE** (arXiv 2608.27496) is compared obligation by obligation, by code at
> `64495e1` and by its stated propositions, in `docs/rope-comparison.md`. Use that
> table, not a feature list: the defensible differences are T4 (consumption
> registered and prepared before the check), fail-closed mediation, and the
> confidentiality scope; the amount bound is policy content and needs an
> equal-policy ROPE arm; record closure (ROPE Lemma 1) is a point in ROPE's favour
> that Axor's `TrustedOrigin.TOOL` leaves as a premise.

- **AgentSpec (Wang, Poskitt, Sun, arXiv:2503.18666)** — *mandatory cite, and the
  closest single-framework prior.* A customizable runtime-enforcement DSL whose
  rules are expressed and enforced against one framework's representation at a time
  (LangChain; re-implemented per domain for embodied / autonomous-driving). The contrast
  is the §4 thesis made concrete: AgentSpec = per-framework rule enforcement; axor = one
  decision core behind a provider-neutral `IntentNormalizer` seam, plus a non-interference
  *theorem* over the projection rather than a rule list. Position axor as *generalizing*
  the runtime-enforcement line across frameworks and giving it a formal guarantee — a
  build-on, not a take-down.
- **Classical noninterference & language-based IFC (the formal lineage — required if
  the theorem is the headline).** K4 is noninterference in the Goguen–Meseguer sense
  applied to the tool-call perimeter of an LLM agent; situate it against language-based
  information-flow control (Sabelfeld & Myers, JSAC 2003) rather than presenting it as
  ex nihilo. The `enum`-supersession rule is, in IFC terms, a **declassification /
  endorsement policy** — a reviewer will map it there instantly, so make the mapping
  ourselves and cite robust declassification (Zdancewic & Myers) and the decentralized
  label model as ancestors. **The stated delta:** (a) the *perimeter* formulation —
  noninterference over a projection of tool-call arguments rather than over program
  variables, with complete mediation as an explicit obligation; (b) the **decidability
  split** on T4 (enum/numeric discharged by a decision procedure vs rich-syntax
  localized to a fuzzing obligation) as the practical shape of the guarantee; (c) the
  swappable trust-model contract under a fixed gate (§5.9). Pin exact citations during
  drafting.
- **CaMeL (Debenedetti et al., arXiv:2503.18813)** — security by construction via an
  interpreter between model and tools; stronger structural provenance and, on the
  model-matched o4-mini comparison, ahead on utility — the full per-suite numbers and
  the partition split (liftable known-payee vs one-off shared-channel) live in §6.3 and
  are not restated here. Axor = a gate in front of an *unmodified* loop; its case is
  adoption cost (§6.5). **Crucially, CaMeL is not only a rival: its structural
  provenance is a conforming `ValueProvenance` backend (§5.9) under axor's gate** — the
  two compose (axor hosts provenance, closing the shared-channel residual while keeping
  the framework-agnostic gate and the K4 guarantee), a *proposed* integration, not
  implemented (§6.5/§8).
- **FIDES** — shares the explicit-flow-only scope boundary (O2); axor inherits the same
  honest limitation on implicit flows.
- **Firewalls (LLM input-firewall)** — the public **T0 counterexample**: a model-produced
  projection is steerable, so it may feed detection only, never the trusted path. Axor's
  T0 obligation is exactly the rule that forbids this.
- **Probabilistic verification for AI agents (Solko-Breslin, Mudrakarta, Christodorescu,
  Jha, Dvijotham, arXiv:2606.20510)** — the sharpest contemporary contrast, and a
  *deliberate inversion* of axor's design axis. They extend Datalog policy verification
  to **admit probabilistic predicates** (imperfect PII classifiers, declassifiers with
  failure rates) into enforcement and, via distributionally robust optimization (no
  independence assumption between detectors), compute a **sound upper bound on the
  probability of policy violation**. Axor makes the *opposite* choice on the trusted
  path: T0 forbids any probabilistic/interpreting producer in the decision loop
  (`tests/invariants/test_pure_allow.py::test_no_probabilistic_component_in_the_loop`),
  so its guarantee is a structural non-interference invariant (K4), not a
  violation-probability bound. State the trade explicitly: *they buy coverage of fuzzy
  predicates at the cost of a probabilistic guarantee; axor buys a deterministic,
  framing-invariant guarantee at the cost of restricting the trusted path to
  non-interpreting projections (the §5.5 decidability split is exactly that restriction
  made precise).* Their work is also the natural **future direction** for axor's two
  probabilistic-adjacent surfaces — the observe-only detection→degradation path
  (`governance-model §8`) and the advisory adjudicator — which could carry such sound
  bounds *without* violating T0, precisely because both are off the trusted path
  (detection may only tighten; the adjudicator only adds a deny). Cite it as the answer
  to the reviewer question "why not just admit a good classifier into the gate?" — they
  show how to do it soundly, and the cost is the probabilistic guarantee axor's
  structural axis avoids.
- Detection-based PI defenses (prompt classifiers) — contrast with the
  enforcement/detection separation: detection is observe-only, may only *tighten*, never
  allow (`governance-model §8`).
- Orchestration frameworks (LangGraph/CrewAI/AutoGen) — orthogonal; axor governs any of
  them.

---

## 8. Limitations & honest scope

Pull together, as first-class content (the kernel-theorem already states these as
*boundaries of the claim*, not gaps):
- Integrity paraphrase residual on the non-liftable partition (§5.6).
- Implicit / control-flow leaks out of scope (O2).
- A-3 (complete mediation) is a premise satisfied by the daemon, not a theorem.
- Availability is outside the safety perimeter (detection→degradation can cost uptime
  under a miscalibrated threshold).
- "Any trust model" demonstrated on 2 instances, not mechanized.
- AgentDojo coverage: one serious stock injection (`important_instructions`) across
  four suites, not the full injection matrix; the cost axis is 7-pass paired on
  banking/slack/workspace but only **2-pass paired on travel** (its 0 is structural —
  zero denials — so more passes change nothing about the mechanism, but state the n);
  ASR benches go dark on robust models (§6.4).
- **Confidentiality (sound) axis — shown as a structural property (Appendix A-floor), not
  a benchmark.** The floor is paraphrase-proof *by construction* (the gate reads a session
  boolean, never the egress bytes; §5.6) and is illustrated by a deterministic unit-level
  demonstration (arm → refuse → paraphrase-still-refused → control-allows), reproduced on
  GPT-4o while running travel. Honest scope, stated as the *named axis asymmetry*: integrity
  is a *measured* axis (§6 cost), confidentiality is a *structural* axis (a by-construction property, shown at unit level) — the
  **stock AgentDojo travel slice does not isolate the floor** (ASR ≈0/noisy; the live benign
  denial was integrity), so there is no benchmark floor-cost and we report none.

**Future direction (one paragraph).** Two concrete extensions, both off-the-shelf given the
existing seams. **(1) A structural-provenance trust-model backend.** The strongest residual in
§6 is the one-off shared-channel partition (a legitimate payee that exists only in an untrusted
read) where axor's content-derivation ledger over-blocks and CaMeL's structural provenance does
not. Because enforcement runs against the swappable `ValueProvenance` contract (§5.9), a
CaMeL-style data-flow/provenance backend can implement `derive_value` and slot in **under** the
unchanged gate — closing that residual while keeping the framework-agnostic gate and the K4
guarantee. This is the highest-value next step (it directly turns §6.3's deficit into a
compose), and it is *unimplemented* — the contract admits it and a custom trust model passes the
substitution test, but the CaMeL backend is not built. **(2) A sound probabilistic predicate off
the trusted path.** The integrity paraphrase residual and the fuzzing fraction (§5.5) are where a
*probabilistic* predicate would help; probabilistic verification (arXiv:2606.20510, §7) admits
one soundly — a bounded violation probability rather than a structural guarantee. Fold it in
*without* weakening K4 by keeping it **off the trusted path**: carry the bound on the observe-only
detection→degradation surface or the advisory adjudicator (both tighten-only), so a fuzzy
classifier can ratchet restrictions under a sound bound while the deterministic gates remain the
only thing that can allow.

---

## 9. Conclusion

Execution governance as a *framework-agnostic layer* with a *secure-by-design*
core: framing-invariant defense from a non-interference theorem whose conditionality is
stated, localized (the fuzzing fraction), and pinned to regressions — validated on
AgentDojo as a **structural guarantee at a measured utility cost**, reported honestly: the
cost is localized to the shared-channel partition (banking ≈−17 ± 7pp, slack ≈−34 ± 7pp,
workspace ≈−15 ± 4pp, all paired, on a capable model; 0 where egress is prompt-driven),
where CaMeL's heavier interpreter is ahead on utility (§6.3). A sound `enum`-supersession
rule deterministically lifts the gate over-block on the known-payee read-derived partition
(+25pp gate-level upper bound; +13.4 ± 9.1pp realized, §6.3/Appendix D); the one-off
shared-channel partition stays with structural provenance. axor's contribution is the
*adoption cost* of a guarantee that drops in front of an unmodified, framework-agnostic
agent loop on any model — not utility. The guarantee is the value; its utility cost is
the honest price. Agents should not self-govern execution.

---

## Appendices

- **A. Obligation → enforcing-test crosswalk** (kernel-theorem §5) — the credibility
  table tying every premise (O1/O2/O3/T0/T4/any-trust-model) to a named CI regression.
  The **T4 / supersession-soundness** row is the enum-only codomain restriction: its
  enforcing regression is `test_numeric_range_does_not_supersede_open_codomain` (kernel
  refuses to supersede on an open codomain) paired with
  `test_value_coincidence_recovery_structural_not_cherry_picked` (the `enum` closed-codomain
  case lifts; attacker IBAN stays denied) and `test_confidentiality_floor_is_not_superseded`
  (supersession is integrity-only) — all in `tests/adversarial/test_decidable_supersession.py`.
- **A-floor. Structural demonstration of the confidentiality floor (unit-level, sits
  beside A — *not* an AgentDojo result).** This is where the sound axis is shown, framed as
  a kernel *property*, not a benchmark number (per §5.6; §6.2 explains why the stock travel
  slice cannot carry it). Two parts:
  1. **The by-construction argument (the load-bearing part — paraphrase-proofness is
     argued, not tested).** The floor's decision is a function of one session boolean —
     *"is a secret read outstanding?"* — and the **outgoing value's content is never an
     input to that decision**. Since the egress bytes are not read, there is **no content
     channel** for an attacker to exploit: *every* re-encoding of the secret (base64,
     paraphrase, homoglyph, sub-fragment) maps to the same decision, because none of them
     can change the boolean. Paraphrase-proofness is therefore a property of the gate's
     *signature* (it ignores the content), not an empirical observation that "the tests we
     tried all failed." Anchor: the floor logic in `policy/gates.py` (`floor_active` →
     deny) and `taint/engine.py` (`confidentiality_floor_active`, armed on a
     `sensitive`-rooted read).
  2. **The illustration (a unit-level demonstration, *not* evidence of the property).** On
     the travel taxonomy, deterministically: a `get_user_information` read arms the floor →
     `send_email` is refused with the floor reason; the **same email with the secret
     paraphrased is *also* refused**; a fresh session with **no** secret read **allows** it.
     This makes the argument concrete; it does not stand in for it. (Reproduced live on
     OpenRouter/GPT-4o while running travel; ship it as a small deterministic listing, the
     way unit tests are shown, beside the crosswalk in A.)
- **B. The full gate sequence** (governance-model §2) with the per-gate rationale.
- **C. Reproduction commands** for every AgentDojo number (`agentdojo_results.md`).
- **D. The `enum`-supersession recovery (contribution #4) — full mechanics.** Kept out of §6.3
  to keep the comparison tight. Contains: the soundness argument (T4 codomain restriction;
  static attacker-inaccessible allowlist; floor not superseded); the recovery-population
  definition and the deterministic 4-task table {3, 4, 6, 15} (strict value-coincidence subset
  {3, 4, 6} vs the broader read-only task 15, with the `US133…`-allowed-in-both / `GB29…`-is-the-
  recovery / attacker-denied breakdown); the **gate ceiling +25pp vs realized +13.4 ± 9.1pp**
  reconciliation as one population (realized {3,4,15} ⊂ gate {3,4,6,15}; task 6 lifts but the
  model fails it 14/14 even undefended) with the wide-CI caveat; and the enforcing regressions
  (`test_value_coincidence_recovery_structural_not_cherry_picked`,
  `test_task15_recovers_via_read_only_known_payee`,
  `test_numeric_range_does_not_supersede_open_codomain`). Source of record:
  `agentdojo_results.md` "Recovering the banking cost."
- **E (planned). Integration-cost table** backing §6.5's adoption claim: per framework,
  normalizer size, agent-loop changes (0), policy changes (0).

---

## Figures / tables to produce

1. The reverse-osmosis gate stack (README) — *the* signature figure.
2. Trust-ring diagram + three-interface adapter seam (§4).
3. The ground → projection → consumption → dispatch → effect chain as a single diagram (§5.2) — the conceptual core.
4. **The primary figure: measured utility cost across suites (o4-mini)** (§6.2 — banking
   ≈ −17 ± 7pp, slack −34.1 ± 7.2pp, workspace −15.5 ± 3.8pp, travel 0pp, all paired)
   with the CaMeL **o4-mini-high** cost overlaid (v2 Table 2: banking +18.8, slack −23.8,
   workspace −7.5, travel +10.0) — shows axor's cost is localized to the shared-channel
   partition and CaMeL is ahead on every suite. Caption the banking bar honestly:
   **CaMeL ahead on the one-off shared-channel half; axor lifts the known-payee
   read-derived half at the gate** (+25pp upper bound / +13.4 ± 9.1pp realized) — not a
   flat "CaMeL ahead on banking." Split the workspace bar visually (shared-channel vs
   consequence-gate denials, 92/12). Integrity axis only; the confidentiality floor is a
   *property*, shown separately (figure 6), not a cost row. Put the ASR-delta table
   (§6.4) *second/smaller*, captioned "where headroom exists," so layout itself signals
   which axis is load-bearing.
5. The decidability split as a classification, next to the negative-control matrix (contract removed → effect; restored → none) (§5.5, §5.7).
6. **The confidentiality-floor listing** (Appendix A-floor) — a small deterministic
   unit-style listing (arm → refuse → paraphrase-still-refused → control-allows), shown
   the way a unit test is shown, *not* as a benchmark bar. Caption names the asymmetry:
   integrity is *measured* (fig. 4), confidentiality is *structural* (fig. 6).
