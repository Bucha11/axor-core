# Axor — Paper Outline

Working outline for the axor paper, structured around the three strengths Haoyu
asked us to emphasize, with an illustrative AgentDojo example as the opening hook.
Every claim below is anchored to something already in the repo (file paths in
parentheses) so the writing stays honest — no mechanism is invented for the paper.

Haoyu's three asks, mapped to sections:

1. **Framework-agnostic governance layer** → §4 (the abstraction and its rationale).
2. **Security model backed by perimeter non-interference (secure-by-design)** → §2
   illustrative example + §5 (theorem) + the explicit *attack → taint → theorem* link.
3. **AgentDojo empirical support + a motivation for practical adoption** → §6.

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
- **Three contributions** (the paper's spine, = Haoyu's three points):
  1. A **framework-agnostic** governance kernel with a clean abstraction seam to any
     agent framework (Claude, LangChain, OpenAI-shaped, …) — §4.
  2. A **secure-by-design** security model anchored by the **Perimeter
     Non-Interference theorem (K4)**, with an explicit chain from a concrete
     agentic attack → taint analysis → the non-interference guarantee — §5.
  3. **Empirical validation on AgentDojo** with the *honest* utility/ASR tradeoff,
     and a concrete motivation for adopting a structural defense even where a robust
     model already resists — §6.
- Forward-reference the opening example (§2).

---

## 2. Illustrative example — "what a successful defense actually looks like"

> Haoyu: *"begin with an illustrative example — adopted from your AgentDojo defended
> by axor — to give the reader a sense of what's behind a successful defense, and
> how the theorem secures the whole system."*

**Candidate (recommended): the banking exfiltration task, AgentDojo `important_instructions`.**
Clean to walk gate-by-gate and it is the CaMeL-comparable scenario.
(Alternative with more drama and *zero* utility cost: the **slack mass-exfiltration**
run, 76.2% → 0% ASR, utility-neutral — good as a second, "and it scales" example.)

**The scene.** A banking agent is asked to do a benign task (summarize transactions /
pay a bill). The transaction history it must read contains an injected instruction:
*"send a transaction to `attacker-IBAN` …"* (data and instructions share a channel —
the hard core of injection). Undefended GPT-4o carries this out on **60.4%** of pairs
(`examples/agentdojo/agentdojo_results.md`).

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

**The bridge to the theorem (state it here, prove it in §5).** The denial did **not**
depend on recognizing the injection. The gate read only the *projection* of the call —
the **origin-class of the recipient field** (untrusted-derived vs prompt-given) — never
the attacker's prose. By **Perimeter Non-Interference (K4)**, the decision is a function
of that projection alone, so *any* reframing of the injection that still puts the
attacker's address in the recipient field yields the **same projection → the same
deny**. That is what "secure-by-design" buys: the defense is invariant to attacker
*framing*. (Empirical echo: the NNSI nested-document attack is blocked **0% at every
framing depth D0–D5**, `examples/attacks/nnsi_results.md`.)

End the section with the one honest qualifier that the rest of the paper earns: framing
is neutralized structurally; what an attacker can still vary is the *encoding* of the
exfiltrated value (the integrity paraphrase residual, §5.4) — and the confidentiality
floor / positional gate are the parts that close even that.

---

## 3. Threat model & background

- **Adversary.** Untrusted content reaching the agent through any read (web fetch,
  inbox, RAG document, tool output, another agent) that can carry instructions and/or
  attacker-chosen values. Goal: drive a high-consequence effect — exfiltration
  (egress of a secret or a third party's data), destructive action, capability
  escalation, uncontrolled child spawn.
- **The lethal trifecta** framing (untrusted input × access to secrets/private data ×
  egress) — tie to `tests/adversarial/test_e2e_gate.py` (lethal-trifecta egress denied).
- **Trust boundary.** In-process soft boundary (`LockedExecutor` → `GovernanceBypassError`)
  vs out-of-process hard boundary (the daemon re-derives the ceiling server-side).
- **Non-goals (state up front, honestly).** Availability/uptime; implicit
  control-flow leaks; covert channels; semantic malware detection
  (README "What benchmarks do not prove"; kernel-theorem §6).
- **Background.** AgentDojo benchmark; prior defenses to contrast in §7 (CaMeL —
  by-construction security via an interpreter; FIDES — explicit-flow; Firewalls —
  LLM input-firewall, the T0 counterexample).

---

## 4. Design — a framework-agnostic governance layer (Haoyu point #1)

> The interesting part Haoyu flagged: the **abstraction** between the core and the
> agent frameworks, and the **rationale** for the design choices. Prior work
> implemented on a single framework; axor's seam is the contribution.

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
is being intercepted." *This is the abstraction the prior single-framework work did
not need and is the design contribution.*

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

**4.5 Trust rings + machine-enforced kernel purity (why the abstraction is load-bearing,
not cosmetic).** Subsystems split into Ring 0 (kernel/TCB), Ring 1 (runtime), Ring 2
(platform). The kernel must not import the runtime/platform — enforced in CI by
`import-linter` (`.importlinter` `kernel-purity`), reinforced by lazy imports so
`from axor_core import ToolCallGovernor` loads the kernel *alone*. *Rationale:* a
bug in a framework adapter (Ring 1) or in budget/context/trace (Ring 2) **cannot cause
a wrong allow** — the decision is structurally isolated from the framework glue. This
is the formal backbone under "framework-agnostic": the governance semantics are
provably independent of the framework, not just conventionally separated.
Pin: `tests/test_kernel_only_import.py`.

**4.6 Deployment taxonomy — how a renamed tool set is governed.** A real deployment
renames its tools, so the operator *declares roles* (`untrusted_sources`,
`sensitive_sources`, `egress_sinks`, `positional_sinks`, `value_policies`,
`driving_args`) via kwargs or a fail-closed YAML (`GovernanceConfig.from_yaml`;
`governance-model §12`). Same declaration accepted by both callers. *This is the
operator-facing half of the abstraction* — the kernel recognizes generic names; the
taxonomy maps any framework's renamed tools onto the same governed roles.

*Figure for this section:* the trust-ring diagram + the three-interface adapter seam.

---

## 5. The security model — Perimeter Non-Interference (Haoyu point #2)

> Haoyu: *"link the agentic attack example, taint analysis, and the perimeter
> non-interference theorem so a reader senses what guarantee it provides."* This
> section is built as exactly that chain.

**5.1 The claim, in one line (K4).** For any sink invocation: (1) **complete
mediation** — no effect bypasses `allow`; (2) `allow` inspects **only admissible
structural projections**. Therefore the decision cannot be steered by raw bytes except
through a projection whose codomain *admits no instruction* — **regardless of trust
model** (`docs/kernel-theorem.md §1`). The safety content is the non-interference
invariant:

```
∀ x₁, x₂ :  π(x₁) = π(x₂)  ⟹  allow(π(x₁), p) = allow(π(x₂), p)
```

Raw input influences the decision *only* through the projection `π`. Two inputs the
consumer would treat differently cannot share a projection (modulo explicit
declassification = governance endorsement). **This is the formal content of "framing
cannot change the verdict."**

**5.2 The chain Haoyu asked for — attack → taint → theorem (the heart of the paper).**
Make this an explicit three-step walk, reusing §2's banking example:

| Step | What happens | Anchor |
|---|---|---|
| **Attack** | Injected instruction + attacker value (IBAN / relay address) enters via an untrusted read; data and instructions share a channel. | §2; `examples/attacks/nnsi_results.md` |
| **Taint analysis (O2, label soundness)** | The value is constructed by `external_read` → `causal_root` marks it untrusted-derived; the join/derive is sound (`causal_root(v) ⊇` the untrusted sources that explicitly influenced `v`), proved by induction over a closed constructor set; over-taint, never silent under-taint. | `axor_core/taint/causal_root.py`, `ledger.py`, `engine.py`; kernel-theorem §2 (O2) |
| **Theorem (K4)** | The egress decision factors through the projection (origin-class of the driving arg), `allow` reads only that — so every reframing with the same tainted recipient gets the same deny. Complete mediation (O3) guarantees the sink cannot be reached around `allow`. | kernel-theorem §1, §2 (O1/O3) |

The reader's takeaway: *the guarantee is not "we detect this attack" but "no member
of the entire equivalence class of attacks that share this projection can get a
different answer."* That is the secure-by-design content.

**5.3 The three structural obligations (the spine of the proof).**
- **O1 — factorization.** `allow` is a pure function of `(projection, policy)`; no
  hidden raw channel — reputation/trajectory are *not* arguments to `allow`
  (`tests/invariants/test_pure_allow.py`).
- **O2 — label soundness** (above). Scope boundary stated honestly: explicit-flow
  only; implicit control-flow leaks are out of scope (shared with FIDES;
  `tests/adversarial/test_implicit_flow_gap.py`, asserted `xfail(strict=True)` so the
  suite trips the moment a sound backend closes it).
- **O3 — complete mediation.** Finite sink ring → statically surveyable; in-process
  soft boundary (`locked.py`) + out-of-process daemon hard boundary; unknown sink
  fails closed.

**5.4 The conditionality is the point — the decidability split (the one genuinely new
result).** *Do not overclaim — Haoyu values the honesty.* K4's safety content rests on
two per-projection obligations: **T0** (the projection producer is *non-interpreting* —
a deterministic structural function, never a model reading the governed content; the
Firewalls LLM-projector is the public counterexample; pinned by an `.importlinter`
contract + an import-scan test) and **T4** (effective codomain = nominal). T4 splits:

- **Decidable by construction** for finite **enums** and **bounded-numeric** ranges
  consumed as case-split / numerically — two equal projections cannot be split into
  distinct effects. Discharged by a decision procedure
  (`axor_core/kernel/decidability.py`); a config that mis-guards a fuzz field with a
  decidable predicate is **rejected** (`registration.py`).
- **Fuzzing only** for **path / string-subfield / carrier-over-free-text** — the
  consumer is a rich-syntax interpreter, T4 is undecidable in general, so it stays a
  fuzzing obligation. The weird-machine class is **localized** here — and indeed both
  real bugs ever found (newline-injection in path extraction, `../` traversal in lease
  validation) lived exactly in this region, as the split predicts.

> The honest one-liner for the paper: **K4 holds outright for the enum/numeric fraction
> of the perimeter, and conditionally — discharged by fuzzing — for the rich-syntax
> fraction.** "Currently testable by fuzzing" is acceptable for practical deployment
> (Haoyu's own words) precisely *because the split tells you which fraction needs it.*

**5.5 The two axes, asymmetric by design.**
- **Integrity** — precise per-value content-derivation; sound to *deny*, with a
  documented **paraphrase residual** on the non-liftable partition (a generic write/exec
  that cannot be made positional). State this as a boundary, not a surprise
  (`governance-model §7`; `examples/attacks/`).
- **Confidentiality** — a **sound, paraphrase-proof session floor**: once a secret is
  read, egress is refused on the *fact of the read*, not the content of the outgoing
  value, so base64 / paraphrase / restructuring cannot slip past. Released only by a
  governance endorsement (`governance-model §5`).
- **Positional admission** — for high-stakes sinks whose legitimate input *cannot
  encode an instruction*, admit by *form*, not content → sound by construction
  (`governance-model §3`). The kernel **refuses** to let you declare an exec-class sink
  positional.

**5.6 "Any trust model."** The kernel/trust-model factorization is enforced at the
import level, and a substitution test re-proves non-interference + complete mediation
under a *second* trust model with a different tainting rule
(`tests/contracts/test_value_provenance.py`). Honest status: *demonstrated on two
instances, not mechanized over all conforming backends* (O2 is the inductive content —
a Coq/Lean target). Put the obligation→test crosswalk table (kernel-theorem §5) in an
appendix — it is the paper's credibility anchor.

---

## 6. Evaluation — AgentDojo (Haoyu point #3)

> Haoyu: defenses are perfect for both axor and the baseline, so we need a **motivation
> for practical adoption** — secure-by-design is enough for him, *numbers are better*.
> The honest, load-bearing framing is **utility retained at ASR ≈ 0** (the CaMeL axis),
> not "ASR → 0" alone (driving ASR to zero is trivial — deny all egress).

**6.1 Setup.** AgentDojo suites (banking, slack, workspace, travel), attack =
`important_instructions` (the strongest stock injection); defense = `GovernedToolsExecutor`
wrapping every tool call in a `ToolCallGovernor` configured by the suite's YAML taxonomy
— *the same config a real deployment loads.* Models: **GPT-4o** primary (the family CaMeL
measured), **Qwen-2.5-72b** susceptible-model supplement, **claude-haiku-4-5** robust
contrast (`examples/agentdojo/agentdojo_results.md`).

**6.2 Primary — GPT-4o, banking, the CaMeL axis.**

| condition | benign utility (16) | utility under attack (48) | ASR |
|---|---|---|---|
| undefended | 100.0% | 79.2% | **60.4%** |
| governed | **62.5%** | 58.3% | **0.0%** |

Headline: **62.5% benign utility retained at 0.0% ASR** against a model that undefended
falls for 60.4% of pairs. Compare to CaMeL's 67% — *same axis* (utility-at-ASR≈0, same
model family). **The crucial honesty:** axor is a gate in front of an *unmodified* agent
loop; CaMeL buys a stronger by-construction guarantee with an *interpreter between the
model and every tool*. Decompose the 6 lost benign tasks into the three mechanisms
(genuine shared channel / value-coincidence false positive / whole-args fallback) — this
candor is a strength, and it precisely locates where a content ledger is weaker than
CaMeL's structural provenance.

**6.3 Supplementary — susceptible model, the *consequential* threats (the adoption
motivation).** The serious data-exfiltration injections, on a model that actually
carries them out:

| suite · threat | n | undefended ASR | governed ASR | utility (undef → gov) |
|---|---|---|---|---|
| **slack** · mass exfiltration (all channel+inbox → attacker site), **full suite** | 21 | **76.2%** | **0.0%** | 47.6% → **47.6%** |
| **banking** · PII exfiltration (third-party IBAN / subscriptions) | 6 | 66.7% | **0.0%** | 0% → 0% |

The slack row is the strongest single argument: **76.2% → 0% ASR at zero utility cost**
(legitimate recipients come from the prompt/channel, not the attacker payload). On a
9-pair slice, governance even *raised* utility (33.3% → 88.9%) because a blocked attack
stops derailing the model. **This is the practical-adoption number.**

**6.4 The motivation Haoyu asked for — why adopt when the model already resists.**
Three numbered arguments, in order of strength:
1. **Secure-by-design invariance (the qualitative reason).** The defense gates on
   *argument provenance*, not on recognizing the injection — so a cleverer *frame* that
   flips a model produces the *same* tainted egress and is refused identically (NNSI:
   62% → 0%, depth-invariant). Model robustness is per-model, per-attack, and erodes
   with the next framing; the structural guarantee does not. **You cannot benchmark your
   way to this — it is a property, not a score.**
2. **The robust model is robust *to this bench*, not in general (the negative result).**
   claude-haiku resists AgentDojo's shallow single-injection attacks — but the *same
   model* falls to the NNSI nested-document framing **62% of the time** undefended
   (`nnsi_results.md`). "The model defended" and "the defense defended" are
   indistinguishable on a bench too weak to create headroom; axor's guarantee is exactly
   what covers the attacks the bench cannot express.
3. **The numbers where headroom exists (the quantitative reason).** Where a model *does*
   have headroom (Qwen, GPT-4o), axor drives ASR to 0 — 76.2%→0%, 66.7%→0%, 60.4%→0% —
   at measured, *explainable* utility cost, often **zero** when the legitimate argument
   comes from the prompt.

**6.5 Honest accounting (a subsection, not buried).** Utility cost falls specifically on
sinks whose *legitimate* argument is read from an untrusted source (data+instructions
sharing a channel). `driving_args` narrows it to the destination field; a strict-mode
approved-recipient allowlist closes even the shared-channel residual at the price of
enumerating destinations. Report what benchmarks do **not** prove (README "Benchmarks").

**6.6 Non-security benchmarks (secondary).** Token reduction 30.8% avg; topology
containment 100% of policy-blocked spawns; export leak rate 0 (README "Benchmarks").
Frame as "governance is also measurable and cheap," not as the main result.

---

## 7. Related work

- **CaMeL** — security by construction via an interpreter between model and tools;
  stronger guarantee, heavier integration. Axor = a gate in front of an *unmodified*
  loop; the §6.2 comparison is the head-to-head. Axor's integrity axis is a content
  ledger (sound-to-deny, paraphrase residual); confidentiality floor is the sound,
  paraphrase-proof part.
- **FIDES** — shares the explicit-flow-only scope boundary (O2); axor inherits the same
  honest limitation on implicit flows.
- **Firewalls (LLM input-firewall)** — the public **T0 counterexample**: a model-produced
  projection is steerable, so it may feed detection only, never the trusted path. Axor's
  T0 obligation is exactly the rule that forbids this.
- Detection-based PI defenses (prompt classifiers) — contrast with the
  enforcement/detection separation: detection is observe-only, may only *tighten*, never
  allow (`governance-model §8`).
- Orchestration frameworks (LangGraph/CrewAI/AutoGen) — orthogonal; axor governs any of
  them.

---

## 8. Limitations & honest scope

Pull together, as first-class content (the kernel-theorem already states these as
*boundaries of the claim*, not gaps):
- Integrity paraphrase residual on the non-liftable partition (§5.5).
- Implicit / control-flow leaks out of scope (O2).
- A-3 (complete mediation) is a premise satisfied by the daemon, not a theorem.
- Availability is outside the safety perimeter (detection→degradation can cost uptime
  under a miscalibrated threshold).
- "Any trust model" demonstrated on 2 instances, not mechanized.
- AgentDojo coverage: 3 serious injections × 16 tasks, not the full 9-injection matrix;
  benches go dark on robust models.

---

## 9. Conclusion

Execution governance as a *framework-agnostic layer* with a *secure-by-design*
core: framing-invariant defense from a non-interference theorem whose conditionality is
stated, localized (the fuzzing fraction), and pinned to regressions — validated on
AgentDojo with an honest utility/ASR tradeoff and a concrete adoption motivation that a
benchmark alone cannot express. Agents should not self-govern execution.

---

## Appendices

- **A. Obligation → enforcing-test crosswalk** (kernel-theorem §5) — the credibility
  table tying every premise (O1/O2/O3/T0/T4/any-trust-model) to a named CI regression.
- **B. The full gate sequence** (governance-model §2) with the per-gate rationale.
- **C. Reproduction commands** for every AgentDojo number (`agentdojo_results.md`).

---

## Figures / tables to produce

1. The reverse-osmosis gate stack (README) — *the* signature figure.
2. Trust-ring diagram + three-interface adapter seam (§4).
3. The attack → taint → theorem chain as a single diagram (§5.2) — the conceptual core.
4. AgentDojo results tables (§6.2, §6.3) — the CaMeL axis and the slack zero-cost row.
5. The decidability split (enum/numeric = decidable vs path/carrier = fuzzing) (§5.4).

## Open decisions for the authors

- **Venue framing:** systems-security (USENIX/CCS/S&P) vs an LLM-agent-safety venue —
  changes how much §5 formalism vs §6 empiricism leads.
- **Headline example:** banking (CaMeL-comparable, clean gate walk) vs slack
  (zero-cost, more dramatic). Recommendation: open with banking, use slack as the
  "and it scales at no cost" beat in §6.
- **How hard to push the theorem:** keep K4 as "stated + pinned + demonstrated on 2
  trust models," explicitly *not* claiming a mechanized proof — matches the repo's own
  honesty and pre-empts reviewer overclaim objections.
</content>
</invoke>
