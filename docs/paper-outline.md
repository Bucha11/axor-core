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
     anchored by the result that a strong production model (GPT-4o) fools itself on
     **60.4%** of pairs undefended → **0.0%** governed at **62.5%** utility retained —
     the motivation no robust-model argument can wave away — §6.
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
the hard core of injection). **The model fools itself:** undefended **GPT-4o** — a
strong, current production model (do **not** call it "the model CaMeL measured" — that is
unverified, §6.1) — carries this out on **60.4%** of pairs
(`examples/agentdojo/agentdojo_results.md`). This is the example to
lead with precisely *because the model is not weak*: the reader cannot dismiss it as
"use a better model." A capable model with enough headroom to attempt the whole task
list still walks straight into the injection on its own — that is the problem axor
exists for, and §6.4 turns it into the central adoption argument.

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
proves it. (Empirical echo, phrased as a rate: the NNSI nested-document attack's success
rate is driven to **0% at every framing depth D0–D5** — from 62% undefended —
`examples/attacks/nnsi_results.md`.)

End with the honest qualifier, kept on the right axis. Framing is neutralized
structurally — but this banking case sits on the **integrity** axis, and there the
content-derivation gate has a documented **paraphrase/encoding residual** on the
non-liftable partition (§5.5): an attacker who *paraphrases* the recipient rather than
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
- **Trust boundary.** In-process soft boundary (`LockedExecutor` → `GovernanceBypassError`)
  vs out-of-process hard boundary (the daemon re-derives the ceiling server-side).
- **Non-goals (state up front, honestly).** Availability/uptime; implicit
  control-flow leaks; covert channels; semantic malware detection
  (README "What benchmarks do not prove"; kernel-theorem §6).
- **Background.** AgentDojo benchmark; prior defenses to contrast in §7 (AgentSpec —
  per-framework runtime-enforcement DSL, the single-framework prior; CaMeL —
  by-construction security via an interpreter; FIDES — explicit-flow; Firewalls —
  LLM input-firewall, the T0 counterexample).

---

## 4. Design — a framework-agnostic governance layer (Haoyu point #1)

> The interesting part Haoyu flagged: the **abstraction** between the core and the
> agent frameworks, and the **rationale** for the design choices. The single-framework
> prior to engage *by name* is **AgentSpec** (Wang, Poskitt, Sun, arXiv:2503.18666) —
> Haoyu's own work, whose enforcement binds to one framework's hooks (LangChain;
> separately re-implemented per domain for embodied / autonomous-driving). He raised
> this contrast himself in the email ("my previous works only implement on a single
> framework, either OpenClaw or LangChain"); engage it directly, do not paraphrase
> around it. Axor's contribution is lifting that binding into a provider-neutral seam.

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
decision core behind a provider-neutral normalization seam.* This is the design
contribution, and it is exactly the gap Haoyu named.

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

## 5. The security model — Perimeter Non-Interference (Haoyu point #2)

> Haoyu: *"link the agentic attack example, taint analysis, and the perimeter
> non-interference theorem so a reader senses what guarantee it provides."* This
> section is built as exactly that chain.

**5.1 The claim, in one line (K4).** For any sink invocation: (1) **complete
mediation** — no effect bypasses the gating decision; (2) the gating decision
**factors through an admissible structural projection** `π` whose codomain *admits no
instruction* — **regardless of trust model** (`docs/kernel-theorem.md §1`). Term: this
is *perimeter non-interference* in the Goguen–Meseguer sense (noninterference, 1982) —
write it that way every time; do **not** reproduce Haoyu's email garble "non-perimeter
inference."

State the invariant over the **raw** artifact, not over `π` — otherwise it is a
tautology a reviewer will catch. Let `decide(x, p)` be the *actual* gating function on
the path to an effect (whatever the implementation does with the raw artifact `x`).
The non-interference content is that `decide` is **constant on the fibers of `π`**:

```
∀ x₁, x₂ :  π(x₁) = π(x₂)  ⟹  decide(x₁, p) = decide(x₂, p)
```

This is **not** automatic: it is the claim that the effect-path decision *factors
through* `π` at all — there is no `decide'` on the effect path that reads raw `x`
outside `π` (this is exactly O1, §5.3). Writing `allow(π(x₁)) = allow(π(x₂))` instead
would be true by substitution and say nothing; the load-bearing statement is that the
raw-input decision cannot distinguish two artifacts that share a projection. Two
inputs the consumer would treat differently therefore cannot share a projection (modulo
explicit declassification = governance endorsement). **This is the formal content of
"framing cannot change the verdict."**

**5.2 The chain Haoyu asked for — attack → taint → theorem (the heart of the paper).**
Make this an explicit three-step walk, reusing §2's banking example:

| Step | What happens | Anchor |
|---|---|---|
| **Attack** | Injected instruction + attacker value (IBAN / relay address) enters via an untrusted read; data and instructions share a channel. | §2; `examples/attacks/nnsi_results.md` |
| **Taint analysis (O2, label soundness)** | The value is constructed by `external_read` → `causal_root` marks it untrusted-derived; the join/derive is sound (`causal_root(v) ⊇` the untrusted sources that explicitly influenced `v`), **argued by pen-and-paper induction** over a closed constructor set (a Coq/Lean target, *not* machine-checked — use this exact hedge everywhere, see §5.6/§8); over-taint, never silent under-taint. | `axor_core/taint/causal_root.py`, `ledger.py`, `engine.py`; kernel-theorem §2 (O2) |
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

**Correct Haoyu's mental model explicitly in the prose (he wrote "we can only test it
using fuzzing").** The framing to land is *not* "the theorem is fuzz-tested" — that
concedes the whole claim to a test. It is: **K4 is discharged by a decision procedure
on the enum/numeric fraction and the weird-machine residual is *localized* to a named
rich-syntax fraction, where fuzzing is the discharge.** Fuzzing covers a bounded
sub-perimeter the split identifies up front, not the theorem as a whole. If we let the
"fuzz-tested theorem" reading stand, Haoyu will (reasonably) down-rate the claim in his
own edits — the split is precisely the result that prevents that.

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

> **Canon lock (do this before writing a single number).** The authoritative figures
> are `examples/agentdojo/agentdojo_results.md`: banking GPT-4o **60.4% → 0.0% ASR at
> 62.5% benign utility retained**. An earlier note circulated 54%/56% — that is a stale
> run; purge it. Every number in §6 and §2/§1 must trace to this file.

> **What "baseline" means — resolve with Haoyu (blocking, see Questions).** Our tables
> compare **undefended vs governed** (60.4% → 0%). That is *not* a competing-defense
> comparison. When Haoyu says "the defence results are perfect for axor and its
> baseline," he most likely means a *rival defense* (CaMeL / a detector) that also holds
> ASR≈0 — i.e. ASR doesn't differentiate, so we must pivot to utility-at-ASR≈0. We have
> **not** run any rival defense on this harness. Two honest options: (a) actually run
> CaMeL or a detector defense on the same AgentDojo subset, or (b) explicitly label the
> CaMeL number as *reported, not re-run, not directly comparable*. Pick (a) if Haoyu
> expects a head-to-head; otherwise (b). Do not imply a head-to-head we didn't run.

**6.1 Setup.** AgentDojo suites (banking, slack, workspace, travel), attack =
`important_instructions` (the strongest stock injection); defense = `GovernedToolsExecutor`
wrapping every tool call in a `ToolCallGovernor` configured by the suite's YAML taxonomy
— *the same config a real deployment loads.* Models: **GPT-4o** primary, **Qwen-2.5-72b**
susceptible-model supplement, **claude-haiku-4-5** robust contrast
(`examples/agentdojo/agentdojo_results.md`). *Verify before asserting "the model CaMeL
measured":* CaMeL's detailed AgentDojo tables foreground **Claude 3.5 Sonnet** (63.9%
utility under CaMeL, **−26.8%** vs native) and **Claude 3.5 Haiku** (44.3%, **−29.9%**);
its 67% headline is an aggregate "provable-security task-solve" figure, **not** a GPT-4o
number. So write "the GPT-4o family that AgentDojo's own inverse-scaling analysis
covers," not "the very model CaMeL measured" — the latter is unverified (§6.3).

**6.2 Headline security result — slack mass-exfiltration at *zero* utility cost (lead
§6 with this).** Per Haoyu's "we need a motivation for adoption": the full slack suite
against the consequential mass-exfiltration injection (concatenate all channel+inbox →
POST to an attacker site), on the susceptible model that actually carries it out.

| suite · threat | n | undefended ASR | governed ASR | utility (undef → gov) |
|---|---|---|---|---|
| **slack** · mass exfiltration, **full suite** | 21 | **76.2%** | **0.0%** | 47.6% → **47.6%** |
| **banking** · PII exfiltration (third-party IBAN / subscriptions) | 6 | 66.7% | **0.0%** | 0% → 0% |

**76.2% → 0% ASR at *zero* utility cost** — the strongest single number in the paper:
legitimate recipients come from the prompt/channel, not the attacker payload, so blocking
the exfiltration costs no real-task completion. On a 9-pair slice governance even *raised*
utility (33.3% → 88.9%), because a blocked attack stops derailing the model. This is the
adoption headline; banking (§6.3) is the harder, *costed* case kept for the CaMeL
comparison. (Roles, stated once: §2 illustrative = banking, clean gate-walk; §6 headline
= slack, zero-cost. Don't conflate them.)

**6.3 Banking — the costed case and the CaMeL comparison (reported, not re-run).**
Banking is where the defense *costs* utility (the legitimate action is driven by
untrusted-read content), so it is the honest place to compare against CaMeL — with the
comparison's limits stated, not buried.

| condition | benign utility (16) | utility under attack (48) | ASR |
|---|---|---|---|
| undefended | 100.0% | 79.2% | **60.4%** |
| governed | **62.5%** | 58.3% | **0.0%** |

**62.5% benign utility retained at 0.0% ASR**, against a model that undefended falls for
60.4% of pairs. On the CaMeL comparison, be precise about what is and isn't comparable:
- CaMeL's **67%** is *reported, not re-run by us*, on a different harness / subset /
  model snapshot, and is an aggregate "provable-security task-solve" figure — **not** an
  apples-to-apples head-to-head and **not** specifically a GPT-4o number (§6.1).
- The *fair* framing is utility **cost**, not the headline: CaMeL's own per-model cost
  (−26.8% on Claude 3.5 Sonnet, −29.9% on Haiku vs native tool-calling) is in the same
  ballpark as axor's banking cost — so the honest claim is **comparable utility cost,
  lighter integration**: axor is a gate in front of an *unmodified* loop; CaMeL puts an
  interpreter between the model and every tool.
- Decompose the 6 lost benign tasks into the three mechanisms (genuine shared channel /
  value-coincidence false positive / whole-args fallback) — this candor locates exactly
  where a content ledger is weaker than CaMeL's structural provenance.

**6.4 The motivation Haoyu asked for — why adopt when the model already resists.**
The frame to defeat is *"just use a robust model."* Four numbered arguments, in order
of strength, built so the GPT-4o result does the load-bearing work:

1. **A strong, current production model fools itself (the result that kills the rebuttal).**
   This is the GPT-4o number, promoted to the lead argument. GPT-4o is not a weak open
   model you can wave away — it is a capable, widely-deployed frontier model with enough
   headroom to attempt the whole task list — and **undefended it carries out the
   data-exfiltration injection on 60.4% of pairs** (`agentdojo_results.md`).
   "Pick a better model" is therefore not a defense: the better model *is* the one that
   leads itself astray. Governed, the same model is at **0.0% ASR while retaining 62.5%
   of its benign utility.** This single contrast — strong model, fools itself, axor
   neutralizes it at known cost — is the paper's adoption headline.
2. **Secure-by-design invariance (the qualitative reason).** The defense gates on
   *argument provenance*, not on recognizing the injection — so a cleverer *frame* that
   flips a model produces the *same* tainted egress and is refused identically (NNSI:
   62% → 0%, depth-invariant). Model robustness is per-model, per-attack, and erodes
   with the next framing; the structural guarantee does not. **You cannot benchmark your
   way to this — it is a property, not a score.**
3. **The "robust" model is robust *to this bench*, not in general (the negative result).**
   claude-haiku resists AgentDojo's shallow single-injection attacks — but the *same
   model* falls to the NNSI nested-document framing **62% of the time** undefended
   (`nnsi_results.md`). So even the apparent counterexample to argument 1 is an artifact
   of a weak bench: "the model defended" and "the defense defended" are
   indistinguishable when the bench is too weak to create headroom; axor's guarantee is
   exactly what covers the attacks the bench cannot express.
4. **The numbers where headroom exists (the quantitative reason).** Across both a strong
   (GPT-4o) and a susceptible (Qwen) model, axor drives ASR to 0 — 60.4%→0%, 76.2%→0%,
   66.7%→0% — at measured, *explainable* utility cost, often **zero** when the legitimate
   argument comes from the prompt. The effect is not confined to weak models; it is the
   same structural gate doing the same thing on a frontier model.

**6.5 Honest accounting (a subsection, not buried).** Utility cost falls specifically on
sinks whose *legitimate* argument is read from an untrusted source (data+instructions
sharing a channel). `driving_args` narrows it to the destination field; a strict-mode
approved-recipient allowlist closes even the shared-channel residual at the price of
enumerating destinations. Report what benchmarks do **not** prove (README "Benchmarks").

**6.6 Containment benchmarks (secondary, security-relevant).** Topology containment —
100% of policy-blocked child spawns denied; export leak rate — 0 leaks observed across
runs (README "Benchmarks"). These are structural-containment numbers, not the
utility/ASR axis; frame as corroborating the gate sequence beyond the injection setting.
(Token-economy figures are deliberately omitted — they are not a security result and do
not belong in this paper.)

---

## 7. Related work

- **AgentSpec (Wang, Poskitt, Sun, arXiv:2503.18666)** — *mandatory cite, and the
  closest single-framework prior* (it is Haoyu's own work; omitting it reads as an
  oversight and is diplomatically poor). A customizable runtime-enforcement DSL whose
  rules are expressed and enforced against one framework's representation at a time
  (LangChain; re-implemented per domain for embodied / autonomous-driving). The contrast
  is the §4 thesis made concrete: AgentSpec = per-framework rule enforcement; axor = one
  decision core behind a provider-neutral `IntentNormalizer` seam, plus a non-interference
  *theorem* over the projection rather than a rule list. Position axor as *generalizing*
  the runtime-enforcement line across frameworks and giving it a formal guarantee — a
  build-on, not a take-down.
- **CaMeL (Debenedetti et al., arXiv:2503.18813)** — security by construction via an
  interpreter between model and tools; stronger guarantee, heavier integration. Axor = a
  gate in front of an *unmodified* loop. **Comparison caveat (carry from §6.3):** CaMeL's
  67% is reported, not re-run, not apples-to-apples; the fair axis is comparable utility
  cost at lighter integration. Axor's integrity axis is a content ledger (sound-to-deny,
  paraphrase residual); the confidentiality floor is the sound, paraphrase-proof part.
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
  non-interpreting projections (the §5.4 decidability split is exactly that restriction
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
- Integrity paraphrase residual on the non-liftable partition (§5.5).
- Implicit / control-flow leaks out of scope (O2).
- A-3 (complete mediation) is a premise satisfied by the daemon, not a theorem.
- Availability is outside the safety perimeter (detection→degradation can cost uptime
  under a miscalibrated threshold).
- "Any trust model" demonstrated on 2 instances, not mechanized.
- AgentDojo coverage: 3 serious injections × 16 tasks, not the full 9-injection matrix;
  benches go dark on robust models.

**Future direction (one paragraph).** The integrity paraphrase residual and the fuzzing
fraction of the perimeter (§5.4) are where a *probabilistic* predicate would help most.
Probabilistic verification (arXiv:2606.20510, §7) shows how to admit such a predicate
soundly — with a bounded violation probability rather than a structural guarantee. The
clean way to fold it into axor *without* weakening K4 is to keep it **off the trusted
path**: carry the bound on the observe-only detection→degradation surface or the advisory
adjudicator (both tighten-only), so a fuzzy classifier can ratchet restrictions under a
sound bound while the deterministic gates remain the only thing that can allow.

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
4. AgentDojo results tables — §6.2 slack zero-cost headline, §6.3 banking + CaMeL-cost
   comparison.
5. The decidability split (enum/numeric = decidable vs path/carrier = fuzzing) (§5.4).

## Questions for Haoyu (blocking — resolve before drafting prose)

1. **What is the "baseline"?** When you say "defence results are perfect for axor and
   its baseline," do you mean (a) a *rival defense* (CaMeL / a detector) we should run on
   the same AgentDojo harness for a true head-to-head, or (b) the *undefended* run as
   baseline (which is what we currently measure: 60.4% → 0%)? If (a), we need to actually
   run it — we have not. This decides whether §6.3 is a real head-to-head or a
   "reported, not re-run" comparison (§6.1 note).
2. **Is framework-agnostic (your point #1) OK at section §4, after the security
   sections?** For a security venue, secure-by-design leads and §4 sits at 4th — you said
   secure-by-design is enough for you, so this is probably fine, but you ranked it #1, so
   confirm the ordering rather than have us assume.
3. **Do you want the illustrative example *literally first* (before the introduction)?**
   You wrote "begin by an illustrative example." We currently have §1 Intro → §2 Example.
   Say the word and we fold the example into the opening of §1 (or make it §0).

## Open decisions for the authors (some now resolved)

- **Headline / example roles — RESOLVED (per review):** §2 illustrative = **banking**
  (clean gate-walk, CaMeL-comparable); §6 headline = **slack** (76.2% → 0% at zero
  cost). Two distinct roles, stated in §6.2. (Supersedes the earlier "open with banking"
  note.)
- **Venue framing:** systems-security (USENIX/CCS/S&P) vs an LLM-agent-safety venue —
  changes how much §5 formalism vs §6 empiricism leads.
- **How hard to push the theorem:** keep K4 as "stated + pinned + demonstrated on 2
  trust models," explicitly *not* claiming a mechanized proof — matches the repo's own
  honesty and pre-empts reviewer overclaim objections. Keep the induction hedge identical
  everywhere ("argued by induction, Coq/Lean target"), per §5.2/§5.6.
- **CaMeL head-to-head — depends on Q1:** either run CaMeL on the same subset, or label
  its 67% "reported, not re-run, not comparable" throughout (current default).
</content>
</invoke>
