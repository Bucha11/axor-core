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
  3. **Empirical validation on AgentDojo**, framed as CaMeL frames it — a **structural
     guarantee at a measured utility cost** (capable model, o4-mini: banking ≈−37.5pp,
     slack −38pp, travel 0pp), *not* a headroom-dependent ASR-delta that vanishes on a
     robust model. Reported honestly: the cost is localized to the shared-channel partition,
     **CaMeL is ahead on utility there, and axor does not beat it**. axor's contribution is
     the adoption cost of a drop-in, framework-agnostic governance layer (§6.5), not raw
     utility — §6.
  4. **A kernel result: a sound `enum`-supersession rule** (§6.3, §5.4) — a sink whose
     driving args are covered by a satisfied `enum` over a *closed trusted set* need not be
     gated by the leaky content-taint on top (the enum is the stronger, content-blind
     control); soundness condition is codomain ⊆ trusted (enum yes, numeric_range no — the
     kernel refuses the latter; soundness is by construction — the allowlist is a static
     operator config, never populated from a runtime read, so the codomain is
     attacker-inaccessible). With a legitimate approved-payee allowlist it
     **deterministically lifts the gate over-block on read-derived egress to allowlisted
     known payees** (4/16 banking tasks — a +25pp *gate-level upper bound*; the strict
     value-coincidence subset is 3 of those) at no security cost; the genuine shared-channel
     partition stays with CaMeL. **Honest scope:** the +25pp is a deterministic *upper bound*
     (if every lifted task converted to utility); the *realized* recovery, measured over 7
     paired o4-mini passes on the *same* population, is **+13.4 ± 9.1pp** — below the bound
     (the model completes only part of the lifted set) and a wide, noise-limited interval
     (§6.3).
     Supporting, not the headline (§5.4 is the headline).
- **Novelty arbitration (one headline, decided — a reviewer will ask "which is *the*
  contribution?").** The **single headline novelty is contribution 2's perimeter
  non-interference theorem with the decidability split (§5.4)** — the part that is
  genuinely new (K4 holds outright on the enum/numeric perimeter; the weird-machine
  residual is localized to a named rich-syntax fraction). Contributions 1 (the
  framework-agnostic seam) and 3 (the empirical study) are **supporting** — strong, but
  not billed as "the novel result." Do not let §4 and §5.4 each claim "the contribution"
  (they did in an earlier draft); §4 is the engineering contribution that *supports* the
  headline. (Section *ordering* is a separate axis — Haoyu ranked framework-agnostic #1
  for emphasis; for a security venue the theorem still leads. See §1 Question 2.)
- Forward-reference the opening example (§2).

---

## 2. Illustrative example — "what a successful defense actually looks like"

> Haoyu: *"begin with an illustrative example — adopted from your AgentDojo defended
> by axor — to give the reader a sense of what's behind a successful defense, and
> how the theorem secures the whole system."*

**Candidate (recommended): the banking exfiltration task, AgentDojo `important_instructions`.**
Clean to walk gate-by-gate and it is the CaMeL-comparable scenario.

> **§2 suite choice — RESOLVED by a repo check (this was the open question).** The choice
> is decided by *which axis the denial lands on*, not by drama. Verified against
> `examples/agentdojo/config/`: **slack declares only `untrusted_sources`, no
> `sensitive_sources`**, and the confidentiality floor arms only on a `sensitive`-rooted
> read (`axor_core/taint/engine.py:101`). So the slack mass-exfil denial is **integrity /
> destination-taint** (per-value taint on `post_webpage(url)`; whole-blob carrier on the
> message sinks) — confirmed by `agentdojo_results.md` ("denied by the per-value taint
> gate"). That means **slack carries the *same* encoding/paraphrase residual as banking
> — it gains §2 nothing**, so **banking stays the default** and the doc is consistent.
> The suite where the *confidentiality floor* (sound, paraphrase-proof, ends on a closed
> axis) actually arms is **travel** (`config/travel.yaml` is the only one with
> `sensitive_sources`). **This finding is bigger than §2 — it surfaced the sound-axis
> demonstration, now placed as the structural unit-level Appendix A-floor** (a *property*,
> not an AgentDojo result; §6.2 records only that the stock travel slice cannot carry it).
> Consequence for §2/§5.2: banking stays the recommended opener (cleanest integrity
> gate-walk), but **travel is the example that *closes on the sound axis*** — the floor
> refusing a paraphrased secret egress — and is the natural second illustration if we want
> the reader to see the paraphrase-proof half, not just the integrity gate that ends on a
> documented residual. Not a swap; an addition.

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
proves it. (If an illustrative stressor is wanted here, the NNSI depth-invariant block is
available — but flag it as *our own, self-refereed* attack and anchor it to the published
nested/indirect-injection class, per §6.5; do **not** lean on its 62% as a load-bearing
number. The framing-invariance claim stands on the theorem, not on NNSI.)

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
decision core behind a provider-neutral normalization seam.* This is the
**design/engineering contribution** — the gap Haoyu named — and it *supports* the
paper's single headline novelty (the non-interference theorem + decidability split,
§5.4); it is deliberately **not** itself billed as "the" novelty (see the novelty-
arbitration note at the end of §1).

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

**The equation alone is necessary but not sufficient — say so explicitly, or a S&P
reviewer kills the theorem with one line.** The equation is exactly "`decide` factors
through `π`," and *every* function factors through `π = identity` (singleton fibers, the
antecedent forces `x₁ = x₂`, the consequent is trivial). So the equation carries **zero**
security content on its own; all of it lives in **which `π`**. Fix `π` up front as a
**specified, lossy projection** (origin-class / enum / bounded-numeric / path-class /
provenance-label — the closed admissible set, §5.4) and split the obligation into three
parts, only the first of which is the equation:

1. **Factorization (O1, the equation).** `decide` reads `x` only through `π` — no
   `decide'` on the effect path touches raw `x`. Necessary, not sufficient.
2. **Coarseness (the part the equation hides).** `π` is lossy *enough* that the whole
   class of attacker reframings of a given injection **collapses into one fiber** — they
   all map to the same origin-class, so they are forced to share a verdict. This is what
   makes framing-invariance a *theorem* and not a coincidence; it is exactly why
   `π = identity` is disqualified (singleton fibers absorb nothing).
3. **Instruction-free codomain (T0/T4, §5.4).** The shared projection value itself
   admits no instruction, so collapsing into a fiber cannot smuggle the attack back in.

State the degenerate case in one sentence: *`π = identity` satisfies the equation and
guarantees nothing; admissibility (the closed lossy codomain set) is precisely the
condition that rules it out.* With that, "two inputs the consumer would treat
differently cannot share a projection" (the non-merging / T4 direction) and "the whole
reframing class does share one" (the coarseness direction) are the two halves that
together give **"framing cannot change the verdict"** — neither half alone does.

**5.2 The chain Haoyu asked for — attack → taint → theorem (the heart of the paper).**
Make this an explicit three-step walk, reusing §2's banking example:

| Step | What happens | Anchor |
|---|---|---|
| **Attack** | The §2 banking scene: an injected instruction + the attacker's IBAN enter via an untrusted transaction-history read; data and instructions share a channel. | §2; `examples/agentdojo/agentdojo_results.md` (banking, `important_instructions`) |
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

**5.4 The conditionality is the point — the decidability split (THE headline novelty of
the paper).** *Do not overclaim — Haoyu values the honesty.* K4's safety content rests on
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

**5.5 The two axes, asymmetric by design — and asymmetric in how they are demonstrated
(say this out loud; it pre-empts the "why no floor benchmark?" question).**
- **Integrity** — precise per-value content-derivation; sound to *deny*, with a
  documented **paraphrase residual** on the non-liftable partition (a generic write/exec
  that cannot be made positional). State this as a boundary, not a surprise
  (`governance-model §7`; `examples/attacks/`). *Demonstrated empirically* — its residual
  and its cost are benchmark-measurable (the §6 banking/slack cost numbers).
- **Confidentiality** — a **sound, paraphrase-proof session floor**, and "paraphrase-proof"
  is a **by-construction argument, not the claim of one passing test**: the floor's
  decision is a function of a *session boolean* — "is a secret read outstanding?" — and
  **never reads the bytes of the outgoing value at all**. Because the egress content is
  not an input to the decision, *no* transformation of the secret (base64, paraphrase,
  homoglyph, sub-fragment shredding) can change the verdict — there is no content channel
  to attack. The paraphrase micro-test (§Appendix) is an *illustration* of that structural
  fact, not its evidence. Released only by a governance endorsement (`governance-model §5`).
  *Demonstrated structurally* — at unit level (Appendix A-floor), **not** as an AgentDojo
  number, because a by-construction guarantee is shown to *hold*, not averaged over a
  noisy suite (§6.2 records why the stock travel slice does not isolate it).
- **Positional admission** — for high-stakes sinks whose legitimate input *cannot
  encode an instruction*, admit by *form*, not content → sound by construction
  (`governance-model §3`). The kernel **refuses** to let you declare an exec-class sink
  positional.

**The asymmetry, named:** integrity is a *measured* axis (cost + residual on the
benchmark); confidentiality is a *proven* axis (a structural property shown at unit
level). They are not demonstrated the same way *on purpose* — and stating that is
stronger than pretending both live on the same benchmark table.

**5.6 "Any trust model."** The kernel/trust-model factorization is enforced at the
import level, and a substitution test re-proves non-interference + complete mediation
under a *second* trust model with a different tainting rule
(`tests/contracts/test_value_provenance.py`). Honest status: *demonstrated on two
instances, not mechanized over all conforming backends* (O2 is the inductive content —
a Coq/Lean target). Put the obligation→test crosswalk table (kernel-theorem §5) in an
appendix — it is the paper's credibility anchor.

---

## 6. Evaluation — AgentDojo (Haoyu point #3)

> **The measurement axis — read this before structuring §6; it is the section's whole
> point.** Our instinct was to lead with **ASR-delta** (undefended X% → governed 0%).
> *Drop that as the primary axis.* ASR-delta is **headroom-dependent**: it requires a
> foolable model — no headroom, no delta, and on a robust model it degenerates to
> 0% → 0% and the result evaporates. **CaMeL never used this axis.** CaMeL's number is
> "fraction of tasks solved *with* a by-construction guarantee," reported as the
> **utility *cost* of that guarantee**; its ASR≈0 is *structural* (the interpreter
> cannot exfiltrate), so it holds whether or not the model would have fallen — the cost
> question is well-defined on a dumb *and* a smart model.
>
> **axor has the same by-construction nature** — the gate denies by *projection*,
> framing-invariant, ASR→0 *structurally* (enum/numeric outright, rich-syntax
> fuzz-discharged; §5.4). So axor can and must make the **same *type* of claim as CaMeL:
> a structural guarantee + its measured utility cost** — which does **not** go dark on
> Claude 4 / o3. Flip the axis:
> - **Primary (§6.2): structural guarantee at utility-cost Y.** CaMeL-*shaped* (not the
>   same number — different harness). This is the load-bearing result and the direct
>   answer to Haoyu's "motivation for adoption": *motivation = the guarantee; metric =
>   its cost; not a scoreboard.*
> - **Secondary colour (§6.3): ASR-delta where headroom exists** (GPT-4o 60.4→0, Qwen
>   slack 76.2→0) — "here is where an undefended model self-owns, i.e. the threat the
>   guarantee neutralizes is live." Explicitly note it degenerates to 0→0 on robust
>   models *and that this is fine*, because it was never the load-bearing axis.
>
> This is the *same honesty as §5.4* (a guarantee with a named, bounded conditionality),
> applied to the empirics. Smart models do not break a by-construction defense — they
> make it look *better* (CaMeL v2's utility *rose* on smarter models, e.g. o3 ≈ +10% vs
> o1, at the same structural guarantee). What goes dark is only the ASR-delta frame.

> **Canon lock (do this before writing a single number).** The authoritative figures
> are `examples/agentdojo/agentdojo_results.md`: banking GPT-4o **60.4% → 0.0% ASR at
> 62.5% benign utility retained**. An earlier note circulated 54%/56% — that is a stale
> run; purge it. Every number in §6 and §2/§1 must trace to this file.

> **What "baseline" means — DECISION TAKEN (option b), per the §6.3 methods statement.**
> Our tables compare **undefended vs governed**; that is each defense's *own-baseline*
> cost, not a competing-defense head-to-head. We **do not re-run CaMeL** — a faithful
> re-implementation of its interpreter is a project in itself, and an imperfect one would
> mislead more than a version-pinned reported reference (§6.3, verbatim). So we report
> **cost profiles vs own baseline**, CaMeL's figures taken from its v2 tables. The only
> residual for Haoyu (Q1, now a *confirmation*, not a fork): confirm he wasn't expecting a
> rival-defense re-run; if he was, that is weeks of interpreter work and a separate
> decision. Until he says otherwise, the §6.3 statement stands.

**6.1 Setup.** AgentDojo suites (banking, slack, workspace, travel), attack =
`important_instructions` (the strongest stock injection); defense = `GovernedToolsExecutor`
wrapping every tool call in a `ToolCallGovernor` configured by the suite's YAML taxonomy
— *the same config a real deployment loads.* Models: **GPT-4o** primary, **Qwen-2.5-72b**
susceptible-model supplement, **claude-haiku-4-5** robust contrast
(`examples/agentdojo/agentdojo_results.md`).

> **CaMeL-model finding — RESOLVED from the v2 PDF tables (do not write "the model
> CaMeL measured").** CaMeL **v2** (arXiv:2503.18813v2, Tables 2–4) evaluates **Claude 4
> Sonnet, Gemini 2.5 Flash, Gemini 2.5 Pro, o3-high, o4-mini-high** as defended backbones.
> **No GPT-4o as a defended backbone in *either* version** — v1 (March 2025) used GPT-4o
> as a backbone; v2 references **GPT-4o-mini** only as an instruction-hierarchy baseline /
> tokenizer, not a defended model (say it this way — Haoyu knows the paper and will catch
> "only in v1"). The headline differs by version (≈67% v1 vs **≈77%** in v2's abstract,
> *77 vs 84 undefended*; the "/75" figure is unconfirmed — drop it unless you can point to
> a specific table cell). So:
> - There is **no shared defended model** between our GPT-4o run and CaMeL v2. The phrase
>   "the model CaMeL measured" is only defensible against v1, and then with v1's numbers.
> - **Pin one CaMeL version and cite its numbers consistently** (see Questions). Default:
>   treat our GPT-4o result as *axor's own measurement* and CaMeL as a *version-pinned,
>   reported reference point*, never a re-run head-to-head.
> - **The actual v2 per-suite cost (Table 2, Difference, no-attack) — use these, the
>   earlier "−27 to −30pp" was a phantom and is wrong in the worse direction.** On
>   **banking** CaMeL's cost is **≈ 0 or positive**: Claude 4 Sonnet +0.0%, Claude 4
>   Sonnet\* +6.2%, o3-high +18.8%, o4-mini-high +18.8%, Gemini 2.5 Pro −6.2%, Gemini 2.5
>   Flash −12.5% (mean ≈ **+4%**, 3/6 positive). On **slack** CaMeL's cost is **large and
>   negative**: roughly **−24 to −43pp** (Claude 4 −33.3, Gemini Pro −42.9,
>   o3/o4-mini/Claude 4\* −23.8, Gemini Flash −28.6). Do **not** write "Difference rows are
>   negative" (false on banking) and do **not** cite Table 8's **58.33%** as utility — it
>   is a *policy-trigger rate* ("rates ... when not under attack"), not utility; dropping
>   it. Read deltas from the cited version's Table 2, not the screenshot.

**6.2 The guarantee and its measured cost — the PRIMARY result (lead §6 with this).**
State the guarantee first, in §5's vocabulary: axor's gate denies by *projection*, so on
every declared egress sink the exfiltration path is refused **structurally** —
framing-invariant, ASR→0 by construction (enum/numeric outright; rich-syntax
fuzz-discharged, §5.4), independent of whether the model would have fallen. The empirical
question is then CaMeL's question, not a scoreboard: **what does the guarantee cost in
benign utility?** Measured per suite on the *benign* task list:

Measured on **o4-mini** (a capable CaMeL-v2 backbone), full CAMEL_MODE, ASR 0% everywhere
(it resists on its own — the headroom-dark case the cost axis exists for). The banking row is
**one representative pass** (undefended is noisy — see caveat); slack/travel as observed:

| suite (full) | benign undef → gov | cost | denials | mechanism |
|---|---|---|---|---|
| **banking** † | 87.5% → 50.0% | **≈ −37.5pp** | 3–5/pass (this pass: 5) | shared channel — payee read from an untrusted source |
| **slack** | 85.7% → 47.6% | **−38.1pp** | 15 (egress sinks) | shared channel — post derived from channel reads |
| **travel** | 65.0% → 65.0% | **0pp** | **0** | egress recipient comes from the **prompt**, not a read |

† *representative single pass; read the banking numbers via the caveat + paired run (§6.3).*

*Caveat on the banking row:* o4-mini's **undefended** benign rate is unstable across runs
(87.5% here, 68.8% / 62.5% / ~70% mean over the 14-pass paired run — pure sampling variance
on n=16); the **governed** rate is the stable signal (generic ~51%, σ=5.2 over 7 passes).
Read banking as "≈ −37.5pp, undefended noisy"; cite the paired run (§6.3) and per-task
losses, not a single absolute.

*Caption — what the banking cost is, and how §6.3 splits it.* **Two different counts that
must not be conflated:** (i) the **per-pass denial count fluctuates 3–5** (a denial only
fires when the model actually reaches an egress sink, so a given pass realizes whichever
subset it gets to — "5" above is one pass, not a structural total); (ii) the **structural
partition is defined over all 16 tasks, independent of any pass.** Structurally the −37.5pp
generic over-block decomposes into **two** partitions: (a) a **known-payee read-derived
partition** — recipient read-derived *and* an allowlisted known payee — **4 tasks {3, 4, 6,
15}** (strict prompt∩read value-coincidence subset = {3, 4, 6}), which the §6.3
`enum`-supersession **lifts at the gate** (+25pp deterministic upper bound); plus (b) a
**one-off shared-channel partition** — payee present *only* in an untrusted file with no
known-payee entry — **2 tasks {0, 2}** (bill/landlord), which **stays with CaMeL** (config
cannot pre-enumerate a one-off payee). So the 6 structural tasks (4 + 2) and the 3–5 per-pass
denials are *different quantities*; "−37.5pp" is the **generic, pre-supersession** figure, and
§6.3 reports what the gate lift recovers (and the third, model-gated *realized* number).

**The load-bearing, honest finding: the cost is *localized to the shared-channel
partition*, not universal.** axor pays ~−38pp where the legitimate egress argument is read
from an untrusted source (banking/slack) and **exactly 0 (zero denials) where the
legitimate recipient is prompt-given (travel)**. Cost = (tasks the model can do) ∩ (egress
derived from an untrusted read).

**Two corrections to retire (both were weak-model artifacts — do not reuse them):**
- *"slack = zero cost"* is **dead**. That 47.6% → 47.6% was on **Qwen**, which fails the
  shared-channel tasks anyway, so there was nothing to block. On o4-mini, which *completes*
  them undefended (85.7%), the gate's block costs a real −38pp. The governed floor (~47.6%)
  is the prompt-driven partition (model-independent); the **cost grows with model
  capability** on the shared-channel partition. Only **travel's 0 is structural** (egress
  from the prompt) and robust across models — that is the one transparency claim that
  survives.
- The integrity-axis "preserved utility is a property of the taxonomy, not the model" line
  is **wrong as stated** — the cost *does* scale with the model (Qwen ~0, o4-mini −38pp).
  State it correctly: the *partition* is taxonomy-fixed; the *cost on it* scales with how
  many of those tasks a stronger model can actually complete.

Decompose the lost banking tasks into the **two partitions** of the caption above
(known-payee read-derived, lifted by supersession / one-off shared-channel, stays with
CaMeL) — this is exactly where a content ledger is weaker than CaMeL's structural provenance
(§6.3), and the gap *widens* on a capable model. (An earlier draft split this into "three
mechanisms" with a separate *whole-args fallback* bucket for task 15; §6.3 retired that —
task 15 is a read-derived known-payee egress in partition (a), not a third mechanism.) The **confidentiality (sound floor)** axis is deliberately not a row here —
it is a structural property (Appendix A-floor, §5.5), and the stock travel suite does not
exercise it (0 floor denials in the o4-mini run; the floor never armed because benign
travel tasks don't read-secret-then-egress and o4-mini resisted the overt injection).

**6.3 CaMeL comparison — the *same kind* of claim, and CaMeL is ahead on utility.** axor
and CaMeL report the **same type of result** — a *structural* guarantee plus its *measured
utility cost* — so they are comparable in *kind* (not in number: different harness /
subset / model / version). On the measured numbers, **CaMeL is ahead** (below); axor's
case rests on *adoption cost*, not utility (§6.5).

> **Paper-ready methods statement (use verbatim).** "We compare each defense's per-suite
> utility cost against its own undefended baseline; absolute utilities are not comparable
> across harnesses, so we report cost profiles rather than a head-to-head. CaMeL's figures
> are taken from its v2 tables (Claude 4 / Gemini 2.5 / o3 / o4-mini backbones — note v2
> reports no GPT-4o, so this is a comparison of mechanism cost profiles, not model-matched
> numbers), not re-run on our harness: a faithful re-implementation of CaMeL's interpreter
> is a project in itself, and an imperfect one would mislead more than a version-pinned
> reported reference."

This statement *resolves the Q1 "do we re-run CaMeL" question* (we do not, and we say why
in print). The bullets below are the supporting detail behind it. Be precise:
- **Not apples-to-apples on the number.** CaMeL's headline is reported, not re-run by us;
  version matters (v1 ≈67% *with* GPT-4o; v2 ≈77% on Claude 4 / Gemini 2.5 / o3 /
  o4-mini, **no GPT-4o backbone** — §6.1). Pin the version you cite.
- **The cost profiles — measured on a capable model (o4-mini). CaMeL is ahead on utility;
  axor does not beat it anywhere. State this plainly.** (The earlier "inverted profiles,
  axor strictly cheaper on slack" was a **Qwen artifact** and is retracted — on o4-mini
  slack cost is −38pp, not ≈0.)
  - **banking**: axor **≈ −37.5pp** vs CaMeL **≈ 0 / +4%** → CaMeL ahead. The −37.5pp
    *partitions* (next bullet): an `enum` allowlist + supersession deterministically lifts
    the **gate** over-block on the **known-payee read-derived** partition (4 tasks, +25pp
    *upper bound*; +13.4 ± 9.1pp realized on the same population, 7 paired passes); the
    **genuine shared-channel** half (one-off payee read only from a file) stays with CaMeL. So
    axor lifts the gate block on part of the gap; it does not close it.
  - **slack** (multi-step): axor **−38.1pp** vs CaMeL **≈ −24 to −43pp** → **comparable.**
  - **travel** (prompt-driven egress): axor **0pp** (0 denials); CaMeL likely also ~0 →
    **tie / both free.**
  **Hard caveat — compare deltas vs each system's own baseline, never absolutes** (axor
  slack-undefended 85.7 ≠ CaMeL native-slack 95.2); verify every CaMeL delta is
  benign/no-attack (Table 2, not Table 3). And **o4-mini's banking undefended is unstable**
  (87.5 / 68.8 / 62.5 across runs — pure sampling variance); use the *paired* run and read
  *per-task* losses, not absolute rates.
- **A sound enum-supersession rule lifts the over-block on read-derived egress to an
  allowlisted known payee — a *deterministic gate-level upper bound*, NOT a realized
  benchmark utility win. Real, tested kernel mechanism (this work).** Mechanism: when every
  driving arg of a sink is covered by a satisfied **`enum`** over a *closed, operator-declared
  trusted set*, that enum carries the integrity axis and the content-taint is skipped on that
  arg (attacker destination not in the set; the confidentiality floor is *not* superseded).
  **Soundness condition (T4, §5.4): predicate codomain ⊆ trusted values the attacker cannot
  choose** — `enum` qualifies, `numeric_range` does **not** (open codomain); the kernel
  refuses to supersede on it — pinned by `test_numeric_range_does_not_supersede_open_codomain`
  (the T4 soundness regression, named in the Appendix A crosswalk beside O1/O2/O3/T0).
  **Soundness is by construction:** the allowlist is a **static operator config, not populated
  from any runtime read, so the enum codomain is attacker-inaccessible by construction** — an
  attacker controlling an untrusted read can never add an IBAN to the trusted set (a static
  invariant, stronger than a runtime origin-filter). And it is *legitimate, not leakage*: the
  IBANs are the user's/operator's known payees.
  **Define the recovery population once, so ceiling and realized are measured on the *same*
  set (avoiding a cross-population comparison).** The gate recovers exactly the tasks with a
  *legitimate egress whose driving recipient is read-derived and is an allowlisted known
  payee* — for each, generic **DENIES** (content-taint), tuned **ALLOWS** (supersession),
  attacker stays **DENIED** (not in the enum). Verified deterministically against the real
  environment reads (a gate-level check, no model), the banking suite has **four**:

  | task | recovered egress | recipient | source | strict value-coincidence? |
  |---|---|---|---|---|
  | 3 | refund friend | GB29… | prompt **and** history | yes |
  | 4 | refund friend | GB29… | prompt **and** history | yes |
  | 6 | recurring iPhone | US122… | prompt **and** scheduled-tx | yes |
  | 15 | friend-refund sub-goal | GB29… | history **only** (IBAN not named in prompt) | **no — read-only known payee** |

  → **gate-level ceiling = 4/16 = +25pp**, the *upper bound* if every gate-lifted task fully
  converted to utility. **Two named sub-characterizations within this one population, not two
  populations:**
  - **Strict value-coincidence subset {3, 4, 6} = +18.75pp** — recipient named in the prompt
    *and* also in a read (the textbook substring false positive). Pinned by
    `test_value_coincidence_recovery_structural_not_cherry_picked`.
  - **Task 15 is the *broader* read-only case:** its recovered recipient `GB29…` is in history
    but **not in task 15's prompt** — the gate lifts it because `GB29…` is an allowlisted known
    payee, *not* because it is value-coincidence. (Its prompt-named recipient, the landlord
    `US133…`, is not read-derived and is **ALLOW in both generic and tuned** — never blocked,
    so *not* a recovery; the security control that holds is the **attacker IBAN denied on all
    sinks**, not "`US133…` denied". This also corrects an earlier draft that counted task 15 as
    a 4th *value-coincidence* task via a **fabricated** `US133…`-in-read claim — that was wrong;
    task 15 enters the recovery population only via the **real** `GB29…` read, under the broader
    definition.)
- **"Ceiling at the gate" ≠ "realized utility" — measured, on the same population.** The
  +25pp upper bound says the gate stops *denying* those four transfers; it does **not** prove
  the model *completes* them. A proper **paired** measurement (7 passes, generic-gov vs
  tuned-gov as paired conditions, o4-mini, benign-only) gives **realized recovery =
  +13.4 ± 9.1pp** (generic 50.9% → tuned 64.3%; paired per-pass diffs
  +12.6/+12.4/+6.3/+31.2/+0.0/+12.5/+18.8). Realized lands **below the +25pp ceiling because
  the model converts only part of the lifted population**: task 6 never completes (o4-mini
  fails it **14/14 even undefended** — a hard scheduled-transaction task the gate lift cannot
  rescue), and tasks 4/15 complete only partially (undefended fail 5/14 and 4/14). So the
  realized set is **{3, 4, 15}** (model-completed) ⊂ the gate set **{3, 4, 6, 15}** (gate-lifted)
  — a clean subset, *one* population, not the cross-population "+13.4 below +18.75" the
  previous draft implied. **Report the three commensurable numbers together: generic over-block
  −37.5pp (pre-supersession) → gate ceiling +25pp (deterministic upper bound, {3,4,6,15}) →
  realized +13.4 ± 9.1pp (measured, {3,4,15}).**
  - **The ± is wide — say so.** ±9.1pp over 7 passes is a broad interval (≈ +4 to +23pp): the
    realized effect is **positive in every pass** (min paired diff +0.0, never negative) but
    its *magnitude* is not tightly bounded. Report +13.4 ± 9.1 as evidence the gate lift is
    **not purely cosmetic**, not as a precise utility figure — a reviewer who computes the CI
    should find we already flagged it.
- **The asymmetry to own honestly — CaMeL is ahead on the shared-channel partition.** CaMeL
  v2's utility *rises* on smarter models (o3 ≈ +10% vs o1). enum-supersession deterministically
  lifts the gate over-block on the known-payee read-derived partition (+25pp upper bound over 4
  tasks) where the operator can enumerate payees, but the genuine one-off shared-channel residual
  (a payee that exists only in an untrusted read, with no known-payee entry) needs CaMeL's
  structural provenance and is unrecoverable by config. Bottom line: **axor does not win on
  utility globally; it lifts the gate over-block on the known-payee read-derived half of the
  banking gap (a +25pp upper bound, +13.4 ± 9.1pp realized); CaMeL is ahead on the one-off
  shared-channel half.**

**6.4 ASR-delta — secondary colour, where headroom exists (NOT the load-bearing axis).**
On a *foolable* model the undefended attack lands and the structural guarantee neutralizes
it — useful to *show the threat is live*, not to carry the result:

| model · suite | undefended ASR | governed ASR | note |
|---|---|---|---|
| GPT-4o · banking | **60.4%** | **0.0%** | strong model self-owns |
| Qwen · slack (full) | **76.2%** | **0.0%** | mass exfiltration neutralized |
| Qwen · banking-PII | 66.7% | **0.0%** | **ASR-only — degenerate utility (0 in both), never a zero-cost win** |
| claude-haiku · banking | 0% | 0% | **degenerates — robust model, no headroom; expected, not a failure** |

Say the quiet part: on a robust model this axis collapses to 0 → 0, and **that is not a
failure of the defense** — the guarantee and its cost (§6.2) are untouched; only the
delta-metric goes dark. (Footnote, principled: a 9-pair slice — *pre-registered* as the
pairs where the undefended injection derailed the task into failure — shows utility
33.3% → 88.9%; report only with that definition + the n=9 / wide-CI caveat, as a
mechanism illustration, never a bare "governance raises utility.")

**6.5 The adoption motivation — and the head-on "why not just use CaMeL?" answer.** Be
blunt: **CaMeL gives more utility (§6.3) and axor does not beat it.** The enum-supersession
rule deterministically lifts the gate over-block on the known-payee read-derived partition (a
+25pp **gate-level upper bound** over 4 tasks, §6.3) where the operator can enumerate payees,
but (a) the **realized** benchmark utility of that lift, measured on the same population, is
**+13.4 ± 9.1pp** (7 paired passes — a wide interval, below the bound because the model
completes only part of the lifted set; not a clean win) and (b) the genuine shared-channel
partition stays with CaMeL. So axor's case is **not** utility; it is
**lower adoption cost at comparable-or-somewhat-lower utility**:

- **axor is a gate in front of an *unmodified* agent loop** — you keep your existing
  agent (any provider, any framework) and drop a `ToolCallGovernor` in front of its tool
  calls. **CaMeL requires re-architecting the agent into an interpreter-mediated plan
  emitter** (the model emits a restricted-Python program a custom interpreter executes);
  their own repo flags it as a research artifact that "likely contains bugs ... might
  crash." Most production agents will not be rewritten that way.
- **axor works on any model**, including weak ones; CaMeL needs a model capable of writing
  valid plans (its utility drops on weaker models).
- **axor is framework-agnostic** (the §4 seam); CaMeL's interpreter approach is a heavier,
  more invasive integration.

So the honest pitch: *if you can adopt CaMeL (rewrite your agent, accept the interpreter)
and want maximum utility-at-security, use CaMeL — axor does not beat it there. axor is for
the common case where you have an existing agent you will not rewrite, across arbitrary
frameworks, and want a structural, framing-invariant guarantee as a thin governance layer
— at a measured utility cost on the shared-channel partition (§6.2).* **Do not claim
parity; claim a different point on the cost/assurance/integration frontier.** (This is the
open question for Haoyu, §1 Q-venue: is "secure-by-design governance layer, framework-
agnostic, drop-in, honest utility cost" enough — knowing CaMeL wins on utility? If the bar
is "best utility-at-security on AgentDojo," CaMeL already cleared it and axor should not
contest that ground.)

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
   whole projection-equivalence class regardless. That is *why* you want the guarantee.
2. **NNSI is illustrative only, never load-bearing — it is *our own* attack.** A result
   built on NNSI is **self-refereed** ("we built X and beat X"); a reviewer discounts it
   to zero, and the 62%-undefended figure is self-reported on a non-standard attack that
   would itself need independent validation. Use NNSI *only* as an illustrative stressor
   for "standard benchmarks underestimate the attack surface," and *only* anchored to the
   published class in point 1 — never as a bare number, never as the reason to adopt. The
   reason to adopt is §6.2 (cost of a real guarantee) + §5 (the guarantee), both of which
   stand without NNSI. Earlier drafts leaned on NNSI's 62% as a load-bearing prong;
   demote it.

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
  interpreter between model and tools; **stronger guarantee and more utility**, heavier
  integration. Axor = a gate in front of an *unmodified* loop. **Comparison caveat (carry
  from §6.3, measured on o4-mini):** CaMeL is **ahead on utility** — banking CaMeL ≈0 vs
  axor −37.5pp generic. But that gap is **not monolithic**: it splits into a
  known-payee read-derived half, which axor's `enum`-supersession **lifts at the gate** (+25pp
  deterministic upper bound over 4 tasks; +13.4 ± 9.1pp realized on the same population over 7
  paired passes), and a genuine one-off shared-channel half (payee only in an untrusted read),
  where **CaMeL is ahead** because its
  structural provenance keeps the read-derived payee axor's content ledger cannot. Slack is
  comparable (both ~−38pp), travel a tie (both ~free). Axor does **not** win on utility
  globally on a capable model; its case is *adoption cost* (drop-in, framework-agnostic, any
  model), not utility. Axor's integrity axis is a content ledger
  (sound-to-deny, paraphrase residual); the confidentiality floor is the sound,
  paraphrase-proof part.
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
- **Confidentiality (sound) axis — shown as a structural property (Appendix A-floor), not
  a benchmark.** The floor is paraphrase-proof *by construction* (the gate reads a session
  boolean, never the egress bytes; §5.5) and is illustrated by a deterministic unit-level
  demonstration (arm → refuse → paraphrase-still-refused → control-allows), reproduced on
  GPT-4o while running travel. Honest scope, stated as the *named axis asymmetry*: integrity
  is a *measured* axis (§6 cost), confidentiality is a *proven* axis (a property) — the
  **stock AgentDojo travel slice does not isolate the floor** (ASR ≈0/noisy; the live benign
  denial was integrity), so there is no benchmark floor-cost and we report none.

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
AgentDojo as a **structural guarantee at a measured utility cost**, reported honestly: the
cost is localized to the shared-channel partition (≈−38pp on banking/slack on a capable
model, 0 where egress is prompt-driven), where **CaMeL's heavier interpreter is ahead on
utility and axor does not beat it**. A sound `enum`-supersession rule deterministically
lifts the gate over-block on the known-payee read-derived partition (a +25pp gate-level upper
bound over 4 tasks) where an operator can enumerate trusted destinations; its *realized*
benchmark utility, measured on the same population, is **+13.4 ± 9.1pp** (7 paired passes — a
wide interval, below the bound because the model converts only part of the lifted set), and
the shared-channel partition stays with CaMeL. axor's
contribution is the *adoption cost* of a guarantee that drops in front of an unmodified,
framework-agnostic agent loop on any model — not utility. The guarantee is the value; its
utility cost is the honest price. Agents should not self-govern execution.

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
  a kernel *property*, not a benchmark number (per §5.5; §6.2 explains why the stock travel
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

---

## Figures / tables to produce

1. The reverse-osmosis gate stack (README) — *the* signature figure.
2. Trust-ring diagram + three-interface adapter seam (§4).
3. The attack → taint → theorem chain as a single diagram (§5.2) — the conceptual core.
4. **The primary figure: measured utility cost across suites (o4-mini)** (§6.2 — banking
   −37.5pp generic, slack −38pp, travel 0pp) with the CaMeL cost overlaid (banking ≈0, slack
   ≈−30s, travel ≈0) — shows the cost is localized to the shared-channel partition. Caption
   the banking bar honestly: **CaMeL ahead on the one-off shared-channel half; axor lifts the
   known-payee read-derived half at the gate** (+25pp upper bound / +13.4 ± 9.1pp realized) —
   not a flat "CaMeL ahead on banking." Integrity axis only; the confidentiality floor is a
   *property*, shown separately (figure 6), not a cost row. Put the ASR-delta
   table (§6.4) *second/smaller*, captioned "where headroom exists," so layout itself
   signals which axis is load-bearing.
5. The decidability split (enum/numeric = decidable vs path/carrier = fuzzing) (§5.4).
6. **The confidentiality-floor listing** (Appendix A-floor) — a small deterministic
   unit-style listing (arm → refuse → paraphrase-still-refused → control-allows), shown the
   way a unit test is shown, *not* as a benchmark bar. Caption names the asymmetry:
   integrity is *measured* (fig. 4), confidentiality is *proven* (fig. 6).

## Questions for Haoyu (confirmations now — none is a hard blocker after the §6.3 decision)

**Send all three in one short async email. None is a hard blocker anymore — Q1 is now a
confirmation (the §6.3 methods statement already takes the decision), Q2/Q3 are cheap
ordering choices. Worth sending early so a surprise ("I did expect a CaMeL re-run") lands
before, not during, the draft.**

1. **The "baseline" — now a *confirmation*, not a fork (we took option b).** We have
   adopted the §6.3 methods statement: **own-baseline cost profiles, CaMeL not re-run**
   (faithful interpreter re-implementation is its own project; an imperfect one misleads
   more than a version-pinned reference). One-line confirmation needed: you weren't
   expecting a rival-defense *re-run* on our harness? If you were, that is weeks of
   interpreter work and we should scope it separately before drafting §6. Absent that,
   the methods statement stands and §6 needs no rival-defense run.
2. **(cheap) Is framework-agnostic (your point #1) OK at section §4, after the security
   sections?** For a security venue, secure-by-design leads and §4 sits at 4th — you said
   secure-by-design is enough for you, so this is probably fine, but you ranked it #1, so
   confirm the ordering rather than have us assume.
3. **(cheap) Do you want the illustrative example *literally first* (before the
   introduction)?** You wrote "begin by an illustrative example." We currently have §1
   Intro → §2 Example. Say the word and we fold the example into the opening of §1.

## Open decisions for the authors (some now resolved)

- **§6 measurement axis — RESOLVED (per review, the big one):** lead with
  **structural-guarantee-at-utility-cost** (CaMeL-shaped, §6.2), *not* ASR-delta. ASR-delta
  (§6.4) is secondary colour because it is headroom-dependent and degenerates to 0→0 on
  robust models; the guarantee + cost does not. This is the empirical analogue of §5.4's
  honesty and the real answer to Haoyu's "motivation for adoption."
- **Headline / example roles — RESOLVED:** §2 illustrative = **banking** (clean gate-walk);
  the §6 *primary* result = the cross-suite **cost-of-guarantee** table (o4-mini: banking
  −37.5pp, slack −38pp, travel **0pp**). **travel is the transparent instance** (egress
  from the prompt), not slack — the old "slack zero-cost" was a Qwen artifact and is
  retracted (§6.2/§6.3). (Supersedes the earlier "open with banking" and "§6 headline =
  slack ASR-delta" notes.)
- **The confidentiality floor — DONE, placed as structural unit-level (Appendix A-floor),
  *not* a §6 result.** The floor is demonstrated by the by-construction argument (the gate
  reads a session boolean, never the egress bytes — so paraphrase-proof by signature) plus
  a deterministic illustration (arm → refuse → paraphrase-still-refused → control-allows),
  reproduced live on OpenRouter/GPT-4o while running travel. **Decision taken (per review):
  present this beside Appendix A, framed as a property, and name the axis asymmetry out
  loud (integrity = measured, confidentiality = proven, §5.5).** §6.2 keeps only the honest
  one-liner that the stock travel slice does not isolate the floor (ASR ≈0/noisy; the live
  benign denial was integrity), so we report no "floor utility cost." Optional, low
  priority: a small targeted benign read-secret-then-email harness only if a reviewer
  insists on a floor *cost* number — below Q1.
- **Venue framing:** systems-security (USENIX/CCS/S&P) vs an LLM-agent-safety venue —
  changes how much §5 formalism vs §6 empiricism leads.
- **How hard to push the theorem:** keep K4 as "stated + pinned + demonstrated on 2
  trust models," explicitly *not* claiming a mechanized proof — matches the repo's own
  honesty and pre-empts reviewer overclaim objections. Keep the induction hedge identical
  everywhere ("argued by induction, Coq/Lean target"), per §5.2/§5.6.
- **CaMeL head-to-head — RESOLVED (no re-run):** we report own-baseline **cost profiles**,
  not a head-to-head; CaMeL's figures are the version-pinned v2 reference, not re-run on
  our harness (a faithful interpreter re-implementation is its own project; an imperfect
  one misleads more). This is fixed by the §6.3 paper-ready methods statement; Q1 is only a
  courtesy confirmation now. (Cost was never the blocker — a full slack run on our harness
  at CaMeL's params is ~$5 on `claude-sonnet-4`; the blocker is re-implementing CaMeL's
  interpreter faithfully.)
- **Pin the CaMeL version — RESOLVED-pending-choice (from the v2 PDF):** v1 (Mar 2025)
  uses **GPT-4o** as a defended backbone at ≈67%; v2 (current arXiv:2503.18813v2) drops
  the GPT-4o backbone for **Claude 4 Sonnet / Gemini 2.5 Flash+Pro / o3 / o4-mini** at
  **≈77%** (v2 references GPT-4o-mini only as a baseline/tokenizer). There is **no shared
  defended model** with our GPT-4o run under v2. Choose one version, cite its table, and
  never write "the model CaMeL measured" except against v1 with v1's numbers.
</content>
</invoke>
