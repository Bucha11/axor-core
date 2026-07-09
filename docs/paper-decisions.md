# Axor paper — decisions log, canon lock & drafting rules

Companion to `docs/paper-outline.md`. The outline is the skeleton of the paper; this
file is the process memory behind it — the canon number lock, resolved decisions with
their rationale, retired numbers that must never reappear, drafting rules, and the open
questions for Haoyu. Nothing here goes into the paper verbatim except where marked.

---

## 0. ACTION ITEMS (blockers before drafting)

1. **Update `examples/agentdojo/agentdojo_results.md` with the new paired runs.** The
   outline now carries slack **−34.1 ± 7.2pp** (paired, n=7, 84.8 → 50.6), workspace
   **−15.5 ± 3.8pp** (paired, n=7, 84.3 → 68.8, denial split 92 taint / 12
   consequence-gate over ~104), and travel **paired n=2 (62.5 → 70.0, 0 denials)** —
   but the results file still records slack as a single run (85.7 → 47.6, −38.1) and
   travel as a single run (65.0 → 65.0), with no workspace cost section. The canon rule
   ("every number traces to `agentdojo_results.md`") is currently violated for these
   three rows. Land the new runs in the results file (with reproduction commands,
   Appendix C) before any §6 prose is drafted.
2. **Consider extending travel to n=7** for symmetry with the other suites. Cheap; its
   0 is structural (zero denials) so the conclusion cannot change, but "7/7/7/2" invites
   a question that "7/7/7/7" doesn't. If not extended, §8 states the n honestly (done).
3. **Pin the missing citations:** the exact nested/recursive-injection follow-up to
   Greshake et al. (§6.5), and the IFC ancestors for the §7 lineage bullet
   (Sabelfeld & Myers JSAC 2003; Zdancewic & Myers robust declassification; DLM).
4. **Build the Appendix E integration-cost table** (per framework: normalizer LOC,
   agent-loop changes = 0, policy changes = 0) — it turns §6.5's adoption claim from
   assertion into data. **Measured LOC (verified 2026-07-09 via tokenize-based count;
   the draft's ~120/~150/~180 were guesses — purge them):**

   | integration | file | raw lines | code lines* |
   |---|---|---|---|
   | Claude normalizer | `tests/normalizers/mock_claude_normalizer.py` | 213 | 181 |
   | LangChain normalizer | `tests/normalizers/mock_langchain.py` | 138 | 113 |
   | OpenAI normalizer | `tests/normalizers/mock_openai_normalizer.py` | 64 | 55 |
   | OpenRouter normalizer | `tests/normalizers/mock_openrouter_normalizer.py` | 71 | 61 |
   | AgentDojo executor shim | `GovernedToolsExecutor`, `examples/agentdojo/agentdojo_adapter.py:293` | 62 | 56 |

   \* code lines = excluding blanks, comments, and docstrings (Python `tokenize`);
   state the counting method in the table caption. Note honestly that the
   per-provider normalizers live in `tests/normalizers/` as the provider-independence
   test shells (§4.3) — the shipped default is `axor_core/policy/normalizer.py`; the
   AgentDojo shim + suite YAML is the one *end-to-end production-shaped* integration
   (it produced all §6 numbers). Rest of the file around the shim
   (`agentdojo_adapter.py` lines 49–292) is LLM-client plumbing for the harness, not
   axor integration — do not count it.

---

## 1. Canon lock — the authoritative numbers

Source of record: `examples/agentdojo/agentdojo_results.md` (after action item 1 lands).
Every number in §1/§2/§6 must trace to it.

**Cost axis (primary, o4-mini, own-baseline paired deltas):**

| suite | undef → gov | cost | n | note |
|---|---|---|---|---|
| banking | 67.9% → 50.9% | **−17.0 ± 7.2pp** | 7 paired | worst single pass −37.5 (not the headline) |
| slack | 84.8% → 50.6% | **−34.1 ± 7.2pp** | 7 paired | supersedes single-run −38.1 |
| workspace | 84.3% → 68.8% | **−15.5 ± 3.8pp** | 7 paired | ~88% shared-channel taint / ~12% `delete` consequence gate (92/12 of ~104 denials) |
| travel | 62.5% → 70.0% | **0 (structural)** | 2 paired | 0 denials/pass; the +7.5pp is sampling noise, never report as a gain |

**Supersession (banking):** gate-level ceiling **+25pp** (deterministic, 4-task
partition {3,4,6,15}); realized **+13.4 ± 9.1pp** (7 paired passes, same population;
task 6 lifts at the gate but the model fails it 14/14 even undefended).

**ASR illustration (never the cost axis):** GPT-4o banking **60.4% → 0.0%**; Qwen slack
**76.2% → 0.0%**; Qwen banking-PII 66.7% → 0.0% (degenerate utility, 0 in both);
claude-haiku banking 0% → 0% (robust model, expected degeneration).

**CaMeL reference (version-pinned, never re-run):** arXiv:2503.18813**v2**, **Table 2
(Difference, no-attack), o4-mini-high row only**: banking **+18.8 ± 1.8**, slack
**−23.8 ± 7.6**, workspace **−7.5 ± 2.9**, travel **+10.0 ± 1.9**, overall −3.1 ± 0.4.
Absolute for the own-baseline caveat: CaMeL native-slack 95.2 (vs our undefended 84.8 —
never compare absolutes).

**Statistics note (do this in the draft):** state once what "±" means (per-pass paired
std over n passes) and n for every interval; a reviewer will compute the CI of
+13.4 ± 9.1 (n=7) themselves — pre-empt with the "wide interval, positive in every
pass" framing.

## 2. Retired / stale numbers — purge on sight

- **slack −38.1pp (single run, 85.7 → 47.6)** — superseded by the paired −34.1 ± 7.2.
- **travel 65.0 → 65.0 single run** — superseded by paired n=2 (62.5 → 70.0, structural 0).
- **"slack = zero cost" (47.6 → 47.6)** — a **Qwen artifact**: Qwen fails the
  shared-channel tasks anyway, so there was nothing to block. Retracted; the partition
  is taxonomy-fixed but its cost scales with model capability. Only travel's 0 is
  structural.
- **"inverted profiles, axor strictly cheaper on slack"** — same Qwen artifact, retracted.
- **54% / 56% utility** (early note) — stale, purge.
- **CaMeL "−27 to −30pp"** — phantom, never existed in the tables.
- **CaMeL cross-model averages ("≈0/+4% banking / −24..−43 slack")** — wrong axis
  (averaging CaMeL over 6 backbones vs axor over o4-mini passes); use the o4-mini-high
  row only. Per-model spread is context only: banking Difference +0.0 (Claude 4) …
  +18.8 (o3/o4-mini) … −12.5 (Gemini Flash); slack −23.8 … −42.9.
- **CaMeL Table 5 defenses-utility row (banking 75.0 / travel 25.0)** — a different,
  weaker model; never cite as o4-mini.
- **CaMeL Table 8's 58.33%** — a policy-trigger rate, not utility; never cite as utility.
- **CaMeL "/75" headline figure** — unconfirmed; drop unless a specific table cell is found.
- **NNSI 62% undefended** — self-refereed, never load-bearing (see drafting rules).
- **"workspace not run for axor"** — stale since the 7-pass workspace study; §6 now
  carries all four suites.

## 3. Resolved decisions (with rationale)

- **Measurement axis (the big one): lead with structural-guarantee-at-utility-cost
  (§6.2), not ASR-delta.** ASR-delta is headroom-dependent and degenerates to 0→0 on
  robust models; the guarantee+cost claim is CaMeL-shaped and survives any model. This
  is the empirical analogue of §5.4's honesty and the real answer to "motivation for
  adoption." ASR-delta (§6.4) is secondary colour.
- **Baseline semantics — option (b), own-baseline cost profiles; CaMeL NOT re-run.**
  A faithful re-implementation of CaMeL's interpreter is a project in itself; an
  imperfect one would mislead more than a version-pinned reported reference. Fixed by
  the §6.3 paper-ready methods statement (use verbatim). Cost was never the blocker
  (~$5 for a slack run); the blocker is faithful interpreter re-implementation.
- **CaMeL version pinned: v2** (arXiv:2503.18813v2). v1 (Mar 2025) used GPT-4o as a
  defended backbone at ≈67%; v2 evaluates Claude 4 Sonnet / Gemini 2.5 Flash+Pro /
  o3-high / o4-mini-high at ≈77% (77 vs 84 undefended) and references GPT-4o-mini only
  as an instruction-hierarchy baseline/tokenizer. **o4-mini(-high) IS the shared
  defended model between our runs and CaMeL v2 — that is why o4-mini is our primary
  cost model and the §6.3 comparison is model-matched.** (This supersedes the earlier
  "no shared defended model" note, which was true only when we ran GPT-4o alone; GPT-4o
  has no v2 counterpart and is axor-only, used for §2/§6.4, never compared to CaMeL.)
  Never write "the model CaMeL measured" except against v1 with v1's numbers.
- **§2 suite choice: banking, resolved by a repo check.** The choice is decided by
  *which axis the denial lands on*, not by drama. Verified against
  `examples/agentdojo/config/`: slack declares only `untrusted_sources`, no
  `sensitive_sources`, and the confidentiality floor arms only on a `sensitive`-rooted
  read (`axor_core/taint/engine.py:101`). So the slack mass-exfil denial is
  integrity/destination-taint (per-value taint on `post_webpage(url)`; whole-blob
  carrier on the message sinks) — confirmed by `agentdojo_results.md` ("denied by the
  per-value taint gate"). Slack therefore carries the *same* encoding/paraphrase
  residual as banking and gains §2 nothing; banking stays the opener (cleanest
  integrity gate-walk). The suite where the floor actually arms is **travel**
  (`config/travel.yaml`, the only one with `sensitive_sources`) — this finding surfaced
  the sound-axis demonstration, now placed as Appendix A-floor (a *property*, not an
  AgentDojo result); travel is the natural second illustration. Not a swap; an addition.
- **Confidentiality floor: placed as structural unit-level (Appendix A-floor), not a §6
  result.** By-construction argument (gate reads a session boolean, never the egress
  bytes — paraphrase-proof by signature) + deterministic illustration (arm → refuse →
  paraphrase-still-refused → control-allows), reproduced live on OpenRouter/GPT-4o
  while running travel. Name the axis asymmetry out loud: integrity = measured,
  confidentiality = proven (§5.5). §6.2 keeps only the one-liner that the stock travel
  slice does not isolate the floor. Optional, low priority: a targeted benign
  read-secret-then-email harness only if a reviewer insists on a floor *cost* number.
- **Novelty arbitration: one headline — the K4 theorem + decidability split (§5.4).**
  §4 (seam) and §6 (empirics) are supporting. An earlier draft let §4 and §5.4 each
  claim "the contribution"; fixed.
- **How hard to push the theorem:** "stated + pinned + demonstrated on 2 trust models,"
  explicitly *not* a mechanized proof. Keep the induction hedge identical everywhere:
  **"argued by pen-and-paper induction over a closed constructor set (a Coq/Lean
  target, not machine-checked)"** (§5.2/§5.6/§8).
- **Enum-supersession placement:** full statement once in §6.3 + Appendix D; §1/§7/§9
  carry one-line references only (it was previously restated in full ~5 times).

## 4. Open decisions

- **Venue framing:** systems-security (USENIX/CCS/S&P) vs an LLM-agent-safety venue —
  changes how much §5 formalism vs §6 empiricism leads. Related: is the §6.5
  "different, composable point on the cost/assurance/integration frontier" framing
  enough for the venue, knowing CaMeL wins raw AgentDojo utility today? (= Q-venue for
  Haoyu.)
- **Section ordering:** Haoyu ranked framework-agnostic #1 for emphasis, but for a
  security venue the theorem leads and §4 sits fourth (see Question 2 below).

## 5. Drafting rules (style & honesty guardrails)

- **Say "CaMeL is ahead on utility" plainly — once, in §6.3** (and the one-line echo in
  §9). Do not re-litigate it in §1/§6.5/§7; repetition turns honesty into
  self-deprecation. Equally: never soften it ("roughly comparable") — −34.1 vs −23.8 is
  CaMeL ahead by ~10pp, say so.
- **Never claim the gate covers "all attacks."** The guarantee covers the
  projection-equivalence (framing) class, with the §5.4 conditionality (enum/numeric
  outright; rich-syntax fuzz-discharged) and the §5.5 integrity paraphrase residual.
- **Do not call the supersession config "tuned."** §6.1 claims the deployed config is
  not benchmark-tuned; the allowlist is a legitimate operator artifact (approved-payee
  list). Use "deployment-allowlist (supersession)" vs "generic."
- **Perimeter non-interference** in the Goguen–Meseguer sense — write it that way every
  time; never reproduce the email garble "non-perimeter inference."
- **Never "the model CaMeL measured"** for GPT-4o (see version pin above).
- **ASR 0% on o4-mini has two distinct causes — attribute both correctly:** undefended
  0% because the model resists the stock injection on its own; governed 0% structurally.
  Never attribute it to "AgentDojo's construction."
- **The two weird-machine bugs (§5.4) are anecdotes** (n=2): offer as consistency with
  the split, never as validation ("as the split predicts" is banned).
- **NNSI:** illustrative stressor only, anchored to the published indirect/nested
  injection class (Greshake et al.); never a bare number, never a reason to adopt.
- **9-pair derailment slice (§6.4 footnote):** report only with the pre-registered
  definition + n=9/wide-CI caveat, as a mechanism illustration — never a bare
  "governance raises utility."
- **Travel +7.5pp:** never report as a utility gain; 0 denials means the gate cannot
  add utility — it is n=2 sampling noise. Travel's result is "structural 0."
- **Workspace −15.5pp:** always state the 92/12 split (shared-channel taint vs
  `delete` consequence gate); the consequence-gate ~12% is an operator-taxonomy choice,
  not the injection defense.
- **Compare own-baseline deltas, never absolute utilities across harnesses** (84.8 ≠
  95.2); verify every CaMeL delta is benign/no-attack (Table 2, not Table 3).
- **§6.3 methods statement is paper-ready — use verbatim** (it lives in the outline).
- **The §6 evaluation ran the synchronous-governor path only — verified in code, not
  from memory.** `examples/agentdojo/run_agentdojo.py:70` builds a `ToolCallGovernor`
  directly; `GovernedToolsExecutor` calls it per tool call. So the benchmark exercised
  the six shared predicates + supersession, and did NOT exercise capability,
  degradation, leases, or the adjudicator (`axor_core/governor.py` docstring: "It
  deliberately does not cover capability/lease/degradation"). Canon was ambiguous on
  this (governance-model §2 listed the sequence without a caller split) — fixed: §2
  now has a "Which steps run where" paragraph (6 shared / 2 caller-owned / 1
  advisory). §6.1 and §8 of the outline state the consequence honestly: ASR=0 does not
  depend on the absent gates (deny-only); the measured utility cost is strictly a
  lower bound for a streaming deployment (expected delta ≈ 0 on this benchmark).
- **"Six structural gates" is the shared pure subset, never the full sequence.** The
  full per-call sequence is nine gates (governance-model §2 / Appendix B): the six
  shared predicates in `policy/gates.py` (consequence, value-policy, SSRF, positional,
  carrier, taint+floor) + two caller-owned state gates (capability, degradation —
  present in `IntentLoop`, deliberately absent from `ToolCallGovernor`) + the optional
  advisory adjudicator. Always present it as 6 + 2 + 1, and scope the anti-drift claim
  to the shared six — the caller-state gates differ between callers *by design*, and a
  reviewer who tests the governor will otherwise call "cannot drift" false.

## 6. Haoyu context (process)

His three asks, mapped: (1) framework-agnostic layer → §4; (2) security model backed by
perimeter non-interference, with the attack → taint → theorem link → §2 + §5;
(3) AgentDojo support + adoption motivation → §6. He asked to "begin with an
illustrative example … to give the reader a sense of what's behind a successful
defense, and how the theorem secures the whole system" (→ §2), flagged the abstraction
between core and frameworks as the interesting design content (→ §4), named his own
single-framework prior himself ("my previous works only implement on a single
framework, either OpenClaw or LangChain" — AgentSpec, arXiv:2503.18666; engage it by
name, a build-on, not a take-down), and wrote "we can only test it using fuzzing" about
K4 — the §5.4 framing (decision procedure on enum/numeric; fuzzing localized to the
rich-syntax fraction) is specifically built to correct that mental model before he
down-rates the claim in his own edits. He values honesty; "currently testable by
fuzzing is acceptable for practical deployment" are his words.

## 7. Questions for Haoyu (one short async email; none is a hard blocker)

1. **Baseline — confirmation, not a fork (we took option b).** We report own-baseline
   cost profiles; CaMeL is a version-pinned reported reference, not re-run (faithful
   interpreter re-implementation is its own project). One-line confirmation: you
   weren't expecting a rival-defense *re-run* on our harness? If you were, that is
   weeks of interpreter work and a separate scoping decision before §6 is drafted.
2. **(cheap) Is framework-agnostic (your point #1) OK at §4, after the security
   sections?** For a security venue the theorem leads; you ranked framework-agnostic #1
   for emphasis, so confirm the ordering rather than have us assume.
3. **(cheap) Do you want the illustrative example *literally first* (before the
   introduction)?** You wrote "begin by an illustrative example." We currently have
   §1 Intro → §2 Example; say the word and we fold the example into the opening of §1.
4. **(new, cheap) Venue check:** is "a different, composable point on the
   cost/assurance/integration frontier" (§6.5) a sufficient contribution framing for
   the target venue, given CaMeL wins raw AgentDojo utility today?
