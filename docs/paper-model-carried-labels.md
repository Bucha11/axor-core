# The producer obligation is load-bearing — a measured negative control

**Chapter draft.** Written to attach to `docs/paper-outline.md` as **5.3b** (immediately
after the three structural obligations, where T0 is stated), with its measurement slice
feeding **6.8** and hooks in **§7 Related work** and **§8 Limitations**. Source of the
measurements: the marker study (`PAPER_DRAFT.md`, internal; experiments
`axor-eval/experiments/attacks/marker_*.py`, OpenRouter, `temperature=1.0`, single-turn
forced-sink harness). Nothing here changes the central claim or adds a mechanism to the
kernel; it supplies the evidence that one premise of the claim is not decorative.

---

## 5.3b.1 Why a chapter about a mechanism we deliberately do not ship

The thesis of the paper is that the **admissibility conditions on a projection and on
its consumption** are what keep the model outside the trusted base of the guarantee. Two
premises carry that:

- **T0 — non-interpreting producer.** Every projection on the trusted path is produced by
  a deterministic structural function, never by a model reading the governed content
  (`docs/kernel-theorem.md §3`; pinned by the `t0-producers-non-interpreting` import
  contract and `tests/invariants/test_t0_producers_non_interpreting.py`).
- **T4 — codomain fidelity.** The check must correspond to how the value is consumed; this
  is where the decidability split lives (§5.4).

A reviewer is entitled to ask whether T0 is load-bearing or merely stated. Answering it
requires an instance in which the codomain is impeccable, the consumption is a trivial
case-split, and the guarantee *still* fails — failing for the producer and nothing else.
We have such an instance, we measured it, and the result is negative. Reporting it is
what makes T0 an obligation rather than a preference.

## 5.3b.2 The construction, restated as a projection

Datamarking (Hines et al., arXiv:2403.14720) interleaves a sentinel character between the
words of untrusted text. It was proposed as a *prompting hint*: marked text makes injected
instructions easier for the model to ignore. The marker study repurposes it as a
**provenance label read by an enforcement gate** — the host marks every untrusted read, a
gate at the tool-call boundary denies any argument still carrying the sentinel, and the
sentinel is trimmed before execution.

In the vocabulary of §5.4, that gate reads the projection

```
π_mark(x) = marked | unmarked          (a two-element enum, consumed as a case-split)
```

which is **admissible and decidable by construction**: the codomain is finite, it admits
no instruction, and the consumer compares rather than re-parses. By T4 alone this layer
looks ideal. Its producer, however, is the model: the sentinel reaches the sink argument
only if the model reproduces it. That is the whole of the finding, and it is why this case
is worth a chapter — **it separates the two premises.** T4 and T0 are independent, and a
flawless codomain does not rescue a projection whose producer is steerable.

## 5.3b.3 What the measurement shows

**Carriage is a property of the model's instruction hierarchy, not of the marker.**
Without a host instruction the sentinel never survives a copy (0/80 — the model strips it
unprompted). With one, three frontier models carry it 110/110, *including* when the
untrusted content itself names the sentinel and demands its removal: the system
instruction beats the tool-content instruction. A fourth model does not carry at all. The
layer is therefore only as present as the model is willing to carry it — and that
willingness is exactly what an attacker contests.

**Coverage is real, and it is coverage of the ledger's benign gap.** Measured where the
model *reformats* a value because the user's own task demanded it (not under attack), the
content-derivation ledger (`axor_core/taint/ledger.py`) misses 62/82, and on identifier
reformatting it is a total miss (0/12 compact IBAN, 0/12 dashed, 0/19 basename, 0/18 split
field) — the documented re-encoding residual of §5.5. The sentinel survives those
transformations structurally, closing 58% of that gap overall and 84% on the models that
carry. The relation is complementary rather than inclusive: on paraphrase the ledger hits
20/20 while the sentinel reaches ~30%.

**Granularity is a per-value-class engineering choice.** Word marking puts the sentinel on
a value's edges, so a sub-part the model extracts and re-types carries nothing; marking at
the identifier's internal delimiters fixes that, and marking *inside* digit groups is
strongly worse.

| value class | mode | marker survives | ledger |
|---|---|---|---|
| delimited identifier | `word` | 62% (73/118) | gap 76% closed |
| delimited identifier | `struct` | 77% (94/122) | gap 90% closed |
| grouped numeric | `word` | 85% (52/61) | 0/61 |
| grouped numeric | `chunk2` | 34% (18/53) | 0/53 |

`word → struct` is +15pp pooled, +25pp [+11, +40], p=0.002 on the scenarios where the
markup actually differs; `word → chunk2` is −51pp [−67, −36], p<0.001.

**The static adversarial result measures the wrong thing, and the search measures the
right one.** Eleven hand-written injections produce 0 bypasses / 277 — but the models
declined the injected formatting instructions wholesale, including harmless ones, so that
number characterises the instruction hierarchy, not the gate. It is the static reading
that arXiv:2510.09023 warns against. An LLM attacker scored on live gate feedback breaks
the preserve-level defence **in round one**, with three of four candidates at 100% bypass.

## 5.3b.4 The bypass is not a marking bug; it is the polarity of the ground

The successful attack never strips the sentinel. It places the target value in the content
*encoded* — JSON `\uXXXX` escapes, the string reversed, ROT13 with a digit shift, hex ASCII
octets — and instructs the model to *derive* the plaintext. The derived string never
appeared in the tool output, so no sentinel was ever on it and the ledger has no fragment
to match. One move, both gates silent, and the model's perfect carriage fidelity is
irrelevant because it *computed* the value rather than copying it.

Stated as a ground for admission, this is the familiar shape:

```
ground for ALLOW  =  the sentinel is ABSENT from this argument
producer of that ground  =  the model
default for an unrecognised surface form  =  clean
```

Marking labels the bytes that are **present**. A value that is merely *derivable* from the
content carries no mark by construction, and an LLM is a general-purpose decoder, so the
space of encodings is unbounded. This is the **same polarity failure** as
content-derivation taint's (§5.5, `docs/rfc-integrity-context-default.md §1`), relocated
rather than repaired, and it needs no privileged access — a phishing author writes their
account number in hex.

## 5.3b.5 The rule the architecture takes from this

The chapter's architectural output is one line, and it is the polarity rule the rest of
the system already obeys:

> **The presence of a model-carried mark may DENY. Its absence may never PERMIT.**

The deny direction is sound: a sentinel on an argument is positive evidence that the value
passed through an untrusted read, and evidence produced by the model *against* the call it
is making costs the attacker nothing to suppress — which is precisely why it cannot be
relied on, and precisely why relying on it only to deny is safe. The admissible placement
is therefore the detection layer, which in this architecture is observe-only and may at
most tighten degradation (`docs/governance-model.md §8`) — never a positive ground:

- not in integrity supersession (`integrity_superseded_by_decidable`,
  `axor_core/policy/gates.py`), whose soundness condition is a *closed operator-declared*
  codomain — the operator declares the enum; nobody declares the absence of a mark;
- not in the closed-codomain classifier (`closed_allowlist_args`,
  `axor_core/policy/value_policy.py`);
- not as a trusted origin (`TrustedOrigin`, `axor_core/contracts/taint.py`).

It is the ⚠️ row this layer would occupy in the audit of permitting inputs
(`docs/kernel-theorem.md §3b`): *ground — absence of a mark; producer — the model;
verdict — inadmissible as a positive ground.*

One implementation detail is worth stating because it is load-bearing and non-obvious: the
construction requires the sentinel to be **trimmed before execution**, which is a payload
rewrite. A rewrite is only sound if every gate ran on the post-rewrite arguments; the loop
resolves a `TRANSFORM` decision before normalization for exactly this reason
(`tests/node/test_transform_is_gated.py`). Without that ordering the marking layer cannot
be expressed soundly at all, whatever its polarity.

## 5.3b.6 The stronger host instruction is a regression, not a repair

The study also tests a stronger instruction that additionally governs values obtained *by
an operation* — decode, reverse, arithmetic, concatenation, reformatting — requiring the
model to wrap the result itself. Against the four found vectors it works: pooled bypass
52% → 0%, and 0/21 on the model where every bypass had been found, with the model wrapping
the value it computed (`•gb29nwbk60161331926819•`).

Under the producer obligation this is not an improvement at a price; it is a move in the
wrong direction. The instruction asks the model to **be the taint tracker**: to recognise
that a value it derived descends from untrusted content, and to label it accordingly. Even
where it succeeds it enlarges the model's share of the decision, trading a *measured*
failure for an *unmeasured dependence*. Two further facts bound the result: the vectors
were optimised against the weaker instruction (an old attack on a new defence), and the
stronger instruction **disabled** one of the four models, which then declined the sink in
29/30 cells — on the benign control as well as the attacks. A search against the stronger
instruction reached 4 of 8 planned rounds with 0 bypasses and the sentinel never moving;
it stopped because the attacker model was safety-flagged, not because the defence held. So
the honest status is "not broken in four rounds" — above the static number, below a
completed search, **not proven to hold**.

## 5.3b.7 Dominance, argued structurally rather than empirically

Context-default integrity inverts the polarity: a value the model generates is denied at a
high-stakes sink unless it provably originates from a trusted source
(`docs/rfc-integrity-context-default.md`; `integrity_default="context"`, the default under
Strict). The producer of *that* default is the host, not the model, and a computed value is
not of trusted origin — so the decode family is denied regardless of encoding, because the
gate never asks what surface form the value has.

The marking layer is therefore **strictly weaker** than the default it is sometimes offered
as an alternative to, and the comparison does not rest on the search result. It follows
from polarity and producer: a blocklist of surface forms whose producer is the adversary's
own instrument cannot dominate an allowlist of origins whose producer is the host. The
measured round-one bypass confirms the ordering; it is not what establishes it.

## 5.3b.8 What the result is worth, scoped

- **Positive and reproducible:** the provenance-read framing (asking whether the mark
  survives onto the *tool argument*, a question the original work never evaluates, since
  its metric inspects the completion for a canary keyword), and the per-value-class
  granularity rule — coarsest marking that still marks the pieces the model keeps:
  `struct` for delimited identifiers, `word` for grouped numerics, nothing for prose.
  These are **recall** results for a host-side ledger on honest reformatting, plus a
  channel canary (verify at runtime that a known marked value returns marked before
  trusting any absence). They are not a control.
- **Negative:** the security claim. The layer cannot carry an enforcement decision.
- **Not a contribution to this paper's thesis:** the granularity numbers. They belong to
  the marker study. Importing them here as a defence would repeat the error this paper
  argues against — promoting a supporting mechanism to a central one.

## 5.3b.9 Availability is not free either

If presence denies, an attacker can sprinkle sentinels into content they author and cause
denials on honest values that happen to pass through it. Availability is outside the safety
perimeter of the claim (`docs/kernel-theorem.md §6`), so this is not a counterexample — but
it must be stated, because it means the sound direction of a model-carried mark has a cost
of its own, and a deployment that wires marking into degradation inherits an
attacker-triggerable tightening lever.

## 5.3b.10 What would overturn the conclusion

- A **completed** adaptive search against the strong instruction (rounds 5–8 need an
  attacker model that is not safety-gated). A 0-bypass completed search would raise the
  empirical standing of the mechanism — it would not change the polarity argument of
  §5.3b.7.
- A marking scheme whose mark is **reconstructible by the host** rather than carried by the
  model — e.g. a label the host re-derives from the value at the sink. That changes the
  producer, and with it the entire analysis; it would no longer be this mechanism.
- Multi-hop evidence (context compression, sub-agent spawn), where each hop is a further
  chance to lose the mark, and where the measured carriage numbers do not apply.

## 5.3b.11 Placement and required artifacts

| Target | Insert |
|---|---|
| `paper-outline.md` **5.3b** | this chapter, compressed to ~½ page: the separation of T0 from T4, the round-one bypass, the polarity rule |
| `paper-outline.md` **6.8** | the measurement slice (carriage, coverage, granularity, search) as a negative-control experiment |
| `paper-outline.md` **§7** | Spotlighting/datamarking as prompting hint vs read-at-the-gate; "The Attacker Moves Second" (arXiv:2510.09023) as the adaptive-search result our bypass instantiates on a provenance gate |
| `paper-outline.md` **§8** | the strong instruction is not adversarially settled; four hosted models, English, single-turn |
| `docs/kernel-theorem.md §3` | one line: T0's load-bearing status has a measured counterexample, not only the Firewalls LLM-projector |
| `docs/kernel-theorem.md §3b` | the ⚠️ row (absence of a mark → producer is the model → inadmissible) |
| `docs/governance-model.md §8` | the polarity rule, as the admissible placement for any model-carried signal |
| `docs/rfc-integrity-context-default.md` | the decode vectors as a **two-sided corpus**: they must be DENIED under `integrity_default="context"` and are ALLOWED under `clean` |

**Artifact gap, to fix before circulation.** The chapter and the marker study both cite
`axor-eval/experiments/attacks/marker_*.py`; that directory currently holds
`drift_curve.py`, `nnsi_openrouter.py` and `split_doc_openrouter.py` only. The marker
harness is either uncommitted or lives at another path, and the raw JSONL is withheld by
design — so every figure above is presently unreproducible from the repository. The
two-sided decode corpus proposed for the RFC is the one piece of this work that belongs in
`axor-core`'s own regression suite rather than in the eval repo, because its expected
verdicts are a property of the kernel's integrity default, not of a model.
