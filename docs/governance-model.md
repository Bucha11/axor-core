# The Governance Model

This is the plain-language reference for what the kernel actually guarantees and how
it decides. It is written to stand on its own — you do not need any external
specification to read it.

The kernel sits between an agent's *intent* (a tool it wants to call) and the
*execution* of that tool. Every tool call is converted into an intent and run
through a fixed sequence of gates. If any gate denies, the call is refused and the
agent receives a denial instead of a result. Nothing downstream can turn that
denial back into an allow.

---

## 1. The two questions, kept separate

The kernel answers two independent questions about every value, and never lets one
answer leak into the other:

- **Integrity** — *could this value carry an instruction or influence from an
  untrusted source?* (e.g. text fetched from the web, a tool output, another
  agent's reply.)
- **Confidentiality** — *could this value carry a secret we read?* (e.g. a key file,
  credentials.)

These are tracked **per value**, on the value's own provenance, not as one
session-wide "the session is tainted" flag. A clean value stays usable even after
the session has touched untrusted data elsewhere — that precision is the whole
point of per-value tracking.

The two axes are handled **asymmetrically**, on purpose, because their risk shapes
differ:

| | Integrity | Confidentiality |
|---|---|---|
| Trigger | any external read | only a *secret* read (sparse) |
| Mechanism | per-value provenance (precise) | a sound session floor (coarse) + per-value precision |
| Cost | precise, but content-derivation has a known gap (see §7) | cheap (fires only after a secret read) and paraphrase-proof |

---

## 2. The gate sequence (per tool call)

In order. Any denial is final.

1. **Capability** — is this tool permitted by the active policy at all? (read/write/
   bash/search/spawn/extra-allowed.) Out-of-policy tools never reach execution.
2. **Consequence** — how irreversible is this *action class*, regardless of its
   arguments? A `shutdown`, `wipe`, or `restart` is catastrophic even when every
   argument looks innocent and the caller is fully trusted. This is content-blind: it
   reads the kind of call, never the argument text. It catches the destructive
   trusted-provenance action that the provenance axes cannot see (there is nothing to
   taint). Sinks above the policy's "unattended ceiling" require a human/operator
   gate (escalation or a capability lease).
3. **Value policies** — operator-registered predicates on *decidable* arguments: an
   `amount` must be a number in a range; a `target` must be in an allowed set. These
   are checked by a decision procedure, not a guess.
4. **Degradation** — if the session has degraded (see §4), the surface is narrowed;
   a call a quarantined source would drive is refused here.
5. **Positional admission** (for declared sinks) — a sink whose legitimate input
   *cannot encode an instruction* (a closed schema, an enum, a number) is admitted
   **only** when the driving value's *form* is instruction-incomplete. This decision
   reads the form, never the content, so a paraphrase that hides an injection cannot
   defeat it. (See §3.)
6. **Carrier** — an untrusted free-text value reaching an instruction-following sink
   (one that would interpret it as a directive — spawn a sub-agent, send a message,
   run code) is the imperative channel, and is refused. Deterministic and structural.
7. **Per-value taint** — the driving argument's own provenance is consulted:
   - *integrity*: an untrusted-derived value flowing into a high-risk operation
     (write outside the workspace, execute generated code, egress to an external
     destination) is refused;
   - *confidentiality*: egress is refused while the **confidentiality floor** is up
     (see §5).
8. **Adjudicator** (optional) — an advisory second opinion (see §6), consulted only
   on the would-approve path, so it can only *add* a deny.
9. **Execute**, then register the output's provenance for later calls. A value
   produced by an external/web/secret read is recorded with the right labels so a
   later sink carrying it is gated.

A separate, opt-in step (federation) decides the provenance of values that arrive
from *other agents* — see §8.

---

## 3. Positional admission: trading precision for soundness where it matters

Content-based tracking is sound but incomplete: an in-process model can paraphrase
or re-encode an untrusted value so a content match misses it. For a chosen set of
high-stakes sinks, the kernel sidesteps this entirely by **changing the question**.

Instead of "does this argument *look* derived from attacker content?" (undecidable
in general), the gate asks "did this argument arrive in a *form that cannot carry an
instruction*?" — a closed schema, an enumerated value, a bounded number. That is a
structural fact about the value's form, independent of its content, so paraphrase is
irrelevant. A value admitted this way is sound by construction.

The cost: this only works for sinks whose legitimate input *fits* such a form. A
sink that genuinely needs free text (a shell command) cannot be lifted — its input
space is instruction-complete by definition, and the kernel **refuses to let you
declare it positional**. Those sinks stay on content-based tracking, with the known
residual in §7.

Declaring a sink positional is opt-in (`positional_sinks=...`) and asserts that its
trusted side constrains the input to that form.

---

## 4. Degradation: a one-way tightening, driven by facts

Repeated refusals and quarantined sources move the session through a monotone ladder
that narrows what it can do:

| Level | Effect |
|---|---|
| Normal | full policy |
| Cautious | a flagged source is soft-blocked from context |
| Restricted | a quarantined source's writes/bash/export are removed |
| Locked | everything frozen except read + escalate; export forced to the most restrictive |
| Terminal | the session is stopped |

Two properties matter:

- **It only tightens.** The level never decreases on its own. The single way down is
  an explicit governance clearance carrying a verifiable authority — a worker path
  cannot lower it. Clearing below "restricted" genuinely releases the quarantine and
  resets the pressure, so the session returns to a clean state rather than carrying a
  cosmetic lower level.
- **It is driven by decidable facts, not scores.** Transitions come from structural
  booleans (a cross-origin export of an untrusted value, a denied dangerous call),
  not from an accumulating counter or a probabilistic detector. Counters are kept as
  telemetry only.

---

## 5. The confidentiality floor

The moment a *secret* source is read, the session is put under an **egress
restriction**: any call to an external destination is refused — on the **fact of the
read**, not on whether the outgoing value textually contains the secret. This is
deliberately coarse, and it is what makes it *sound*: a paraphrased, base64'd, or
re-structured secret cannot slip past, because the gate never looks at the content
of the egress value.

It is **sparse** — it fires only after a sensitive read, which is rare — so the cost
is low. It is released only by a governance **endorsement** of the specific secret
(or a wholesale clearance); merely "looking clean" never lifts it. A short secret
that is too small to fingerprint still arms the floor, because the floor counts the
read, not a stored fragment.

Per-value confidentiality precision still runs on top (a value that *does* carry a
registered secret is refused even at a non-egress sink), but the floor is the sound
baseline.

---

## 6. The advisory adjudicator

An optional, pluggable second opinion. The kernel ships **no implementation and
makes no model or network calls** — you supply one (a rule engine, an external
policy service, anything) or leave it off.

Three hard guarantees, enforced in the kernel regardless of what you plug in:

- **It sees only the projection** — a content-free summary of the call (kinds, enums,
  bucketed sizes, hashed paths), never raw content. It cannot be steered by the
  governed text.
- **Its verdict is memoized by the projection** — equal projections get equal
  verdicts, so the advice is a deterministic function of the projection and an
  expensive oracle is asked at most once per distinct call shape.
- **It can only tighten.** It is consulted only when the kernel would already
  approve, so it can add a deny but never override a hard deny. If it errors, it
  abstains.

---

## 7. The one acknowledged gap

The integrity per-value gate is content-derivation: it matches the registered
untrusted content inside a sink argument. It is sound in the deny direction but
**incomplete** against an in-process model that paraphrases an untrusted value
before passing it to a sink that *cannot* be made positional (a generic write or
exec). On that partition, a paraphrased untrusted influence is not caught.

This is recorded honestly — there are tests that assert the *sound* behaviour and
are marked expected-to-fail, so the suite trips the moment a sound per-value
interpreter backend closes the gap. The confidentiality floor (§5) and the positional
gate (§3) already close their share; this residual is the integrity, non-liftable
partition only.

---

## 8. Detection is observe-only

Reputation, anomaly, and behavioural-drift signals are **telemetry**. They never
return an allow/deny decision and never feed the gates above. A poisoned reputation
score cannot, by itself, cause an action.

The single, opt-in exception: a reputation reading crossing a registered threshold is
a *decidable fact* (a number compared to a bound), and may **tighten** degradation —
never loosen it, never allow. It is per-tenant isolated: a poisoned score for one
tenant cannot tighten another.

This separation is deliberate: enforcement is driven by structural facts, detection
informs humans and may only ratchet restrictions.

---

## 9. Children and other agents

- **Spawned children** run as their own governed nodes. A child's capabilities can
  never exceed the parent's, and a child shares the parent's degradation state (it
  cannot start below the parent's current level). By default a child's returned
  output is re-minted *untrusted* in the parent — a child cannot launder a value it
  read by returning it.
- **Federated peers** (§ Federation). When you opt in, a value arriving from another
  agent carries a **signed receipt** attesting its provenance. The kernel decides:
  - a valid receipt from an authenticated peer running a compatible kernel in a
    federated domain → **restore** the peer's provenance (trust its labels);
  - an authentic receipt from an incompatible kernel or non-federated domain →
    **re-mint untrusted**;
  - a forged, tampered, or unknown-peer receipt → **reject** the value outright.

  Restoring provenance is the *only* place an external claim can lower a value's
  trust, and it is gated on cryptography plus explicit configuration. With federation
  on, in-process children become same-kernel peers and their actual provenance is
  restored instead of being blanket-untrusted.

---

## 10. Modes

| Mode | Isolation | Policy from task text | On ambiguity | Egress allowlist |
|---|---|---|---|---|
| Library | none (same process) | yes (classifier on) | escalate | optional |
| Production | bypass attempts raise an error | yes | escalate | optional |
| Strict | production + audit-required trace | no — operator sets policy | deny | **required** |

Strict removes content-derived policy decisions entirely and fails closed. It adds
one more obligation: **every declared egress sink must carry a destination
allowlist** (an `enum` value policy on its destination argument). The per-value
taint gate on an egress sink is content-derivation — sound in the deny direction
but with the §7 paraphrase residual; an `enum` allowlist is content-blind and
provenance-independent (membership, not derivation), so it closes that residual.
Strict refuses to construct a session whose egress sink relies on the leaky gate
alone — the misconfiguration fails closed at construction, not at run time. In
Library/Production the allowlist stays optional (the content-derivation gate still
applies); Strict makes the sound control mandatory where it matters most.

---

## 11. The guarantees, in one place

- A denial at any structural gate is final; nothing downstream loosens it.
- Detection cannot cause an allow. It only observes, or (opt-in) tightens degradation.
- Workers cannot clear taint or lower degradation. Only a verifiable governance
  authority can.
- Degradation is monotone; it returns to normal only by governance clearance.
- The carrier and positional gates are deterministic and structural — no model reads
  content to make an enforcement decision.
- The adjudicator and detection see only a content-free projection.
- The confidentiality floor is sound against paraphrase; the integrity per-value gate
  has a documented paraphrase residual (§7) on the non-liftable partition.
- A child cannot exceed its parent or launder a value across the spawn boundary;
  cross-agent trust is only ever raised by a verified signed receipt.
- Unknown sinks fail closed under the high-assurance (strict) ceiling.
- The core has zero required dependencies; everything provider-, transport-, or
  model-specific lives outside it behind a small interface.
- The gates exist as one shared implementation; the streaming path and the
  synchronous `ToolCallGovernor` both call it, so the decision cannot drift between
  them.
- The kernel (the gate logic and the data it reasons over) does not depend on the
  runtime or the platform — enforced in CI, so a budget/context/trace bug cannot
  reach a decision, and the kernel can be imported and used on its own.

---

## 12. Declaring the tool set

The kernel recognises generic tool names on its own, but a real deployment renames
its tools. The operator declares their roles so the kernel can govern them:

- **`untrusted_sources`** — reads whose output can carry injected content (an inbox,
  a web fetch, a document store). Their output is registered untrusted.
- **`sensitive_sources`** — reads of a secret (a credential store). Their output is
  registered untrusted *and* arms the confidentiality floor.
- **`egress_sinks`** — calls that leave the trust boundary (send an email, post to a
  URL, move money). Gated when driven by an untrusted/secret value.
- **`positional_sinks`**, **`value_policies`** — as in §3 and the gate sequence.

A declared role takes precedence over the built-in heuristic; undeclared tools still
get the heuristic. The same declaration is accepted by `GovernedSession` and
`ToolCallGovernor`.
