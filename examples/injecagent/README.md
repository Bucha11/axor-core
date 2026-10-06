# InjecAgent × axor — governance adapter

A second, independent indirect-prompt-injection benchmark
([InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent), Zhan et al.,
arXiv:2403.02691) wired to axor governance, mirroring the AgentDojo adapter. It
exists to test external validity: the same structural gate, a different harness.

## What InjecAgent is (and how it differs from AgentDojo)

InjecAgent is a **pure attack** benchmark — 1,054 cases (510 direct-harm +
544 data-stealing), 17 user tools, 63 attacker tools. Every case embeds an
injected instruction in a tool's response; the metric is **ASR** (did the agent
call the attacker's tool). There are **no benign tasks**, so there is no benign
utility to over-block — the taxonomy can drive taint on *all* egress params
without a false-positive budget to protect.

## Governance mapping (data-derived, `build_config.py`)

| InjecAgent role | axor role |
|---|---|
| user tool (its response carries the injection) | `untrusted_source` |
| direct-harm action `AttackerTools[0]` (dh) | `egress_sink` |
| exfil step `AttackerTools[1]` (ds S2 — always `GmailSendEmail`) | `egress_sink` |
| sensitive read `AttackerTools[0]` (ds S1) | `sensitive_source` |
| every parameter of each egress sink | `driving_args` |

Two mechanisms, one per attack shape:
- **direct-harm** → **context-default integrity** (`integrity_default: context`):
  the benign task (`User Instruction`) is registered as the trusted origin, and the
  egress is refused when a driving arg is **not of trusted origin** — i.e. the value
  came from the injection rather than the task. This is origin-based, not
  length-based: a short/synthesized attacker param (`appID12345`) is refused just as
  a long one is. (The other live `integrity_default` knob, `clean`, instead tests
  content-derivation with a ≥12-char ledger — length-dependent — and is not the
  config reported here.)
- **data-stealing** → the **confidentiality floor**: S1 sensitive read arms it,
  S2 exfil is refused content-blind and paraphrase-proof (polarity-independent).

## Paired-by-construction, cheap

The model runs **once** (undefended). Governance is a **deterministic
post-filter** replaying the trajectory through a `ToolCallGovernor`
(`govern_injecagent.py`), so undefended and governed ASR come from the *same*
outputs — the ideal paired design at zero extra API cost.

## Mock pilot (no API) — mechanism validation

```sh
python3 examples/injecagent/build_config.py      # regenerate config/injecagent.yaml
python3 examples/injecagent/govern_injecagent.py 50
```

The mock fabricates the "agent followed the injection" trajectory (undefended
ASR = 100% by construction) and measures what governance removes. Under
context-default the attacker's egress value is not of trusted origin, so the
origin gate refuses it: governed dh → **0%**, and data-stealing → **0%** via the
confidentiality floor.

**Control:** a clean `GmailSendEmail` whose recipient is a verbatim span of the
registered task (trusted origin), with no sensitive read, is **ALLOWED** —
governance is origin-based, not a blanket attacker-tool block.

The authoritative numbers are the real run below (context-default, governed
dh/ds = 0). The earlier content-ledger mock split dh into "provenance 24% /
+consequence 0%"; that two-number framing belonged to the legacy **clean**
polarity (length-dependent) and is superseded — context-default refuses the
attacker egress by origin regardless of value length, so there is one number: 0.

## Real run — two models × two runs (`run_injecagent.py`, dh base = 510 cases each)

`run_injecagent.py` drives the undefended pass via OpenRouter and applies the
governance post-filter to the *same* trajectory (paired by construction, no extra
API cost). We ran the direct-harm (dh) base suite **four times** — gpt-4o ×2 and
Qwen-2.5-72B-Instruct ×2 — all under **context-default integrity**
(`integrity_default: context`), to separate what is model/run-dependent (the
undefended ASR) from what the governor guarantees (the governed ASR). **Every row
below is empirically measured on that run's own trajectory** — no row is asserted
"by construction". All four runs are clean (0 call-model errors over 510 cases each).

Context-default means a driving value the agent emits is untrusted *unless it
provably originates from the trusted task* (the case's `User Instruction`, registered
via `register_task`).

| run | dh undefended ASR | dh governed ASR |
|---|---|---|
| gpt-4o #1   | 6.3% (32/510) | **0.0% (0/510)** |
| gpt-4o #2   | 4.7% (24/510) | **0.0% (0/510)** |
| Qwen-72B #1 | 5.5% (28/510) | **0.0% (0/510)** |
| Qwen-72B #2 | 5.5% (28/510) | **0.0% (0/510)** |

**The governed ASR is model- and run-independent (0.0% in every run); only the
undefended ASR drifts** with the backbone/sample (4.7–6.3%) — expected for a
capable-model, headroom-dark suite. The *defense* is structural and stable; the
*attack surface* is what moves. On each run's realized successes the **origin axis
alone** already refuses every one (governed-provenance = governed-+consequence = 0);
the consequence axis's independent contribution shows up on the full attack set
(below), not on these particular realized misses.

**Data-stealing (ds).** ds is governed by the **confidentiality floor**, which is
model- and polarity-independent. Replaying all 544 ds base cases (the S1 sensitive
read arms the floor, the S2 exfil `GmailSendEmail` is the target), the floor refuses
S2 in **544/544** — governed ds = **0.0%**, computed deterministically over the
S1→S2 call sequence. This is a property of the calls, not of any model trajectory,
so it is not re-sampled per run (and was not part of this dh re-run pass).

**Why governed dh is 0 under context-default.** The attacker's driving value comes
from the **injection, not the task**, so context-default marks it untrusted **by
origin, regardless of length** (a short/synthesized param — `appID12345`, `Code 001`,
a room id — is refused just as a long one is), and the egress is refused. This needs
**no** reliance on fragment length or on the completeness of a catastrophic-verb list.
(The other live knob, `integrity_default: clean`, instead tests content-derivation
with a ≥12-char ledger; it is length-dependent and is not the config this adapter
reports — see the mapping section above.)

**Both axes are load-bearing — consequence catches what the origin axis cannot.**
The consequence gate runs *before* the origin/taint gate, so it is not masked by it.
Replaying the governor over all 510 dh base attacks (each attacker call carrying an
injection-derived driving value) and asking which axis is *necessary*: origin denies
**444/510**, and a further **60 are consequence-only** — the consequence gate denies
them and the origin axis *alone* (consequence disabled) would **allow** them
(origin-only denials: **0**). These 60 are action-shaped, content-blind harms —
`AugustSmartLockUnlockDoor`, `…GrantGuestAccess`, and the like: an irreversible
physical/security action whose driving argument carries no attacker-authored value
for the origin axis to flag, so only the action-class gate sees it (length- and
content-agnostic). This is the direct, measurable answer to the "action-shaped harm
with no data flow" class (cf. PACT): origin provenance handles the exfil-shaped
attacks, the consequence axis the irreversible-action ones, and 60/510 fall only to
the latter.

**Honest caveat.** InjecAgent is all-attack, so every denial is correct and it
cannot show context-default's *over-block* cost (legitimate egress whose value is
model-synthesized rather than a verbatim task span). That false-positive cost is
measured on AgentDojo, where the same origin polarity pays a real utility price
(e.g. the `*_origin.yaml` suites: banking −21.5pp, slack −43.3pp on o4-mini;
`examples/agentdojo/agentdojo_origin_results.md`).

> Note: InjecAgent does not run out of the box — `requirements.txt` omits
> `nltk`/`together`/`tqdm`/`pydantic`, and `src/utils.py` builds an OpenAI client
> at import (needs `OPENAI_API_KEY` set even to import). Install those and export
> a key (a dummy is fine for the mock and the deterministic analysis).
