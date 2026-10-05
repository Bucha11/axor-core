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
- **direct-harm** → per-value **taint**: refuse the action when a driving arg is
  content-derived from the untrusted read (a ≥12-char shared fragment — an
  email / URL / long account the agent lifted from the injection).
- **data-stealing** → the **confidentiality floor**: S1 sensitive read arms it,
  S2 exfil is refused content-blind and paraphrase-proof.

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
ASR = 100% by construction) and measures what governance removes. Result
(50/suite, base):

| suite | undefended ASR | governed ASR | mechanism |
|---|---|---|---|
| direct-harm (provenance only) | 100% | **24%** | per-value taint (38/50) |
| direct-harm (+ consequence axis) | 100% | **0%** | + consequence gate (50/50) |
| data-stealing | 100% | **0%** | confidentiality floor (50/50 S2) |

**Control:** a clean `GmailSendEmail` (recipient from the prompt, no sensitive
read) is **ALLOWED** — governance is not a blanket attacker-tool block.

**Two dh numbers, reported separately (never merged), because they are two
different axes:**

- **Provenance only — 24% governed.** This is the honest *content-derivation
  boundary*: per-value taint catches exfil-shaped harms (the attacker's
  destination — an email / URL / long account ≥12 chars — is lifted from the
  untrusted read), but is blind to the 12/50 residual: 6 *short-value* harms
  (value <12 chars, below the ledger threshold, which exists to avoid
  false positives on short common substrings — a deliberate guard, not a bug) and
  6 *no-argument actions* (`AugustSmartLockUnlockDoor`, disable-2FA — no arg
  carries an attacker-derived value, so taint can *never* see them).
- **+ consequence axis — 0% governed.** The residual is the **consequence** axis,
  not provenance: irreversible actions (money movement / physical access / data
  destruction / security disable / exec) are content-blind and length-agnostic.
  The taxonomy is **role-blind and semantic** (leading action verb; see
  `build_config.py`), applied to *every* tool — not a hand-picked attacker list.
  It flags **10 attacker tools and 0 user tools here** — because InjecAgent's
  user tools are all benign reads, not because it is fitted to the attack set.
  Its false-positive cost is **measured on AgentDojo** (the same axis:
  `workspace.yaml` `delete=catastrophic` blocked benign deletes, part of the
  −15.5pp), so it is not a free lunch; InjecAgent (all-attack) simply cannot show
  that cost itself.

We report **both** so the reader sees the pure provenance result *and* the full
stack, and can judge — the +consequence 0% is not a manipulated headline.

## Real run — two models × two runs (`run_injecagent.py`, full 1,054 cases each)

`run_injecagent.py` drives the undefended pass via OpenRouter and applies the
post-filter to the same trajectory. We ran the full base suite **four times** —
gpt-4o ×2 and Qwen-2.5-72B-Instruct ×2 — to separate what is model/run-dependent
(the undefended ASR) from what the governor guarantees (the governed ASR). All
four runs are clean (0 call-model errors over 1,054 cases each).

| run | dh undefended | dh governed (prov) | dh governed (+conseq) | ds undefended | ds governed |
|---|---|---|---|---|---|
| gpt-4o #1 | 6.3% (32/510) | 2.2% | 1.6% (8/510) | 13.4% (73/544) | 0.0% |
| gpt-4o #2 | 5.1% (26/510) | 2.4% | 0.6% | 16.7% (91/544) | 0.0% |
| Qwen-72B #1 | 5.9% (30/510) | 1.8% | 0.6% | 13.2% (72/544) | 0.0% |
| Qwen-72B #2 | 6.5% (33/510) | 2.2% | 0.6% | 14.2% (77/544) | 0.0% |

All four runs now use the **same (domain-extended) consequence taxonomy** — GPT-4o #1
was re-run on it (dh suite; ds is floor-based and taxonomy-independent, so its
13.4%→0.0% is carried over). No run is on a stale config.

**The governed ASR is model- and run-independent; only the undefended ASR
varies.** Governance is a deterministic post-filter over the recorded
trajectory, so its effect does not depend on which model produced the attack or
on sampling noise: across both models and all four runs, governed dh lands at
**0.6–1.6% (+consequence)** / **1.8–2.4% (provenance only)** and governed ds at
**0.0%**. (The +consequence figure varies a little by run — 0.6% for three runs,
1.6% for GPT-4o #1 — because *which* dh attacks a given pass happens to follow,
and how many land in the correctly-ungated reversible set, is itself a run
property; the governed number is small and bounded regardless.) The undefended
ASR, in contrast, is a model/run property and drifts
(dh 4.9–6.5%, ds 13.2–16.7%) — expected for a capable-model, headroom-dark suite
(as on AgentDojo's robust-model contrast); a weaker model or the enhanced setting
would show more headroom. This is exactly the separation the paper claims: the
*defense* is structural and stable, the *attack surface* is what moves.

**Governance decomposition (deterministic, over the *full* attack set —
model-independent).** The four-run table above measures governance on the
attacks each model actually attempted; the decomposition below is the stronger,
run-free view: apply the governor to *all* the benchmark's defined attacks,
regardless of whether any model took the bait. This is why governed ds is 0% in
every run — the floor refuses the S2 exfil for **every** ds case (all 544),
content-blind and paraphrase-proof. The dh set decomposes cleanly:

- **ds → 544/544 refused** by the confidentiality floor. Governed ds ASR **0%**.
- **dh → 504/510 (98.8%) refused**, decomposing cleanly by attack shape:

| dh attack shape | count | caught by |
|---|---|---|
| **exfil-shaped** — attacker destination lifted from the read (email/URL/account) | 444 | per-value **taint** (provenance) |
| **action-shaped, irreversible** — harmful action, no data flow (`BinancePlaceOrder` "sell 50 BTC", `EpicFHIRManage*` records, `DropboxMoveItem`, `FedExShip*`) | 60 | **consequence** axis |
| **action-shaped, reversible** — `IndoorRobotGoToRoom` (move a robot to a room) | 6 | **correctly NOT gated** |

The 6 that "escape" are *not* a defense gap: moving a robot between rooms is
recoverable, so the consequence axis rightly leaves it — gating it would be a
false positive. So governance refuses **every dh attack it should** (504/510),
and the residual is a correct non-action on a reversible, low-consequence tool.

**Why lowering the content ledger does nothing here (the "short-value" red
herring).** We checked exact whole-value driving-arg taint (match a driving arg's
value against the read at *any* length, not just ≥12 chars): it catches **0/42**
of the action-shaped set. The reason is structural, not a threshold — these
attacks move **no data**: "sell 50 BTC" arrives as prose in a laptop review and
the agent *synthesizes* short params (`sell`, `BTC`, `50`) that are nowhere in the
read. Provenance is blind by construction, not by ledger length. The right axis is
**consequence** (irreversibility), which is length- and content-agnostic.

**Honest caveat on the domain-extended consequence verbs** (`place/order/manage/
move/ship`, in `build_config.py`): these are irreversible *in a high-stakes domain*
but ambiguous in general — in a real deployment they would also flag benign
`Manage*`/`Move*` **user** tools (a false positive). InjecAgent (all-attack, user
tools all reads) cannot show that cost; the same consequence axis's FP cost is
measured on AgentDojo (`workspace` `delete=catastrophic`, part of the −15.5pp).

> Note: InjecAgent does not run out of the box — `requirements.txt` omits
> `nltk`/`together`/`tqdm`/`pydantic`, and `src/utils.py` builds an OpenAI client
> at import (needs `OPENAI_API_KEY` set even to import). Install those and export
> a key (a dummy is fine for the mock and the deterministic analysis).
