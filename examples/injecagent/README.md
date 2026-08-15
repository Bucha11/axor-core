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

## Real run (needs a model + key)

The mock proves the post-filter; a real ASR number needs a susceptible model
(headroom) run undefended once, then the same post-filter. `run_injecagent.py`
(TODO, after a budget top-up) will drive InjecAgent's own harness for the
undefended pass and apply `govern_injecagent` for the governed column.

> Note: InjecAgent does not run out of the box — `requirements.txt` omits
> `nltk`/`together`/`tqdm`/`pydantic`, and `src/utils.py` builds an OpenAI client
> at import (needs `OPENAI_API_KEY` set even to import). Install those and export
> a key (dummy is fine for the mock).
