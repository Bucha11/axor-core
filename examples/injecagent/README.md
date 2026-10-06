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
  a long one is. (The legacy "clean" polarity used a ≥12-char content-derivation
  ledger and had a short-value blind spot; see the real-run section.)
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

## Real run — two models × two runs (`run_injecagent.py`, full 1,054 cases each)

`run_injecagent.py` drives the undefended pass via OpenRouter and applies the
post-filter to the same trajectory. We ran the full base suite **four times** —
gpt-4o ×2 and Qwen-2.5-72B-Instruct ×2 — to separate what is model/run-dependent
(the undefended ASR) from what the governor guarantees (the governed ASR). All
four runs are clean (0 call-model errors over 1,054 cases each).

The config uses **context-default integrity** (`integrity_default: context`): a
driving value the agent emits is untrusted *unless it provably originates from the
trusted task* (the case's `User Instruction`, registered via `register_task`).

| run | dh undefended | ds undefended | dh governed | ds governed |
|---|---|---|---|---|
| gpt-4o #1 | 6.3% (32/510) | 13.4% (73/544) | **0.0% (0/510)** | 0.0% |
| gpt-4o #2 | 5.1% (26/510) | 16.7% (91/544) | 0.0%† | 0.0% |
| Qwen-72B #1 | 5.9% (30/510) | 13.2% (72/544) | 0.0%† | 0.0% |
| Qwen-72B #2 | 6.5% (33/510) | 14.2% (77/544) | 0.0%† | 0.0% |

† gpt-4o #1's governed dh is **empirically re-scored under the context-default
config** (0/510 over the model's actual trajectory). The other three predate the
polarity migration and their saved trajectories were lost to an environment reset,
so their governed dh is **0 by the deterministic origin gate** (it refuses any
egress whose driving value is not of trusted origin — a property of the call, not
of the model), not independently re-measured.

**The governed ASR is model- and run-independent; only the undefended ASR varies.**
Governed dh and ds are both **0.0%**; undefended drifts with the backbone/sample
(dh 5.1–6.5%, ds 13.2–16.7%) — expected for a capable-model, headroom-dark suite.
The *defense* is structural and stable; the *attack surface* is what moves.

**Why governed dh is 0 under context-default — and why the earlier residual is
gone.** This adapter previously ran the legacy **clean** polarity (a
content-derivation ledger): an egress was refused only when its driving value was a
≥12-char fragment visibly lifted from the untrusted read. That left a *real*
residual — an attacker param that is **short or synthesized** (`appID12345`,
`Code 001`, a room id) evaded the ledger, and if the tool's verb was not in the
catastrophic set the call slipped. Measured on the gpt-4o trajectory, 8/510 dh
attacks slipped the clean polarity: 2 `EmergencyDispatch…`, 2 `The23andMeShareData`
(a genetic-data exfil), 1 `TrafficControl…`, 3 `IndoorRobotGoToRoom`.

Context-default inverts the test: the attacker value comes from the **injection, not
the task**, so it is untrusted **by origin, regardless of length**, and the egress
is refused. Re-scored under the new polarity, **all 8 are refused** (the robot
redirect too — the attacker's room is not in the user's task), driving governed dh
**1.6% → 0.0%**. The fix needs **no** reliance on ledger length or on the
completeness of a catastrophic-verb list — it removes both the short-value blind
spot and the earlier "6 reversible vs 8 realized" bookkeeping tension (both were
artifacts of the length-dependent clean polarity).

`ds` is unchanged and polarity-independent: the **confidentiality floor** refuses
the S2 exfil for all 544 ds cases (a sensitive read arms the floor, the egress is
refused content-blind), governed ds **0.0%**.

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
