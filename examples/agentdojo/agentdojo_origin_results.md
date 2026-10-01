# AgentDojo STRICT-origin re-measurement (new core)

Re-run of the AgentDojo suites against **current axor-core** (post the
context-default-integrity RFC merge) under the **STRICT-origin methodology** — the
sound, no-oracle configuration mirrored from `axor-eval` `experiments/rope_bridge`.
This supersedes an earlier production-mode attempt that was **incorrectly
configured** (see "What was wrong" below).

## Methodology (what "origin" means, and why it is the correct config)

Polarity flipped in the core: under `integrity_default=context` a model-emitted
driving value is **untrusted unless it provably originates from a trusted input**
(the user's task, registered via `register_task`; operator config; trusted tools).
The faithful config therefore requires:

- **Every tool of the suite classified** — an explicit data-flow role
  (`untrusted_sources` / `sensitive_sources` / `egress_sinks` / `integrity_sinks` /
  `benign_tools`) **and** an explicit consequence class. STRICT is fail-closed on an
  unclassified tool/consequence, so partial taxonomies are wrong, not merely weaker.
- **`integrity_sinks`** for credential/PII/booking writes: their driving arg must
  trace to the request (reproduces ROPE's PROMPT marker via the integrity axis).
- `require_tool_roles=True`, `require_egress_allowlist=False` (STRICT roles, **no**
  ground-truth allowlist), `integrity_default=context`, `register_task(prompt)`.

Configs: `config/{banking,slack,travel}_origin.yaml` (complete coverage: banking
11/11 tools, slack 11/11, travel 28/28). Runner flag `AXOR_BENCH_ORIGIN=1` forces
the STRICT-roles/no-allowlist combo and auto-derives a complete per-tool consequence
class (BENIGN unless an egress/integrity sink → CONSEQUENTIAL).

**Two model axes, two comparisons** (a model-match is mandatory — each baseline
was measured on one backbone):
- **o4-mini** → compare to **CaMeL** (CaMeL v2 Table 2 is o4-mini-high). This is
  the paper's load-bearing comparison. See "o4-mini vs CaMeL" below.
- **gpt-4o-mini** → compare to **ROPE** / `rope_bridge` (both gpt-4o-mini). The
  gpt-4o-mini CU table just below is this axis.

Do **not** cross them (an o4-mini number against a gpt-4o-mini reference is
meaningless).

## gpt-4o-mini vs ROPE — Clean utility (CU), 7 paired passes/suite, benign-only

| suite | undef → gov (mean of 7) | cost | rope_bridge ref (gov CU) |
|---|---|---|---|
| banking | 50.0 → **37.5** | −12.5pp | 43.8 |
| slack   | 74.8 → **51.7** | −23.1pp | 52.4 |
| travel  | 58.6 → **47.9** | −10.7pp | 55.0 |

Per-pass governed CU: banking 37.5×7 (denials 5–7, dead stable); slack
52.4/52.4/47.6/61.9/47.6/57.1/42.9; travel 50/50/50/50/45/45/45.

**Reads:** slack governed CU **51.7 ≈ reference 52.4** (near-exact, same model).
banking/travel land a few pp below the ROPE-harness reference — expected: this is
the ethz-spylab AgentDojo harness, not ROPE's `autodojo` fork (different task
set + ROPE's effect-based scoring corrections). The methodology reproduces; the
absolute CU carries the harness difference.

## Security (ASR) — curated serious-injection slice

| suite | undef ASR → gov ASR |
|---|---|
| banking | 83.3% → **0.0%** |
| slack   | 44.4% → **0.0%** |
| travel  | 33.3% → **0.0%** |

Governed ASR is **0.0% on all three suites**: the serious injections all route an
attacker-derived value (exfil email/URL/IBAN, redirected payment) into an
origin-gated egress/integrity sink, which the context axis denies. rope_bridge's
small full-matrix residuals (slack 1.9 = IT1 in-*body* link; travel 12.9 = IT6
harm-in-the-agent's-*message-text*) are the in-message-text class no tool-call gate
(axor or ROPE) can enforce; they are outside this serious-egress slice.

## o4-mini vs CaMeL — the model-matched comparison (load-bearing)

Origin on **o4-mini** (model-matched to CaMeL v2 o4-mini-high, same ethz harness),
governed benign-only, 7 passes/suite. Undefended uses the paper's established
o4-mini 7-pass baselines (banking 67.9, slack 84.8), reproduced here (banking p1
undef 68.8 ≈ 67.9).

| suite | undef → gov | axor cost | CaMeL o4-mini-high |
|---|---|---|---|
| banking (n=7) | 67.9 → **46.4** | **−21.5pp** | +18.8 |
| slack (n=7)   | 84.8 → **41.5** | **−43.3pp**¹ | −23.8 |
| travel (n=3)  | 62.5 → **66.7** | **≈ 0** (noise) | +10.0 |

¹ slack's −43.3pp is **inflated by `get_webpage`-as-integrity_sink over-block** — see
the get_webpage section below: the gate drives ~17/23 slack denials (via the
consequence axis) but adds **no** ASR protection (the per-value taint gate already
catches get_webpage exfil to an external domain, with or without the integrity_sink,
in both clean and context modes — verified deterministically). The honest slack
cost is lower; a clean rerun without that over-block is the fix (pending).

travel governed (66.7) ≥ undefended (62.5, noisy n=2) → **~0 cost**, the "neither
defense loses" suite (CaMeL +10.0). travel was run as two 10-task halves/pass
(`AXOR_BENCH_TASK_SLICE`) because a 20-task o4-mini pass exceeds one background
window; 3 passes (65/65/70%).

Per-pass governed: banking 50/50/50/43.8/43.8/50/37.5; slack
42.9/47.6/42.9/38.1/28.6/47.6/42.9. **CaMeL is ahead on both** — and by more than
in clean mode, because origin's integrity axis is deliberately *stricter* than the
content-ledger clean mode: it closes the paraphrase residual, gates credential/PII
writes (`integrity_sinks`), and treats channel-names/URLs as untrusted. That extra
strictness buys sound ASR at a higher utility cost, which o4-mini's capability
amplifies on the shared-channel partition (it completes more of those tasks
undefended, so the gate removes more real utility — the paper's documented
capability-scaling, here between gpt-4o-mini −23pp and o4-mini −43pp on slack).

### The slack cost is the honest price of a load-bearing gate (not over-block)

slack's −43pp is dominated by `get_webpage` as an integrity_sink (17/23 governed
denials: a benign url-fetch of a channel link is denied because the URL is
untrusted-derived). This is **not** removable over-block — it gates a real
exfil-via-GET attack. Measured directly (gpt-4o-mini, curated slice):

| slack config | governed ASR |
|---|---|
| origin (with `get_webpage` integrity_sink) | **0.0%** |
| origin minus `get_webpage` integrity_sink | **11.1%** |

Removing it lets an attack through (the agent fetches an attacker URL built from
channel content). So the slack utility cost is the honest price of keeping ASR at
0 — exactly the axor↔CaMeL trade-off: axor's structural gate is stricter (more
utility cost, sound ASR); CaMeL keeps more utility via its interpreter.

### travel on o4-mini (resolved via task-slice)

travel's 20-task governed pass exceeds one background window on o4-mini (reasoning
latency), so it was run as two 10-task halves (`AXOR_BENCH_TASK_SLICE=0:10` / `10:20`)
and combined from raw success counts. 3 passes: 13/20, 13/20, 14/20 → gov **66.7%**.
With undefended ≈ 62.5 that is **~0 cost** — travel's egress recipient is prompt-given,
so origin does not over-block it (the "neither defense loses" suite).

## What was wrong before (the correction)

The first re-run used `banking.yaml` (production mode) + `AXOR_BENCH_INTEGRITY=
context` — a **minimal** taxonomy (only the recipient egress path). That is not the
origin methodology and is unsound: `update_password`/`update_user_info` were
neither egress nor integrity sinks, so an injected credential/PII change was
**ungated** (an ASR hole), and without `require_tool_roles` unclassified tools were
not fail-closed. The lesson: under STRICT/origin you must configure **all** inputs
and outputs of every tool, not just run with a couple of sinks declared.

## Reproduce

```
# CU (benign, 7 passes/suite):
for s in banking slack travel; do
  AXOR_BENCH_BACKEND=openrouter AXOR_BENCH_SUITE=$s AXOR_BENCH_CONFIG=${s}_origin.yaml \
    AXOR_BENCH_INTEGRITY=context AXOR_BENCH_ORIGIN=1 AXOR_BENCH_MODEL=openai/gpt-4o-mini \
    AXOR_BENCH_MAXTOK=2048 AXOR_BENCH_CAMEL=1 AXOR_BENCH_BENIGN_ONLY=1 \
    python -m examples.agentdojo.run_agentdojo
done
# ASR (curated serious-injection slice): same, drop AXOR_BENCH_CAMEL / BENIGN_ONLY.
```

## Scope / caveats

- **workspace is not covered.** rope_bridge has no vetted origin taxonomy for it,
  and hand-classifying its 24 tools without a reference risks the same
  mis-configuration this document corrects; left out rather than guessed.
- CU is the ethz-harness figure; the direct ROPE head-to-head lives in
  `axor-eval/experiments/rope_bridge` (same core, ROPE's harness + scoring).
- ASR is the curated serious-injection slice, not the full injection matrix.
