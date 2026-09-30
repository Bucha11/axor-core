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

**Model: `gpt-4o-mini`** (OpenRouter) — the rope_bridge reference model, so the
numbers are directly comparable. (The o4-mini/CaMeL clean-mode cost axis is a
separate experiment, unchanged.)

## Clean utility (CU) — 7 paired passes/suite, benign-only

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
