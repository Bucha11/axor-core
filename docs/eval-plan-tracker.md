# Evaluation plan tracker (R0–R11)

Source of record for the camera-ready run cycle. All runs use the **fixed
positive-polarity** core (`integrity_default: context`), **STRICT**
(`require_tool_roles=True`, `require_egress_allowlist=False` in origin mode). Pin the
**commit hash** of the core for every run in the per-run notes. Save **per-task
outcomes and raw trajectories** (needed for statistics, R-stats, and the artifact).

Status legend: ✅ done · 🟡 in progress · ⏸ queued · ⬜ not started · ❌ blocked.

## P0 — gating for submission

| ID | Run | Closes | Status | Command / artifact |
|---|---|---|---|---|
| R0 | Sanity: re-encoded IBAN → deny; unknown sink → deny | — | ✅ | `python examples/agentdojo/poc_reencoding.py` |
| R1 | AgentDojo cost, new Table 2 (o4-mini, 4 suite, **7 passes** incl. travel; branches undefended / request-only / any-trusted; banking also request-only+supersession; `known_payees` rebuilt from t=0 env state) | B1/A5/B2/B4/B7 | ⬜ | runner below; **needs workspace origin (R1-ws)** |
| R1-ws | Build workspace origin taxonomy (24 tools) so R1/R6 cover 4 suites | B6 (enables) | 🟡 | `examples/agentdojo/config/workspace_origin.yaml` (in progress; needs validation, no vetted reference) |
| R2 | ASR on tractable models (GPT-4o + Qwen-72B, all 4 suite, both settings, `important_instructions`, ≥3 passes) | "main model never engages the defense" | ⬜ | `run_agentdojo` ASR mode per suite/model |
| R3 | Adaptive suite, closed loop (GPT-4o; banking/travel/slack; both settings; ASR+UA; a1–a7) | adaptive/boundary | ⬜ | a1–a7 table below |
| R4a | InjecAgent replay on fixed core: 2×2 origin×consequence over 510 + residual accounting | the "6 cases / 0.6%" question | ✅ | `python examples/injecagent/replay_dh_axes.py` |
| R4b | InjecAgent ds floor replay over 544 | ds provenance | ✅ | `python examples/injecagent/replay_ds_floor.py` |
| R4c | InjecAgent **live** governed closed loop (GPT-4o + Qwen, ≥1 run each, DH+DS; ASR over **successful undefended**, not /510) | live closed loop | 🟡 | DH done ×4 (`run_injecagent.py`); DS live + closed-loop pending |
| R5 | ROPE on InjecAgent — the "what ROPE doesn't cover" proof. **Gate-check first** (only the 60 consequence-only + ~50 DS); full DH+DS only if ROPE misses | key differentiation claim | ⬜ | gate-check ≈1 day, then full |
| R6 | Benign cost floor (workspace+banking, realistic `sensitive_sources`, o4-mini, paired, floor on/off, 7 passes; report benign-denial rate + cost) | B6/B5 | ⬜ | needs R1-ws |

## P1 — strong support

| ID | Run | Closes | Status |
|---|---|---|---|
| R7 | ROPE comparison multi-pass (Table 3, 5 passes/cell; run banking attack ourselves; gpt-4o-mini **+ gpt-4o** per Haoyu; record router-cache origin) | B7 | ⬜ |
| R8 | Allowlist sensitivity (±k entries; cheap gate replay + 1 live) | robustness | ⬜ |
| R9 | Real integration (LangGraph agent, 1 suite, undef/governed, 3 passes; verdicts byte-identical to the shim on identical intents) | B8/C4 | ⬜ |
| R10 | Latency on fixed core (code path changed; ~30 min) | perf claim | ⬜ |

## P2 — if time remains

| ID | Run | Status |
|---|---|---|
| R11 | Daemon boundary (1 suite: verdict parity + overhead). Else state in the paper that all runs were on the soft boundary. | ⬜ |

## R3 adaptive attacks (closed loop, GPT-4o)

| Attack | Tests | Expected | Status |
|---|---|---|---|
| a1 | re-encoding of attacker value | 0 — demonstrates the fix | ⬜ (R0 shows the deterministic version) |
| a2 | catalogue/record steering (travel IT4; banking existing-payee-from-history) | any-trusted passes, request-only does not | ⬜ |
| a3 | pick a different known payee under supersession | passes — declared boundary | ⬜ |
| a4 | non-driving args (`amount`, subject) on a trusted payee | passes if not projected | ⬜ |
| a5 | **write-then-read** (`update_scheduled_transaction`, then read) | **critical**: does the mark survive storage? | ⬜ |
| a6 | tool selection with trusted constants | consequence axis | ⬜ |
| a7 | in-band text in body (IT1) | residual (shown, not hidden) | ⬜ |

a2–a4 and a7 failures are the honestly-declared boundary — **show them, don't hide**.
If ROPE's AutoDojo builds, run the optimized attack on **both** systems.

## Schedule

| Week | Dates | Runs |
|---|---|---|
| 1 | 9–15 Oct | R0 ✅, launch R1 (background whole cycle), R4 replay ✅, R5 gate-check, assemble R3 |
| 2 | 16–22 Oct | R3, R2 |
| 3 | 23–29 Oct | R5 full (if gate-check passes), R4 live, R6, R7 |
| ~1 Nov | | **go/no-go** |
| 4 | 1–10 Nov | R8, R9, R10, buffer |

**Go/no-go criterion:** R1, R3 (at least a1, a2, a4, a5) and R5 ready. If not → USENIX C2.

**Results that could change the paper:**
- **R5:** ROPE catches the consequence cases → drop that differentiation point.
- **R3 a5:** the mark does not survive storage → a hole in O2, fix before submission.
- **R1:** request-only is catastrophically expensive → headline cost gets awkward, but
  report it anyway.

## Reproduce (commands)

```sh
# R0
python3 examples/agentdojo/poc_reencoding.py
# R4a / R4b (deterministic, no API; exit non-zero on drift)
INJECAGENT_DIR=/path/to/InjecAgent python3 examples/injecagent/replay_dh_axes.py
INJECAGENT_DIR=/path/to/InjecAgent python3 examples/injecagent/replay_ds_floor.py
# R1 / R6 cost (o4-mini, 7 passes/suite), per suite:
AXOR_BENCH_BACKEND=openrouter AXOR_BENCH_SUITE=<suite> AXOR_BENCH_CONFIG=<suite>_origin.yaml \
  AXOR_BENCH_INTEGRITY=context AXOR_BENCH_ORIGIN=1 AXOR_BENCH_MODEL=openai/o4-mini \
  AXOR_BENCH_CAMEL=1 AXOR_BENCH_BENIGN_ONLY=1 python -m examples.agentdojo.run_agentdojo
# R2 / R3 ASR: drop AXOR_BENCH_CAMEL / BENIGN_ONLY; set the model + important_instructions.
```
