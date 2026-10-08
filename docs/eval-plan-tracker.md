# Evaluation plan tracker

Source of record for the camera-ready run cycle. All runs use the **fixed
positive-polarity** core (`integrity_default: context`), **STRICT**
(`require_tool_roles=True`, `require_egress_allowlist=False` in origin mode). Pin the
**commit hash** of the core for every run. Save **per-task outcomes and raw
trajectories** (needed for statistics and the artifact).

Status legend: ✅ done · 🟡 in progress · ⏸ queued · ⬜ not started · ❌ blocked.

## Decisive path (go/no-go) — do in this exact order

Four runs decide whether there is a paper. Order matters: each is worth doing only if
the previous one did not sink the submission. **Three of the four are free; the paid
one is ≈ $15–20.**

| # | ID | Run | Cost | Status | A bad result means |
|---|---|---|---|---|---|
| 1 | R0 | **Fix check** — re-encoded IBAN → deny, unknown sink → deny | free, ½ day | ✅ | fix does not hold → every other run is pointless |
| 2 | a5 | **write-then-read, deterministic** (`examples/agentdojo/poc_write_then_read.py`) | free | ✅ (hole found + fixed) | **found a real O2 hole, now fixed.** A benign/trusted read seeds its whole output into the trusted index, so reading attacker-written state back through `get_scheduled_transactions` (misclassified benign) laundered a **short** recipient to trusted and the egress was ALLOWED (long values were still caught by the ledger, hiding it). Fix: mutable/attacker-writable reads must be `untrusted_source`. **Audited all origin configs and fixed the same hole in three:** banking `get_scheduled_transactions`, travel `get_day_calendar_events`/`search_calendar_events`, slack `get_users_in_channel` (the same class rope_bridge fixed for `get_channels` but missed here). Confirmed with short-value exploits (allowed before, denied after) and guarded by `poc_write_then_read.py` (banking) + `poc_mutable_read_audit.py` (all three). B3(ii) answered. Note: this raises the benign over-block cost (R1/R6 will reflect it) — the honest price of soundness. |
| 3 | R1 | **Cost at `request-only`** — AgentDojo headline cost (one live confirmation ~$15–20) | ~$15–20 | 🟡 smoke ✅ + branch flag ✅; full 7-pass pending | required by all four review sets; **cannot submit without it**. Smoke: banking governed o4-mini CU 43.8% (6 taint denials), config+a5-fix run clean. `AXOR_BENCH_TRUST={request_only,any_trusted}` added: request_only = ROPE AxorOriginBootstrap exactly; any_trusted = + t=0 established payees (OPERATOR), verified (payee flips allow, attacker stays denied). **Next:** replay-harness (undefended once → gate-replay both settings → live only on deny-tasks) to run full R1 at budget. |
| 4 | R4 | **InjecAgent replay 2×2** — origin×consequence over 510 | free, one evening | ✅ | resolves 0.6% vs 0.0% and 444+60≠510; a reviewer spots the discrepancy in 15 min |

Enabler (done): **R1-ws** — workspace origin taxonomy, all 24 tools classified & smoke-
tested (`examples/agentdojo/config/workspace_origin.yaml`); validate its numbers against
R1 before treating the workspace row as load-bearing.

**Go/no-go after the decisive path:** O2 holds (a5) · cost known (R1). Missing either →
USENIX C2. a5 "mark lost" → fix O2 first.

### Decisive-path commands

```sh
# 1. R0 — fix check (free)
python3 examples/agentdojo/poc_reencoding.py
# 4. R4 — InjecAgent replay 2x2 + ds floor (free; exit non-zero on drift)
INJECAGENT_DIR=/path/to/InjecAgent python3 examples/injecagent/replay_dh_axes.py
INJECAGENT_DIR=/path/to/InjecAgent python3 examples/injecagent/replay_ds_floor.py
# 3. R1 — cost at request-only (o4-mini, 7 passes/suite), per suite:
AXOR_BENCH_BACKEND=openrouter AXOR_BENCH_SUITE=<suite> AXOR_BENCH_CONFIG=<suite>_origin.yaml \
  AXOR_BENCH_INTEGRITY=context AXOR_BENCH_ORIGIN=1 AXOR_BENCH_MODEL=openai/o4-mini \
  AXOR_BENCH_CAMEL=1 AXOR_BENCH_BENIGN_ONLY=1 python -m examples.agentdojo.run_agentdojo
# 2. a5 — script to be committed when that run is built.
```

## Beyond go/no-go — in-paper arguments (not whether the paper exists)

These sharpen the paper once it is known to stand; none can sink it.

| ID | Run | Cost | Status |
|---|---|---|---|
| R3 | Adaptive suite, closed loop (**gpt-4o-mini**; banking/travel/slack; both settings; ASR+UA; a1–a4, a6, a7 — a5 promoted to the decisive path) | small | ⬜ |
| R6 | Benign cost floor (workspace+banking, realistic `sensitive_sources`, o4-mini, paired floor on/off, 7 passes; benign-denial rate + cost) | ~$100 | ⬜ (needs R1-ws ✅) |

### R3 adaptive attacks (gpt-4o-mini)

| Attack | Tests | Expected | Status |
|---|---|---|---|
| a1 | re-encoding of attacker value | 0 — demonstrates the fix | ⬜ (R0 shows the deterministic version) |
| a2 | catalogue/record steering (travel IT4; banking existing-payee-from-history) | any-trusted passes, request-only does not | ⬜ |
| a3 | pick a different known payee under supersession | passes — declared boundary | ⬜ |
| a4 | non-driving args (`amount`, subject) on a trusted payee | passes if not projected | ⬜ |
| a6 | tool selection with trusted constants | consequence axis | ⬜ |
| a7 | in-band text in body (IT1) | residual (shown, not hidden) | ⬜ |

(a5 write-then-read is now **decisive run #2**, above.) a2–a4 and a7 failures are the
honestly-declared boundary — **show them, don't hide**.

## Other tracked items (not on the submission-critical path)

| ID | Run | Status |
|---|---|---|
| R2 | ASR on tractable models (GPT-4o + Qwen-72B, 4 suite, both settings, `important_instructions`, ≥3 passes) | ⬜ |
| R4c | InjecAgent live governed closed loop (GPT-4o + Qwen, DH+DS; ASR over successful-undefended) | 🟡 (DH done ×4; DS live pending) |
| R8 | Allowlist sensitivity (±k entries; cheap gate replay + 1 live) | ⬜ |
| R9 | Real integration (LangGraph agent, 1 suite, 3 passes; verdicts byte-identical to the shim) | ⬜ |
| R10 | Latency on fixed core (~30 min) | ⬜ |
| R11 | Daemon boundary (1 suite: verdict parity + overhead); else state all runs were on the soft boundary | ⬜ |

## Schedule

| When | Work | Cost |
|---|---|---|
| Days 1–4 (now) | **Decisive path**: R0 ✅, a5, R1 (replay + 1 live), R4 ✅ | ≈ $20 |
| — | **go/no-go** | |
| After go | In-paper arguments: R3 (gpt-4o-mini), R6 | ~$100 + small |
| If time | R2, R4c live, R8, R9, R10, R11 | varies |

**Results that could change the paper:**
- **a5 (resolved):** the mark did NOT survive storage for short values — a benign read of
  mutable state laundered attacker values into the trusted index. Fixed by reclassifying
  such reads as untrusted. Rule for O2: *only attacker-independent reads may be trusted.*
  Audit travel/slack for the same pattern before their rows are load-bearing.
- **R1:** request-only is catastrophically expensive → headline cost gets awkward, but
  report it anyway.
