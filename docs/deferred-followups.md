# Deferred follow-ups — governance v4 review

These were the findings the review pass first left unchanged. Most have since been
worked through; the status of each is marked below. They are kept here with enough
context to pick up cold. Ordered by value/effort.

Status legend: severity is from the original review; "blast radius" is what a
careless fix could break.

## Status summary

- **Done:** item 1 step 1 (EscalationManager extraction), item 2 (dead protocol
  removal, scoped to classifier-only), item 3 (deferred lease/grant consumption),
  item 4 (bounded confidentiality-floor map), item 5 (spawn routed through the
  shared carrier gate).
- **Deliberately not done:** item 1 step 2 (telemetry extraction) — low value, see
  note; item 6 (in-core SSRF host residuals) — documented in governance-model §7,
  the code option stays opt-in by design.

---

## 1. Split `node/intent_loop.py` (god-module) — MAJOR (maintainability)

**Where:** `axor_core/node/intent_loop.py` (1369 lines; `_resolve_tool_intent`
alone is ~165 lines, 453–617).

**Problem:** the gate orchestrator absorbed ~10 concerns — stream pumping,
escalation grants + flood guard, lease validation, spawn dispatch, token
accounting, degradation glue, density telemetry, federation ingress, adjudicator
consultation, trajectory observers. The pure-gate extraction (`policy/gates.py`)
and the shared arming map (`policy/provenance.py`) already pulled logic out; this
is the remaining structural debt.

**Plan (incremental, behaviour-preserving — no semantic change in any commit):**
1. **DONE** — `EscalationManager` (`node/escalation.py`): grants + leases + flood
   guard + `grant_from_intent` + `resolve`/`evaluate` + `covers`. The loop delegates;
   intent_loop dropped 1369→~1190. Unit-tested directly.
2. **NOT DONE (low value)** — token/telemetry recording into a `_LoopTelemetry`
   helper. Skipped on purpose: the trace helpers (`_record_token_event`,
   `SinkDensityEvent` emission, `_record_degradation_signal`, `_run_trajectory_observers`)
   are already small cohesive methods, and a helper would have to thread
   `trace_events` + `normalizer` + `degradation_engine` + `taint_engine` and would
   pull the genuinely loop-coupled degradation glue into a "telemetry" class — more
   indirection than cleanup. Revisit only if these grow.
3. **DONE** — spawn now routes through the shared `carrier_gate` (see item 5).
4. Target `< 700` lines was **not** reached and is not realistically reachable
   without splitting `_resolve_tool_intent`'s gate cascade itself — which is high
   risk for low marginal benefit now that the gate *logic* already lives in
   `policy/gates.py`. The file is meaningfully better; further splitting is not
   worth the hot-path risk.

**Guardrail (followed):** full suite + `lint-imports` + determinism sweep green
after each extraction; no behaviour change mixed into the move commits.

**Blast radius:** high (this file is the hot path). Mitigated by doing it as
pure moves with the suite as the oracle.

---

## 2. `AnomalyDetector` / `LLMVerifier` — prune dead surface — **DONE (scoped)**

**Resolved:** the cross-repo grep showed `AnomalyDetector` and `LLMVerifier` are
implemented only by `axor-classifier-simple` (being retired), while
`AnomalyResult` / `AnomalyClass` / `ReputationEnricher` and the
reputation→degradation path are still imported by the live `axor-sentinel`. So the
two classifier-only Protocols were removed and their exports dropped; the
sentinel-facing detection surface was kept. `contracts/normalizer.py`
(`ProviderNormalizer`) was kept — it is the live provider-normalizer contract used
by the cross-provider mock-normalizer tests, not dead.

---

## 3. Lease/grant consumed before the taint gates — **DONE**

**Resolved:** `EscalationManager.evaluate` now decides without mutating and returns
a deferred `_PendingConsumption`; the loop commits it only after every gate passes
(`intent_loop` approval point). A call denied by a later gate burns nothing, and a
last-op grant stays present (`covers()` True) through the whole cascade. Covered by
`tests/node/test_escalation_consumption.py`.

---

## 4. `_outstanding` confidentiality-floor map is unbounded — **DONE**

**Resolved:** capped at `_MAX_OUTSTANDING_SECRETS`. Past the cap a new secret read
does not grow the map but flips a sticky `_floor_saturated` flag that forces the
floor active (fail-closed), inherited by children, reset only by
`clear_by_governance`. Logs a warning on the flip. Covered by
`tests/test_class_b_floor.py`. (A dedicated trace event was *not* added — that needs
a new `TraceEventKind` + collector whitelist entry, out of scope for the fix; the
logged warning is the operator signal for now.)

---

## 5. Spawn path bypasses the shared gate sequence — **DONE (carrier gate)**

**Resolved:** `_spawn_taint_reason` now delegates to the shared
`policy.gates.carrier_gate` (spawn_child is in `IMPERATIVE_SINKS`; `is_imperative_sink`
is None-safe for the `normalized` arg), with whole-args derivation preserved. The
spawn branch and the regular tool path can no longer drift on the carrier check.
(Spawn still does not run the consequence/value-policy gates on its args; that is by
design — `spawn_child` is a kernel-internal intent with its own capability/flood
controls, and the child inherits the per-value ledger so its *own* sinks are gated.
Running the full sink cascade on the spawn intent itself is left as a possible
future tightening if a spawn arg ever needs value-policy treatment.)

---

## 6. SSRF host-classification residuals — code option (currently doc-only)

**Where:** `policy/normalizer.py` `_URL_PATTERN` is `http(s)`-only; no redirect
re-check; no DNS resolution. Documented in `governance-model.md §7` as residuals.

**Plan (only if a deployment needs it in-core rather than via tool config):**
- broaden internal-IP literal detection to any `scheme://host` (catch
  `gopher://2852039166/`, `dict://169.254.169.254/`) without widening the URL
  match used for benign classification;
- add an optional, opt-in resolver hook so an operator can enable DNS-time
  classification (off by default — it adds I/O to the hot path).
Each needs its own fuzz cases in `test_property_fuzz.py`.

**Blast radius:** normalizer heuristics feed many tests — high false-positive risk;
that is why it was documented rather than changed. Treat as opt-in.

---

## Not planned (accepted as-is, with rationale)

- **`GovernanceAuthority` is a plain dataclass** (no crypto). Safe because the
  worker→kernel boundary passes JSON, not Python objects, so nothing deserializes
  worker input into this type. Keep the invariant; the existing test asserting no
  deserialization path targets it is the guard. Revisit only if a normalizer ever
  maps tool args to this type.
- **`fingerprint.py` `default=str` collision** — narrow, requires influence over
  the signer's in-memory non-JSON types; wire values are JSON scalars. Not worth
  the complexity of a typed encoder.
- **Same shared HMAC key reused across peers enables impersonation** — inherent to
  symmetric MACs; the right answer is ed25519 per peer, already supported. A config
  lint warning when one key ref is reused across peers would be a nice-to-have.
