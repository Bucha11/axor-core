# Deferred follow-ups — governance v4 review

These are the findings the review pass intentionally **did not** change, because
each is either a large refactor, a cross-repo coordination, or an availability-only
issue where a hasty change carries more risk than the finding. They are recorded
here with enough context to pick up cold. Ordered by value/effort.

Status legend: severity is from the original review; "blast radius" is what a
careless fix could break.

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
1. Extract an `EscalationManager` (grants + leases + flood guard): the
   `_handle_escalation` path (916–1026), the `_GrantState`/lease state, and the
   `increment_use`/`ops_remaining` bookkeeping. ~250 lines. Unit-test the manager
   directly (flood, expiry, lease exhaustion) instead of only through the loop.
2. Extract token/telemetry recording (`_record_token_event`, density
   `SinkDensityEvent` emission, `_record_degradation_signal`) into a small
   `_LoopTelemetry` helper that takes the trace list.
3. Move spawn handling (`resolve_spawn_intent`, `_spawn_taint_reason`) onto the
   shared gate path (see item 5) so it stops re-implementing carrier ad hoc.
4. After 1–3, `_resolve_tool_intent` should read as: normalize → run gate sequence
   → register provenance → telemetry. Target < 700 lines for the file.

**Guardrail:** the full suite + `lint-imports` after every extraction; the
determinism sweep (seeds 0/1/42/1337) must stay green. Do **not** combine with any
behaviour fix.

**Blast radius:** high (this file is the hot path). Mitigated by doing it as
pure moves with the suite as the oracle.

---

## 2. `AnomalyDetector` / `contracts.reputation` / `contracts.normalizer` —
   prune dead surface — MINOR, but **cross-repo**

**Where:** `axor_core/contracts/__init__.py` exports `AnomalyDetector`;
`contracts/normalizer.py` and `contracts/reputation.py` are at 0% coverage and
unwired in the loop (the `anomaly_detector` hook was removed).

**Do NOT just delete.** `axor-classifier-simple` *implements* this Protocol:
`axor_classifier_simple/anomaly_detector.py:32` imports `AnomalyDetector` from
axor-core, and `intent_loop.py:73` still imports `ReputationEnricher` from
`contracts.reputation`. So the surface is dead *as enforcement* but live *as a
published contract*.

**Plan:**
1. Decide the intent: is observe-only ML anomaly scoring a supported extension
   point or fully removed? The docs say detection is observe-only telemetry — so
   the Protocol should stay but be clearly marked "telemetry, never gates".
2. If keeping: add a docstring + one wiring test that drives a fake
   `AnomalyDetector` through the observe-only path and asserts it cannot deny.
   Remove only the genuinely-unreferenced `contracts.normalizer` if nothing (incl.
   downstream) imports it — re-run the cross-repo grep first.
3. If removing: coordinate a major-version bump and a PR in
   `axor-classifier-simple` that drops the import in the same release train.

**Blast radius:** downstream import breakage — must grep all `axor-*` repos before
any removal.

---

## 3. Lease/grant consumed before the taint gates — MINOR (correctness)

**Where:** `intent_loop.py:879` (`lease.increment_use()`) and `:890`
(`grant.ops_remaining -= 1`) run inside `_evaluate_tool_intent` *before* the
carrier/consequence/taint gates decide. A call later denied by taint still burns a
lease use; and a valid lease with no matching grant falls through to a deny after
incrementing (latent, currently unreachable since both are created together at the
grant site).

**Plan:** move the decrement/increment to *after* a decision is known to be ALLOW
(or roll it back on deny). Cleanest once item 1 lands — the `EscalationManager`
should expose `try_consume()` that only commits on an allowed decision. Add a test:
a lease-covered call that is then taint-denied leaves `ops_remaining` unchanged.

**Blast radius:** escalation/lease tests; do after the manager extraction.

---

## 4. `_outstanding` confidentiality-floor map is unbounded — MINOR (availability)

**Where:** `taint/engine.py:78,93`. One entry per distinct sensitive-read
fingerprint, never capped (contrast the ledger's `_MAX_TOTAL_SEGMENTS`). Many
distinct secret reads grow memory unboundedly. It fails **closed** (more entries
keep the floor up), so this is availability-only, not a bypass.

**Plan:** cap the map; on overflow flip a `_floor_saturated` flag that forces
`confidentiality_floor_active()` to return `True` unconditionally (stay fail-closed
— never drop an entry that would lower the floor). Mirror the ledger's saturation
pattern and emit a trace event so operators see it. Add a flood test asserting the
floor stays up at the cap.

**Blast radius:** low (additive, fail-closed). The only care: saturation must be
sticky and only clearable by governance, like the ledger.

---

## 5. Spawn path bypasses the shared gate sequence — MINOR (consistency)

**Where:** `intent_loop.py:1102` `_spawn_taint_reason` re-implements `carrier_gate`
ad hoc; spawn args do not run consequence/value-policy/taint gates.

**Plan:** route spawn through the same `gates.py` predicates (with spawn-specific
`positional_sinks`/`imperative_sinks` config) so there is genuinely one enforcement
path. Fold into item 1/step 3. Keep the existing whole-args derivation for spawn
(it is correctly *not* narrowed by `driving_args`).

**Blast radius:** spawn/federation-spawn tests.

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
