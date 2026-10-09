# The confidentiality floor: what C2 rests on, and what it does not

Companion to `docs/paper-outline.md` (§5.5, Appendix A-floor) and
`docs/paper-decisions.md`. This file is the audited dependency list behind the
paper's claim C2 on the confidentiality axis — written so every row names the
code it is true of and the test that keeps it true.

The paper's central construction is *per guarantee, its own set of trusted
dependencies*. This is that set, for one guarantee.

---

## 1. The claim, stated at the precision the code supports

> **C2 (confidentiality axis).** Untrusted data does not extend the
> pre-established set of permitted effects for the declared obligation
> "a secret read is outstanding ⟹ no egress". For a tool the operator declared a
> sensitive source and a tool the operator declared an egress sink, the refusal
> is invariant under an **arbitrary** output of the integrity labeler — right,
> wrong, inverted, or adversarial, on any value.

Two things that claim deliberately does **not** say:

* It is not "the floor is correct". It is "the floor's decision is not a
  function of the integrity labeler". Those are different statements and only
  the second is proved.
* It is not asserted for an **undeclared** tool. Arming then falls to a
  structural classifier over the read's own arguments (row **D2**), which the
  model writes; that is best-effort arming and a rich-syntax residual, covered
  by fuzzing, not by this claim.

### Why it is an argument about a signature

```python
# axor_core/policy/gates.py
def confidentiality_risk(
    tool_name: str,
    normalized: NormalizedIntent,
    floor_active: bool,
    egress_sinks: frozenset[str] | set[str] = frozenset(),
) -> bool:
    return floor_active and is_exfil(tool_name, normalized, egress_sinks)
```

There is no `CausalRoot` parameter. The integrity labeler's output
(`TaintEngine.derive_value`) cannot reach this decision because it is not in its
domain. `driving_root.sensitive` is likewise **not** an enforcement input
anywhere in the kernel: every gate reads `driving_root.is_tainted` (integrity)
and the session-wide `floor_active`, and nothing else
(`axor_core/policy/gates.py`, `axor_core/governor.py`).

The floor itself is armed on the **fact** of a read, at the read boundary, and
lifted only by capability:

| step | code | nature |
|---|---|---|
| arm | `policy/provenance.py:38` → `taint/engine.py:146-161` | keyed on the tool name against `sensitive_sources`; the value's fingerprint is the registry **key for release**, never the predicate for arming |
| hold | `taint/engine.py:164-168` | `_floor_saturated or bool(_outstanding)` — one session boolean |
| lift | `taint/engine.py:331-352` | `endorse_value` under an unforgeable `GovernanceAuthority`, by fingerprint identity — *released by identity, not by `derive()`* |

So the floor and the per-value ledger cannot desynchronise: they never consult
each other.

---

## 2. The trusted dependency set

**Guaranteed rows** — C2 is stated over these. Each is content-blind and
attacker-independent: an attacker influences a call's arguments, never a
deployment's declarations.

| # | dependency | nature | failure mode if wrong | pinned by |
|---|---|---|---|---|
| **A1** | `sensitive_sources` — the operator's declaration of which tools read secrets | operator config, keyed on tool name at the read boundary | a secret-reading tool left undeclared never arms the floor | `tests/policy/test_floor_arming_dependencies.py::test_declared_sensitive_source_arms_the_floor_whatever_the_arguments` |
| **A2** | `egress_sinks` — the operator's declaration of which tools exfiltrate | operator config, keyed on tool name | an egress tool left undeclared is not `is_exfil` unless the normalizer recognises its destination (row D1) | `::test_declared_egress_sink_is_exfil_with_no_recognisable_destination` |
| **A3** | complete mediation of reads — every executed tool's output passes `register_output` / `_register_value_taint` | integration contract of the wrapper | an unmediated read arms nothing | `tests/test_both_wrap_paths_emit.py`, `tests/test_wrap_paths_record_alike.py` |
| **A4** | `GovernanceAuthority` is unforgeable and is the only way to lift the floor | capability, checked in `taint/engine.py` | floor lifted without governance | `tests/test_class_b_floor.py` |

**Residual rows** — real dependencies in the *dangerous* direction, present in
the implementation, excluded from C2 by construction and covered by fuzzing
instead. Each is attacker-reachable because the model writes the arguments the
classifier reads.

| # | dependency | direction of failure | reach | pinned by |
|---|---|---|---|---|
| **D1** | `normalized.destination_kind` as the second ground of `is_exfil` | a missed destination **narrows** `is_exfil`, so the floor does not fire | only `url`, `command`, `cmd` are inspected (an endpoint under any other argument name reads `none`); the first matched URL decides, so a command opening with a benign `http://localhost/` classifies `localhost`, which is in neither `EXFIL_DESTINATIONS` nor `ssrf_gate`'s internal set | `::test_destination_kind_residual_first_url_wins`, `::test_destination_kind_residual_unrecognised_argument_name` |
| **D2** | `reads_secret_like_data` / `target_kind == "secret"` as the arming fallback for an **undeclared** tool | a missed secret read means the floor **never arms** | pattern match over the read's own `path` / `command`; misses an unrevealing name (`creds.yaml`, `prod.vars`, a symlink) | `::test_structural_fallback_misses_are_a_residual_not_a_guarantee` |

D1 is a **widening** of the declared egress set and never a backstop for it: it
can only add exfil-ness, and it fails to `destination_kind == "none"` whenever it
cannot see a destination. The host classifier underneath it is sound (userinfo,
integer-encoded and short-form IPv4, IPv4-mapped IPv6, and an unparsable DNS name
all resolve to the real class or fall to `external_url` —
`axor_core/security/net.py`, pinned by
`::test_destination_kind_is_not_fooled_by_an_obfuscated_host`); the residual is
in URL *extraction* and the argument-name surface, not in host classification.

The remedy for every residual row is the same and is a deployment statement, not
a code change: declare the tool
(`::test_declaration_closes_the_fallback_gap`).

**Scope rows** — true statements that bound the claim rather than support it.

| # | statement | code |
|---|---|---|
| **S1** | Saturation is sticky and fail-closed: past `_MAX_OUTSTANDING_SECRETS` distinct outstanding secrets the floor is forced active until governance clears it, so `floor_active` has two grounds and must appear in the theorem as one symbol, not as "⟺ a secret is outstanding" | `taint/engine.py:129, 150-161` |
| **S2** | The floor protects **our** secrets in **our** session. A value crossing a process boundary is re-minted `sensitive=False` by design, and on the peer path a peer's `sensitive=True` survives only at ladder level L2 within a declared discount class; at L0/L1 it is dropped. A peer's secrets are the peer's floor | `taint/causal_root.py:83-93`, `federation/ladder.py:98-151`, `node/messaging.py:297` |
| **S3** | Within one trust domain the floor **is** transitive across write→read: an intra-domain message folds the carried root intact, so a sensitive value arriving at another node arms that node's floor | `node/messaging.py:264-268`, `kernel/replay.py` MESSAGE_RECEIVED fold |
| **S4** | A child session inherits the parent's floor, saturation included, or the child would be a bypass | `taint/engine.py:265-272` |

---

## 3. The regression: labeler fault injection over recorded traces

The signature argument above is the proof. The fault injection is its
**regression in the implementation** — it cannot strengthen the claim, it can
only catch the code drifting away from it. Both go in the paper, in different
places: the signature argument in §5, this crosswalk in Appendix A.

`axor_core/kernel/labeler_fault.py` adds five arms to counterfactual replay,
selected by `KernelConfig.labeler_fault` (+ `labeler_fault_seed`):

| mode | labeler behaviour |
|---|---|
| `NONE` | control arm — the identity; a replay under it must reproduce the recorded run with zero divergence |
| `ALL_TRUSTED` | every value reads back trusted: all integrity gating goes silent at once |
| `ALL_TAINTED` | every value reads back maximally tainted |
| `INVERTED` | the label is flipped per value |
| `FLIP_BY_REF` | a deterministic per-`(seed, ref)` coin — wrong at a bounded rate |

*(These are the five arms the code implements. If the paper's table names them
differently, rename here — the invariant below quantifies over
`tuple(LabelerFault)`, so adding or renaming an arm does not change the test.)*

Two placement rules make the experiment measure what it claims, and both are
enforced, not just intended:

1. **The fault is integrity-only.** `CausalRoot` carries both axes in one frozen
   value and `register_value` branches on each separately, so an injection
   written as "return a clean root" erases the floor as well — the run would then
   show C2 holding for a system with no confidentiality enforcement left. Every
   mode rewrites `sources` only and carries `sensitive` through; `_preserve`
   raises if a mode ever violates that.
2. **The injection sits at the labeler, not at the arming map.** It replaces the
   driving root in `replay._derive_driving_root`'s result — the labeler's verdict
   at the decision point. `state.floor_active`, folded from the recorded
   TOOL_RESULT roots, is untouched. Injecting into `output_root` instead would
   kill the arming, not the labeler.

### The invariant, and the trap it avoids

> For every fault mode, the **set of calls with `confidentiality_risk == True`**
> is identical.

It is stated over the predicate, per call — **never** over denial counts or
denial categories. Integrity and confidentiality are decided in the same gate,
and `carrier_gate` (which does read `is_tainted`) runs *before* it, so:

* a call the carrier gate refuses is reported under `carrier_gate`;
* silence the labeler and the carrier gate goes quiet, so the same call falls
  through to the floor and is now reported as a **confidentiality** denial.

The count of confidentiality denials therefore **rises** under `ALL_TRUSTED`
while C2 holds exactly. An invariant written over counts would fail a
correctly-behaving system. `tests/kernel/test_floor_labeler_independence.py::
test_axis_of_a_denial_moves_under_fault_while_the_predicate_does_not` pins that
trap open.

### Cases that must be in any fault-injection run

* **Integrity-only denials.** A trace must contain calls refused on the
  integrity axis *alone* (a declared integrity sink that is not an egress sink;
  an `executes_generated_code` call), and `ALL_TRUSTED` must turn them ALLOW.
  Without them a broken integrity labeler hides behind gate overlap and the run
  proves nothing — it shows "the defence holds" when what holds is only the
  floor. Pinned by `::test_integrity_only_denials_vanish_when_the_labeler_goes_blind`.
* **Benign (no-attack) trajectories under `ALL_TRUSTED`.** Utility must rise
  toward undefended, less the consequence-gate refusals, which must remain.
  Otherwise "the defence holds" can mean "everything is refused". Needs captured
  AgentDojo trajectories, available from the pass in which capture is enabled.
* **Attribution of the InjecAgent cases where the refusal rests on origin
  alone**, by the same replay.

---

## 4. Anchors

| fact | code |
|---|---|
| the confidentiality predicate, without provenance in its domain | `axor_core/policy/gates.py` — `confidentiality_risk`, `is_exfil` |
| `driving_root.sensitive` is not an enforcement input | `axor_core/policy/gates.py` (only `is_tainted` is read, lines 175 and 318) |
| the shared arming map, declaration first | `axor_core/policy/provenance.py` — `output_root` |
| both wrapping paths arm through it | `axor_core/governor.py` — `register_output`; `axor_core/node/intent_loop.py` — `_register_value_taint` |
| arming, holding, lifting the floor | `axor_core/taint/engine.py` — `register_value`, `confidentiality_floor_active`, `endorse_value` |
| request-side arming fallback | `axor_core/policy/normalizer.py` — `_reads_secret`, `_classify_destination`, `_classify_url_target` |
| the fault modes and their integrity-only contract | `axor_core/kernel/labeler_fault.py` |
| the invariant over traces | `tests/kernel/test_floor_labeler_independence.py` |
| the dependency rows | `tests/policy/test_floor_arming_dependencies.py` |
