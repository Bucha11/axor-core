"""Every input that can PERMIT a call — as a checkable specification.

T0 says no model produces a trusted-path projection. The claim the kernel makes is
wider: the model is outside the trusted base of the guarantee. That holds only if
every input that can turn a denial into an allow has a ground the model cannot
manufacture. ``docs/kernel-theorem.md`` §3b audited those inputs in prose; this
module is the same audit as data, so it can be checked rather than re-read:

  * every ``verified_by`` symbol must import (the mechanism exists);
  * every ``pinned_by`` test must exist (the behaviour on absence is exercised);
  * the table in the theorem document is rendered from these records
    (:func:`render_markdown`) and a test fails when the two differ.

For each ground the record states who may issue it, what it permits, how it is
checked, what happens when it is absent or fails, and what it does NOT override.
The last column is the composition claim: no ground here relaxes an obligation
it is not about.

Determinism is not correctness. A deterministic matcher can still permit the
wrong call; each record therefore names its *check*, not only its producer, and a
ground whose check cannot be made in core says so (``status="premise"``).

Two columns keep what the kernel establishes apart from what it is told:
``check`` is what core verifies; ``premise`` is what the ground additionally
rests on and core cannot verify. A consumption *contract* and a consumption
*attestation* are separate rows for that reason: the first is checked against the
predicates and applied to every call, the second is the operator's sentence and
nothing more.

This is a SPECIFICATION, and the tests hold each row to the code. They do not
show the list is COMPLETE — that every input able to turn a denial into an allow
has a row. Completeness is argued in kernel-theorem §3b by enumerating what the
allow path reads; a new input added to that path without a row here is not
caught by any test.
"""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = ("enforced", "enforced-under-strict", "premise", "inadmissible")


@dataclass(frozen=True)
class PermittingGround:
    name: str
    issuer: str            # who may issue it
    permits: str           # what it permits
    check: str             # how it is checked
    on_absence: str        # what happens when it is absent or fails the check
    does_not_override: str  # obligations it leaves in force
    premise: str           # what it also rests on that core cannot verify ("—" if nothing)
    status: str
    verified_by: tuple[str, ...]  # "module:symbol"
    pinned_by: tuple[str, ...]    # "tests/path.py::test_name"


GROUNDS: tuple[PermittingGround, ...] = (
    PermittingGround(
        name="Operator policy (A₀: tools, paths, consequence ceiling)",
        issuer="operator configuration (policy= / default_policy=)",
        permits="the capability surface and the unattended consequence ceiling",
        check="STRICT refuses to run without an operator-supplied policy",
        on_absence="STRICT: PolicyError, nothing runs; other modes: classifier-selected, not covered",
        does_not_override="every per-call gate",
        premise="the operator's policy is the intended one",
        status="enforced-under-strict",
        verified_by=("axor_core.worker.session:GovernedSession",),
        pinned_by=(
            "tests/worker/test_strict_mode.py::test_strict_refuses_to_derive_authority_from_classification",
        ),
    ),
    PermittingGround(
        name="Delegation ceiling (G)",
        issuer="operator configuration (escalation_policy.grantable_tools)",
        permits="which tools a grant or approval may ever reach",
        check="grants and leases validated against available ∪ grantable",
        on_absence="no escalation surface: an over-ceiling tool stays unreachable",
        does_not_override="per-call gates; a grant only adds the tool to the capability set",
        premise="the operator's ceiling is the intended one",
        status="enforced",
        verified_by=("axor_core.node.escalation:EscalationManager",),
        pinned_by=(
            "tests/policy/test_capability_on_demand.py::test_without_operator_ceiling_no_escalation_surface",
            "tests/node/test_call_bound_authority.py::test_grantable_tools_bounds_the_call_path",
        ),
    ),
    PermittingGround(
        name="Action class (consequence axis)",
        issuer="built-in table + operator consequence_overrides, keyed on the sink name",
        permits="an unattended call whose class is within the ceiling",
        check="exact-name lookup; STRICT: unclassified ⇒ CATASTROPHIC",
        on_absence="STRICT: refused at construction / treated as catastrophic",
        does_not_override="value, taint, floor, carrier, SSRF gates",
        premise="the declared class matches the tool's real irreversibility",
        status="enforced-under-strict",
        verified_by=("axor_core.policy.consequence:consequence_class",),
        pinned_by=(
            "tests/adversarial/test_consequence_hardening.py::test_strict_governor_refuses_an_unclassified_tool_and_says_why",
        ),
    ),
    PermittingGround(
        name="Sink roles (egress / integrity / imperative / positional / benign)",
        issuer="operator declaration",
        permits="which gates a tool is subject to; a positional sink admits only structured carriers",
        check="STRICT role completeness; an instruction-complete sink cannot be declared positional",
        on_absence="STRICT: an unclassified tool is refused",
        does_not_override="the floor and the value gates",
        premise="the declared role matches what the tool does (a mis-declared sensitive source arms no floor)",
        status="enforced-under-strict",
        verified_by=(
            "axor_core.kernel.registration:validate_role_completeness",
            "axor_core.kernel.registration:tool_is_classified",
        ),
        pinned_by=(
            "tests/adversarial/test_consequence_hardening.py::test_strict_intent_loop_gates_an_unclassified_tool",
            "tests/adversarial/test_positional_admission.py::test_exec_sink_cannot_be_declared_positional",
        ),
    ),
    PermittingGround(
        name="Consumption contract (T4 for a guarded field)",
        issuer="operator declaration per sink field (field_contracts)",
        permits="a value predicate to count as faithful for the field it guards",
        check="predicate role vs field kind at registration; every call prepared into the contract before the first gate; the prepared set is what executes",
        on_absence="see the attestation row; outside STRICT, nothing — reported 'unverified'",
        does_not_override="every gate after preparation",
        premise="the handler consumes each field as its contract states (its defaults, aliases and parsing are the ones declared)",
        status="enforced",
        verified_by=(
            "axor_core.kernel.consumption:validate_consumption_contracts",
            "axor_core.kernel.consumption:prepare_args",
        ),
        pinned_by=(
            "tests/adversarial/test_check_vs_consumer.py::test_absent_amount_with_contract_is_refused_and_the_full_call_runs",
            "tests/adversarial/test_check_vs_consumer.py::test_path_contract_refuses_that_configuration",
            "tests/integration/test_negative_control_matrix.py::test_removing_the_condition_produces_the_effect_restoring_it_removes_it",
        ),
    ),
    PermittingGround(
        name="Consumption attestation",
        issuer="operator statement per sink (consumption_attested)",
        permits="a guarded sink with no contract to construct under STRICT",
        check="none beyond presence: core does not compare the predicates with any field description and prepares nothing",
        on_absence="STRICT: construction refused for a guarded sink with neither contract nor attestation",
        does_not_override="every gate — but the gates decide the raw proposal, which is also what executes",
        premise="the whole of T4 for that sink: the handler consumes the guarded fields exactly as the predicates assume",
        status="premise",
        verified_by=("axor_core.kernel.consumption:conformance_obligations",),
        pinned_by=(
            "tests/kernel/test_consumption.py::test_strict_requires_a_contract_or_an_attestation_per_guarded_sink",
            "tests/integration/test_end_to_end_chain.py::test_adaptive_adversary_without_contract_reaches_a_forbidden_effect",
        ),
    ),
    PermittingGround(
        name="Integrity supersession",
        issuer="operator, by declaring a closed allowlist on every driving arg",
        permits="skipping the origin (integrity-taint) check: membership in the pre-authorised set satisfies that obligation instead",
        check="every driving arg present and covered by a closed codomain (enum / tuple_enum / keyed key)",
        on_absence="origin check applies; numeric_range never supersedes",
        does_not_override="the confidentiality floor, consequence, value ranges, capability",
        premise="the operator's set is the intended one; T4 for the covered arg rests on that sink's contract, or on its attestation",
        status="enforced",
        verified_by=("axor_core.policy.gates:integrity_superseded_by_decidable",),
        pinned_by=(
            "tests/adversarial/test_decidable_supersession.py::test_numeric_range_does_not_supersede_open_codomain",
            "tests/adversarial/test_decidable_supersession.py::test_confidentiality_floor_is_not_superseded",
            "tests/integration/test_end_to_end_chain.py::test_supersession_relaxes_integrity_only",
        ),
    ),
    PermittingGround(
        name="Confidentiality floor down",
        issuer="no sensitive read outstanding; lifted only by GovernanceAuthority endorsement",
        permits="egress",
        check="outstanding-read map; endorsement verifies the authority token, per value fingerprint",
        on_absence="egress denied while a sensitive read is outstanding",
        does_not_override="integrity, consequence, value gates",
        premise="the governance authority is not reachable by the agent",
        status="enforced",
        verified_by=(
            "axor_core.taint.engine:TaintEngine.confidentiality_floor_active",
            "axor_core.taint.engine:TaintEngine.endorse_value",
        ),
        pinned_by=(
            "tests/test_inprocess_gaps.py::test_endorse_rejects_invalid_authority",
            "tests/integration/test_end_to_end_chain.py::test_floor_holds_when_the_integrity_labeler_fails",
        ),
    ),
    PermittingGround(
        name="Trusted value (TrustedOrigin)",
        issuer="TASK: the principal's task at depth 0 · OPERATOR: allowlist members · TOOL: a declared benign tool · ENDORSED: governance",
        permits="a model-emitted value to escape the context root at an integrity sink",
        check="register_trusted from kernel-side code only; TOOL requires a benign declaration under STRICT",
        on_absence="the value carries the context root (STRICT) and is denied at an integrity sink",
        does_not_override="the floor, consequence, value gates",
        premise="a declared benign tool's output is in fact not attacker-writable",
        status="enforced-under-strict",
        verified_by=(
            "axor_core.taint.engine:TaintEngine.register_trusted",
            "axor_core.policy.provenance:is_trusted_tool",
        ),
        pinned_by=(
            "tests/adversarial/test_context_default_integrity.py::test_context_mode_strict_trusts_only_declared_benign_tools",
            "tests/adversarial/test_context_default_integrity.py::test_a_worker_cannot_reach_register_trusted_through_tool_args",
        ),
    ),
    PermittingGround(
        name="Call-bound approval",
        issuer="the operator-supplied EscalationCallback",
        permits="one call over the consequence ceiling",
        check="call digest over every prepared argument; volatile state re-checked after the answer; claimed atomically, one use",
        on_absence="the consequence denial stands",
        does_not_override="every other gate (asked only after all of them pass)",
        premise="the callback is a trusted party or a checkable rule — not a model",
        status="premise",
        verified_by=(
            "axor_core.policy.effect:call_digest",
            "axor_core.node.escalation:EscalationManager.claim_call",
        ),
        pinned_by=(
            "tests/node/test_call_bound_authority.py::test_a_denied_approval_denies_the_call",
            "tests/node/test_call_bound_authority.py::test_a_floor_armed_during_the_wait_refuses_the_call",
            "tests/node/test_call_bound_authority.py::test_two_concurrent_calls_cannot_both_spend_one_authorisation",
            "tests/integration/test_end_to_end_chain.py::test_a_model_backed_approver_is_a_ground_the_model_controls",
        ),
    ),
    PermittingGround(
        name="Projection features (carrier, path class, provenance label, normalized intent)",
        issuer="the T0 producer modules",
        permits="the per-value gates' inputs",
        check="import contract (no model surface) + determinism tests",
        on_absence="—",
        does_not_override="—",
        premise="—",
        status="enforced",
        verified_by=(
            "axor_core.security.carrier:classify_carrier",
            "axor_core.policy.normalizer:IntentNormalizer",
        ),
        pinned_by=(
            "tests/invariants/test_t0_producers_non_interpreting.py::test_producer_imports_no_interpreting_surface",
            "tests/invariants/test_t0_producers_non_interpreting.py::test_carrier_classifier_is_deterministic",
        ),
    ),
    PermittingGround(
        name="Peer provenance label",
        issuer="a peer, by signed receipt",
        permits="restoring a peer value's trust level",
        check="signature, nonce, binding to the value; across keysets only a bounded discount",
        on_absence="the value is untrusted",
        does_not_override="every gate",
        premise="the peer's key is held by the peer",
        status="enforced",
        verified_by=("axor_core.federation.gateway:FederationGateway",),
        pinned_by=(
            "tests/adversarial/test_federation.py::test_forged_receipt_is_denied",
            "tests/adversarial/test_federation.py::test_replay_of_consumed_receipt_is_denied",
        ),
    ),
    PermittingGround(
        name="Absence of a model-carried mark",
        issuer="the model (it must reproduce the mark onto the argument)",
        permits="nothing — inadmissible as a positive ground",
        check="—",
        on_absence="admissible only in the deny direction, as detection",
        does_not_override="—",
        premise="—",
        status="inadmissible",
        verified_by=(),
        pinned_by=(
            "tests/integration/test_end_to_end_chain.py::test_a_model_carried_mark_is_not_a_ground",
        ),
    ),
)


def render_markdown() -> str:
    """The theorem document's table, generated from :data:`GROUNDS`."""
    head = (
        "| Permitting input | Issued by | Permits | Checked by core | Absent / fails | "
        "Does not override | Premise (not checked by core) | Status |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = "".join(
        f"| {g.name} | {g.issuer} | {g.permits} | {g.check} | {g.on_absence} | "
        f"{g.does_not_override} | {g.premise} | {g.status} |\n"
        for g in GROUNDS
    )
    return head + rows
