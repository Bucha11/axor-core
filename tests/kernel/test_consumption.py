"""Consumption contracts — preparation, registration, config and replay parity.

The end-to-end consequences (a real handler behind the real loop) are in
``tests/adversarial/test_check_vs_consumer.py``; this file pins the mechanism.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axor_core.config import GovernanceConfig
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.policy import PolicyDecision, PolicyDecisionKind
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.consumption import (
    FieldContract,
    FieldKind,
    SinkContract,
    parse_field_contracts,
    prepare_args,
    prepare_call,
    validate_consumption_contracts,
)
from axor_core.kernel.replay import (
    GovernanceState,
    KernelConfig,
    evaluate_call,
    normalized_from_payload,
)
from axor_core.policy.value_policy import enum, numeric_range
from axor_core.taint.causal_root import CausalRoot
from tests.consumers import ALLOWED_IBAN, Bank, SendMoney, call, envelope, loop

C = SinkContract(fields={
    "to": FieldContract(FieldKind.IDENTIFIER),
    "cc": FieldContract(FieldKind.IDENTIFIER_LIST, default=[]),
    "n": FieldContract(FieldKind.NUMBER, integral=True, decimal_string=True),
    "urgent": FieldContract(FieldKind.FLAG, default=False),
    "note": FieldContract(FieldKind.TEXT, unused_when_absent=True, null_is_absent=True),
})


# ── FieldContract: what the handler does on absence must be stated ───────────


def test_required_is_inferred_from_the_absence_statement():
    assert FieldContract(FieldKind.TEXT).required is True
    assert FieldContract(FieldKind.TEXT, default="").required is False
    assert FieldContract(FieldKind.TEXT, unused_when_absent=True).required is False


@pytest.mark.parametrize("kw", [
    {"required": False},                                   # absent ⇒ ? unstated
    {"required": False, "default": 1, "unused_when_absent": True},  # both
    {"required": True, "default": 1},
    {"required": True, "null_is_absent": True},
    {"required": True, "falsy_is_absent": True},
])
def test_an_ambiguous_absence_statement_is_refused(kw):
    with pytest.raises(ValueError):
        FieldContract(FieldKind.NUMBER, **kw)


def test_numeric_options_only_on_numbers():
    with pytest.raises(ValueError, match="NUMBER"):
        FieldContract(FieldKind.IDENTIFIER, integral=True)


# ── preparation ──────────────────────────────────────────────────────────────


def test_defaults_are_materialised_and_unused_fields_stay_absent():
    out, why = prepare_args({"to": "a", "n": 3}, C)
    assert why is None
    assert out == {"to": "a", "n": 3, "cc": [], "urgent": False}


def test_null_is_absent_drops_an_explicit_none():
    out, _ = prepare_args({"to": "a", "n": 3, "note": None}, C)
    assert "note" not in out


# AgentDojo's stock update_scheduled_transaction tests each field by truthiness
# (``if amount:``), so amount=0 / recipient="" leave the record unchanged.
UPDATE = SinkContract(fields={
    "id": FieldContract(FieldKind.NUMBER, integral=True),
    "recipient": FieldContract(FieldKind.IDENTIFIER, unused_when_absent=True, falsy_is_absent=True),
    "amount": FieldContract(FieldKind.NUMBER, unused_when_absent=True, falsy_is_absent=True),
})


@pytest.mark.parametrize("args, kept", [
    ({"id": 1, "amount": 0, "recipient": ""}, {"id": 1}),
    ({"id": 1, "amount": None}, {"id": 1}),
    ({"id": 1, "amount": 5, "recipient": "X"}, {"id": 1, "amount": 5, "recipient": "X"}),
])
def test_falsy_is_absent_drops_what_a_truthiness_handler_ignores(args, kept):
    out, why = prepare_args(args, UPDATE)
    assert why is None and out == kept


@pytest.mark.parametrize("args, fragment", [
    ({"n": 1}, "required argument 'to'"),
    ({"to": "a", "n": 1, "bcc": "x"}, "['bcc']"),
    ({"to": ["a"], "n": 1}, "single identifier"),
    ({"to": "a", "n": 1, "cc": "b"}, "list of identifier"),
    ({"to": "a", "n": 1, "cc": ["b", 2]}, "list of identifier"),
    ({"to": "a", "n": True}, "not a number"),
    ({"to": "a", "n": 2.5}, "truncated"),
    ({"to": "a", "n": float("nan")}, "not finite"),
    ({"to": "a", "n": "1e3"}, "plain decimal"),
    ({"to": "a", "n": " 12"}, "plain decimal"),
    ({"to": "a", "n": "1_000"}, "plain decimal"),
    ({"to": "a", "n": 1, "urgent": 1}, "boolean"),
    ({"to": "a", "n": 1, "note": 5}, "not a string"),
])
def test_out_of_contract_calls_are_refused(args, fragment):
    out, why = prepare_args(args, C)
    assert out is None and fragment in why


def test_declared_conversions():
    assert prepare_args({"to": "a", "n": "12"}, C)[0]["n"] == 12
    assert prepare_args({"to": "a", "n": 4.0}, C)[0]["n"] == 4
    assert isinstance(prepare_args({"to": "a", "n": 4.0}, C)[0]["n"], int)
    assert prepare_args({"to": "a", "n": 10**30}, C)[0]["n"] == 10**30


def test_open_contract_passes_undeclared_fields_through():
    open_c = SinkContract(fields={"to": FieldContract(FieldKind.IDENTIFIER)}, closed=False)
    assert prepare_args({"to": "a", "body": "hi"}, open_c) == ({"to": "a", "body": "hi"}, None)


def test_a_tool_without_contract_is_unchanged():
    assert prepare_call("other", {"x": 1}, {"send": C}) == ({"x": 1}, None)


_scalar = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True),
    st.text(max_size=6), st.lists(st.text(max_size=3), max_size=3),
)


@settings(max_examples=400, deadline=None)
@given(st.dictionaries(st.sampled_from(["to", "cc", "n", "urgent", "note", "zz"]), _scalar))
def test_preparation_is_idempotent(args):
    """What lets replay re-run preparation on a recorded (already prepared) call
    and reach the runtime's verdict. Property-tested, not proved."""
    out, why = prepare_args(args, C)
    if why is None:
        assert prepare_args(out, C) == (out, None)


# ── registration ─────────────────────────────────────────────────────────────


def test_strict_requires_a_contract_or_an_attestation_per_guarded_sink():
    pols = {"send": [enum("to", {"a"})]}
    errs = validate_consumption_contracts(pols, {}, strict=True)
    assert errs and "consumption_attested" in errs[0]
    assert validate_consumption_contracts(pols, {}, attested={"send"}, strict=True) == []
    assert validate_consumption_contracts(pols, {"send": C}, strict=True) == []
    # outside STRICT the premise is merely unverified (reported, not refused)
    assert validate_consumption_contracts(pols, {}) == []


def test_contract_and_attestation_together_is_ambiguous():
    errs = validate_consumption_contracts({}, {"send": C}, attested={"send"})
    assert errs and "one ground" in errs[0]


def test_a_default_outside_its_own_contract_is_refused():
    bad = SinkContract(fields={"n": FieldContract(FieldKind.NUMBER, default="ten")})
    errs = validate_consumption_contracts({}, {"send": bad})
    assert errs and "declared default" in errs[0]


def test_relational_predicates_are_checked_field_by_field():
    from axor_core.policy.value_policy import keyed_range, tuple_enum
    ok = validate_consumption_contracts(
        {"send": [keyed_range("to", "n", {"a": (0, 5)}), tuple_enum(("to", "urgent"), {("a", True)})]},
        {"send": C},
    )
    assert ok == []
    bad = validate_consumption_contracts(
        {"send": [keyed_range("note", "to", {"a": (0, 5)})]}, {"send": C},
    )
    assert len(bad) == 2  # a member check on TEXT, a range on an IDENTIFIER


def test_every_wrapping_path_validates_at_construction():
    pols = {"send": [enum("note", {"x"})]}
    contracts = {"send": C}
    with pytest.raises(ValueError, match="invalid consumption contracts"):
        ToolCallGovernor(value_policies=pols, field_contracts=contracts)
    with pytest.raises(ValueError, match="invalid consumption contracts"):
        loop(value_policies=pols, field_contracts=contracts)


# ── config ───────────────────────────────────────────────────────────────────


def test_config_round_trip_into_both_paths():
    cfg = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send"],
        "driving_args": {"send": ["to"]},
        "consequence_overrides": {"send": "consequential"},
        "value_policies": {"send": [{"arg": "to", "kind": "enum", "allowed": ["a"]}]},
        "field_contracts": {"send": {"fields": {
            "to": {"kind": "identifier"},
            "n": {"kind": "number", "decimal_string": True},
            "note": {"kind": "text", "unused_when_absent": True},
        }}},
    })
    assert cfg.field_contracts["send"].fields["note"].required is False
    gov = ToolCallGovernor(**cfg.as_governor_kwargs())
    d = gov.evaluate("send", {"to": "a", "n": "7"})
    assert d.allowed and d.args == {"to": "a", "n": 7}
    assert [r.ground for r in gov.conformance_obligations] == ["contract"]
    assert cfg.as_session_kwargs()["field_contracts"] is not None


@pytest.mark.parametrize("raw, fragment", [
    ({"send": {"fields": {}}}, "non-empty"),
    ({"send": {"fields": {"to": {"kind": "regex"}}}}, "unknown kind"),
    ({"send": {"fields": {"to": {"kind": "identifier", "alias": "x"}}}}, "unknown key"),
    ({"send": {"fields": {"to": {"kind": "number", "required": False}}}}, "exactly one"),
    ({"send": {"fields": {"to": {"kind": "number", "integral": "yes"}}}}, "booleans"),
    ({"send": {"fields": {"to": {"kind": "number"}}, "closed": "no"}}, "boolean"),
    ({"send": {"strict": True, "fields": {"to": {"kind": "number"}}}}, "unknown key"),
])
def test_malformed_contract_config_fails_closed(raw, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_field_contracts(raw)


def test_consumption_attested_is_a_config_key():
    cfg = GovernanceConfig.from_dict({"consumption_attested": ["send"]})
    assert cfg.consumption_attested == frozenset({"send"})


# ── replay parity ────────────────────────────────────────────────────────────


def test_replay_reaches_the_runtime_verdict_on_recorded_calls():
    pols = {"send_money": [enum("recipient", {ALLOWED_IBAN}), numeric_range("amount", 0, 10)]}
    contracts = {"send_money": SinkContract(fields={
        "recipient": FieldContract(FieldKind.IDENTIFIER),
        "amount": FieldContract(FieldKind.NUMBER, decimal_string=True),
    })}
    classes = {"send_money": ConsequenceClass.CONSEQUENTIAL}
    gov = ToolCallGovernor(value_policies=pols, field_contracts=contracts,
                           consequence_overrides=classes)
    runtime = [
        gov.evaluate("send_money", {"recipient": ALLOWED_IBAN, "amount": "5"}),
        gov.evaluate("send_money", {"recipient": ALLOWED_IBAN}),
        gov.evaluate("send_money", {"recipient": ALLOWED_IBAN, "amount": "50"}),
    ]
    config = KernelConfig(value_policies=pols, field_contracts=contracts,
                          consequence_overrides=classes)
    for decision, event in zip(runtime, gov.trace_events):
        payload = dict(event.payload)
        replayed = evaluate_call(
            "send_money", payload["args"], normalized_from_payload("send_money", payload),
            CausalRoot.constant(), False, config, GovernanceState(),
        )
        assert (replayed is None) == decision.allowed
        if replayed is not None:
            assert replayed.category == decision.category


# ── a rewritten call is held to the same contract ────────────────────────────


@pytest.mark.asyncio
async def test_transform_outside_the_contract_is_refused():
    handler = SendMoney(Bank())
    contracts = {"send_money": SinkContract(fields={
        "recipient": FieldContract(FieldKind.IDENTIFIER),
        "amount": FieldContract(FieldKind.NUMBER),
    })}
    lp = loop(handler, field_contracts=contracts,
              consequence_overrides={"send_money": ConsequenceClass.CONSEQUENTIAL})
    original = lp._evaluate_tool_intent

    def _patched(intent, env):
        _d, pending = original(intent, env)
        return PolicyDecision(
            kind=PolicyDecisionKind.TRANSFORM, reason="rewritten",
            transformed_payload={"recipient": ALLOWED_IBAN},  # amount dropped
        ), pending

    lp._evaluate_tool_intent = _patched  # type: ignore[method-assign]
    resolved = await lp._resolve_tool_intent(
        call("send_money", {"recipient": ALLOWED_IBAN, "amount": 1}), envelope(["send_money"])
    )
    assert not resolved.approved and "'amount' is absent" in resolved.reason
    assert handler.received == []


# ── a call named in advance is bound in its prepared representation ──────────


@pytest.mark.asyncio
async def test_an_approval_requested_for_a_named_call_fits_the_prepared_call():
    from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

    seen = []

    async def approver(request):
        seen.append(dict(request.args))
        return True

    handler = SendMoney(Bank())
    contracts = {"send_money": SinkContract(fields={
        "recipient": FieldContract(FieldKind.IDENTIFIER),
        "amount": FieldContract(FieldKind.NUMBER, decimal_string=True),
    })}
    lp = loop(handler, field_contracts=contracts, escalation_callback=approver,
              consequence_overrides={"send_money": ConsequenceClass.CONSEQUENTIAL})
    env = envelope(["send_money"], ceiling=ConsequenceClass.REVERSIBLE, allow_escalation=True)
    named = {"recipient": ALLOWED_IBAN, "amount": "40"}  # the model's representation

    def _escalate(args):
        return ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "escalate_policy", "tool_use_id": "e1",
                     "args": {"tool": "send_money", "reason": "pay", "paths": [],
                              "args": args}},
            node_id="n1",
        )

    granted = await lp._escalation.grant_from_intent(_escalate(named), env, lp._trace_events)
    assert granted.get("scope") == "call"
    assert seen == [{"recipient": ALLOWED_IBAN, "amount": 40}]  # shown prepared
    resolved = await lp._resolve_tool_intent(call("send_money", dict(named)), env)
    assert resolved.approved and handler.received == [{"recipient": ALLOWED_IBAN, "amount": 40}]

    refused = await lp._escalation.grant_from_intent(
        _escalate({"recipient": ALLOWED_IBAN}), env, lp._trace_events
    )
    assert refused["error"] == "escalation_denied" and "'amount' is absent" in refused["reason"]


def test_a_denial_carries_the_set_the_kernel_judged():
    """A host that executes a denied call (observe-only) must be able to tell the
    judged set from what it ran: prepared when preparation succeeded, the
    proposal as received when preparation itself refused it."""
    pols = {"send_money": [enum("recipient", {ALLOWED_IBAN}), numeric_range("amount", 0, 10)]}
    contracts = {"send_money": SinkContract(fields={
        "recipient": FieldContract(FieldKind.IDENTIFIER),
        "amount": FieldContract(FieldKind.NUMBER, decimal_string=True),
    })}
    gov = ToolCallGovernor(value_policies=pols, field_contracts=contracts,
                           consequence_overrides={"send_money": ConsequenceClass.CONSEQUENTIAL})
    over = gov.evaluate("send_money", {"recipient": ALLOWED_IBAN, "amount": "50"})
    assert not over.allowed and over.category == "value_policy"
    assert over.args == {"recipient": ALLOWED_IBAN, "amount": 50}
    absent = gov.evaluate("send_money", {"recipient": ALLOWED_IBAN})
    assert absent.category == "consumption_contract"
    assert absent.args == {"recipient": ALLOWED_IBAN}


def test_structured_field_carries_a_json_value_and_takes_no_predicate():
    c = SinkContract(fields={
        "to": FieldContract(FieldKind.IDENTIFIER_LIST),
        "attachments": FieldContract(FieldKind.STRUCTURED, unused_when_absent=True, null_is_absent=True),
    })
    att = [{"type": "file", "file_id": "7"}]
    out, why = prepare_args({"to": ["a"], "attachments": att}, c)
    assert why is None and out["attachments"] == att
    _, why = prepare_args({"to": ["a"], "attachments": [object()]}, c)
    assert why and "JSON" in why
    from axor_core.policy.value_policy import enum
    errs = validate_consumption_contracts({"s": [enum("attachments", ["x"])]}, {"s": c})
    assert errs
