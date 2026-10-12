"""T4 against a real consumer: the predicate's verdict versus what the handler did.

``validate_value_policies`` classifies a predicate by its OWN kind: an ``enum`` is
membership, a ``numeric_range`` a comparison, both decidable. That makes the CHECK
exact. Whether the property checked is the property consumed depends on the
handler, which the classifier never looks at. These tests put a real handler
(:mod:`tests.consumers`) behind the real loop and read its own state.

Each case runs twice:

  * **without a consumption contract** — the configuration ``validate_value_policies``
    accepts today. Where the handler then produces an effect outside the
    predicate's codomain, the test asserts that effect: it is the measured
    negative control, not a hypothetical.
  * **with a contract** (:mod:`axor_core.kernel.consumption`) — registration or
    preparation refuses the incompatible call or configuration, and the correct
    variant of the same call executes with the value that was checked.

Not every candidate is a gap. A comma-joined recipient string and an exponent
string are already refused by the exact predicates; those cases are kept as
controls so the claim stays proportional: the contract closes presence, aliasing,
shape and path-resolution gaps, and turns some over-denials into correct
executions. It does not prove the handler honours the contract.
"""
from __future__ import annotations

import os

import pytest

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.consumption import (
    FieldContract,
    FieldKind,
    SinkContract,
    conformance_obligations,
    validate_consumption_contracts,
)
from axor_core.kernel.registration import validate_value_policies
from axor_core.policy.value_policy import enum, numeric_range
from tests.consumers import (
    ALLOWED_IBAN,
    ATTACKER_IBAN,
    Bank,
    Publish,
    SendMoney,
    call,
    envelope,
    loop,
)

LIMIT = 1000
POLICIES = {
    "send_money": [enum("recipient", {ALLOWED_IBAN}), numeric_range("amount", 0, LIMIT)],
}
CONTRACTS = {
    "send_money": SinkContract(fields={
        "recipient": FieldContract(FieldKind.IDENTIFIER),
        "amount": FieldContract(FieldKind.NUMBER, decimal_string=True),
        "subject": FieldContract(FieldKind.TEXT, default=""),
        "date": FieldContract(FieldKind.TEXT, unused_when_absent=True, null_is_absent=True),
    }),
}
CLASSES = {"send_money": ConsequenceClass.CONSEQUENTIAL, "publish": ConsequenceClass.CONSEQUENTIAL}


def _bank_loop(contracted: bool):
    bank = Bank()
    handler = SendMoney(bank)
    lp = loop(
        handler,
        value_policies=POLICIES,
        consequence_overrides=CLASSES,
        field_contracts=CONTRACTS if contracted else None,
    )
    return lp, handler, bank


async def _send(lp, args: dict):
    return await lp._resolve_tool_intent(call("send_money", args), envelope(["send_money"]))


def _within_policy(transfer: tuple[str, float]) -> bool:
    """The property the predicates were registered to establish, read off the
    bank's own ledger."""
    to, amount = transfer
    return to == ALLOWED_IBAN and 0 <= amount <= LIMIT


# ── registration: the classifier accepts what the field contract refuses ─────


def test_both_configurations_pass_the_predicate_only_classifier():
    """The baseline: the predicate kinds are decidable, so the existing check has
    nothing to object to — for the bank tool and for an enum on a resolved path."""
    assert validate_value_policies(POLICIES) == []
    assert validate_value_policies({"publish": [enum("path", {"reports/q1.txt"})]}) == []


def test_an_enum_on_a_resolved_path_is_refused_by_the_field_contract():
    errs = validate_consumption_contracts(
        {"publish": [enum("path", {"reports/q1.txt"})]},
        {"publish": SinkContract(fields={"path": FieldContract(FieldKind.PATH)})},
    )
    assert len(errs) == 1
    assert "fuzz obligation" in errs[0] and "path" in errs[0]


@pytest.mark.parametrize("kind", [FieldKind.INTERPRETED, FieldKind.TEXT])
def test_no_decidable_predicate_guards_an_interpreted_field(kind):
    errs = validate_consumption_contracts(
        {"fetch": [enum("url", {"https://api.example.com/v1"})]},
        {"fetch": SinkContract(fields={"url": FieldContract(kind)})},
    )
    assert errs and "fuzz obligation" in errs[0]


def test_a_range_on_an_identifier_decides_nothing_the_handler_uses():
    errs = validate_consumption_contracts(
        {"send_money": [numeric_range("recipient", 0, 10)]},
        {"send_money": SinkContract(fields={"recipient": FieldContract(FieldKind.IDENTIFIER)})},
    )
    assert errs and "decides nothing" in errs[0]


def test_a_predicate_on_a_field_the_contract_does_not_name_is_refused():
    errs = validate_consumption_contracts(
        {"send_money": [enum("to", {ALLOWED_IBAN})]}, CONTRACTS,
    )
    assert errs and "does not name" in errs[0]


def test_the_bank_contract_registers():
    assert validate_consumption_contracts(POLICIES, CONTRACTS, strict=True) == []


# ── presence: an absent argument passes the predicate, the handler defaults ──


@pytest.mark.asyncio
async def test_absent_amount_without_contract_sweeps_the_account():
    lp, handler, bank = _bank_loop(contracted=False)
    resolved = await _send(lp, {"recipient": ALLOWED_IBAN})
    assert resolved.approved  # the range predicate does not constrain an absent value
    assert bank.transfers == [(ALLOWED_IBAN, 50_000.0)]
    assert not _within_policy(bank.transfers[0])  # the effect the policy forbade


@pytest.mark.asyncio
async def test_absent_amount_with_contract_is_refused_and_the_full_call_runs():
    lp, handler, bank = _bank_loop(contracted=True)
    refused = await _send(lp, {"recipient": ALLOWED_IBAN})
    assert not refused.approved
    assert "required argument 'amount' is absent" in refused.reason
    assert bank.transfers == []

    ok = await _send(lp, {"recipient": ALLOWED_IBAN, "amount": 500})
    assert ok.approved
    assert bank.transfers == [(ALLOWED_IBAN, 500.0)]
    # the declared default was materialised before the gates, and reached the handler
    assert handler.received == [{"recipient": ALLOWED_IBAN, "amount": 500, "subject": ""}]


# ── aliasing: the handler reads a field no predicate saw ─────────────────────


@pytest.mark.asyncio
async def test_alias_field_without_contract_redirects_the_transfer():
    lp, _, bank = _bank_loop(contracted=False)
    args = {"recipient": ALLOWED_IBAN, "amount": 10, "iban": ATTACKER_IBAN}
    assert (await _send(lp, args)).approved
    assert bank.transfers == [(ATTACKER_IBAN, 10.0)]
    assert not _within_policy(bank.transfers[0])


@pytest.mark.asyncio
async def test_alias_field_with_closed_contract_is_refused():
    lp, _, bank = _bank_loop(contracted=True)
    args = {"recipient": ALLOWED_IBAN, "amount": 10, "iban": ATTACKER_IBAN}
    resolved = await _send(lp, args)
    assert not resolved.approved
    assert "['iban']" in resolved.reason
    assert bank.transfers == []


# ── shape: a list where the handler consumes a scalar ────────────────────────


@pytest.mark.asyncio
async def test_list_in_a_scalar_field_without_contract_books_an_unvalidated_account():
    """The enum decides a collection element by element, which is right for a
    list-valued destination. For a scalar field the handler formats whatever it
    gets, so the account it books is a string no predicate ever checked."""
    lp, _, bank = _bank_loop(contracted=False)
    assert (await _send(lp, {"recipient": [ALLOWED_IBAN], "amount": 10})).approved
    assert bank.transfers == [(f"['{ALLOWED_IBAN}']", 10.0)]
    assert not _within_policy(bank.transfers[0])


@pytest.mark.asyncio
async def test_list_in_a_scalar_field_with_contract_is_refused_and_the_scalar_runs():
    lp, _, bank = _bank_loop(contracted=True)
    refused = await _send(lp, {"recipient": [ALLOWED_IBAN], "amount": 10})
    assert not refused.approved and "single identifier" in refused.reason
    assert (await _send(lp, {"recipient": ALLOWED_IBAN, "amount": 10})).approved
    assert bank.transfers == [(ALLOWED_IBAN, 10.0)]


# ── representation: numeric conversion done once, before the check ───────────


def test_the_handler_would_parse_an_exponent_string():
    """Why a string must never reach this handler: its own float() reads "1e5" as
    100000 — a value no range check was shown."""
    bank = Bank()
    bank.send_money(recipient=ALLOWED_IBAN, amount="1e5")
    assert bank.transfers == [(ALLOWED_IBAN, 100_000.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("contracted", [False, True])
async def test_exponent_string_is_refused_either_way(contracted):
    """Control: the exact predicate already refuses a non-number, so this is not
    a gap today. With the contract it is refused one step earlier, at
    preparation, by the declared conversion rule."""
    lp, handler, bank = _bank_loop(contracted)
    resolved = await _send(lp, {"recipient": ALLOWED_IBAN, "amount": "1e5"})
    assert not resolved.approved
    assert bank.transfers == [] and handler.received == []
    if contracted:
        assert "plain decimal" in resolved.reason


@pytest.mark.asyncio
async def test_decimal_string_is_over_denied_without_contract_and_runs_with_one():
    lp, _, bank = _bank_loop(contracted=False)
    assert not (await _send(lp, {"recipient": ALLOWED_IBAN, "amount": "500"})).approved

    lp, handler, bank = _bank_loop(contracted=True)
    assert (await _send(lp, {"recipient": ALLOWED_IBAN, "amount": "500"})).approved
    assert handler.received[0]["amount"] == 500  # a number: the handler parses nothing
    assert bank.transfers == [(ALLOWED_IBAN, 500.0)]


@pytest.mark.asyncio
async def test_decimal_string_over_the_limit_is_denied_after_conversion():
    lp, _, bank = _bank_loop(contracted=True)
    resolved = await _send(lp, {"recipient": ALLOWED_IBAN, "amount": "5000"})
    assert not resolved.approved and "fails range" in resolved.reason
    assert bank.transfers == []


# ── a string the check and a consumer would split differently ────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("contracted", [False, True])
async def test_comma_joined_recipients_are_refused_either_way(contracted):
    """Control: a handler that split on commas would pay both accounts, but exact
    membership refuses the joined string as a whole. Not a gap; kept so the claim
    does not overreach."""
    lp, _, bank = _bank_loop(contracted)
    args = {"recipient": f"{ALLOWED_IBAN},{ATTACKER_IBAN}", "amount": 10}
    assert not (await _send(lp, args)).approved
    assert bank.transfers == []


# ── path resolution: exact membership on a string the resolver interprets ────


@pytest.fixture
def reports(tmp_path):
    root = tmp_path / "reports_root"
    (root / "reports").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"API_KEY=s3cr3t")
    (root / "reports" / "q2.txt").write_bytes(b"q2 numbers")
    # the allowlisted name is a symlink out of the root
    os.symlink(secret, root / "reports" / "q1.txt")
    return str(root)


@pytest.mark.asyncio
async def test_allowlisted_path_publishes_whatever_it_resolves_to(reports):
    handler = Publish(reports)
    lp = loop(handler, value_policies={"publish": [enum("path", {"reports/q1.txt"})]},
              consequence_overrides=CLASSES)
    resolved = await lp._resolve_tool_intent(
        call("publish", {"path": "reports/q1.txt"}), envelope(["publish"])
    )
    assert resolved.approved
    assert handler.public == [b"API_KEY=s3cr3t"]  # equal strings, a different effect


def test_path_contract_refuses_that_configuration(reports):
    with pytest.raises(ValueError, match="fuzz obligation"):
        loop(Publish(reports), value_policies={"publish": [enum("path", {"reports/q1.txt"})]},
             field_contracts={"publish": SinkContract(fields={"path": FieldContract(FieldKind.PATH)})})


@pytest.mark.asyncio
async def test_path_contract_without_a_decidable_guard_registers_and_runs(reports):
    """The corrected configuration declares the field for what it is and puts no
    decidable predicate on it: the path rides the fuzz/path obligation, and an
    ordinary call executes."""
    handler = Publish(reports)
    lp = loop(handler, consequence_overrides=CLASSES,
              field_contracts={"publish": SinkContract(fields={"path": FieldContract(FieldKind.PATH)})})
    resolved = await lp._resolve_tool_intent(
        call("publish", {"path": "reports/q2.txt"}), envelope(["publish"])
    )
    assert resolved.approved and handler.public == [b"q2 numbers"]


# ── the synchronous path carries the prepared set to the caller ──────────────


def test_governor_hands_back_the_prepared_arguments():
    gov = ToolCallGovernor(value_policies=POLICIES, field_contracts=CONTRACTS,
                           consequence_overrides=CLASSES)
    refused = gov.evaluate("send_money", {"recipient": ALLOWED_IBAN})
    assert not refused.allowed and refused.category == "consumption_contract"

    ok = gov.evaluate("send_money", {"recipient": ALLOWED_IBAN, "amount": "250"})
    assert ok.allowed
    assert ok.args == {"recipient": ALLOWED_IBAN, "amount": 250, "subject": ""}
    bank = Bank()
    bank.send_money(**ok.args)
    assert all(_within_policy(t) for t in bank.transfers)


def test_without_contract_the_governor_returns_the_proposed_arguments():
    gov = ToolCallGovernor(value_policies=POLICIES, consequence_overrides=CLASSES)
    d = gov.evaluate("send_money", {"recipient": ALLOWED_IBAN, "amount": 5})
    assert d.allowed and d.args == {"recipient": ALLOWED_IBAN, "amount": 5}


# ── the obligation report ────────────────────────────────────────────────────


def test_conformance_report_names_the_ground_of_each_guarded_argument():
    pols = {**POLICIES, "email": [enum("to", {"a@b.c"})], "post": [enum("url", {"u"})]}
    rows = conformance_obligations(pols, CONTRACTS, attested={"email"})
    grounds = {(r.sink, r.arg): r.ground for r in rows}
    assert grounds == {
        ("send_money", "recipient"): "contract",
        ("send_money", "amount"): "contract",
        ("email", "to"): "attested",
        ("post", "url"): "unverified",
    }
