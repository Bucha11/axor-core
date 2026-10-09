"""EXPERIMENTAL prototype: grant identity and the shared-witness check.

Scope of what these tests establish: the STORAGE and the CHECK. Grants are
issued structurally here — the test states the pairing — so nothing below says
anything about deriving grants from a request. That is a separate problem with
its own failure modes, and the prototype deliberately contains none of it.

Two tests exist to mark the boundary rather than to pass:
``test_one_grant_covering_both_payments_admits_mixing`` shows the mechanism
cannot detect a grant issued too coarsely, and
``test_scoped_never_widens_the_existing_origin_check`` shows a shared witness is
an additional requirement, never a replacement for the origin checks.
"""

from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintSource, TrustedOrigin
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine
from axor_core.taint.grants import MAX_GRANTS, Grant, GrantIndex, canonical

ALICE = "GB29NWBK60161331926819"
BOB = "US122000000121212121212"
RENT = 2200
REFUND = 50

SEND = "send_money"
SCHEDULE = "schedule_transaction"
GROUP = frozenset({"recipient", "amount"})

REQUEST = f"Pay the rent of {RENT} to {ALICE}, and refund {REFUND} to {BOB}."


def _index() -> GrantIndex:
    """The two grants the request licenses, issued structurally — one per
    instruction, each naming the position of each value."""
    index = GrantIndex()
    index.issue(Grant.of("rent-instruction", SEND, {"recipient": ALICE, "amount": RENT}))
    index.issue(Grant.of("refund-instruction", SEND, {"recipient": BOB, "amount": REFUND}))
    return index


def _call(recipient: str, amount: object) -> dict[str, object]:
    return {"recipient": recipient, "amount": amount, "subject": "payment"}


# ── the pairs the request licensed, and the ones it did not ──────────────────

@pytest.mark.parametrize(("recipient", "amount"), [(ALICE, RENT), (BOB, REFUND)])
def test_the_licensed_pairs_share_a_grant(recipient: str, amount: int) -> None:
    assert _index().jointly_granted(SEND, _call(recipient, amount), GROUP) is True


@pytest.mark.parametrize(("recipient", "amount"), [(ALICE, REFUND), (BOB, RENT)])
def test_the_mixed_pairs_share_no_grant(recipient: str, amount: int) -> None:
    """Each value is licensed somewhere; the pair is licensed nowhere."""
    index = _index()
    assert index.witnesses(SEND, "recipient", recipient)
    assert index.witnesses(SEND, "amount", amount)
    assert index.jointly_granted(SEND, _call(recipient, amount), GROUP) is False


# ── position binding: the same token in the wrong role ───────────────────────

def test_a_value_licensed_as_an_amount_does_not_license_a_recipient() -> None:
    """A numeric account id equal to an approved amount must not inherit it."""
    index = GrantIndex()
    index.issue(Grant.of("g1", SEND, {"recipient": ALICE, "amount": 2200}))
    assert index.witnesses(SEND, "amount", 2200)
    assert index.witnesses(SEND, "recipient", 2200) == frozenset()


def test_a_grant_for_one_sink_does_not_license_another_operation() -> None:
    index = _index()
    assert index.jointly_granted(SEND, _call(ALICE, RENT), GROUP) is True
    assert index.jointly_granted(SCHEDULE, _call(ALICE, RENT), GROUP) is False


# ── repeated values across grants ────────────────────────────────────────────

def test_a_recipient_in_two_grants_keeps_each_grant_s_own_amount() -> None:
    index = GrantIndex()
    index.issue(Grant.of("jan", SEND, {"recipient": ALICE, "amount": 2200}))
    index.issue(Grant.of("feb", SEND, {"recipient": ALICE, "amount": 2300}))
    assert index.shared_witnesses(SEND, _call(ALICE, 2200), GROUP) == {"jan"}
    assert index.shared_witnesses(SEND, _call(ALICE, 2300), GROUP) == {"feb"}
    # An amount from neither grant shares nothing, even though the recipient is
    # licensed by both.
    assert index.jointly_granted(SEND, _call(ALICE, 9999), GROUP) is False


def test_the_same_amount_under_two_recipients_does_not_cross_over() -> None:
    index = GrantIndex()
    index.issue(Grant.of("g1", SEND, {"recipient": ALICE, "amount": 100}))
    index.issue(Grant.of("g2", SEND, {"recipient": BOB, "amount": 100}))
    assert index.shared_witnesses(SEND, _call(ALICE, 100), GROUP) == {"g1"}
    assert index.shared_witnesses(SEND, _call(BOB, 100), GROUP) == {"g2"}


# ── unknown witness, and the fail-closed edges ───────────────────────────────

def test_a_value_in_no_grant_gets_no_scoped_permission() -> None:
    index = _index()
    attacker = "DE89370400440532013000"
    assert index.witnesses(SEND, "recipient", attacker) == frozenset()
    assert index.jointly_granted(SEND, _call(attacker, RENT), GROUP) is False


def test_an_empty_group_grants_nothing() -> None:
    """An unsupported call must not gain a permission by naming no group."""
    assert _index().jointly_granted(SEND, _call(ALICE, RENT), frozenset()) is False


def test_a_group_argument_absent_from_the_call_grants_nothing() -> None:
    index = _index()
    assert index.jointly_granted(SEND, {"recipient": ALICE}, GROUP) is False


@pytest.mark.parametrize("value", [None, True, False, ""])
def test_a_value_with_no_comparable_form_carries_no_witness(value: object) -> None:
    assert canonical(value) is None
    index = GrantIndex()
    index.issue(Grant.of("g", SEND, {"recipient": ALICE, "amount": value}))
    assert index.witnesses(SEND, "amount", value) == frozenset()


def test_issuance_is_bounded_and_stops_rather_than_growing() -> None:
    index = GrantIndex()
    for i in range(MAX_GRANTS):
        assert index.issue(Grant.of(f"g{i}", SEND, {"recipient": f"R{i}", "amount": i}))
    assert index.issue(Grant.of("overflow", SEND, {"recipient": ALICE, "amount": RENT})) is False
    assert index.saturated is True
    assert index.jointly_granted(SEND, _call(ALICE, RENT), GROUP) is False


# ── canonicalisation must agree with the existing index ──────────────────────

@pytest.mark.parametrize("amount", [2200, 2200.0, "2200", "2,200", "2200.00"])
def test_the_amount_s_spelling_does_not_change_the_grant(amount: object) -> None:
    """Disagreement here would deny licensed tuples for a reason unrelated to
    authority, so the module reuses the trusted index's own canonicalisation."""
    assert _index().jointly_granted(SEND, _call(ALICE, amount), GROUP) is True


# ── origin mode ──────────────────────────────────────────────────────────────

def test_a_tool_issued_grant_does_not_clear_a_group_under_request_only() -> None:
    index = GrantIndex()
    index.issue(Grant.of(
        "from-a-read", SEND, {"recipient": ALICE, "amount": RENT},
        origin=TrustedOrigin.TOOL,
    ))
    args = _call(ALICE, RENT)
    assert index.jointly_granted(SEND, args, GROUP) is True          # any origin
    assert index.jointly_granted(
        SEND, args, GROUP,
        accept=frozenset({TrustedOrigin.TASK, TrustedOrigin.OPERATOR}),
    ) is False                                                        # request-only


# ── the two boundary tests ───────────────────────────────────────────────────

def test_one_grant_covering_both_payments_admits_mixing() -> None:
    """THE LIMIT OF THE MECHANISM, pinned on purpose.

    A single grant naming all four values — what a coarse issuance step would
    produce from "pay 2200 to Alice and 50 to Bob" read as one instruction —
    licenses every pairing. The intersection is non-empty and the check is
    behaving exactly as specified; the error is in issuance, and no property of
    this index can see it.

    So a shared witness proves common provenance of the grant, never that the
    grant authorises the combination. Any claim about authorisation has to be
    discharged where grants are issued, and that is not this module.
    """
    coarse = GrantIndex()
    coarse.issue(Grant.of(
        "one-sentence", SEND,
        {"recipient": ALICE, "amount": RENT, "recipient2": BOB, "amount2": REFUND},
    ))
    # Issued against the same positions a mixed call would use:
    coarse.issue(Grant.of("one-sentence-b", SEND, {"recipient": BOB, "amount": RENT}))
    assert coarse.jointly_granted(SEND, _call(BOB, RENT), GROUP) is True


def test_scoped_never_widens_the_existing_origin_check() -> None:
    """A shared witness is an extra requirement, not a substitute.

    The grant here names a recipient the request never contained, so the engine's
    own origin check refuses it while the grant index is satisfied. The scoped
    rule may only narrow the conjunction.
    """
    attacker = "DE89370400440532013000"
    index = GrantIndex()
    index.issue(Grant.of("bogus", SEND, {"recipient": attacker, "amount": RENT}))
    assert index.jointly_granted(SEND, _call(attacker, RENT), GROUP) is True

    engine = TaintEngine(integrity_default="context", integrity_origins="request-only")
    engine.register_trusted(REQUEST, TrustedOrigin.TASK)
    engine.register_value("an untrusted note", CausalRoot.external_read(TaintSource.WEB))
    covered = engine.is_trusted(
        {"recipient": attacker, "amount": RENT},
        include_scalars=True, integrity_sink=True,
    )
    assert covered is False
    # The conjunction the prototype proposes: existing origin check AND grant.
    assert (covered and index.jointly_granted(SEND, _call(attacker, RENT), GROUP)) is False


def test_the_check_is_finite_id_equality_which_classify_already_admits() -> None:
    """Compatibility of the CHECK, which is all it is.

    `classify` takes only (codomain kind, consumption mode), so comparing
    witness ids — a finite enum consumed as a case split — classifies decidable
    without any change to the classifier. That says nothing about whether the
    request was divided into the right grants or the values bound to the right
    arguments; correct issuance is a separate premise, not a corollary.
    """
    from axor_core.kernel.decidability import (
        CodomainKind,
        ConsumptionMode,
        DecidabilityVerdict,
        classify,
    )

    result = classify(CodomainKind.ENUM, ConsumptionMode.CASE_SPLIT)
    assert result.verdict is DecidabilityVerdict.DECIDABLE_PASS
