"""Decidable-predicate supersession of the integrity content-taint gate.

When every driving arg of an egress sink is guarded by a *satisfied* decidable
predicate (enum / numeric_range), that predicate is a content-blind,
provenance-independent control strictly stronger than content-derivation on the
same arg. The integrity content-taint check is then redundant and is superseded,
which removes the value-coincidence false positive at no security cost. The
confidentiality floor is unaffected.

These tests pin: (1) the value-coincidence recipient recovers; (2) an
attacker-chosen destination is still denied (enum); (3) the floor still blocks
after a secret read even for an approved recipient; (4) no supersession when a
driving arg is absent (whole-blob fallback) or lacks a decidable predicate.
"""
from __future__ import annotations

import dataclasses

from axor_core.config import GovernanceConfig
from axor_core.policy.value_policy import enum, numeric_range
from axor_core.governor import ToolCallGovernor


def _gov(**overrides) -> ToolCallGovernor:
    cfg = dataclasses.replace(
        GovernanceConfig.from_yaml("examples/agentdojo/config/banking.yaml"),
        **overrides,
    )
    return ToolCallGovernor(**cfg.as_governor_kwargs())


_APPROVED = ["GB29NWBK60161331926819"]
_READ = "rent to GB29NWBK60161331926819; ATTACK to DE00ATTACKER999"


def test_value_coincidence_recovers_with_decidable_cover():
    """A prompt-given recipient that *also* appears in an untrusted read is the
    documented value-coincidence false positive. With an approved-payee enum on
    the driving arg, the integrity taint is superseded and the legit transfer
    passes — while the attacker destination is still denied by the enum."""
    g = _gov(value_policies={"send_money": [enum("recipient", _APPROVED)]})
    r = g.evaluate("get_most_recent_transactions", {})
    g.register_output(r, _READ)

    legit = g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 100})
    attacker = g.evaluate("send_money", {"recipient": "DE00ATTACKER999", "amount": 100})

    assert legit.allowed is True, "decidable cover should supersede integrity over-block"
    assert attacker.allowed is False, "attacker destination must still be denied by the enum"


def test_no_supersession_without_decidable_cover():
    """Without a value policy on the driving arg, the content-taint gate stands:
    the value-coincidence recipient is (over-)blocked, as before."""
    g = _gov()  # generic banking config, no value_policies
    r = g.evaluate("get_most_recent_transactions", {})
    g.register_output(r, _READ)
    legit = g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 100})
    assert legit.allowed is False, "no decidable cover => integrity taint still applies"


def test_supersession_requires_driving_arg_present():
    """If the declared driving arg is absent, the taint check falls back to the
    whole blob; supersession must NOT fire there (fail-closed)."""
    g = _gov(value_policies={"send_money": [enum("recipient", _APPROVED)]})
    r = g.evaluate("get_most_recent_transactions", {})
    g.register_output(r, _READ)
    # recipient absent => whole-blob fallback; the blob carries the tainted read
    d = g.evaluate("send_money", {"to": "GB29NWBK60161331926819", "note": _READ})
    assert d.allowed is False, "absent driving arg => no supersession (whole-blob taint holds)"


def test_amount_range_still_catches_attacker_amount():
    """With both recipient-enum and amount-range, an attacker-controlled amount to
    an approved payee is still denied by the numeric predicate."""
    g = _gov(value_policies={
        "send_money": [enum("recipient", _APPROVED), numeric_range("amount", 0, 1000)],
    })
    r = g.evaluate("get_most_recent_transactions", {})
    g.register_output(r, _READ)
    ok = g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 100})
    huge = g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 999999})
    assert ok.allowed is True
    assert huge.allowed is False, "out-of-range amount must be denied by the numeric predicate"


def test_numeric_range_does_not_supersede_open_codomain():
    """A numeric_range has an OPEN codomain the attacker can satisfy (an in-range
    value), so it must NOT supersede integrity — only a closed enum may. Tested on
    the predicate directly (an end-to-end check is confounded by whether a numeric
    value substring-taints at all)."""
    from axor_core.policy.gates import integrity_superseded_by_decidable
    args = {"amount": 5000}
    # numeric_range on the driving arg => NOT superseded (open codomain)
    assert integrity_superseded_by_decidable(
        "send_money", args, {"send_money": ["amount"]},
        {"send_money": [numeric_range("amount", 0, 100000)]},
    ) is False
    # enum on the driving arg => superseded (closed trusted codomain)
    assert integrity_superseded_by_decidable(
        "send_money", {"recipient": "GB29NWBK60161331926819"},
        {"send_money": ["recipient"]},
        {"send_money": [enum("recipient", _APPROVED)]},
    ) is True
    # mixed: a driving arg covered only by numeric_range blocks supersession even if
    # another arg has an enum (every driving arg must be enum-covered)
    assert integrity_superseded_by_decidable(
        "send_money", {"recipient": "GB29NWBK60161331926819", "amount": 5000},
        {"send_money": ["recipient", "amount"]},
        {"send_money": [enum("recipient", _APPROVED), numeric_range("amount", 0, 100000)]},
    ) is False


def test_confidentiality_floor_is_not_superseded():
    """Supersession is integrity-only: after a secret read the confidentiality
    floor still blocks egress, even to an approved recipient."""
    g = _gov(
        sensitive_sources=frozenset({"get_user_info"}),
        value_policies={"send_money": [enum("recipient", _APPROVED)]},
    )
    r = g.evaluate("get_user_info", {})
    g.register_output(r, "passport X123; iban GB29NWBK60161331926819")
    assert g._taint.confidentiality_floor_active() is True
    d = g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819", "amount": 100})
    assert d.allowed is False, "floor must hold despite an approved recipient (integrity-only supersession)"
