"""Relational value predicates — the policy can pin the PAIRING, not just the fields.

Independent per-argument predicates constrain the product of their codomains. An
allowlist of recipients and a range of amounts admit every recipient with every
in-range amount, including pairings the operator never intended — the swapped-payment
counterexample. A decidable projection makes each check sound; it does not make the
policy a description of the effect. `tuple_enum` and `keyed_range` close that gap
while staying inside the decidable fraction (case-split, then numeric).

Also pinned here: the three callers that must agree on what counts as a CLOSED
operator-declared codomain (integrity supersession, the STRICT egress obligations)
now read one classifier, ``closed_allowlist_args``.
"""
from __future__ import annotations

import pytest

from axor_core.config import GovernanceConfig
from axor_core.kernel.registration import (
    validate_driving_arg_allowlists,
    validate_egress_allowlists,
    validate_value_policies,
)
from axor_core.policy.gates import integrity_superseded_by_decidable
from axor_core.policy.value_policy import (
    check_value_policies,
    closed_allowlist_args,
    enum,
    keyed_range,
    numeric_range,
    tuple_enum,
)

_SINK = "send_money"


class TestSwappedPayments:
    """The counterexample, as a test: per-argument predicates admit the swap."""

    PRODUCT = {
        _SINK: [
            enum("recipient", {"acme", "globex"}),
            numeric_range("amount", 0, 1000),
        ]
    }
    RELATION = {
        _SINK: [
            keyed_range("recipient", "amount", {"acme": (0, 1000), "globex": (0, 50)})
        ]
    }

    def test_product_admits_the_swap(self):
        """Both calls satisfy every per-argument predicate: the allowlisted
        recipients and the in-range amounts are simply re-paired."""
        assert check_value_policies(_SINK, {"recipient": "acme", "amount": 40}, self.PRODUCT) is None
        assert check_value_policies(_SINK, {"recipient": "globex", "amount": 900}, self.PRODUCT) is None

    def test_relation_rejects_the_swap(self):
        assert check_value_policies(_SINK, {"recipient": "acme", "amount": 900}, self.RELATION) is None
        assert check_value_policies(_SINK, {"recipient": "globex", "amount": 40}, self.RELATION) is None
        denial = check_value_policies(_SINK, {"recipient": "globex", "amount": 900}, self.RELATION)
        assert denial is not None and "globex" in denial

    def test_tuple_enum_pins_a_finite_pairing(self):
        pol = {_SINK: [tuple_enum(("recipient", "currency"), {("acme", "EUR"), ("globex", "USD")})]}
        assert check_value_policies(_SINK, {"recipient": "acme", "currency": "EUR"}, pol) is None
        assert check_value_policies(_SINK, {"recipient": "acme", "currency": "USD"}, pol) is not None


class TestFailClosed:

    def test_partial_tuple_is_denied_not_skipped(self):
        """A per-argument predicate skips an absent argument (presence is another
        gate's concern). A relational one cannot: with part of the tuple missing the
        relation is unverifiable, and passing it would admit the pairing the
        predicate exists to forbid."""
        pol = {_SINK: [tuple_enum(("recipient", "currency"), {("acme", "EUR")})]}
        denial = check_value_policies(_SINK, {"recipient": "acme"}, pol)
        assert denial is not None and "missing" in denial

    def test_absent_tuple_entirely_is_not_a_violation(self):
        pol = {_SINK: [tuple_enum(("recipient", "currency"), {("acme", "EUR")})]}
        assert check_value_policies(_SINK, {"memo": "hi"}, pol) is None

    def test_unknown_key_fails_closed(self):
        pol = {_SINK: [keyed_range("recipient", "amount", {"acme": (0, 10)})]}
        denial = check_value_policies(_SINK, {"recipient": "evil", "amount": 1}, pol)
        assert denial is not None and "no declared range" in denial

    def test_both_relational_kinds_are_decidable(self):
        """validate_value_policies rejects a predicate whose projection is not
        decidable; a relational predicate must not be rejected by it."""
        assert validate_value_policies(TestSwappedPayments.RELATION) == []
        assert validate_value_policies(
            {_SINK: [tuple_enum(("a", "b"), {("x", "y")})]}
        ) == []


class TestClosedCodomainClassification:
    """One classifier, three callers. A closed codomain is one the attacker cannot
    choose a value outside of; an open interval is not."""

    def test_enum_and_tuple_enum_are_closed(self):
        assert closed_allowlist_args([enum("recipient", {"acme"})]) == {"recipient"}
        assert closed_allowlist_args(
            [tuple_enum(("recipient", "currency"), {("acme", "EUR")})]
        ) == {"recipient", "currency"}

    def test_numeric_range_is_not_closed(self):
        assert closed_allowlist_args([numeric_range("amount", 0, 10)]) == frozenset()

    def test_keyed_range_is_closed_on_its_key_only(self):
        """The key set doubles as an allowlist (an undeclared key is a denial); the
        VALUE is an open interval the attacker can satisfy, so it is not."""
        preds = [keyed_range("recipient", "amount", {"acme": (0, 10)})]
        assert closed_allowlist_args(preds) == {"recipient"}

    def test_empty_allowlists_are_not_closed_codomains(self):
        assert closed_allowlist_args([enum("recipient", set())]) == frozenset()
        assert closed_allowlist_args([tuple_enum(("a", "b"), set())]) == frozenset()

    def test_supersession_accepts_a_relational_allowlist(self):
        args = {"recipient": "acme", "currency": "EUR"}
        drivers = {_SINK: ["recipient", "currency"]}
        assert integrity_superseded_by_decidable(
            _SINK, args, drivers,
            {_SINK: [tuple_enum(("recipient", "currency"), {("acme", "EUR")})]},
        )

    def test_supersession_still_refuses_a_numeric_range(self):
        assert not integrity_superseded_by_decidable(
            _SINK, {"amount": 5}, {_SINK: ["amount"]},
            {_SINK: [numeric_range("amount", 0, 10)]},
        )

    def test_strict_obligations_accept_a_relational_allowlist(self):
        pol = {_SINK: [tuple_enum(("recipient", "currency"), {("acme", "EUR")})]}
        assert validate_egress_allowlists({_SINK}, pol) == []
        assert validate_driving_arg_allowlists({_SINK}, {_SINK: ["recipient"]}, pol) == []

    def test_strict_obligations_still_reject_an_unconstrained_destination(self):
        pol = {_SINK: [numeric_range("amount", 0, 10)]}
        assert validate_egress_allowlists({_SINK}, pol) != []


class TestConfigSurface:
    """Declarable in YAML/dict config, not only in Python kwargs — and the same
    fail-closed-at-load discipline as the per-argument kinds."""

    def test_round_trip(self):
        cfg = GovernanceConfig.from_dict({
            "value_policies": {
                _SINK: [
                    {"kind": "tuple_enum", "args": ["recipient", "currency"],
                     "allowed": [["acme", "EUR"], ["globex", "USD"]]},
                    {"kind": "keyed_range", "args": ["recipient", "amount"],
                     "ranges": {"acme": [0, 500]}},
                ]
            },
        })
        preds = cfg.value_policies[_SINK]
        assert [p.kind for p in preds] == ["tuple_enum", "keyed_range"]
        # Two relational predicates over overlapping tuples mean a call must carry
        # every argument both of them name — the partial-tuple denial composes.
        full = {"recipient": "acme", "currency": "EUR", "amount": 100}
        assert check_value_policies(_SINK, full, cfg.value_policies) is None
        assert check_value_policies(
            _SINK, {**full, "currency": "USD"}, cfg.value_policies
        ) is not None
        assert check_value_policies(
            _SINK, {**full, "amount": 900}, cfg.value_policies
        ) is not None
        partial = check_value_policies(
            _SINK, {"recipient": "acme", "currency": "EUR"}, cfg.value_policies
        )
        assert partial is not None and "missing" in partial

    @pytest.mark.parametrize("bad,match", [
        ({"kind": "tuple_enum", "args": ["recipient"], "allowed": [["a"]]}, "at least"),
        ({"kind": "tuple_enum", "args": ["a", "b"], "allowed": []}, "non-empty"),
        ({"kind": "tuple_enum", "args": ["a", "b"], "allowed": [["x"]]}, "exactly 2"),
        ({"kind": "tuple_enum", "args": ["a", "a"], "allowed": [["x", "y"]]}, "duplicates"),
        ({"kind": "keyed_range", "args": ["a", "b"], "ranges": {}}, "non-empty"),
        ({"kind": "keyed_range", "args": ["a", "b"], "ranges": {"k": [5, 1]}}, "inverted"),
        ({"kind": "keyed_range", "args": ["a", "b"], "ranges": {"k": ["x", 1]}}, "must be"),
        ({"kind": "keyed_range", "args": ["a", "b", "c"], "ranges": {"k": [0, 1]}}, "exactly two"),
        ({"kind": "tuple_enum", "args": ["a", "b"], "allowed": [["x", "y"]], "lo": 1}, "unknown"),
    ])
    def test_malformed_relational_predicates_fail_at_load(self, bad, match):
        with pytest.raises(ValueError, match=match):
            GovernanceConfig.from_dict({"value_policies": {_SINK: [bad]}})
