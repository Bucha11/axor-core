"""Effect signature, effect digest, and how completely a policy describes an effect.

The digest is the equality key an authorisation is bound to (see
capability/approvals). Two properties matter, in this order of severity:

  * it must NOT collide for different effects — that is the direction in which a
    collision spends one approval on another call. Hence no normalisation of
    values: type-tagged, exact.
  * it must be stable for the same effect — otherwise an approval cannot be
    matched to the call it was issued for.

`validate_effect_coverage` is the other half: it reports what the registered
predicates do not pin, including the product-vs-relation gap that makes a fully
decidable policy still not a description of the effect.
"""
from __future__ import annotations

from axor_core.policy.effect import (
    call_digest,
    effect_signature,
    validate_effect_coverage,
)
from axor_core.policy.value_policy import enum, keyed_range, numeric_range, tuple_enum

_SINK = "send_money"
_CALL = {"recipient": "acme", "amount": 100, "memo": "invoice"}


class TestSignature:

    def test_declared_drivers_win_and_are_sorted(self):
        assert effect_signature(_SINK, _CALL, {_SINK: ["recipient", "amount"]}) == (
            "amount",
            "recipient",
        )

    def test_undeclared_falls_back_to_every_present_arg(self):
        """Fail-closed: over-binding only narrows what an authorisation covers."""
        assert effect_signature(_SINK, _CALL, None) == ("amount", "memo", "recipient")

    def test_a_declared_driver_absent_from_the_call_stays_in_the_signature(self):
        sig = effect_signature(_SINK, {"recipient": "acme"}, {_SINK: ["recipient", "amount"]})
        assert sig == ("amount", "recipient")


class TestDigest:

    def test_same_call_same_digest(self):
        assert call_digest(_SINK, dict(_CALL)) == call_digest(_SINK, dict(_CALL))

    def test_argument_order_does_not_matter(self):
        reordered = {k: _CALL[k] for k in reversed(list(_CALL))}
        assert call_digest(_SINK, reordered) == call_digest(_SINK, _CALL)

    def test_any_value_change_changes_the_digest(self):
        base = call_digest(_SINK, _CALL)
        for k, v in [("recipient", "globex"), ("amount", 101), ("memo", "other")]:
            assert call_digest(_SINK, {**_CALL, k: v}) != base

    def test_an_added_or_removed_argument_changes_the_digest(self):
        base = call_digest(_SINK, _CALL)
        assert call_digest(_SINK, {**_CALL, "currency": "EUR"}) != base
        assert call_digest(_SINK, {"recipient": "acme", "amount": 100}) != base

    def test_the_sink_is_part_of_the_digest(self):
        assert call_digest("other_sink", _CALL) != call_digest(_SINK, _CALL)

    def test_types_are_not_conflated(self):
        """`1` and `"1"` are different effects; so are None and "None"."""
        assert call_digest(_SINK, {"amount": 1}) != call_digest(_SINK, {"amount": "1"})
        assert call_digest(_SINK, {"amount": 1}) != call_digest(_SINK, {"amount": True})
        assert call_digest(_SINK, {"x": None}) != call_digest(_SINK, {"x": "None"})

    def test_values_are_not_normalised(self):
        """Case-folding or NFKC would merge two DIFFERENT identifiers into one
        digest — the one collision direction that must never happen."""
        for other in ("ACME", " acme", "acme ", "acme​"):
            assert call_digest(_SINK, {"recipient": other}) != call_digest(
                _SINK, {"recipient": "acme"}
            )

    def test_nested_values_are_covered(self):
        a = call_digest(_SINK, {"items": [{"to": "acme"}]})
        b = call_digest(_SINK, {"items": [{"to": "globex"}]})
        assert a != b
        assert a == call_digest(_SINK, {"items": [{"to": "acme"}]})

    def test_unserialisable_values_do_not_raise(self):
        class Opaque:
            def __repr__(self) -> str:
                return "<opaque 1>"

        assert call_digest(_SINK, {"x": Opaque()})

    def test_the_authority_scope_is_part_of_the_digest(self):
        """Evidence cannot travel to another authority scope: the same call made in
        another session hashes differently, so an approval issued in one place
        cannot be spent in another."""
        assert call_digest(_SINK, _CALL, scope="node-1") != call_digest(
            _SINK, _CALL, scope="node-2"
        )
        assert call_digest(_SINK, _CALL, scope="node-1") != call_digest(_SINK, _CALL)
        assert call_digest(_SINK, _CALL, scope="node-1") == call_digest(
            _SINK, dict(_CALL), scope="node-1"
        )

    def test_the_default_digest_covers_every_argument(self):
        """What authorisations are bound with: no dependence on driving_args being
        declared, let alone declared correctly."""
        assert call_digest(_SINK, _CALL) != call_digest(
            _SINK, {**_CALL, "memo": "different"}
        )

    def test_the_narrower_signature_scoped_digest_is_available_but_not_the_default(self):
        """Passing driving_args gives the deliberately narrower, signature-scoped
        key (for tooling that wants it). It ignores arguments outside the
        signature — which is exactly why binding does not use it."""
        drivers = {_SINK: ["recipient", "amount"]}
        assert call_digest(_SINK, _CALL, driving_args=drivers) == call_digest(
            _SINK, {**_CALL, "memo": "different"}, driving_args=drivers
        )
        narrow = call_digest(_SINK, _CALL, driving_args={_SINK: ["recipient"]})
        wide = call_digest(_SINK, _CALL, driving_args={_SINK: ["recipient", "amount"]})
        assert narrow != wide


class TestCoverage:

    def test_complete_when_a_relation_covers_the_signature(self):
        assert validate_effect_coverage(
            {_SINK: ["recipient", "amount"]},
            {_SINK: [keyed_range("recipient", "amount", {"acme": (0, 500)})]},
            require_relational=True,
        ) == []

    def test_reports_an_unconstrained_signature_argument(self):
        errors = validate_effect_coverage(
            {_SINK: ["recipient", "amount"]},
            {_SINK: [enum("recipient", {"acme"})]},
        )
        assert len(errors) == 1 and "'amount'" in errors[0]

    def test_reports_the_product_versus_relation_gap(self):
        """Every argument constrained, every projection decidable — and the policy
        still admits any pairing. This is the swapped-payment gap, reported."""
        per_arg = {_SINK: [enum("recipient", {"acme", "globex"}), numeric_range("amount", 0, 1000)]}
        assert validate_effect_coverage({_SINK: ["recipient", "amount"]}, per_arg) == []
        errors = validate_effect_coverage(
            {_SINK: ["recipient", "amount"]}, per_arg, require_relational=True
        )
        assert len(errors) == 1 and "PRODUCT" in errors[0]

    def test_a_single_argument_signature_needs_no_relation(self):
        assert validate_effect_coverage(
            {_SINK: ["recipient"]},
            {_SINK: [enum("recipient", {"acme"})]},
            require_relational=True,
        ) == []

    def test_a_wider_relation_covers_a_narrower_signature(self):
        assert validate_effect_coverage(
            {_SINK: ["recipient", "amount"]},
            {_SINK: [tuple_enum(("recipient", "amount", "currency"), {("acme", 1, "EUR")})]},
            require_relational=True,
        ) == []

    def test_no_declaration_reports_nothing(self):
        assert validate_effect_coverage(None, None) == []
        assert validate_effect_coverage({_SINK: []}, {}) == []
