"""Value-policy validators: the decidability classifier is wired into registration
so that only guardable predicates are accepted, and the null-byte path check fails
closed instead of crashing."""

from __future__ import annotations

import pytest

from axor_core.capability.executor import CapabilityExecutor
from axor_core.kernel.registration import (
    field_obligation,
    predicate_is_decidable,
    validate_value_policies,
)
from axor_core.kernel.decidability import CodomainKind, ConsumptionMode
from axor_core.policy.value_policy import ValuePredicate, enum, numeric_range
from axor_core.node.intent_loop import IntentLoop
from axor_core.security.paths import path_matches_allowlist, path_within

pytestmark = pytest.mark.adversarial


def test_decidable_predicates_validate():
    pols = {
        "transfer": [numeric_range("amount", 0, 1000)],
        "deploy": [enum("target", ["staging", "prod"])],
    }
    assert validate_value_policies(pols) == []
    assert predicate_is_decidable(pols["transfer"][0]) is True
    assert predicate_is_decidable(pols["deploy"][0]) is True


def test_unknown_predicate_kind_is_rejected():
    bad = ValuePredicate(arg="cmd", kind="regex_match")   # not a decidable projection
    errors = validate_value_policies({"shell": [bad]})
    assert errors and "shell" in errors[0]
    assert predicate_is_decidable(bad) is False


def test_intent_loop_rejects_fuzz_required_predicate():
    bad = ValuePredicate(arg="cmd", kind="regex_match")
    with pytest.raises(ValueError, match="value_policies"):
        IntentLoop(capability_executor=CapabilityExecutor(), trace_events=[],
                   value_policies={"shell": [bad]})


def test_field_obligation_split():
    # decidable codomains → guardable by a predicate; rich-syntax → fuzz.
    assert field_obligation(CodomainKind.ENUM, ConsumptionMode.CASE_SPLIT) == "predicate"
    assert field_obligation(CodomainKind.BOUNDED_NUMERIC, ConsumptionMode.NUMERIC) == "predicate"
    assert field_obligation(CodomainKind.PATH_CLASS, ConsumptionMode.PATH_RESOLVE) == "fuzz"
    assert field_obligation(CodomainKind.CARRIER_OVER_TEXT, ConsumptionMode.INTERPRET) == "fuzz"
    # An enum HANDED TO an interpreter is fuzz despite the low-capacity codomain.
    assert field_obligation(CodomainKind.ENUM, ConsumptionMode.INTERPRET) == "fuzz"


def test_null_byte_path_fails_closed_not_crash():
    # A null byte makes Path.resolve raise; the governance check must DENY
    # (fail closed), never propagate the exception.
    assert path_within("x\x00y", "/repo") is False
    assert path_matches_allowlist("/repo/\x00etc", ["/repo"]) is False
