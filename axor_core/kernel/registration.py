"""Sink-projection registration validated by the Thm. 0 classifier (K3.6 / build
order X4 — the "validators" step).

A sink declares, per argument, the admissible projection it consumes (a codomain
kind + a consumption mode). The kernel then decides the T4 obligation via
``kernel.t4.classify``:

  • DECIDABLE  — the argument may be guarded by a value-policy PREDICATE, discharged
                 by a decision procedure (verify_enum / verify_bounded_numeric).
  • FUZZ_REQ   — the argument is a rich-syntax field (path / interpreted string /
                 carrier over free text); it CANNOT be soundly discharged by a
                 predicate and must instead ride the fuzz obligation (K5/T4) and/or
                 the positional D_high gate.

This makes the classifier load-bearing: ``validate_value_policies`` rejects a
configuration that tries to guard a fuzz-required field with a decidable predicate
(the silent-misconfiguration that would give false assurance).
"""

from __future__ import annotations

from axor_core.kernel.t4 import (
    CodomainKind,
    ConsumptionMode,
    T4Verdict,
    classify,
    is_t4_decidable,
)
from axor_core.policy.value_policy import ValuePredicate

# Each value-predicate kind implies the projection it consumes. Both are in the
# DECIDABLE class by construction; this map is what `validate_value_policies`
# checks against the Thm. 0 classifier, and it is the single place a new predicate
# kind must declare its projection (or it fails closed as unknown).
_PREDICATE_PROJECTION: dict[str, tuple[CodomainKind, ConsumptionMode]] = {
    "numeric_range": (CodomainKind.BOUNDED_NUMERIC, ConsumptionMode.NUMERIC),
    "enum": (CodomainKind.ENUM, ConsumptionMode.CASE_SPLIT),
}


def predicate_is_decidable(pred: ValuePredicate) -> bool:
    """True iff the predicate's implied projection is T4-decidable (Thm. 0)."""
    proj = _PREDICATE_PROJECTION.get(pred.kind)
    if proj is None:
        return False  # unknown predicate kind → not provably decidable
    return is_t4_decidable(*proj)


def validate_value_policies(
    policies: "dict[str, list[ValuePredicate]] | None",
) -> list[str]:
    """Validate operator-registered value policies against Thm. 0.

    Returns a list of configuration errors (empty == valid). A predicate is an
    error when its projection is not T4-decidable: a decision procedure cannot
    soundly discharge a rich-syntax (fuzz-required) field, so guarding one that way
    is false assurance and must be surfaced, not silently accepted.
    """
    errors: list[str] = []
    if not policies:
        return errors
    for sink, preds in policies.items():
        for pred in preds:
            proj = _PREDICATE_PROJECTION.get(pred.kind)
            if proj is None:
                errors.append(
                    f"sink {sink!r} arg {pred.arg!r}: unknown predicate kind "
                    f"{pred.kind!r} — projection undeclared, cannot classify (T4)"
                )
                continue
            result = classify(*proj)
            if result.verdict != T4Verdict.DECIDABLE_PASS:
                errors.append(
                    f"sink {sink!r} arg {pred.arg!r}: projection "
                    f"{proj[0].value}/{proj[1].value} is {result.verdict.value} — "
                    "a value predicate cannot discharge a fuzz-required field"
                )
    return errors


def field_obligation(kind: CodomainKind, mode: ConsumptionMode) -> str:
    """Operator-facing helper: 'predicate' if a sink field can be guarded by a
    decidable value predicate, else 'fuzz' (must ride the fuzz/positional path)."""
    return "predicate" if is_t4_decidable(kind, mode) else "fuzz"
