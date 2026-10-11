"""Value-policy predicates.

The fixed part of the system is the set of admissible *projections* of a value;
the extensible part is the set of *predicates* over them. This is that extensible
layer: per-sink, per-arg predicates over an admissible projection — e.g.
`transfer(amount)` is admissible only if the amount is a bounded number in range.
Predicates consume an admissible projection (number / enum), never raw content,
and are checked by decidable comparisons.

Operator-supplied (the `value_policies` knob), like attaching security policies
to a value. Absent → no value-policy constraint (the coarse default).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from axor_core.kernel.decidability import verify_bounded_numeric, verify_enum


def _enum_elements(value: object) -> "list[object]":
    """The values an enum predicate must decide for this argument.

    A scalar decides itself; a collection decides element by element (all of them).
    Kept as one function so every composite predicate added later inherits the same
    rule instead of re-deriving it — and so a crash can never again stand in for a
    verdict.
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


@dataclass(frozen=True)
class ValuePredicate:
    """A predicate over one argument's admissible projection of a sink.

    kind="numeric_range": arg must be a real in [lo, hi].
    kind="enum":          arg must be in `allowed`.
    """
    arg: str
    kind: str
    lo: Real | None = None
    hi: Real | None = None
    allowed: frozenset = frozenset()

    def check(self, args: dict) -> str | None:
        """Return a denial reason if the predicate fails, else None.

        A missing argument is not a violation here (capability/consequence handle
        presence); the predicate only constrains a *present* value.
        """
        if self.arg not in args:
            return None
        value = args[self.arg]
        if self.kind == "numeric_range":
            r = verify_bounded_numeric(value, self.lo, self.hi)
            if not r.is_pass:
                return f"value policy: arg '{self.arg}'={value!r} fails range [{self.lo}, {self.hi}]"
            return None
        if self.kind == "enum":
            # A COLLECTION-valued argument names several destinations at once
            # (`send_email(recipients=[...])`, `share_file(emails=[...])`): EVERY
            # element must be a member, else an extra destination rides in next to a
            # legitimate one. Checking the collection as a whole is not an option —
            # `[a] in {a}` is False, and an unhashable value raised TypeError out of
            # `evaluate`, which is a gate failure, not a verdict. An empty
            # collection names no destination and so violates nothing.
            for element in _enum_elements(value):
                try:
                    r = verify_enum(element, self.allowed)
                except TypeError:
                    # A nested collection (or any unhashable element) is not a
                    # scalar an enum codomain can decide: fail closed, loudly, in
                    # the verdict rather than as an exception.
                    return (
                        f"value policy: arg '{self.arg}' element {element!r} is not a "
                        f"hashable scalar; an enum allowlist decides over scalars only"
                    )
                if not r.is_pass:
                    return (
                        f"value policy: arg '{self.arg}'={element!r} not in "
                        f"{sorted(map(str, self.allowed))}"
                    )
            return None
        # Unknown predicate kind — fail closed.
        return f"value policy: unknown predicate kind {self.kind!r} on '{self.arg}'"


def numeric_range(arg: str, lo: Real, hi: Real) -> ValuePredicate:
    return ValuePredicate(arg=arg, kind="numeric_range", lo=lo, hi=hi)


def enum(arg: str, allowed) -> ValuePredicate:
    return ValuePredicate(arg=arg, kind="enum", allowed=frozenset(allowed))


def check_value_policies(
    tool: str,
    args: dict,
    policies: "dict[str, list[ValuePredicate]] | None",
) -> str | None:
    """Check all predicates registered for `tool`. Return the first denial reason,
    else None. Content-blind: predicates read admissible projections of args."""
    if not policies:
        return None
    for pred in policies.get(tool, ()):  # operator-registered predicates for this sink
        reason = pred.check(args or {})
        if reason is not None:
            return reason
    return None
