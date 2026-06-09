"""Value-policy predicates (TM3.1 predicate layer).

The closed thing is the set of admissible *projections* (Def. 3b); the open thing
is the set of *predicates* over them. This is that open layer: per-sink, per-arg
predicates over an admissible projection — e.g. `transfer(amount)` admissible iff
the amount is a bounded number in range. Predicates consume an admissible
projection (number / enum), never raw content, and are discharged by the Thm. 0
decision procedures (decidable T4).

Operator-supplied (the `value_policies` knob), analogous to CaMeL's security
policies over a value. Absent → no value-policy constraint (coarse default).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

from axor_core.kernel.t4 import verify_bounded_numeric, verify_enum


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
            r = verify_enum(value, self.allowed)
            if not r.is_pass:
                return f"value policy: arg '{self.arg}'={value!r} not in {sorted(map(str, self.allowed))}"
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
