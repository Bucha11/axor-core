"""Value-policy predicates.

The fixed part of the system is the set of admissible *projections* of a value;
the extensible part is the set of *predicates* over them. This is that extensible
layer: per-sink, per-arg predicates over an admissible projection — e.g.
`transfer(amount)` is admissible only if the amount is a bounded number in range.
Predicates consume an admissible projection (number / enum), never raw content,
and are checked by decidable comparisons.

Operator-supplied (the `value_policies` knob), like attaching security policies
to a value. Absent → no value-policy constraint (the coarse default).

Two shapes of predicate:

  * **per-argument** (`enum`, `numeric_range`) — constrains one argument on its own.
    A set of these constrains the PRODUCT of their codomains: an allowlisted
    recipient AND an in-range amount, in any pairing.
  * **relational** (`tuple_enum`, `keyed_range`) — constrains a TUPLE of arguments
    jointly, so the pairing itself is policy. Both stay decidable: a tuple enum is
    a case-split over a finite set of tuples, and a keyed range is a case-split on
    the key followed by a numeric comparison — neither reads content semantics, and
    both are discharged by the same decision procedures
    (:mod:`axor_core.kernel.decidability`).

A relational predicate fails CLOSED on a partially-present tuple: if some but not
all of its arguments are in the call, the relation cannot be verified, so the call
is denied rather than passed as "unconstrained".
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Mapping, Sequence

from axor_core.kernel.decidability import verify_bounded_numeric, verify_enum


@dataclass(frozen=True)
class ValuePredicate:
    """A predicate over the admissible projection(s) of a sink's argument(s).

    Per-argument kinds (``arg`` names the argument):
        kind="numeric_range": arg must be a real in [lo, hi].
        kind="enum":          arg must be in `allowed`.

    Relational kinds (``args`` names the tuple; ``arg`` is unused):
        kind="tuple_enum":    the tuple of values must be in `allowed`.
        kind="keyed_range":   args = (key_arg, value_arg); the key selects the
                              admissible range from `ranges`, an unknown key is a
                              denial.
    """
    arg: str = ""
    kind: str = ""
    lo: Real | None = None
    hi: Real | None = None
    allowed: frozenset = frozenset()
    # Relational kinds only. Tuples keep the dataclass frozen and hashable.
    args: tuple[str, ...] = ()
    ranges: tuple[tuple[object, Real, Real], ...] = ()

    def check(self, args: dict) -> str | None:
        """Return a denial reason if the predicate fails, else None.

        A missing argument is not a violation here (capability/consequence handle
        presence); the predicate only constrains a *present* value.
        """
        if self.args:
            return self._check_relational(args)
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

    def _check_relational(self, args: dict) -> str | None:
        """Relational kinds. A partially-present tuple is a DENIAL: the relation
        cannot be verified, and passing it would admit exactly the pairing the
        predicate exists to forbid."""
        present = [a for a in self.args if a in args]
        if not present:
            return None  # the sink was not called with these args at all
        if len(present) != len(self.args):
            missing = [a for a in self.args if a not in args]
            return (
                f"value policy: {self.kind} over {list(self.args)} cannot be checked — "
                f"missing {missing}"
            )
        if self.kind == "tuple_enum":
            value = tuple(args[a] for a in self.args)
            r = verify_enum(value, self.allowed)
            if not r.is_pass:
                return (
                    f"value policy: {list(self.args)}={value!r} is not an allowed "
                    f"combination ({len(self.allowed)} permitted)"
                )
            return None
        if self.kind == "keyed_range":
            key_arg, value_arg = self.args
            key, value = args[key_arg], args[value_arg]
            for k, lo, hi in self.ranges:
                if k == key:
                    r = verify_bounded_numeric(value, lo, hi)
                    if not r.is_pass:
                        return (
                            f"value policy: {value_arg}={value!r} fails range "
                            f"[{lo}, {hi}] for {key_arg}={key!r}"
                        )
                    return None
            return (
                f"value policy: {key_arg}={key!r} has no declared range — "
                f"unknown key fails closed"
            )
        return f"value policy: unknown relational kind {self.kind!r} on {list(self.args)}"


def numeric_range(arg: str, lo: Real, hi: Real) -> ValuePredicate:
    return ValuePredicate(arg=arg, kind="numeric_range", lo=lo, hi=hi)


def enum(arg: str, allowed) -> ValuePredicate:
    return ValuePredicate(arg=arg, kind="enum", allowed=frozenset(allowed))


def tuple_enum(args: "Sequence[str]", allowed) -> ValuePredicate:
    """Allow only these combinations of these arguments.

    ``tuple_enum(("recipient", "currency"), {("acme", "EUR"), ("globex", "USD")})``
    admits an allowlisted recipient only in the pairing the operator declared —
    where two independent ``enum`` predicates would admit either recipient with
    either currency.
    """
    return ValuePredicate(
        kind="tuple_enum",
        args=tuple(args),
        allowed=frozenset(tuple(a) for a in allowed),
    )


def keyed_range(key_arg: str, value_arg: str, ranges: "Mapping[object, tuple]") -> ValuePredicate:
    """Per-key numeric bound: ``keyed_range("recipient", "amount", {"acme": (0, 500)})``.

    The key is a case-split over the declared keys (closed codomain); the value is
    a numeric comparison inside the selected branch. An undeclared key is a
    denial, so the key set doubles as an allowlist.
    """
    return ValuePredicate(
        kind="keyed_range",
        args=(key_arg, value_arg),
        ranges=tuple((k, lo, hi) for k, (lo, hi) in ranges.items()),
    )


# ── what a predicate set pins, for the obligations that have to ask ──────────
#
# Three callers need this and must agree: the integrity-supersession rule
# (policy/gates.py), the STRICT allowlist obligations (kernel/registration.py),
# and effect coverage (policy/effect.py). Keeping the answer here means a new
# predicate kind is classified once.


def closed_allowlist_args(preds: "Iterable[ValuePredicate]") -> frozenset[str]:
    """Args whose codomain is a CLOSED, operator-declared set of trusted values.

    This is the soundness condition for integrity supersession and for a
    destination allowlist: the attacker cannot choose a value outside the set, so
    the constraint is content-blind and provenance-independent. A
    ``numeric_range`` does NOT qualify (an open interval the attacker can satisfy);
    a ``keyed_range`` qualifies for its KEY only, for the same reason.
    """
    out: set[str] = set()
    for p in preds:
        kind = getattr(p, "kind", None)
        if kind == "enum" and len(p.allowed) > 0:
            out.add(p.arg)
        elif kind == "tuple_enum" and len(p.allowed) > 0:
            out.update(p.args)
        elif kind == "keyed_range" and len(p.ranges) > 0:
            out.add(p.args[0])
    return frozenset(out)


def constrained_args(preds: "Iterable[ValuePredicate]") -> frozenset[str]:
    """Every arg any predicate constrains at all (closed or not)."""
    out: set[str] = set()
    for p in preds:
        if getattr(p, "args", ()):
            out.update(p.args)
        elif getattr(p, "arg", ""):
            out.add(p.arg)
    return frozenset(out)


def relational_arg_groups(preds: "Iterable[ValuePredicate]") -> list[frozenset[str]]:
    """The arg tuples constrained JOINTLY (one entry per relational predicate)."""
    return [frozenset(p.args) for p in preds if getattr(p, "args", ()) and len(p.args) > 1]


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
