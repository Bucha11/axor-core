"""Effect specification for one governed sink — written in authority/effect terms.

This module states which sequences of executed effects a policy permits, without
reference to how the kernel decides. It imports nothing from ``axor_core`` and
mentions no gate, projection, taint label or consequence class: the vocabulary is
tools, arguments, approvals given by a trusted approver, a budget, and the fact
that a secret has been delivered to the agent.

    Allowed(P, s, e)      may event e happen in spec state s under policy P?
    Update(P, s, e) = s'  the spec state after e

The spec is nondeterministic in one place: when an executed call is covered both
by a call-scoped and by a class-scoped approval, either may be the one consumed.
``Checker`` therefore tracks the SET of spec states consistent with the observed
prefix, and a trace is accepted iff that set never becomes empty (trace inclusion).

What it deliberately does NOT say: anything about provenance. A transfer to an
allowlisted recipient that an injection chose is Allowed here. Provenance is a
mechanism the implementation may add on top; it can only make the implementation
refuse more, which trace inclusion permits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from fractions import Fraction


# ── policy ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Policy:
    """The operator's declaration, in effect terms."""

    callable_tools: frozenset[str]           # tools the agent may invoke at all
    needs_approval: frozenset[str]           # each execution needs a trusted approval
    grantable: frozenset[str]                # tools an approval may authorise
    egress: frozenset[str]                   # tools that send data out
    secret_sources: frozenset[str]           # tools whose output is a secret
    allowed_recipients: frozenset[str]       # for egress tools' `recipient`
    amount_bounds: tuple[float, float]       # for egress tools' `amount`
    budget: int | None                       # max executed calls, any tool


# ── value semantics (what an argument means as an effect) ────────────────────


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def canon(v: object) -> object:
    """Effect identity of an argument value: two values denote the same effect iff
    their canon is equal. Numbers compare by exact value (100 == 100.0), never with
    booleans; strings compare exactly; containers structurally."""
    if isinstance(v, bool):
        return ("bool", v)
    if _is_number(v):
        if isinstance(v, float) and not math.isfinite(v):
            return ("nonfinite", repr(v))
        return ("num", Fraction(v))
    if isinstance(v, str):
        return ("str", v)
    if v is None:
        return ("none",)
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(canon(x) for x in v))
    if isinstance(v, (set, frozenset)):
        return ("set", frozenset(canon(x) for x in v))
    if isinstance(v, dict):
        return ("map", frozenset((k, canon(x)) for k, x in v.items()))
    return ("other", repr(v))


def call_key(tool: str, args: dict) -> object:
    return (tool, canon(args))


def recipients_of(args: dict) -> list[object] | None:
    """The destinations a call sends to. None = the call names no recipient."""
    if "recipient" not in args:
        return None
    r = args["recipient"]
    if isinstance(r, (list, tuple, set, frozenset)):
        return list(r)
    return [r]


def amount_ok(p: Policy, args: dict) -> bool:
    if "amount" not in args:
        return True          # the declaration constrains a present amount only
    a = args["amount"]
    if not _is_number(a):
        return False
    if isinstance(a, float) and not math.isfinite(a):
        return False
    lo, hi = p.amount_bounds
    return Fraction(lo) <= Fraction(a) <= Fraction(hi)


def recipients_ok(p: Policy, args: dict) -> bool:
    rs = recipients_of(args)
    if rs is None:
        return True          # the declaration constrains a present recipient only
    return all(isinstance(r, str) and r in p.allowed_recipients for r in rs)


# ── state and events ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class State:
    spent: int = 0
    secret_delivered: bool = False
    # Call-scoped approvals not yet spent: multiset as a sorted tuple of keys.
    call_auth: tuple = ()
    # Class-scoped approvals: tuple of (tool, remaining uses), one per approval.
    class_auth: tuple = ()


@dataclass(frozen=True)
class Approve:
    """A trusted approver said yes. scope 'call' → one execution of exactly
    (tool, args); scope 'class' → up to max_ops executions of tool."""
    scope: str
    tool: str
    args: dict = field(default_factory=dict)
    max_ops: int = 1


@dataclass(frozen=True)
class Exec:
    """A tool handler ran with these arguments — the effect itself."""
    tool: str
    args: dict


@dataclass(frozen=True)
class Deliver:
    """A tool's output was handed to the agent."""
    tool: str


# ── Allowed / Update ─────────────────────────────────────────────────────────


def _remove_one(t: tuple, item) -> tuple:
    lst = list(t)
    lst.remove(item)
    return tuple(lst)


def successors(p: Policy, s: State, e) -> list[State]:
    """All s' with Allowed(P, s, e) and s' = Update(P, s, e) under some resolution
    of the spec's nondeterminism. Empty list ⇔ not Allowed."""
    if isinstance(e, Approve):
        if e.tool not in p.grantable:
            return [s]                                   # confers nothing
        if e.scope == "call":
            key = call_key(e.tool, e.args)
            return [replace(s, call_auth=tuple(sorted(s.call_auth + (key,), key=repr)))]
        if e.max_ops <= 0:
            return [s]
        return [replace(s, class_auth=s.class_auth + ((e.tool, e.max_ops),))]

    if isinstance(e, Deliver):
        if e.tool in p.secret_sources:
            return [replace(s, secret_delivered=True)]
        return [s]

    assert isinstance(e, Exec)
    t, a = e.tool, e.args
    if t not in p.callable_tools and t not in p.grantable:
        return []
    if p.budget is not None and s.spent >= p.budget:
        return []
    if t in p.egress:
        if s.secret_delivered:
            return []
        if not (recipients_ok(p, a) and amount_ok(p, a)):
            return []
    base = replace(s, spent=s.spent + 1)

    if t not in p.needs_approval and t in p.callable_tools:
        return [base]

    # Needs an approval: consume one call-scoped match or one class-scoped use.
    out: list[State] = []
    key = call_key(t, a)
    if key in s.call_auth:
        out.append(replace(base, call_auth=_remove_one(s.call_auth, key)))
    for i, (tool, left) in enumerate(s.class_auth):
        if tool == t and left > 0:
            ca = list(s.class_auth)
            ca[i] = (tool, left - 1)
            out.append(replace(base, class_auth=tuple(ca)))
    return out


class Violation(AssertionError):
    pass


class Checker:
    """Trace-inclusion checker: feed observed events in order."""

    def __init__(self, policy: Policy) -> None:
        self.p = policy
        self.states: set[State] = {State()}
        self.prefix: list = []

    def step(self, e) -> None:
        self.prefix.append(e)
        nxt: set[State] = set()
        for s in self.states:
            nxt.update(successors(self.p, s, e))
        if not nxt:
            raise Violation(
                "observed trace leaves the specification at event "
                f"#{len(self.prefix) - 1}: {e!r}\nprefix:\n"
                + "\n".join(f"  {i}: {x!r}" for i, x in enumerate(self.prefix))
                + f"\nspec states before it: {sorted(map(repr, self.states))}"
            )
        self.states = nxt
