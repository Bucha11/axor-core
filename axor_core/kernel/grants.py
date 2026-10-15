"""Task-scoped grants (Ring 0): authority that no model output can create.

A GRANT is a tuple constraint over the driving arguments of ONE sink, with a use
counter, an expiry and a source::

    Grant = (task, sink, {arg: constraint}, uses, expiry, source ∈ {L1, L2, L3, L4})

A constraint is one of

  * ``exact``  — the argument's canonical value is this one value
  * ``enum``   — the argument's canonical value is in a closed set
  * ``limit``  — a numeric argument is at most a bound. A limit is NOT an
                 endorsement of the value; it is a bounded-damage guarantee and is
                 refused at registration unless the sink.arg is declared
                 ``bounded_damage``.

A call is covered by a grant iff EVERY constrained argument is present and satisfies
its constraint — the tuple is matched as a whole. Two grants (Bob, 100) and
(Alice, 50) do not cover (Bob, 50); per-argument rights would.

Invariants this module carries (docs/task-scoped-grants.md):

  O4  no amplification: a grant enters the store only through ``GrantMinter``,
      whose every method takes a trusted input (the request text, an operator
      template, a structured intent, a confirmation). Nothing on the
      ``register_output`` path imports this module (import-linter contract).
  O5  atomic spend: ``reserve`` is a linearizable check-and-decrement under one
      lock; a reservation is then ``commit``ted after the effect ran or
      ``release``d if it did not. The idempotency key of a spend is
      ``<grant id>#<spend number>`` and is handed to the tool handler, so a retry
      after an unknown outcome is answered by reconciliation, never by re-running.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping

from axor_core.kernel.handles import (
    HandleTable,
    TrustedChannel,
    canonicalize,
    is_handle_ref,
    split_kind,
)
from axor_core.kernel.request_literals import literal_values

UNLIMITED = None  # uses=None: an operator template, good for the whole task


class ConstraintKind(enum.Enum):
    EXACT = "exact"
    ENUM = "enum"
    LIMIT = "limit"
    # L1 string: any word-bounded fragment of the request that is no typed literal
    # (values holds the request text). Uses are counted per distinct value.
    FRAGMENT = "fragment"
    # MODEL.md §9.20: a list-valued argument; every element in the values (clause literals ∪ O);
    # a scalar value is never admitted by it (scalars keep their own constraints)
    LIST = "list"


@dataclass(frozen=True)
class ArgConstraint:
    kind: ConstraintKind
    handle_kind: str
    values: frozenset[str] = frozenset()  # canonical values (exact: one; enum: many)
    limit: Decimal | None = None
    # An optional arg may be absent (None / missing / empty collection); a value that
    # is present must still be admitted. Optional with no values = absence only.
    optional: bool = False

    @classmethod
    def exact(cls, handle_kind: str, value: Any) -> "ArgConstraint":
        return cls(ConstraintKind.EXACT, handle_kind, frozenset({canonicalize(handle_kind, value)}))

    @classmethod
    def enum(cls, handle_kind: str, values: Iterable[Any]) -> "ArgConstraint":
        vs = frozenset(canonicalize(handle_kind, v) for v in values)
        if not vs:
            raise ValueError("an enum constraint needs at least one value")
        return cls(ConstraintKind.ENUM, handle_kind, vs)

    @classmethod
    def fragment(cls, request: str, optional: bool = False) -> "ArgConstraint":
        return cls(ConstraintKind.FRAGMENT, "string", frozenset({request}), optional=optional)

    @classmethod
    def at_most(cls, bound: Any) -> "ArgConstraint":
        return cls(ConstraintKind.LIMIT, "money", limit=Decimal(canonicalize("money", bound)))

    def admits(self, value: Any, table: HandleTable | None) -> bool:
        if self.kind is ConstraintKind.FRAGMENT:
            from axor_core.kernel.request_literals import is_request_fragment
            raw = value
            if is_handle_ref(value):
                h = table.get(value) if table is not None else None
                if h is None:
                    return False
                raw = h.raw
            if not isinstance(raw, str):
                return False
            return any(is_request_fragment(req, raw) for req in self.values)
        if self.kind is ConstraintKind.LIMIT:
            if is_handle_ref(value):
                c = table.resolve(value, "money") if table is not None else None
            else:
                try:
                    c = canonicalize("money", value)
                except ValueError:
                    return False
            if c is None:
                return False
            try:
                return Decimal(c) <= self.limit  # type: ignore[operator]
            except InvalidOperation:
                return False
        if table is not None:
            c = table.resolve(value, self.handle_kind)
        else:
            try:
                c = canonicalize(self.handle_kind, value)
            except ValueError:
                c = None
        return c is not None and c in self.values


@dataclass(frozen=True)
class Grant:
    id: str
    task: str
    sink: str
    constraints: Mapping[str, ArgConstraint]
    uses: int | None
    source: TrustedChannel
    expiry: float | None = None  # wall-clock seconds; None = the task's lifetime
    note: str = ""
    # L4: the digest of the whole confirmed call (kernel.confirm.call_digest). When
    # set, the grant covers only that exact call — rich fields included.
    call_digest: str | None = None
    grant_args: Mapping[str, str] | None = None
    # ... and the canonical call itself (kernel.confirm.canonical_call), kept with
    # the grant so an audit shows exactly what was confirmed, rich fields included.
    call_canonical: str | None = None

    def value_key(self, args: Mapping[str, Any], table: HandleTable | None = None) -> str | None:
        """For a grant with string-fragment args: the canonical tuple of those values
        (its uses are counted per key); None for any other grant."""
        frag = sorted(a for a, c in self.constraints.items() if c.kind is ConstraintKind.FRAGMENT)
        if not frag:
            return None
        parts = []
        for a in frag:
            v = args.get(a)
            vals = v if isinstance(v, (list, tuple, set, frozenset)) else [v]
            parts.append(tuple(sorted("" if x is None else canonicalize("string", table.get(x).raw
                                      if (table is not None and is_handle_ref(x) and table.get(x)) else x)
                                      for x in vals)))
        return repr(tuple(parts))

    def covers(self, sink: str, args: Mapping[str, Any], table: HandleTable | None) -> bool:
        if sink != self.sink:
            return False
        if self.call_digest is not None:
            from axor_core.kernel.confirm import call_digest
            if call_digest(sink, args, self.grant_args or {}, table) != self.call_digest:
                return False
        for arg, c in self.constraints.items():
            v = args.get(arg)
            if v is None or (isinstance(v, (list, tuple, set, frozenset)) and not v):
                if c.optional:
                    continue  # absent: nothing flows through this slot
                return False
            if c.kind is not ConstraintKind.LIMIT and not c.values:
                return False  # absence-only constraint, and a value was passed
            # A collection-valued argument (recipients=[...]) is covered only if
            # every element is admitted — a grant for Bob does not cover [Bob, Eve].
            if isinstance(v, (list, tuple, set, frozenset)):
                if not v or not all(c.admits(e, table) for e in v):
                    return False
            elif c.kind is ConstraintKind.LIST:
                return False  # §9.20: a list constraint admits lists only
            elif not c.admits(v, table):
                return False
        return True


class SpendState(enum.Enum):
    RESERVED = "reserved"
    COMMITTED = "committed"
    RELEASED = "released"


@dataclass
class Reservation:
    key: str             # idempotency key: "<grant id>#<spend number>"
    grant_id: str
    call_key: str | None
    state: SpendState
    at: float
    value_key: str | None = None  # fragment grants: the value whose use this is


@dataclass
class _Slot:
    grant: Grant
    remaining: int | None
    spends: int = 0
    # fragment grants: uses left per distinct string value (absent key = full quota)
    by_value: dict = field(default_factory=dict)


class GrantStore:
    """The grant set G_t of one task, and its counters: a pure state machine.

    Ring 0 has no locks and no clock, so time is passed in (``now``; None skips
    expiry) and serialization is the host's job: ``capability.grant_store`` wraps
    this in one lock for the process that executes effects (the daemon side).
    ``find`` is a pure read (used by ``decide``); ``reserve``/``commit``/``release``
    are the spend path (used by the executor, after an allow)."""

    def __init__(self) -> None:
        self._slots: dict[str, _Slot] = {}
        self._order: list[str] = []
        self._res: dict[str, Reservation] = {}
        self._by_call: dict[str, str] = {}
        self._next = 0

    # -- issue: reachable only through GrantMinter (O4) --------------------------

    def _issue(self, grant: Grant) -> Grant:
        if grant.id in self._slots:
            raise ValueError(f"duplicate grant id {grant.id!r}")
        self._slots[grant.id] = _Slot(grant, grant.uses)
        self._order.append(grant.id)
        return grant

    def next_id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}-{self._next}"

    # -- decide (pure) -------------------------------------------------------------

    @staticmethod
    def _live(slot: _Slot, now: float | None, key: str | None = None) -> bool:
        g = slot.grant
        if g.expiry is not None and now is not None and now > g.expiry:
            return False
        if key is not None and g.uses is not None:  # fragment grant: per-value quota
            return slot.by_value.get(key, g.uses) > 0
        return slot.remaining is None or slot.remaining > 0

    def find(self, sink: str, args: Mapping[str, Any], table: HandleTable | None = None,
             now: float | None = None) -> Grant | None:
        """The first live grant covering the call, in issue order; None if none."""
        for gid in self._order:
            s = self._slots[gid]
            if s.grant.covers(sink, args, table) and self._live(s, now, s.grant.value_key(args, table)):
                return s.grant
        return None

    def has_grants_for(self, sink: str) -> bool:
        return any(self._slots[g].grant.sink == sink for g in self._order)

    # -- spend (O5) ------------------------------------------------------------------

    def reserve(self, sink: str, args: Mapping[str, Any], table: HandleTable | None = None,
                call_key: str | None = None, grant_id: str | None = None,
                now: float | None = None) -> Reservation | None:
        """Pick a covering live grant and take one use from it (check-and-decrement
        in one step; atomic under the host's lock).

        ``call_key`` makes the reservation idempotent per call: a retried reserve
        for the same call returns the same reservation instead of spending again."""
        if call_key is not None and call_key in self._by_call:
            return self._res[self._by_call[call_key]]
        for gid in ([grant_id] if grant_id else self._order):
            s = self._slots.get(gid)
            if s is None or not s.grant.covers(sink, args, table):
                continue
            vkey = s.grant.value_key(args, table)
            if not self._live(s, now, vkey):
                continue
            if vkey is not None and s.grant.uses is not None:
                s.by_value[vkey] = s.by_value.get(vkey, s.grant.uses) - 1
            elif s.remaining is not None:
                s.remaining -= 1
            s.spends += 1
            r = Reservation(f"{gid}#{s.spends}", gid, call_key, SpendState.RESERVED,
                            now if now is not None else 0.0, value_key=vkey)
            self._res[r.key] = r
            if call_key is not None:
                self._by_call[call_key] = r.key
            return r
        return None

    def commit(self, key: str) -> None:
        r = self._res[key]
        if r.state is SpendState.RELEASED:
            raise ValueError(f"spend {key} was released; it cannot commit")
        r.state = SpendState.COMMITTED

    def release(self, key: str) -> None:
        r = self._res[key]
        if r.state is SpendState.COMMITTED:
            raise ValueError(f"spend {key} committed; it cannot be released")
        if r.state is SpendState.RELEASED:
            return
        r.state = SpendState.RELEASED
        s = self._slots[r.grant_id]
        if r.value_key is not None and s.grant.uses is not None:
            s.by_value[r.value_key] = s.by_value.get(r.value_key, s.grant.uses) + 1
        elif s.remaining is not None:
            s.remaining += 1

    def state(self, key: str) -> SpendState:
        return self._res[key].state

    def reconcile(self, key: str, executed: Callable[[str], bool | None]) -> SpendState:
        """Resolve a reservation whose outcome is unknown (executor died, timeout).

        ``executed(key)`` asks the tool handler's idempotency log whether an effect
        carrying this key ran. True -> commit, False -> release, None (the handler
        cannot say) -> stay RESERVED: the use stays spent, which over-restricts but
        never double-executes."""
        if self._res[key].state is not SpendState.RESERVED:
            return self._res[key].state
        ran = executed(key)
        if ran is True:
            self.commit(key)
        elif ran is False:
            self.release(key)
        return self._res[key].state

    def stale(self, older_than: float, now: float) -> list[str]:
        return [k for k, r in self._res.items()
                if r.state is SpendState.RESERVED and now - r.at > older_than]

    # -- introspection ----------------------------------------------------------------

    def remaining(self, grant_id: str) -> int | None:
        return self._slots[grant_id].remaining

    def grants(self) -> list[Grant]:
        return [self._slots[g].grant for g in self._order]

    def reservations(self) -> list[Reservation]:
        return list(self._res.values())

    def snapshot(self) -> tuple:
        return tuple(
            (gid, s.grant.sink, s.grant.source.value,
             tuple(sorted((a, c.kind.value, tuple(sorted(c.values)), str(c.limit))
                          for a, c in s.grant.constraints.items())),
             s.remaining)
            for gid, s in ((g, self._slots[g]) for g in self._order)
        )


_FRAGMENT = object()  # sentinel: an L1 string arg (fragment constraint)

# MODEL.md §9.14: clause boundaries — a sentence end (a punctuation mark followed by a space or the
# end of the text; a period inside a number / address / URL is not one) or a sequencing conjunction
# (9.14a — the complete list: . ; ? ! before whitespace or the end, a line break, and the conjunctions
# "and then", "then", "also", "after that", "и затем", "потом")
_CLAUSE_END = re.compile(r"[.;!?](?=\s|$)|\n|\b(?:and then|after that|also|then)\b|(?<!\w)(?:и затем|потом)(?!\w)",
                         re.I)


def clauses(request: str) -> list[str]:
    """The request split into clauses (§9.14)."""
    return [c for c in (x.strip() for x in _CLAUSE_END.split(request or "")) if c]


# ── minting: the only way into a GrantStore ────────────────────────────────────


@dataclass
class GrantMinter:
    """Builds grants from trusted inputs only (the L1–L4 sources).

    ``grant_args``: sink -> {arg: handle kind} — which driving args a grant pins.
    ``tuple_grants=False`` is the per-argument ablation: each arg gets its own enum
    and the grant admits the product of them."""

    store: GrantStore
    table: HandleTable
    grant_args: Mapping[str, Mapping[str, str]]
    task: str = "task"
    tuple_grants: bool = True
    default_uses: int | None = 1
    # Operator rule: how many uses an L1 (request) grant gets, per sink. A sink not
    # listed gets default_uses. A model never sets this.
    quotas: Mapping[str, int] = field(default_factory=dict)
    # Access-control sinks (a credential change): authority only from a structured
    # request (L3) or a confirmation (L4) — never a request literal (L1) or a
    # template (L2).
    access_control: frozenset[str] = frozenset()
    # L2 tuple templates: sink -> {driving arg: O-set (frozenset) | None (= request literal)}.
    tuple_templates: dict[str, dict[str, Any]] = field(default_factory=dict)
    # §9.14: literals of different kinds form a tuple only within one clause of the request
    by_clause: bool = False
    # §9.20: a list argument passes if every element is a literal of the clause or in O
    list_union: bool = False
    issued: list[Grant] = field(default_factory=list)

    def _clause_combos(self, request: str, kinds: dict[str, str], opt: dict[str, bool],
                       literal_args: list[str], other_kinds: tuple = ()) -> list[tuple] | None:
        """§9.14a: the tuples of request literals, bound by clause, over ``literal_args`` (in that
        order). ``other_kinds``: kinds of the tuple's own (O) components — a clause with two literals
        of such a kind is ambiguous too. None when the rule does not apply (the tuple combines fewer
        than two kinds)."""
        if not self.by_clause or len({kinds[a] for a in literal_args} | set(other_kinds)) < 2:
            return None  # the rule binds literals of different kinds; same-kind args are unchanged
        out: list[tuple] = []
        for cl in clauses(request):
            if any(len(set(literal_values(cl, k))) > 1 for k in other_kinds):
                continue  # ambiguous: two values for an own component's argument in this clause
            vals: dict[str, list] = {}
            for a in literal_args:
                vs = []
                for v in literal_values(cl, kinds[a]):
                    try:
                        vs.append(canonicalize(kinds[a], v))
                    except ValueError:
                        continue
                vals[a] = list(dict.fromkeys(vs))
            if any(not vals[a] and not opt[a] for a in literal_args):
                continue  # a required argument has no literal in this clause
            if any(len(vals[a]) > 1 for a in literal_args):
                continue  # ambiguous: two values of one kind in the clause
            combo = tuple(vals[a][0] if vals[a] else None for a in literal_args)
            if any(v is not None for v in combo) and combo not in out:
                out.append(combo)
        return out

    def _mint(self, sink: str, constraints: dict[str, ArgConstraint], uses: int | None,
              source: TrustedChannel, note: str = "") -> Grant:
        g = Grant(self.store.next_id(source.value), self.task, sink, constraints, uses, source, note=note)
        self.store._issue(g)
        self.issued.append(g)
        return g

    def from_request(self, request: str, *, uses: int | None = None) -> list[Grant]:
        """L1: grants over the literals the user wrote.

        For a sink whose every grant arg has at least one literal of its kind, a
        grant per tuple of literals (or one per-arg grant in the ablation). A sink
        with a grant arg the request does not spell out gets no L1 grant."""
        out: list[Grant] = []
        explicit = uses
        for sink, args in self.grant_args.items():
            if sink in self.access_control:
                continue  # access control: L3/L4 only
            uses = explicit if explicit is not None else self.quotas.get(sink, self.default_uses)
            cands: dict[str, list[str]] = {}
            opt: dict[str, bool] = {}
            for arg, spec in args.items():
                kind, optional = split_kind(spec)
                opt[arg] = optional
                if kind == "string":
                    # revision 4: any word-bounded request fragment (one constraint)
                    cands[arg] = [_FRAGMENT]
                    continue
                vals = []
                for v in literal_values(request, kind):
                    try:
                        vals.append(canonicalize(kind, v))
                    except ValueError:
                        continue
                vals = list(dict.fromkeys(vals))
                if not vals and not optional:
                    break
                cands[arg] = vals
                for v in vals:
                    self.table.mint_trusted(v, kind, TrustedChannel.REQUEST)
            else:
                kinds = {a: split_kind(args[a])[0] for a in cands}
                if not self.tuple_grants:
                    out.append(self._mint(
                        sink, {a: ArgConstraint(ConstraintKind.ENUM, kinds[a], frozenset(vs),
                                                optional=opt[a])
                               for a, vs in cands.items()},
                        None, TrustedChannel.REQUEST, "per-arg"))
                    continue
                names = list(cands)
                lit = [a for a in names if cands[a] != [_FRAGMENT]]
                bound = self._clause_combos(request, kinds, opt, lit)
                if bound is not None:  # §9.14: the literal args come from one clause together
                    combos = [tuple(dict(zip(lit, c)).get(a, _FRAGMENT) for a in names) for c in bound]
                else:
                    combos = list(itertools.product(*((cands[a] or [None]) for a in names)))
                # an optional arg with no literal contributes "absent" (an empty set)
                for combo in combos:
                    if all(v is None for v in combo):
                        continue  # an empty leading tuple: no L1 grant (origin/class checks decide)
                    out.append(self._mint(
                        sink, {a: (ArgConstraint.fragment(request, optional=opt[a]) if v is _FRAGMENT
                                   else ArgConstraint(ConstraintKind.EXACT, kinds[a],
                                                      frozenset() if v is None else frozenset({v}),
                                                      optional=opt[a]))
                               for a, v in zip(names, combo)},
                        uses, TrustedChannel.REQUEST))
        out += self._mixed_from_request(request, explicit)
        if self.list_union:
            out += self._list_union_from_request(request, explicit)
        return out

    def _list_union_from_request(self, request: str, explicit: int | None) -> list[Grant]:
        """§9.20: for each sink whose L2 tuple template gives an argument an O-set, and each clause
        holding request literals of that argument's kind, one grant whose constraint on that argument
        admits a list of elements each in (the clause's literals ∪ O); the other arguments as the
        template gives them (O-set) or a single literal of the same clause."""
        out: list[Grant] = []
        cls = clauses(request) if self.by_clause else [request]
        for sink, comps in self.tuple_templates.items():
            if sink in self.access_control:
                continue
            uses = explicit if explicit is not None else self.quotas.get(sink, self.default_uses)
            for arg, oset in comps.items():
                if oset is None or (isinstance(oset, Mapping) and "max" in oset):
                    continue
                kind, optional = split_kind(self.grant_args[sink][arg])
                if kind == "string":
                    continue
                for cl in cls:
                    lits = []
                    for v in literal_values(cl, kind):
                        try:
                            lits.append(canonicalize(kind, v))
                        except ValueError:
                            continue
                    lits = list(dict.fromkeys(lits))
                    if not lits:
                        continue  # nothing beyond O: the plain O grant covers it
                    o_vals = []
                    for v in oset:
                        try:
                            o_vals.append(canonicalize(kind, v))
                        except ValueError:
                            continue
                    for v in lits:
                        self.table.mint_trusted(v, kind, TrustedChannel.REQUEST)
                    cs: dict[str, ArgConstraint] = {
                        arg: ArgConstraint(ConstraintKind.LIST, kind, frozenset(lits) | frozenset(o_vals),
                                           optional=optional)}
                    ok = True
                    for a, vs in comps.items():
                        if a == arg:
                            continue
                        if vs is not None:
                            cs[a] = self._o_constraint(sink, a, vs)
                            continue
                        k, opt = split_kind(self.grant_args[sink][a])
                        if k == "string":
                            cs[a] = ArgConstraint.fragment(request, optional=opt)
                            continue
                        lv = list(dict.fromkeys(literal_values(cl, k)))
                        if len(lv) > 1 or (not lv and not opt):
                            ok = False  # ambiguous, or a required argument with no literal here
                            break
                        cs[a] = ArgConstraint(ConstraintKind.EXACT, k,
                                              frozenset({canonicalize(k, lv[0])}) if lv else frozenset(),
                                              optional=opt)
                    if ok:
                        out.append(self._mint(sink, cs, uses, TrustedChannel.REQUEST, "list-union"))
        return out

    def _o_constraint(self, sink: str, arg: str, values: Any) -> ArgConstraint:
        kind, optional = split_kind(self.grant_args[sink][arg])
        if isinstance(values, Mapping) and "max" in values:  # a magnitude limit component
            return dataclasses.replace(ArgConstraint.at_most(values["max"]), optional=optional)
        for v in values:
            self.table.mint_trusted(v, kind, TrustedChannel.OPERATOR)
        return dataclasses.replace(ArgConstraint.enum(kind, values), optional=optional)

    def from_tuple_templates_start(self) -> list[Grant]:
        """L2, all-O tuple templates: one grant per template at session start, uses
        from the operator quota (default_uses when none is set)."""
        out: list[Grant] = []
        for sink, comps in self.tuple_templates.items():
            if any(v is None for v in comps.values()):
                continue
            cs = {a: self._o_constraint(sink, a, vs) for a, vs in comps.items()}
            out.append(self._mint(sink, cs, self.quotas.get(sink, self.default_uses),
                                  TrustedChannel.OPERATOR, "tuple"))
        return out

    def _mixed_from_request(self, request: str, explicit: int | None) -> list[Grant]:
        """L2, mixed tuple templates: each R component is a request literal (as L1),
        each O component the O-set; one grant per tuple of R literals."""
        out: list[Grant] = []
        for sink, comps in self.tuple_templates.items():
            r_args = [a for a, v in comps.items() if v is None]
            if not r_args or len(r_args) == len(comps):
                continue  # all-O: issued at start; all-R: plain L1
            uses = explicit if explicit is not None else self.quotas.get(sink, self.default_uses)
            cands: dict[str, list] = {}
            opt: dict[str, bool] = {}
            kinds: dict[str, str] = {}
            for arg in r_args:
                kind, optional = split_kind(self.grant_args[sink][arg])
                opt[arg], kinds[arg] = optional, kind
                if kind == "string":
                    cands[arg] = [_FRAGMENT]
                    continue
                vals = []
                for v in literal_values(request, kind):
                    try:
                        vals.append(canonicalize(kind, v))
                    except ValueError:
                        continue
                vals = list(dict.fromkeys(vals))
                if not vals and not optional:
                    break
                cands[arg] = vals
                for v in vals:
                    self.table.mint_trusted(v, kind, TrustedChannel.REQUEST)
            else:
                o_cs = {a: self._o_constraint(sink, a, vs) for a, vs in comps.items() if vs is not None}
                lit = [a for a in r_args if cands[a] != [_FRAGMENT]]
                o_kinds = tuple(split_kind(self.grant_args[sink][a])[0] for a, vs in comps.items()
                                if vs is not None and not (isinstance(vs, Mapping) and "max" in vs))
                bound = self._clause_combos(request, kinds, opt, lit, o_kinds)
                combos = ([tuple(dict(zip(lit, c)).get(a, _FRAGMENT) for a in r_args) for c in bound]
                          if bound is not None else
                          list(itertools.product(*((cands[a] or [None]) for a in r_args))))
                for combo in combos:
                    cs = dict(o_cs)
                    for a, v in zip(r_args, combo):
                        cs[a] = (ArgConstraint.fragment(request, optional=opt[a]) if v is _FRAGMENT
                                 else ArgConstraint(ConstraintKind.EXACT, kinds[a],
                                                    frozenset() if v is None else frozenset({v}),
                                                    optional=opt[a]))
                    out.append(self._mint(sink, cs, uses, TrustedChannel.OPERATOR, "tuple"))
        return out

    def from_templates(self, templates: Mapping[str, list[Mapping[str, Any]]]) -> list[Grant]:
        """L2: operator templates. Each entry maps arg -> value | [values] | {max: n};
        uses are unlimited (the existing enum supersession is this with uses = ∞)."""
        out: list[Grant] = []
        for sink, entries in templates.items():
            if sink in self.access_control:
                raise ValueError(f"template for {sink}: an access-control sink takes L3/L4 grants only")
            kinds = self.grant_args.get(sink, {})
            for entry in entries:
                if not entry:
                    continue  # an empty leading tuple: no L2 grant
                cs: dict[str, ArgConstraint] = {}
                for arg, spec in entry.items():
                    if isinstance(spec, Mapping) and "max" in spec:
                        cs[arg] = ArgConstraint.at_most(spec["max"])
                        continue
                    spec_kind = kinds.get(arg)
                    if spec_kind is None:
                        raise ValueError(f"template for {sink}.{arg}: not a grant arg")
                    kind = split_kind(spec_kind)[0]
                    vals = list(spec) if isinstance(spec, (list, tuple, set, frozenset)) else [spec]
                    for v in vals:
                        self.table.mint_trusted(v, kind, TrustedChannel.OPERATOR)
                    cs[arg] = (ArgConstraint.exact(kind, vals[0]) if len(vals) == 1
                               else ArgConstraint.enum(kind, vals))
                out.append(self._mint(sink, cs, UNLIMITED, TrustedChannel.OPERATOR))
        return out

    def from_description(self, request: str, sink: str, arg: str, values: Any, *, members: bool = False,
                         note: str = "description", uses: "int | None" = None) -> list[Grant]:
        """MODEL.md §9.15: grants on a value the request described (computed by the kernel from
        trusted structured fields), as L1 — the other arguments as L1 (request literals / a fragment)
        or as the sink's L2 tuple template (its O-set), with the sink's quota. ``members``: one
        grant whose argument admits any subset of the values."""
        kinds = self.grant_args.get(sink) or {}
        if arg not in kinds or sink in self.access_control:
            return []
        kind, optional = split_kind(kinds[arg])
        canon = []
        for v in values:
            try:
                canon.append(canonicalize(kind, v))
            except ValueError:
                continue
        canon = list(dict.fromkeys(canon))
        if not canon:
            return []
        for v in canon:
            self.table.mint_trusted(v, kind, TrustedChannel.REQUEST)
        targets = ([ArgConstraint(ConstraintKind.ENUM, kind, frozenset(canon), optional=optional)] if members
                   else [ArgConstraint(ConstraintKind.EXACT, kind, frozenset({v}), optional=optional)
                         for v in canon])
        options: dict[str, list[ArgConstraint]] = {}
        tmpl = self.tuple_templates.get(sink) or {}
        for a, spec in kinds.items():
            if a == arg:
                continue
            k, opt = split_kind(spec)
            opts: list[ArgConstraint] = []
            if k == "string":
                opts.append(ArgConstraint.fragment(request, optional=opt))
            else:
                vals = []
                for v in literal_values(request, k):
                    try:
                        vals.append(canonicalize(k, v))
                    except ValueError:
                        continue
                for v in dict.fromkeys(vals):
                    self.table.mint_trusted(v, k, TrustedChannel.REQUEST)
                    opts.append(ArgConstraint(ConstraintKind.EXACT, k, frozenset({v}), optional=opt))
                if not vals and opt:
                    opts.append(ArgConstraint(ConstraintKind.EXACT, k, frozenset(), optional=True))
            if tmpl.get(a) is not None:
                opts.append(self._o_constraint(sink, a, tmpl[a]))
            if not opts:
                return []  # a required argument with no ground
            options[a] = opts
        uses = self.quotas.get(sink, self.default_uses) if uses is None else uses
        out: list[Grant] = []
        others = list(options)
        for t in targets:
            for combo in itertools.product(*(options[a] for a in others)):
                cs = dict(zip(others, combo))
                cs[arg] = t
                out.append(self._mint(sink, cs, uses, TrustedChannel.REQUEST, note))
        return out

    def from_structured(self, sink: str, args: Mapping[str, Any], *, uses: int | None = None) -> Grant | None:
        """L3: a typed intent the application passed (never model text)."""
        return self._exact(sink, args, TrustedChannel.STRUCTURED,
                           self.default_uses if uses is None else uses)

    def from_confirmation(self, sink: str, args: Mapping[str, Any]) -> Grant | None:
        """L4: a human confirmed this exact action. One use, bound to the digest of
        the whole canonical call — not just its driving arguments."""
        from axor_core.kernel.confirm import call_digest, canonical_call

        g = self._exact(sink, args, TrustedChannel.CONFIRMATION, 1, dry=True)
        if g is None:
            return None
        kinds = dict(self.grant_args.get(sink, {}))
        g = dataclasses.replace(g, call_digest=call_digest(sink, args, kinds, self.table),
                                call_canonical=canonical_call(sink, args, kinds, self.table),
                                grant_args=kinds)
        self.store._issue(g)
        self.issued.append(g)
        return g

    def _exact(self, sink: str, args: Mapping[str, Any], channel: TrustedChannel,
               uses: int | None, dry: bool = False) -> Grant | None:
        kinds = self.grant_args.get(sink)
        if not kinds:
            return None
        cs: dict[str, ArgConstraint] = {}
        for arg, spec in kinds.items():
            kind, optional = split_kind(spec)
            v = args.get(arg)
            if v is None or (isinstance(v, (list, tuple, set, frozenset)) and not v):
                # the confirmed/structured action omitted this arg: the grant admits
                # only its absence, never a value added later
                cs[arg] = ArgConstraint(ConstraintKind.ENUM, kind, frozenset(), optional=True)
                continue
            if is_handle_ref(v):
                h = self.table.get(v)
                if h is None:
                    return None
                v = h.raw
            vals = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
            try:
                for x in vals:
                    self.table.mint_trusted(x, kind, channel)
                c = (ArgConstraint.exact(kind, vals[0]) if len(vals) == 1
                     else ArgConstraint.enum(kind, vals))
                cs[arg] = dataclasses.replace(c, optional=optional)
            except (ValueError, IndexError):
                return None
        if dry:  # built, not issued: the caller completes it first
            return Grant(self.store.next_id(channel.value), self.task, sink, cs, uses, channel)
        return self._mint(sink, cs, uses, channel)


def compile_tuple_templates(
    value_policies: Mapping[str, Any] | None,
    grant_args: Mapping[str, Mapping[str, str]],
    driving_args: Mapping[str, Any] | None,
    access_control: "frozenset[str] | set[str]" = frozenset(),
    *,
    operator_sets: "Mapping[str, Mapping[str, Any]] | None" = None,
    template_limits: "Mapping[str, Mapping[str, Any]] | None" = None,
) -> dict[str, dict[str, Any]]:
    """L2 tuple templates (residuals_rules.md revision 9): for each grant sink that is not
    access-control, each driving arg that is a grant arg takes the O-set of an operator
    ``enum`` on that (sink, arg), else a request literal (None). A sink with no O-set is
    plain L1 and gets no template."""
    out: dict[str, dict[str, Any]] = {}
    for sink, kinds in grant_args.items():
        if sink in access_control:
            continue
        driving = [a for a in ((driving_args or {}).get(sink) or ()) if a in kinds]
        osets: dict[str, frozenset] = {}
        for p in (value_policies or {}).get(sink, ()) or ():
            if getattr(p, "kind", None) == "enum" and p.arg in driving:
                vs = frozenset(p.allowed)
                osets[p.arg] = osets[p.arg] & vs if p.arg in osets else vs
        # operator-declared sets (revision 12): O-sets that refuse nothing
        for a, vals in ((operator_sets or {}).get(sink) or {}).items():
            if a in driving and vals:
                vs = frozenset(vals)
                osets[a] = osets[a] & vs if a in osets else vs
        if osets:
            lim = (template_limits or {}).get(sink) or {}
            out[sink] = {a: (sorted(osets[a], key=repr) if a in osets
                             else {"max": lim[a]} if a in lim else None) for a in sorted(driving)}
    return out
