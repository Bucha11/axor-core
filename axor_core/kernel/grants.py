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
    def at_most(cls, bound: Any) -> "ArgConstraint":
        return cls(ConstraintKind.LIMIT, "money", limit=Decimal(canonicalize("money", bound)))

    def admits(self, value: Any, table: HandleTable | None) -> bool:
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

    def covers(self, sink: str, args: Mapping[str, Any], table: HandleTable | None) -> bool:
        if sink != self.sink:
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


@dataclass
class _Slot:
    grant: Grant
    remaining: int | None
    spends: int = 0


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
    def _live(slot: _Slot, now: float | None) -> bool:
        g = slot.grant
        if g.expiry is not None and now is not None and now > g.expiry:
            return False
        return slot.remaining is None or slot.remaining > 0

    def find(self, sink: str, args: Mapping[str, Any], table: HandleTable | None = None,
             now: float | None = None) -> Grant | None:
        """The first live grant covering the call, in issue order; None if none."""
        for gid in self._order:
            s = self._slots[gid]
            if self._live(s, now) and s.grant.covers(sink, args, table):
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
            if s is None or not self._live(s, now) or not s.grant.covers(sink, args, table):
                continue
            if s.remaining is not None:
                s.remaining -= 1
            s.spends += 1
            r = Reservation(f"{gid}#{s.spends}", gid, call_key, SpendState.RESERVED,
                            now if now is not None else 0.0)
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
        if s.remaining is not None:
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
    issued: list[Grant] = field(default_factory=list)

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
        uses = self.default_uses if uses is None else uses
        for sink, args in self.grant_args.items():
            cands: dict[str, list[str]] = {}
            opt: dict[str, bool] = {}
            for arg, spec in args.items():
                kind, optional = split_kind(spec)
                opt[arg] = optional
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
                # an optional arg with no literal contributes "absent" (an empty set)
                for combo in itertools.product(*((cands[a] or [None]) for a in names)):
                    out.append(self._mint(
                        sink, {a: ArgConstraint(ConstraintKind.EXACT, kinds[a],
                                                frozenset() if v is None else frozenset({v}),
                                                optional=opt[a])
                               for a, v in zip(names, combo)},
                        uses, TrustedChannel.REQUEST))
        return out

    def from_templates(self, templates: Mapping[str, list[Mapping[str, Any]]]) -> list[Grant]:
        """L2: operator templates. Each entry maps arg -> value | [values] | {max: n};
        uses are unlimited (the existing enum supersession is this with uses = ∞)."""
        out: list[Grant] = []
        for sink, entries in templates.items():
            kinds = self.grant_args.get(sink, {})
            for entry in entries:
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

    def from_structured(self, sink: str, args: Mapping[str, Any], *, uses: int | None = None) -> Grant | None:
        """L3: a typed intent the application passed (never model text)."""
        return self._exact(sink, args, TrustedChannel.STRUCTURED,
                           self.default_uses if uses is None else uses)

    def from_confirmation(self, sink: str, args: Mapping[str, Any]) -> Grant | None:
        """L4: a human confirmed this exact action. One use."""
        return self._exact(sink, args, TrustedChannel.CONFIRMATION, 1)

    def _exact(self, sink: str, args: Mapping[str, Any], channel: TrustedChannel,
               uses: int | None) -> Grant | None:
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
        return self._mint(sink, cs, uses, channel)
