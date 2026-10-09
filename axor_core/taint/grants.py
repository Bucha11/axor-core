"""EXPERIMENTAL — grant identity, so a permission can be bound to a tuple.

Not wired into any gate. Nothing in the kernel imports this; it exists to test
one mechanism on one microtest, and it is removable.

The problem it addresses
-----------------------
``TrustedValueIndex`` maps a canonical key to an *origin kind*
(``key -> TrustedOrigin``). It records THAT a value is attacker-independent, not
WHICH authorisation made it so. So two leaves of one call can each be trusted
via a different authorisation and the call still passes: the request "pay the
rent of 2200 to A, and refund 50 to B" licenses two tuples, while independent
per-leaf membership admits all four — the product of the projections. See
``experiments/floor_independence/analysis/product_closure.py``.

What this module adds, and what it does NOT
-------------------------------------------
It adds identity: a :class:`Grant` names an authorisation and the
(sink, argument) positions it licenses, and :meth:`GrantIndex.shared_witnesses`
intersects the witness sets of a declared group of arguments. A non-empty
intersection says **the arguments of this call trace to a common grant**.

It does **not** establish that the combination is authorised. That rests
entirely on the grant having been *issued* correctly, and issuance is out of
scope here: grants are supplied structurally, by a caller that already knows the
pairing. Three things follow and are each pinned by a test:

* **A grant that covers too much admits mixing.** One grant naming all four
  values of a two-payment request authorises every pairing. The intersection is
  non-empty and the mechanism is working exactly as specified — the error is in
  the issuance, and no property of this index can detect it
  (``test_one_grant_covering_both_payments_admits_mixing``).
* **No text is parsed.** Deriving grants from a request is a separate problem
  with its own failure modes (a single clause naming two payments; a pairing
  that spans two sentences). A clause heuristic is not a parser of user
  authority and this module does not pretend to contain one.
* **Finite-id equality is a decidable CHECK, not correct issuance.**
  ``kernel.decidability.classify`` takes only (codomain kind, consumption mode),
  so comparing witness ids classifies ``DECIDABLE_PASS`` without change — which
  is compatibility of the check. It says nothing about whether a request was
  divided into the right grants, whether values were bound to the right
  arguments, or whether one id has merged two authorisations.

Relation to enumerating the tuples
----------------------------------
Issuing grants and enumerating permitted tuples are not different in authority:
verified grants can be materialised into tuples, and a wrongly issued grant
widens permissions just as a wrong tuple would. The advantage claimed here is
narrower — keeping the link inside the structure the engine already has, and a
cheap check — not that an implicit relation is automatically safer.

Composition
-----------
A shared grant is an ADDITIONAL, narrower requirement. It is never a substitute
for the origin checks in ``TrustedValueIndex``: a tuple with a common witness
whose values are not trusted values must still be refused by those. Pinned by
``test_scoped_never_widens_the_existing_origin_check``.
"""

from __future__ import annotations

from dataclasses import dataclass

from axor_core.contracts.taint import TrustedOrigin
# Deliberate reuse of the existing canonicalisation: a grant and a call must
# agree that 2200, 2200.0 and "2,200" are the same amount, or the mechanism
# denies legitimate tuples for a reason that has nothing to do with authority.
from axor_core.taint.trusted import _generic, _num

#: Bound on stored grants, mirroring the trusted index's own bounding policy.
#: Past it, issuance stops rather than growing without limit, so a flood of
#: grants cannot exhaust memory — and a value with no stored grant simply has no
#: scoped permission, which is the fail-closed direction.
MAX_GRANTS = 4096


def canonical(value: object) -> str | None:
    """The key a value is stored and looked up under.

    ``None`` for a value with no comparable form (booleans, ``None``, an empty
    string) — such a value can never carry a witness, so it can never be the
    one that satisfies a group.
    """
    if value is None or isinstance(value, bool):
        return None
    number = _num(value)
    if number is not None:
        return f"num:{number}"
    text = _generic(str(value))
    return f"text:{text}" if text else None


@dataclass(frozen=True)
class Grant:
    """One authorisation, and the argument positions it licenses.

    ``witness_id`` identifies the authorisation itself — a specific instruction
    or invoice, not the document or source it arrived in. Two instructions in
    one document are two grants; one id covering both is the issuance error the
    module docstring names.

    ``bindings`` maps an argument name to the value this grant licenses *in that
    position*, for ``sink`` only. The position matters: a token that appears as
    an amount under one grant must not support a recipient under another, and a
    grant for one operation must not license a different one.
    """

    witness_id: str
    sink: str
    bindings: tuple[tuple[str, object], ...]
    origin: TrustedOrigin = TrustedOrigin.TASK

    @classmethod
    def of(
        cls,
        witness_id: str,
        sink: str,
        bindings: dict[str, object],
        origin: TrustedOrigin = TrustedOrigin.TASK,
    ) -> "Grant":
        return cls(witness_id, sink, tuple(sorted(bindings.items(), key=repr)), origin)


class GrantIndex:
    """Which grants license a value in a given (sink, argument) position."""

    __slots__ = ("_by_position", "_grants", "saturated")

    def __init__(self) -> None:
        # (sink, arg, canonical value) -> witness ids licensing it there
        self._by_position: dict[tuple[str, str, str], set[str]] = {}
        self._grants: dict[str, Grant] = {}
        self.saturated = False

    def __len__(self) -> int:
        return len(self._grants)

    def issue(self, grant: Grant) -> bool:
        """Record a grant. Returns False if it was not stored (bound reached).

        Structured issuance only: the caller states the pairing. Nothing here
        infers one.
        """
        if grant.witness_id in self._grants:
            return True
        if len(self._grants) >= MAX_GRANTS:
            self.saturated = True
            return False
        self._grants[grant.witness_id] = grant
        for arg, value in grant.bindings:
            key = canonical(value)
            if key is None:
                continue
            self._by_position.setdefault((grant.sink, arg, key), set()).add(
                grant.witness_id
            )
        return True

    def witnesses(self, sink: str, arg: str, value: object) -> frozenset[str]:
        """Grants licensing ``value`` for ``sink``'s ``arg`` — and no other
        position, and no other sink."""
        key = canonical(value)
        if key is None:
            return frozenset()
        return frozenset(self._by_position.get((sink, arg, key), ()))

    def shared_witnesses(
        self, sink: str, args: dict[str, object], group: "frozenset[str] | set[str]"
    ) -> frozenset[str]:
        """Grants licensing EVERY argument of ``group`` in its own position.

        Fail-closed in three ways, so an unsupported call gains nothing: an
        empty group yields no witness; an argument of the group missing from the
        call yields no witness; an argument with no grant in its position
        empties the intersection.
        """
        if not group:
            return frozenset()
        shared: frozenset[str] | None = None
        for arg in sorted(group):
            if arg not in args:
                return frozenset()
            here = self.witnesses(sink, arg, args[arg])
            shared = here if shared is None else (shared & here)
            if not shared:
                return frozenset()
        return shared or frozenset()

    def jointly_granted(
        self,
        sink: str,
        args: dict[str, object],
        group: "frozenset[str] | set[str]",
        *,
        accept: "frozenset[TrustedOrigin] | None" = None,
    ) -> bool:
        """Whether some single grant licenses the whole group.

        ``accept`` restricts which grant origins count, mirroring
        ``integrity_origins``: a grant issued off a tool read should not clear a
        group under ``request-only``.

        True means the arguments share a grant. It does NOT mean the
        combination is authorised — that is the issuance premise, stated in the
        module docstring.
        """
        shared = self.shared_witnesses(sink, args, group)
        if accept is None:
            return bool(shared)
        return any(self._grants[w].origin in accept for w in shared)
