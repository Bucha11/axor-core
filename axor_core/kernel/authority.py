"""The authority state machine: what is in force, what may ever be granted, and
which moves between them are legal.

The kernel's safety claim about authority is not "the composition contains no
union". That phrasing cannot even describe a system with lawful recovery: an
escalation legitimately *extends* what is in force. The claim is a containment
invariant plus a closed set of transitions:

    A_t ⊆ B       at every step, for every axis

where the three sets are deliberately distinct (see
[governance-model](../../docs/governance-model.md) §2, "Three tool sets, kept
apart"):

    A₀   what the policy makes callable now
    G    what the operator declared grantable (the delegation ceiling)
    B    the trusted ceiling for this turn — A₀ ∪ G, after parent and deployment
         composition have narrowed both
    A_t  what is in force right now — A₀ plus whatever live authority adds

Only four moves exist, and each has its own legality condition:

    NARROW         A_t shrinks; B unchanged. Degradation, adaptive
                   re-classification, a grant expiring or being exhausted.
    RESTORE_CLASS  A_t grows by a tool with a live class grant. Legal only if the
                   added tools are in G and A_t ⊆ B after.
    RESTORE_CALL   A_t grows by a tool with unspent call authorisation — for one
                   call. Same condition.
    SPEND          a call authorisation moves from unspent to spent; A_t does not
                   grow. The retraction half of RESTORE_CALL, and the reason a
                   call authority cannot accumulate.
    SET_CEILING    B itself moves. This is the one move the untrusted side can
                   never cause: it is a trusted event (a new policy for the turn,
                   an operator profile change), and legality is the caller's to
                   establish — the classifier only reports that it happened.

This module is pure Ring 0: it defines the view, the invariant and the
classification over plain data. Building a view from the live runtime objects is
:mod:`axor_core.node.authority_probe` (Ring 1), so the definition cannot drift
into a parallel bookkeeping copy of the state it is supposed to check.

A note on what this does NOT claim. Containment of authority is one half of the
safety argument (the other is that an executed effect is within the authority in
force — the gate sequence and complete mediation). An invariant over these sets
says nothing about whether the policy's *description* of an effect is precise
enough; that is :func:`axor_core.policy.effect.validate_effect_coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.security.paths import paths_within

# Kernel-internal intents, not governed sinks: the loop handles them as intent
# kinds and they never reach the executor as tools. Degradation deliberately adds
# them at LOCKED (everything frozen except read + escalate), which is a narrowing
# of the governed surface even though it adds these names. The carve-out is
# declared here and asserted by the state-machine test, so a real tool appearing
# in that list fails the build rather than riding the exemption.
KERNEL_INTERNAL_INTENTS = frozenset({"escalate", "escalate_policy"})


class TransitionKind(str, Enum):
    IDLE = "idle"
    NARROW = "narrow"
    RESTORE_CLASS = "restore_class"
    RESTORE_CALL = "restore_call"
    SPEND = "spend"
    SET_CEILING = "set_ceiling"


@dataclass(frozen=True)
class AuthorityView:
    """A derived, read-only snapshot of the authority situation.

    Every field is *read* from the live objects, never maintained here.
    """

    # ── B: the trusted ceiling ────────────────────────────────────────────────
    ceiling_tools: frozenset[str] = frozenset()
    grantable_tools: frozenset[str] = frozenset()
    ceiling_paths: tuple[str, ...] = ()
    ceiling_consequence: ConsequenceClass = ConsequenceClass.CATASTROPHIC

    # ── A_t: in force right now ───────────────────────────────────────────────
    in_force_tools: frozenset[str] = frozenset()
    in_force_paths: tuple[str, ...] = ()
    in_force_consequence: ConsequenceClass = ConsequenceClass.BENIGN

    # ── the authority records that back any extension beyond A₀ ───────────────
    class_grants: frozenset[str] = frozenset()
    unspent_calls: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    spent_calls: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    @property
    def ceiling_key(self) -> tuple:
        """The part of the view that only a trusted event may change."""
        return (
            self.ceiling_tools,
            self.grantable_tools,
            tuple(self.ceiling_paths),
            self.ceiling_consequence,
        )


def _governed(tools: frozenset[str]) -> frozenset[str]:
    return frozenset(tools) - KERNEL_INTERNAL_INTENTS


def violations(view: AuthorityView) -> list[str]:
    """``A_t ⊆ B`` on every axis. Empty list == the invariant holds."""
    out: list[str] = []

    excess = _governed(view.in_force_tools) - view.ceiling_tools
    if excess:
        out.append(
            f"tools in force outside the ceiling: {sorted(excess)} "
            f"(ceiling {sorted(view.ceiling_tools)})"
        )

    if view.ceiling_paths:
        if not view.in_force_paths:
            out.append(
                "paths in force are unrestricted while the ceiling restricts to "
                f"{tuple(view.ceiling_paths)!r}"
            )
        elif not paths_within(view.in_force_paths, view.ceiling_paths):
            out.append(
                f"paths in force {tuple(view.in_force_paths)!r} are not within the "
                f"ceiling {tuple(view.ceiling_paths)!r}"
            )

    if view.in_force_consequence > view.ceiling_consequence:
        out.append(
            f"unattended consequence in force {view.in_force_consequence.name} "
            f"exceeds the ceiling {view.ceiling_consequence.name}"
        )

    both = view.unspent_calls & view.spent_calls
    if both:
        out.append(f"call authorisations both spent and unspent: {sorted(both)}")

    return out


def check(
    before: AuthorityView, after: AuthorityView, declared: TransitionKind
) -> list[str]:
    """Verify a transition against the kind its CAUSE says it is.

    A transition is typed by what caused it, and the cause is not always visible
    in the two states: a trusted re-assertion of the turn's policy is a
    ``SET_CEILING`` even when the ceiling value happens to be unchanged, and from
    the states alone that is indistinguishable from authority appearing out of
    nowhere. So the caller — which knows whether a trusted event occurred —
    declares the kind, and this function checks the declaration. Use
    :func:`classify` only where no cause is available (then an unexplained
    widening is reported, which is the silent-widening detector).

    Returns the list of reasons the declaration does not hold (empty == legal).
    """
    errors = violations(after)
    grew = _governed(after.in_force_tools) - _governed(before.in_force_tools)
    consequence_grew = after.in_force_consequence > before.in_force_consequence

    if declared is TransitionKind.SET_CEILING:
        # The one move only a trusted event may cause. It may widen or narrow
        # anything, provided containment still holds afterwards (checked above).
        # Legality of the CAUSE is the caller's obligation, not something a pair
        # of states can establish.
        return errors

    if declared is TransitionKind.IDLE:
        if before != after:
            errors.append("declared IDLE but the authority situation changed")
        return errors

    if declared is TransitionKind.NARROW:
        if grew:
            errors.append(f"declared NARROW but authority grew by {sorted(grew)}")
        if consequence_grew:
            errors.append(
                f"declared NARROW but the unattended ceiling in force rose "
                f"{before.in_force_consequence.name} → {after.in_force_consequence.name}"
            )
        if before.ceiling_key != after.ceiling_key:
            errors.append("declared NARROW but the ceiling moved (that is SET_CEILING)")
        return errors

    if declared is TransitionKind.SPEND:
        newly_spent = (after.spent_calls - before.spent_calls) & before.unspent_calls
        if not newly_spent:
            errors.append("declared SPEND but no unspent call authorisation was spent")
        if grew or consequence_grew:
            errors.append(
                f"declared SPEND but authority grew by {sorted(grew)}"
                if grew else "declared SPEND but the unattended ceiling in force rose"
            )
        if after.unspent_calls & newly_spent:
            errors.append(f"a spent call authorisation is unspent again: {sorted(newly_spent)}")
        return errors

    # RESTORE_CLASS / RESTORE_CALL
    new_class = after.class_grants - before.class_grants
    new_calls = after.unspent_calls - before.unspent_calls
    if declared is TransitionKind.RESTORE_CLASS and not new_class:
        errors.append("declared RESTORE_CLASS but no class grant appeared")
    if declared is TransitionKind.RESTORE_CALL and not new_calls:
        errors.append("declared RESTORE_CALL but no call authorisation appeared")
    if before.ceiling_key != after.ceiling_key:
        errors.append("declared a restoration but the ceiling moved")
    granted_tools = frozenset(new_class) | {t for t, _ in new_calls}
    outside = _governed(granted_tools) - after.ceiling_tools
    if outside:
        errors.append(
            f"restored authority over tools outside the ceiling: {sorted(outside)}"
        )
    # A restoration may only reach tools the operator declared grantable, or ones
    # already callable — never something neither set names.
    unbacked = _governed(granted_tools) - (
        after.grantable_tools | _governed(before.in_force_tools)
    )
    if unbacked:
        errors.append(
            f"restored authority over tools that are neither in force nor "
            f"grantable: {sorted(unbacked)}"
        )
    unexplained = grew - _governed(granted_tools)
    if unexplained:
        errors.append(
            f"authority grew by {sorted(unexplained)}, which no new record authorises"
        )
    if frozenset(after.spent_calls) & new_calls:
        errors.append("a call authorisation was minted already spent")
    return errors


def classify(
    before: AuthorityView, after: AuthorityView
) -> tuple[TransitionKind, list[str]]:
    """Infer the move from the two states, and whether it was legal.

    For the case where no cause is declared — an audit over a recorded sequence,
    or a check that nothing moved behind the system's back. Where the cause IS
    known, :func:`check` is the stronger tool: inference cannot see a trusted
    re-assertion that leaves the ceiling value unchanged, and will report the
    authority it restores as unexplained.

    Returns ``(kind, errors)``; a move with no legal reading is reported as the
    kind it most resembles, with the reason it is illegal.
    """
    errors = violations(after)

    if before.ceiling_key != after.ceiling_key:
        # Only a trusted event may move B. The classifier cannot verify WHO moved
        # it (that is the caller's obligation); it reports the move so a test or
        # an audit can insist on a trusted cause.
        return TransitionKind.SET_CEILING, errors

    newly_spent = (after.spent_calls - before.spent_calls) & before.unspent_calls
    if newly_spent:
        if not _governed(after.in_force_tools) <= _governed(before.in_force_tools):
            errors.append(
                "spending a call authorisation widened what is in force: "
                f"{sorted(_governed(after.in_force_tools) - _governed(before.in_force_tools))}"
            )
        if after.unspent_calls & newly_spent:
            errors.append(f"a spent call authorisation is unspent again: {sorted(newly_spent)}")
        return TransitionKind.SPEND, errors

    new_class = after.class_grants - before.class_grants
    new_calls = after.unspent_calls - before.unspent_calls
    if new_class or new_calls:
        kind = TransitionKind.RESTORE_CLASS if new_class else TransitionKind.RESTORE_CALL
        granted_tools = frozenset(new_class) | {t for t, _ in new_calls}
        outside = _governed(granted_tools) - after.ceiling_tools
        if outside:
            errors.append(
                f"restored authority over tools outside the ceiling: {sorted(outside)}"
            )
        # A restoration may only reach tools the operator declared grantable, or
        # ones already callable — never something neither set names.
        unbacked = _governed(granted_tools) - (
            after.grantable_tools | _governed(before.in_force_tools)
        )
        if unbacked:
            errors.append(
                f"restored authority over tools that are neither in force nor "
                f"grantable: {sorted(unbacked)}"
            )
        if frozenset(after.spent_calls) & new_calls:
            errors.append("a call authorisation was minted already spent")
        return kind, errors

    if _governed(after.in_force_tools) < _governed(before.in_force_tools) or (
        after.in_force_consequence < before.in_force_consequence
    ):
        return TransitionKind.NARROW, errors

    if (
        _governed(after.in_force_tools) != _governed(before.in_force_tools)
        or after.in_force_consequence != before.in_force_consequence
        or tuple(after.in_force_paths) != tuple(before.in_force_paths)
    ):
        # In force changed without any authority record to back it and without the
        # ceiling moving: there is no legal move with that shape.
        errors.append(
            "what is in force changed with no authority record and no ceiling "
            f"change: tools {sorted(_governed(before.in_force_tools))} → "
            f"{sorted(_governed(after.in_force_tools))}, consequence "
            f"{before.in_force_consequence.name} → {after.in_force_consequence.name}, "
            f"paths {tuple(before.in_force_paths)!r} → {tuple(after.in_force_paths)!r}"
        )
        return TransitionKind.RESTORE_CLASS, errors

    return TransitionKind.IDLE, errors


# Which moves may extend what is in force. Named so a reader does not have to
# infer the asymmetry from the code above.
EXTENDING = frozenset({TransitionKind.RESTORE_CLASS, TransitionKind.RESTORE_CALL})
CONTRACTING = frozenset({TransitionKind.NARROW, TransitionKind.SPEND})
