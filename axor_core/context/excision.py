"""Apply a context-repair proposal as an authorised governance action.

A behavioral monitor (axor-probe) localises the context fragments that cause a
regime escape and proposes an excision; *removing* them is an irreversible memory
mutation, so it happens here, gated on a GovernanceAuthority — the same authority
model that gates degradation clearance.

The authority TYPE is the operator-in-loop boundary: an ``automated_policy`` may
remove only the clean, pure-tainted ``auto_excise`` set; the ``escalate`` set
(collateral or diffuse) requires a ``human_operator`` (or ``trusted_boundary``) and
is otherwise deferred — recorded, not removed. axor-core never imports axor-probe:
the proposal crosses as plain fragment ids.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from axor_core.contracts.context import ContextFragment
from axor_core.contracts.degradation import GovernanceAuthority

_VALID_AUTHORITIES = frozenset({"human_operator", "automated_policy", "trusted_boundary"})


class ExcisionError(Exception):
    """Raised when excision is attempted without a valid GovernanceAuthority."""


@dataclass(frozen=True)
class ExcisionResult:
    repaired_fragments: tuple[ContextFragment, ...]
    excised: tuple[str, ...]      # fragment ids removed
    deferred: tuple[str, ...]     # ids an automated policy may not remove — left for an operator


def _default_key(f: ContextFragment) -> str:
    """Identify a fragment by its taint canary (the localised candidates are tainted)."""
    return f.taint_mark or ""


def apply_excision(
    fragments: Sequence[ContextFragment],
    *,
    auto_excise: Iterable[str],
    escalate: Iterable[str] = (),
    authority: GovernanceAuthority,
    key: Callable[[ContextFragment], str] = _default_key,
) -> ExcisionResult:
    """Remove the proposed fragments under the given authority. Raises ExcisionError
    on an invalid authority. An ``automated_policy`` removes only ``auto_excise`` and
    defers ``escalate``; a human/trusted authority removes both."""
    if not authority.authority_id.strip():
        raise ExcisionError("excision requires a non-blank GovernanceAuthority.authority_id")
    if not authority.reason_code.strip():
        raise ExcisionError("excision requires a non-blank reason_code")
    if authority.authority_type not in _VALID_AUTHORITIES:
        raise ExcisionError(f"unknown authority_type {authority.authority_type!r}")

    auto = set(auto_excise)
    operator = set(escalate)
    if authority.authority_type == "automated_policy":
        remove, deferred = auto, operator
    else:
        remove, deferred = auto | operator, set()

    present = {key(f) for f in fragments}
    repaired = tuple(f for f in fragments if key(f) not in remove)
    return ExcisionResult(
        repaired_fragments=repaired,
        excised=tuple(sorted(remove & present)),
        deferred=tuple(sorted(deferred)),
    )
