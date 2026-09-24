"""ValueProvenance — the per-value trust-model interface.

Enforcement depends on THIS contract, not on a concrete engine. A trust model
maps a value to its `causal_root` (the structural provenance projection) and
records produced values. Any backend satisfies it: a content-derivation taint
tracker, a data-flow interpreter, or a label-propagating tracker.

Typing enforcement against this Protocol is a convention expressed in types:
engine internals NOT in the contract (whole-session taint, cross-session
persistence) are structurally outside the enforcement boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from axor_core.contracts.taint import TrustedOrigin
from axor_core.taint.causal_root import CausalRoot


@runtime_checkable
class ValueProvenance(Protocol):
    """Per-value provenance contract consumed by the kernel for enforcement."""

    def register_value(self, content: object, root: CausalRoot) -> None:
        """Record that a value with the given causal_root produced this content."""
        ...

    def derive_value(self, value: object) -> CausalRoot:
        """Per-value causal_root of `value` (constant/clean if untainted)."""
        ...

    def inherit_value_ledger(self, parent: "ValueProvenance") -> None:
        """Fold a parent's per-value provenance into this (child) backend so the
        child's gate sees values the parent marked tainted/sensitive (the spawn
        boundary). A trust-model backend must support child inheritance; how it
        folds the state is the backend's business."""
        ...

    def confidentiality_floor_active(self) -> bool:
        """True while a sensitive read is outstanding — the content-blind
        confidentiality floor that gates egress until governance endorses release.

        This is part of the contract, not an optional extra: the kernel gates
        confidentiality on THIS, not on a value's derived ``sensitive`` label,
        precisely because the floor is sound (armed on the fact of the read) while
        per-value derivation is paraphrase-leaky. A backend that omits it would
        silently downgrade the kernel's headline confidentiality guarantee, so the
        enforcement paths call it directly and a non-conforming backend fails loudly
        rather than degrading in silence."""
        ...


@runtime_checkable
class ContextProvenance(ValueProvenance, Protocol):
    """A trust model that supports the context-default integrity mode.

    Under ``integrity_default == "context"`` a value the model generates carries
    the node's :meth:`context_root` — every untrusted source registered so far —
    unless it provably originates from a trusted source registered with
    :meth:`register_trusted`. Proving trusted origin instead of untrusted origin
    is what makes re-encoding an attacker value useless: any spelling that is not
    literally a trusted value is tainted (docs/rfc-integrity-context-default.md).
    """

    integrity_default: str

    def register_trusted(self, content: object, origin: "TrustedOrigin") -> None:
        """Record values of ``content`` as having an origin the attacker cannot
        author (the user's task, operator config, a trusted tool, an endorsement)."""
        ...

    def context_root(self) -> CausalRoot:
        """Join of the untrusted sources this node's model has been shown."""
        ...

    def derive_carried(self, value: object) -> CausalRoot:
        """What ``value`` visibly carries, without the context root."""
        ...

    def trusted_origin(self, value: object) -> "TrustedOrigin | None":
        """The origin that makes ``value`` trusted, or None if it has none."""
        ...
