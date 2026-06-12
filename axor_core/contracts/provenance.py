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
