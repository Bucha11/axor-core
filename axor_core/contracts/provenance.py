"""ValueProvenance — the per-value trust-model interface (spec K3 / Part II).

Enforcement depends on THIS contract, not on a concrete engine. A trust model
maps a value to its `causal_root` (the structural provenance projection) and
records produced values. Replaceable: our content-derivation `TaintEngine`, a
CaMeL-style interpreter, or a FIDES-style labeler all satisfy it.

Typing enforcement against this Protocol is a *convention expressed in types*:
engine internals NOT in the contract (legacy session-taint, cross-session
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
