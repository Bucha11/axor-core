"""Per-value causal root — the provenance object the kernel decides on.

This is the provenance of a single value (as opposed to a whole session). It is
the ENFORCEMENT object: ``ValueProvenance.derive_value`` returns a ``CausalRoot``
for a sink's driving argument, and the per-value gate (``IntentLoop``) denies on
its labels (integrity ``is_tainted`` / confidentiality ``sensitive``). The density
meter reads the same object observe-only to compare per-value tracking against a
coarse session-wide flag, but the object itself is load-bearing when deciding
whether to allow a call.

The causal root of a value is the set of external sources that influenced it:

    causal_root(constant)        = {}      (a literal carries no external source)
    causal_root(external_read s) = { s }
    causal_root(mint(v1..vn))    = union of causal_root(vi)   # over-taint, the safe direction
    causal_root(parse(v))        = causal_root(v)             # schema parse passes provenance through
    causal_root(cross_process_in)= { unknown-external }, non-sensitive   # re-minted at the boundary

A value is tainted exactly when its causal root contains any external source.
No source is trusted by default, so ``is_tainted`` is simply "carries any
external source". Integrity (tainted) and confidentiality (sensitive) are tracked
as two independent labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource


@dataclass(frozen=True)
class CausalRoot:
    """Provenance of a single value (not a session).

    sources    — external sources that explicitly influenced this value.
                 Empty means trusted / constant.
    sensitive  — confidentiality label: True if the value carries a sensitive
                 source (e.g. a secret read) and is harmful to let leave.
    """

    sources: frozenset[TaintSource] = field(default_factory=frozenset)
    sensitive: bool = False

    @property
    def is_tainted(self) -> bool:
        # Tainted exactly when any external source is present (no source is
        # trusted by default).
        return bool(self.sources)

    # ── constructors ────────────────────────────────────────────────────────

    @classmethod
    def constant(cls) -> "CausalRoot":
        """A literal / user-trusted value: no external sources."""
        return cls()

    @classmethod
    def external_read(cls, source: TaintSource, *, sensitive: bool = False) -> "CausalRoot":
        """A value read from an external source: its causal root is {source}."""
        return cls(sources=frozenset({source}), sensitive=sensitive)

    @classmethod
    def mint(cls, *roots: "CausalRoot") -> "CausalRoot":
        """A value derived from others: union of sources, OR of sensitivity.

        This over-taints in the safe direction: a minted value is at least as
        tainted as any of its inputs.
        """
        sources: frozenset[TaintSource] = frozenset()
        sensitive = False
        for r in roots:
            sources |= r.sources
            sensitive = sensitive or r.sensitive
        return cls(sources=sources, sensitive=sensitive)

    @classmethod
    def parse(cls, root: "CausalRoot") -> "CausalRoot":
        """Schema parse of a value: causal root is preserved unchanged."""
        return root

    @classmethod
    def cross_process_in(cls) -> "CausalRoot":
        """A value arriving across a process boundary is re-minted: maximal
        integrity taint, explicitly non-sensitive (it carries none of our own
        secrets).
        """
        return cls(sources=frozenset({TaintSource.UNKNOWN_EXTERNAL}), sensitive=False)
