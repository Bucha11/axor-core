"""Per-value causal_root — the per-value provenance object the kernel decides on.

This is the *per-value* provenance object the spec's TM2 defines. It is now the
ENFORCEMENT object: ``ValueProvenance.derive_value`` returns a ``CausalRoot`` for a
sink's driving argument, and the per-value gate (``IntentLoop``) denies on its
labels (integrity ``is_tainted`` / confidentiality ``sensitive``). The density
meter (TM3.3) still reads the same object observe-only to measure per-value vs
session-sticky separation, but the object itself is load-bearing in ``allow``.

Semantics follow TM2 exactly:

    causal_root(constant)        = ∅
    causal_root(external_read s) = { s }
    causal_root(mint(v₁..vₙ))    = ⋃ causal_root(vᵢ)        # over-taint (safe direction)
    causal_root(parse(v))        = causal_root(v)           # schema parse is passthrough
    causal_root(cross_process_in)= { ⊤_untrusted, non-sensitive }   # re-mint (TM4.1)

    tainted(v) ⟺ causal_root(v) ⊄ Trusted

Here ``Trusted`` is empty (no source is trusted by default), so ``is_tainted``
is simply "carries any external source". Integrity and sensitivity are tracked as
the two independent labels TM2 calls for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource


@dataclass(frozen=True)
class CausalRoot:
    """Provenance of a single value (not a session).

    sources    — external sources that *explicitly* influenced this value (TM2).
                 Empty == trusted/constant.
    sensitive  — confidentiality label: True if the value carries a sensitive
                 source (e.g. a secret read) and is harmful to let leave.
    """

    sources: frozenset[TaintSource] = field(default_factory=frozenset)
    sensitive: bool = False

    @property
    def is_tainted(self) -> bool:
        # TM2: tainted ⟺ causal_root ⊄ Trusted; Trusted is empty here.
        return bool(self.sources)

    # ── TM2 constructors ────────────────────────────────────────────────────

    @classmethod
    def constant(cls) -> "CausalRoot":
        """A literal / user-trusted value: causal_root = ∅."""
        return cls()

    @classmethod
    def external_read(cls, source: TaintSource, *, sensitive: bool = False) -> "CausalRoot":
        """A value read from an external source: causal_root = {source}."""
        return cls(sources=frozenset({source}), sensitive=sensitive)

    @classmethod
    def mint(cls, *roots: "CausalRoot") -> "CausalRoot":
        """A value derived from others: union of sources, OR of sensitivity.

        This is the over-taint point (TM2): the safe direction. A minted value is
        at least as tainted as any of its inputs.
        """
        sources: frozenset[TaintSource] = frozenset()
        sensitive = False
        for r in roots:
            sources |= r.sources
            sensitive = sensitive or r.sensitive
        return cls(sources=sources, sensitive=sensitive)

    @classmethod
    def parse(cls, root: "CausalRoot") -> "CausalRoot":
        """Schema parse of a value: causal_root is preserved unchanged (TM2)."""
        return root

    @classmethod
    def cross_process_in(cls) -> "CausalRoot":
        """A value arriving across a process boundary is re-minted (TM4.1):
        maximal integrity taint, explicitly *non*-sensitive (it carries none of
        *our* secrets).
        """
        return cls(sources=frozenset({TaintSource.UNKNOWN_EXTERNAL}), sensitive=False)
