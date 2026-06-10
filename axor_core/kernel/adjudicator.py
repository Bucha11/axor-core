"""Advisory adjudicator — projection-only, memoized by projection hash, tightening.

An adjudicator is a pluggable advisory layer (e.g. an LLM judge / external policy
oracle). The kernel keeps three hard guarantees over it:

  • PROJECTION-ONLY: it sees only the CanonicalizedIntent (the projection) —
    enums, ints, bucketed lengths, hashed paths — never raw content. It cannot be
    steered by the governed content's semantics.
  • MEMOIZED by projection hash: equal projections get equal verdicts, queried
    once. The advice is therefore a deterministic function of the projection (no
    per-call drift), and an expensive/external oracle is consulted at most once per
    distinct projection.
  • ADVISORY / TIGHTENING-ONLY: it can only ADD a deny. A kernel HARD-DENY is never
    overridden by an adjudicator ALLOW, and an adjudicator that errors ABSTAINS
    (advice is a bonus restriction on top of the kernel, never a relaxation).

Core defines the protocol and the memoizing wrapper; implementations live outside.
"""

from __future__ import annotations

import hashlib
from dataclasses import fields
from enum import Enum
from typing import Protocol, runtime_checkable

from axor_core.contracts.canonical import CanonicalizedIntent


class AdjudicationVerdict(str, Enum):
    ADVISE_ALLOW = "advise_allow"
    ADVISE_DENY = "advise_deny"
    ABSTAIN = "abstain"      # no opinion (also the fail-closed-for-availability default)


@runtime_checkable
class Adjudicator(Protocol):
    """An advisory oracle over the projection. Must not raise (the wrapper guards
    anyway); must read ONLY the projection."""

    def adjudicate(self, projection: CanonicalizedIntent) -> AdjudicationVerdict:
        ...


def projection_hash(projection: CanonicalizedIntent) -> str:
    """Stable digest of a projection for audit / cross-process memoization.

    Deterministic over the dataclass fields in declaration order; values are enums/
    ints/strs/bools, so their str() is stable. No raw content is present by
    construction (CanonicalizedIntent carries only canonical features)."""
    parts = []
    for f in fields(projection):
        v = getattr(projection, f.name)
        v = v.value if isinstance(v, Enum) else v
        parts.append(f"{f.name}={v}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


class MemoizingAdjudicator:
    """Wraps an advisory Adjudicator: memoizes verdicts by projection hash and
    enforces the tightening-only contract."""

    def __init__(self, inner: Adjudicator) -> None:
        self._inner = inner
        self._cache: dict[CanonicalizedIntent, AdjudicationVerdict] = {}

    def verdict(self, projection: CanonicalizedIntent) -> AdjudicationVerdict:
        """Memoized advisory verdict for this projection. Equal projection → equal
        verdict. A raising adjudicator ABSTAINS (advisory layer must not break the
        loop)."""
        cached = self._cache.get(projection)
        if cached is not None:
            return cached
        try:
            v = self._inner.adjudicate(projection)
        except Exception:
            v = AdjudicationVerdict.ABSTAIN
        if not isinstance(v, AdjudicationVerdict):
            v = AdjudicationVerdict.ABSTAIN
        self._cache[projection] = v
        return v

    def apply(self, projection: CanonicalizedIntent, kernel_allowed: bool) -> bool:
        """Effective allow after advice. Tightening-only:

        - kernel HARD-DENY (kernel_allowed=False) → stays denied; advice ignored
          (an adjudicator can never override a hard deny).
        - kernel allowed → an ADVISE_DENY tightens to deny; ALLOW/ABSTAIN keep it
          allowed.
        """
        if not kernel_allowed:
            return False
        return self.verdict(projection) != AdjudicationVerdict.ADVISE_DENY
