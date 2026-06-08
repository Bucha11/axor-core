"""Thm. 0 — the T4 decidability split (v4.12, K3.6).

T4 is the faithfulness obligation: a projection's *effective* codomain equals its
*nominal* one (no weird machine / residual channel — Def. 4). Whether T4 is
*decidable* depends on the codomain **and** the consumer, and it splits cleanly:

  • DECIDABLE — discharged by a decision procedure (not fuzzing) — for
    low-capacity codomains: a **finite enum** consumed as a finite case-split, and
    a **bounded numeric** range consumed *numerically* (comparison/arithmetic/
    range-check only). No pair π(x₁)=π(x₂) can be split by the consumer into
    distinct effects, so no residual channel can exist — by construction.

  • FUZZING ONLY — undecidable in general (Rice / LangSec weird-machine) — for
    rich-syntax codomains: **path** (a filesystem resolver: `..`, symlinks,
    newline, unicode), **string subfields** (a shell/SQL/URL/template interpreter)
    and **carrier over free text**. Here T4 stays a fuzz obligation (K5/T4); the
    two real bugs (newline, `../`) live exactly here, as the split predicts.

Decidability of the enum/numeric branch is **conditional on K2**: the consumer's
*consumption mode* must be a registered, surveyable property of a finite sink —
otherwise deciding it would be whole-program analysis (Rice), undecidable.

This module is the decision procedure for the decidable branch and the classifier
for which branch a (codomain, consumption-mode) pair falls into.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Iterable


class CodomainKind(str, Enum):
    """The admissible codomain kinds (Def. 3b), tagged by T4 tractability."""
    ENUM = "enum"                       # decidable
    BOUNDED_NUMERIC = "bounded_numeric"  # decidable
    ORIGIN_CLASS = "origin_class"        # decidable (finite enum of origins)
    PROVENANCE_LABEL = "provenance_label"  # decidable (finite enum)
    PATH_CLASS = "path_class"            # fuzzing (rich-syntax consumer)
    CARRIER_OVER_TEXT = "carrier_over_text"  # fuzzing
    STRING_SUBFIELD = "string_subfield"  # fuzzing


# Consumption modes a registered sink (K2) may declare for a projection.
class ConsumptionMode(str, Enum):
    CASE_SPLIT = "case_split"      # finite enum compared/branched — decidable
    NUMERIC = "numeric"            # compare/arithmetic/range only — decidable
    PATH_RESOLVE = "path_resolve"  # handed to a filesystem resolver — fuzzing
    INTERPRET = "interpret"        # handed to a shell/SQL/template interpreter — fuzzing


_DECIDABLE_KINDS = frozenset({
    CodomainKind.ENUM,
    CodomainKind.BOUNDED_NUMERIC,
    CodomainKind.ORIGIN_CLASS,
    CodomainKind.PROVENANCE_LABEL,
})

_DECIDABLE_MODES = frozenset({ConsumptionMode.CASE_SPLIT, ConsumptionMode.NUMERIC})


class T4Verdict(str, Enum):
    DECIDABLE_PASS = "decidable_pass"
    DECIDABLE_FAIL = "decidable_fail"
    FUZZ_REQUIRED = "fuzz_required"


@dataclass(frozen=True)
class T4Result:
    verdict: T4Verdict
    reason: str

    @property
    def is_pass(self) -> bool:
        return self.verdict == T4Verdict.DECIDABLE_PASS


def is_t4_decidable(kind: CodomainKind, mode: ConsumptionMode) -> bool:
    """Thm. 0 classifier: is T4 decidable for this (codomain, consumption) pair?

    Decidable iff the codomain is low-capacity AND the consumer treats it as a
    finite case-split / numeric value. If the consumer re-parses or resolves it
    (rich-syntax), T4 is a fuzzing obligation regardless of the nominal codomain
    (this is why FIDES's `string`-typed field is fuzzing, not decidable).
    """
    return kind in _DECIDABLE_KINDS and mode in _DECIDABLE_MODES


def verify_enum(value: object, admissible: Iterable[object]) -> T4Result:
    """Decision procedure for an enum codomain consumed as a case-split.

    Decidable: membership in a finite admissible set. effective ⊆ nominal iff the
    value is in the declared set; nothing the consumer does to an in-set value can
    manufacture an out-of-set effect (finite case-split).
    """
    admissible_set = set(admissible)
    if value in admissible_set:
        return T4Result(T4Verdict.DECIDABLE_PASS, f"{value!r} ∈ admissible enum")
    return T4Result(
        T4Verdict.DECIDABLE_FAIL, f"{value!r} ∉ admissible enum {sorted(map(repr, admissible_set))}"
    )


def verify_bounded_numeric(value: object, lo: Real, hi: Real) -> T4Result:
    """Decision procedure for a bounded-numeric codomain consumed numerically.

    Decidable: a real in [lo, hi]. Numeric consumption (compare/arithmetic/range)
    cannot decode a number into an instruction, so effective = nominal by
    construction — provided the value is genuinely numeric and in range.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return T4Result(T4Verdict.DECIDABLE_FAIL, f"{value!r} is not a real number")
    if lo <= value <= hi:
        return T4Result(T4Verdict.DECIDABLE_PASS, f"{value!r} ∈ [{lo}, {hi}]")
    return T4Result(T4Verdict.DECIDABLE_FAIL, f"{value!r} ∉ [{lo}, {hi}]")


def classify(kind: CodomainKind, mode: ConsumptionMode) -> T4Result:
    """Return the obligation a projection must discharge for T4 under Thm. 0:
    a decision procedure (decidable branch) or a fuzz obligation (rich-syntax).
    """
    if is_t4_decidable(kind, mode):
        return T4Result(
            T4Verdict.DECIDABLE_PASS,
            f"T4 decidable by construction for ({kind.value}, {mode.value}); "
            "discharge with verify_enum / verify_bounded_numeric (given K2).",
        )
    return T4Result(
        T4Verdict.FUZZ_REQUIRED,
        f"T4 is a fuzzing obligation for ({kind.value}, {mode.value}): the consumer "
        "is a rich-syntax interpreter; effective may exceed nominal (K5/T4).",
    )
