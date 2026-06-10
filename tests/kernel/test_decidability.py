"""Thm. 0 — T4 decidability split (v4.12 Phase 4).

The decidable branch (enum / bounded-numeric consumed as case-split / numeric)
is discharged by a decision procedure; the rich-syntax branch (path / string /
carrier) is classified FUZZ_REQUIRED. Plus a fuzz coverage-floor for the path
normalizer — the fuzzing region the split predicts bugs must live in.
"""

from __future__ import annotations

from axor_core.kernel.decidability import (
    CodomainKind,
    ConsumptionMode,
    T4Verdict,
    classify,
    is_t4_decidable,
    verify_bounded_numeric,
    verify_enum,
)


# ── classifier ──────────────────────────────────────────────────────────────────

def test_enum_and_numeric_are_decidable():
    assert is_t4_decidable(CodomainKind.ENUM, ConsumptionMode.CASE_SPLIT)
    assert is_t4_decidable(CodomainKind.BOUNDED_NUMERIC, ConsumptionMode.NUMERIC)
    assert is_t4_decidable(CodomainKind.PROVENANCE_LABEL, ConsumptionMode.CASE_SPLIT)


def test_rich_syntax_is_fuzz_required():
    assert not is_t4_decidable(CodomainKind.PATH_CLASS, ConsumptionMode.PATH_RESOLVE)
    assert not is_t4_decidable(CodomainKind.STRING_SUBFIELD, ConsumptionMode.INTERPRET)
    assert classify(CodomainKind.PATH_CLASS, ConsumptionMode.PATH_RESOLVE).verdict == T4Verdict.FUZZ_REQUIRED


def test_enum_consumed_by_an_interpreter_falls_to_fuzzing():
    # A low-capacity codomain handed to a rich-syntax consumer is NOT decidable:
    # decidability is a property of (codomain, consumer), per Thm. 0 (and why
    # FIDES's string-typed field is fuzzing, not a clean pass).
    assert not is_t4_decidable(CodomainKind.ENUM, ConsumptionMode.INTERPRET)


# ── decision procedures ───────────────────────────────────────────────────────

def test_verify_enum():
    assert verify_enum("read", {"read", "write", "exec"}).is_pass
    assert not verify_enum("rm -rf", {"read", "write", "exec"}).is_pass


def test_verify_bounded_numeric():
    assert verify_bounded_numeric(10, 0, 100).is_pass
    assert not verify_bounded_numeric(999999, 0, 100).is_pass
    # a number is admissible only if genuinely numeric (consumer-relativity):
    assert not verify_bounded_numeric("10", 0, 100).is_pass
    assert not verify_bounded_numeric(True, 0, 100).is_pass  # bool is not a real here


# ── fuzz coverage floor (the rich-syntax region Thm. 0 sends to fuzzing) ──────────

def test_path_normalizer_fuzz_floor_no_escape():
    """Coverage floor for the path projection (K5/T4 PLANNED). The decidable
    procedure does not apply here; we fuzz that lexical normalization keeps `../`
    escapes out of the allow-listed root — the class the two real bugs were in.
    """
    from axor_core.security.paths import path_matches_allowlist

    root = "/work/repo"
    # a spread of traversal / encoding shapes — the effective-codomain region
    escapes = [
        "../etc/passwd",
        "../../etc/passwd",
        "/work/repo/../../../etc/passwd",
        "/work/repo/sub/../../escape",
        "....//....//etc/passwd",
        "/work/repo/ok/../../../../root/.ssh/id_rsa",
        "/etc/passwd",
        "foo/../../bar",
    ]
    floor = 8
    assert len(escapes) >= floor  # explicit budget floor
    for cand in escapes:
        assert path_matches_allowlist(cand, [root]) is False, f"escape leaked into root: {cand!r}"

    # and genuinely-inside paths are accepted (effective = nominal both ways)
    for inside in ["/work/repo/a.py", "/work/repo/sub/b.txt", "/work/repo/./c"]:
        assert path_matches_allowlist(inside, [root]) is True, f"in-root path rejected: {inside!r}"
