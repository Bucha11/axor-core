"""Degradation recompute (spec decision #9): level = max(severity(uncovered))."""
from __future__ import annotations

from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.degradation import compute_level
from axor_core.kernel.events import Fact


def _fact(fid: str, sev: int) -> Fact:
    return Fact(fact_id=fid, fact_type="denial", severity=sev)


def _attest(fid: str, covers: tuple[str, ...], revokes: str | None = None) -> Fact:
    return Fact(fact_id=fid, fact_type="operator_attestation", severity=0,
                covers=covers, revokes=revokes, operator="op", reason="r", sig="s")


def _facts(*items: Fact) -> dict[str, Fact]:
    return {f.fact_id: f for f in items}


def test_level_is_max_uncovered() -> None:
    assert compute_level(_facts(_fact("f1", 1), _fact("f2", 2))) is DegradationLevel.RESTRICTED


def test_attestation_descends_level() -> None:
    assert compute_level(_facts(_fact("f1", 2), _attest("a1", ("f1",)))) is DegradationLevel.NORMAL


def test_partial_coverage_partial_descent() -> None:
    facts = _facts(_fact("f1", 2), _fact("f2", 1), _attest("a1", ("f1",)))
    assert compute_level(facts) is DegradationLevel.CAUTIOUS


def test_revocation_restores_level() -> None:
    facts = _facts(_fact("f1", 2), _attest("a1", ("f1",)), _attest("a2", (), revokes="a1"))
    assert compute_level(facts) is DegradationLevel.RESTRICTED


def test_severity_clamped_to_terminal() -> None:
    assert compute_level(_facts(_fact("f1", 99))) is DegradationLevel.TERMINAL


def test_empty_facts_is_normal() -> None:
    assert compute_level({}) is DegradationLevel.NORMAL
