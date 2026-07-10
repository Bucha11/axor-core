"""Desired-state lattice (protocol note v0.2, section 3) and the excision guard."""
from __future__ import annotations

from axor_core.kernel.state import DesiredState, Excision, excision_refused_refs


def test_stopped_is_absorbing() -> None:
    s = DesiredState(version=1, stopped=True)
    merged = s.merge(DesiredState(version=2, paused=True))
    assert merged.stopped and not merged.paused and merged.version == 2


def test_lww_by_version() -> None:
    s = DesiredState(version=3, paused=True)
    assert s.merge(DesiredState(version=2, paused=False)) is s
    assert s.merge(DesiredState(version=4, paused=False)).paused is False


def _exc(*refs: str) -> Excision:
    return Excision(excision_id="e1", target_refs=refs, reason="r",
                    operator="op", sig="s")


def test_excision_guard_refuses_operator_config_provenance() -> None:
    refused = excision_refused_refs(
        _exc("v1", "v2"), {"v1": "runtime", "v2": "operator_config"}
    )
    assert refused == ("v2",)


def test_excision_guard_fails_closed_on_unknown_provenance() -> None:
    assert excision_refused_refs(_exc("v1"), {}) == ("v1",)


def test_excision_guard_passes_runtime_provenance() -> None:
    assert excision_refused_refs(_exc("v1", "v2"),
                                 {"v1": "runtime", "v2": "runtime"}) == ()
