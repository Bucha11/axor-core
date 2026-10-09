"""Phase 0, P5 and P6: the two holes the audit found, pinned closed.

**P5 — one endorsement releases one read.** ``_outstanding`` counts un-released
reads of the same secret (``register_value`` increments per read), but
``endorse_value`` used to drop the whole entry, so a single endorsement lifted a
floor armed by N reads: governance named one value and silently released every
outstanding read of it. A counter that can only be cleared is not a counter, and
the floor is the one obligation the paper's C2 is stated over — so the release
now decrements, and these tests are what keep it decrementing.

**P6 — executed equals checked.** The gate cascade decides on the call's own
arguments; a TRANSFORM decision would hand the handler a payload that passed no
gate. No gate in this kernel produces TRANSFORM, so the loop refuses one rather
than executing it — an unreachable path kept unreachable, since the thing it
would break is exactly the identity P6 asserts.
"""

from __future__ import annotations

import pytest

from axor_core.contracts.degradation import GovernanceAuthority
from axor_core.contracts.taint import TaintSource
from axor_core.errors.exceptions import TaintClearanceError
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import _MAX_OUTSTANDING_SECRETS, TaintEngine

_GA = GovernanceAuthority("operator", "human_operator", "reviewed")
_SECRET = "CNRY7PQ4ZK2M8XJD3VB6WLT9"
_OTHER = "CNRYQ9WE5RT7YU1IO3PA8SDF"


def _read(engine: TaintEngine, secret: str = _SECRET) -> None:
    engine.register_value(secret, CausalRoot.external_read(TaintSource.FILE, sensitive=True))


# ── P5: release accounting ────────────────────────────────────────────────────

def test_one_endorsement_does_not_release_two_reads() -> None:
    """The hole as found: two reads, one release, floor stays up."""
    engine = TaintEngine()
    _read(engine)
    _read(engine)
    engine.endorse_value(_SECRET, _GA)
    assert engine.confidentiality_floor_active() is True


def test_one_endorsement_per_read_releases_the_floor() -> None:
    engine = TaintEngine()
    _read(engine)
    _read(engine)
    engine.endorse_value(_SECRET, _GA)
    engine.endorse_value(_SECRET, _GA)
    assert engine.confidentiality_floor_active() is False


@pytest.mark.parametrize("reads", [1, 2, 3, 7])
def test_release_takes_exactly_as_many_endorsements_as_reads(reads: int) -> None:
    engine = TaintEngine()
    for _ in range(reads):
        _read(engine)
    for i in range(reads):
        assert engine.confidentiality_floor_active() is True, f"lifted after {i} releases"
        engine.endorse_value(_SECRET, _GA)
    assert engine.confidentiality_floor_active() is False


def test_extra_endorsements_do_not_go_negative() -> None:
    """A release of something not outstanding is a no-op, not a credit.

    Otherwise an operator could bank releases before the read and pre-clear the
    floor — a release of a future read is not a release.
    """
    engine = TaintEngine()
    _read(engine)
    engine.endorse_value(_SECRET, _GA)
    engine.endorse_value(_SECRET, _GA)          # nothing outstanding
    engine.endorse_value(_OTHER, _GA)           # never read
    assert engine.confidentiality_floor_active() is False
    _read(engine)                               # a NEW read must re-arm
    assert engine.confidentiality_floor_active() is True


def test_releasing_one_secret_leaves_another_outstanding() -> None:
    engine = TaintEngine()
    _read(engine, _SECRET)
    _read(engine, _OTHER)
    engine.endorse_value(_SECRET, _GA)
    assert engine.confidentiality_floor_active() is True
    engine.endorse_value(_OTHER, _GA)
    assert engine.confidentiality_floor_active() is False


def test_release_still_requires_a_governance_authority() -> None:
    engine = TaintEngine()
    _read(engine)
    with pytest.raises(TaintClearanceError):
        engine.endorse_value(_SECRET, object())  # type: ignore[arg-type]
    assert engine.confidentiality_floor_active() is True


def test_saturation_is_not_released_by_endorsing_everything() -> None:
    """P9, kept beside P5 because the fix touches the same registry.

    Past the cap the floor is sticky: the reads that saturated it were never
    tracked, so releasing the tracked ones cannot account for them.
    """
    engine = TaintEngine()
    secrets = [f"CNRY-secret-{i:05d}-padding" for i in range(_MAX_OUTSTANDING_SECRETS + 1)]
    for secret in secrets:
        _read(engine, secret)
    for secret in secrets:
        engine.endorse_value(secret, _GA)
    assert engine.confidentiality_floor_active() is True


# ── P6: executed equals checked ───────────────────────────────────────────────

def test_no_kernel_gate_produces_a_transform_decision() -> None:
    """The guard in the loop refuses a TRANSFORM payload; this pins the premise
    that nothing produces one, so the guard costs no behaviour today.

    If this fails, a producer appeared and the loop now refuses it: re-gate the
    transformed payload in that producer and approve the re-gated intent. Do not
    delete the guard — it is the executed-equals-checked identity.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "axor_core"
    producers = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "PolicyDecisionKind.TRANSFORM" in line and "==" not in line
    ]
    assert not producers, f"a TRANSFORM producer appeared: {producers}"
