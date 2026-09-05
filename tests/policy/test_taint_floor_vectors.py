"""The shared taint-floor vectors, proved against the kernel that owns the gate.

`axor_core/vectors/taint_floor.json` is the statement every consumer checks
itself against. Two of them used to reimplement this gate instead — the Lab's
reference kernel and the Control Plane's export converter — which is exactly the
duplication Rule 0 forbids and exactly the thing that drifts silently. The file
is shipped inside this package so a consumer reads it by import, not by reaching
into another product's checkout.

This test is the producing half: the real `taint_gate` must decide every vector
the way the file says. The consuming halves live in the Control Plane and the
Lab, over the same file.
"""

from __future__ import annotations

import pytest

from axor_core.policy.from_record import (
    causal_root_from_record,
    normalized_from_record,
)
from axor_core.policy.gates import taint_gate
from axor_core.vectors import TAINT_FLOOR

CASES = TAINT_FLOOR()["vectors"]


@pytest.mark.parametrize("vec", CASES, ids=[v["name"] for v in CASES])
def test_the_kernel_decides_every_vector_as_written(vec: dict) -> None:
    decision = taint_gate(
        vec["tool"],
        normalized_from_record(vec["tool"], vec["normalized"]),
        causal_root_from_record(vec["root"]),
        floor_active=bool(vec.get("floor_active")),
        egress_sinks=frozenset(vec.get("egress_sinks") or ()),
        integrity_superseded=bool(vec.get("integrity_superseded")),
    )
    if vec["expect"] == "allow":
        assert decision is None, f"expected ALLOW, got DENY: {decision}"
        return
    assert decision is not None, "expected DENY, the gate allowed"
    assert decision.category == vec["category"]
    # The axis is load-bearing: integrity and confidentiality deny for different
    # reasons and a caller may only lift one of them.
    assert vec["axis"] in decision.reason


def test_the_vector_file_covers_both_axes_and_both_verdicts() -> None:
    """A vector file that only ever expects one answer proves nothing."""
    verdicts = {v["expect"] for v in CASES}
    axes = {v.get("axis") for v in CASES if v["expect"] == "deny"}
    assert verdicts == {"allow", "deny"}
    assert axes == {"integrity", "confidentiality"}
