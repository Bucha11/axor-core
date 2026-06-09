"""The four previously-unconnected in-process features:
carrier/imperative gate (TM1), endorsement (TM4), live density (TM3.3),
value-policy predicates (TM3.1).

NOTE: committed as a checkpoint on the 7a5ab03 base after an environment reset
lost the later cleanup commits (drop-ML, session-taint removal, ValueProvenance,
profiles, children). These features are correct and independent; they will be
re-based onto the cleaned architecture during recovery.
"""

from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintSource
from axor_core.errors.exceptions import TaintClearanceError
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine


# ── value-policy predicates (TM3.1) ──────────────────────────────────────────

def test_value_policy_numeric_range_and_enum():
    from axor_core.policy.value_policy import numeric_range, enum, check_value_policies
    pols = {
        "transfer": [numeric_range("amount", 0, 1000)],
        "set_mode": [enum("mode", {"read", "write"})],
    }
    assert check_value_policies("transfer", {"amount": 10}, pols) is None
    assert check_value_policies("transfer", {"amount": 999999}, pols) is not None
    assert check_value_policies("transfer", {"amount": "10"}, pols) is not None  # not numeric
    assert check_value_policies("set_mode", {"mode": "read"}, pols) is None
    assert check_value_policies("set_mode", {"mode": "rm -rf"}, pols) is not None
    # absent arg / absent policy → no constraint
    assert check_value_policies("transfer", {}, pols) is None
    assert check_value_policies("other", {"amount": 9}, pols) is None


# ── endorsement (TM4) — governed per-value release ───────────────────────────

SECRET = "SECRET_TOKEN_abcdef123456"


def test_endorse_releases_one_value_under_governance():
    eng = TaintEngine()
    eng.register_value(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    assert eng.derive_value(SECRET).is_tainted is True

    removed = eng.endorse_value(SECRET, "operator", "human_operator", "reviewed")
    assert removed >= 1
    assert eng.derive_value(SECRET).is_tainted is False  # released


def test_endorse_rejects_invalid_authority():
    eng = TaintEngine()
    eng.register_value(SECRET, CausalRoot.external_read(TaintSource.WEB))
    with pytest.raises(TaintClearanceError):
        eng.endorse_value(SECRET, "", "worker", "")  # not a governance authority
    assert eng.derive_value(SECRET).is_tainted is True  # still tainted


def test_endorse_is_per_value_not_whole_ledger():
    eng = TaintEngine()
    other = "OTHER_TAINTED_fragment_xyz"
    eng.register_value(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    eng.register_value(other, CausalRoot.external_read(TaintSource.WEB))
    eng.endorse_value(SECRET, "op", "human_operator", "reviewed")
    assert eng.derive_value(SECRET).is_tainted is False   # released
    assert eng.derive_value(other).is_tainted is True     # untouched


# ── live density metric (TM3.3) ──────────────────────────────────────────────

def test_density_metric_from_sink_events():
    from axor_core.contracts.trace import SinkDensityEvent, TraceEventKind
    from axor_core.trace.metrics import GovernanceMetrics
    evs = [
        SinkDensityEvent(kind=TraceEventKind.SINK_DENSITY, node_id="n", sequence=i,
                         operation="bash", tainted=t)
        for i, t in enumerate([True, False, True, False])
    ]
    m = GovernanceMetrics.from_events(evs)
    assert m.sink_firings_by_op["bash"] == 4
    assert m.sink_tainted_by_op["bash"] == 2
    assert m.density == 0.5
