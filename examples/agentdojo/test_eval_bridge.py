"""Unit tests for the pure logic of eval_bridge (no model calls)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import eval_bridge as eb  # noqa: E402
from axor_eval.contracts import DeviationType  # noqa: E402


# ── property attribution ────────────────────────────────────────────────────────

def test_property_known_data_flow():
    assert eb.property_for("banking", "injection_task_1") == eb.PROPERTY_DATA_FLOW
    assert eb.property_for("slack", "injection_task_4") == eb.PROPERTY_DATA_FLOW


def test_property_known_instruction():
    assert eb.property_for("slack", "injection_task_1") == eb.PROPERTY_INSTRUCTION


def test_property_unknown_defaults():
    assert eb.property_for("banking", "injection_task_99") == eb.DEFAULT_PROPERTY


# ── evidence cases ───────────────────────────────────────────────────────────────

def _outcome(governed, succeeded, it="injection_task_1", util=True):
    return eb.TaskOutcome(
        suite="banking", user_task="user_task_0", injection_task=it,
        governed=governed, utility=util, attack_succeeded=succeeded,
    )


def test_benign_outcome_no_case():
    o = eb.TaskOutcome(suite="banking", user_task="user_task_0", injection_task=None,
                       governed=True, utility=True, attack_succeeded=False)
    assert eb.build_evidence_case(o) is None


def test_attack_succeeded_is_deterministic_violation():
    case = eb.build_evidence_case(_outcome(governed=False, succeeded=True))
    assert case is not None
    assert case.deviation == DeviationType.DIRECT_POLICY_VIOLATION
    assert case.verdict_source == "deterministic"
    assert case.confidence == 1.0
    assert case.observed_reality["property"] == eb.PROPERTY_DATA_FLOW


def test_defended_pair_is_clean_case():
    case = eb.build_evidence_case(_outcome(governed=True, succeeded=False))
    assert case is not None
    assert case.deviation is None  # governance held → evidence of a clean run


# ── Wilson CI ────────────────────────────────────────────────────────────────────

def test_wilson_zero_successes_has_zero_lower_bound():
    lo, hi = eb.wilson_ci(0, 48)
    assert lo == 0.0
    assert 0.0 < hi < 0.12  # 0/48 → tight upper bound near 7.5%


def test_wilson_midrange():
    lo, hi = eb.wilson_ci(29, 48)  # ~60.4%
    assert lo < 0.604 < hi
    assert 0.45 < lo and hi < 0.75


def test_wilson_empty():
    assert eb.wilson_ci(0, 0) == (0.0, 0.0)


# ── aggregation ──────────────────────────────────────────────────────────────────

def test_summarize_splits_by_condition_and_property():
    outcomes = []
    # undefended: data-flow attack succeeds 3/4, instruction 1/2
    outcomes += [_outcome(False, True), _outcome(False, True), _outcome(False, True), _outcome(False, False)]
    outcomes += [
        eb.TaskOutcome("slack", "user_task_3", "injection_task_1", False, True, True),
        eb.TaskOutcome("slack", "user_task_3", "injection_task_1", False, True, False),
    ]
    # governed: everything blocked
    outcomes += [_outcome(True, False) for _ in range(4)]
    outcomes += [eb.TaskOutcome("slack", "user_task_3", "injection_task_1", True, True, False)]

    s = eb.summarize(outcomes)
    assert s["undefended"]["by_property"][eb.PROPERTY_DATA_FLOW]["attacks_succeeded"] == 3
    assert s["undefended"]["by_property"][eb.PROPERTY_DATA_FLOW]["n"] == 4
    assert s["undefended"]["by_property"][eb.PROPERTY_INSTRUCTION]["attacks_succeeded"] == 1
    assert s["governed"]["overall"]["asr"] == 0.0
    assert s["governed"]["overall"]["attacks_succeeded"] == 0
    # CI present
    assert len(s["undefended"]["overall"]["asr_ci95"]) == 2
