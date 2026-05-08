"""Tests for BudgetPolicyEngine 0.5.0 extensions: suggest_tier_shift, on_threshold_crossed."""
from __future__ import annotations

import pytest

from axor_core.budget import BudgetEstimator, BudgetPolicyEngine, BudgetTracker


@pytest.fixture()
def tracker():
    return BudgetTracker()


@pytest.fixture()
def engine(tracker):
    tr = tracker
    tr.register_node("n1", None, 0)
    return BudgetPolicyEngine(
        tracker=tr,
        estimator=BudgetEstimator(),
        soft_limit=100_000,
    )


# ── suggest_tier_shift ────────────────────────────────────────────────────────

def test_suggest_tier_shift_zero_when_no_limit(tracker):
    engine = BudgetPolicyEngine(
        tracker=tracker, estimator=BudgetEstimator(), soft_limit=None
    )
    assert engine.suggest_tier_shift() == 0


def test_suggest_tier_shift_plus1_when_very_low_spend(tracker, engine):
    # No tokens recorded — ratio=0, which is < compress*0.5 (0.30) → return 1
    assert engine.suggest_tier_shift() == 1


def test_suggest_tier_shift_zero_at_moderate_spend(tracker, engine):
    tracker.record("n1", input_tokens=40_000, output_tokens=0)
    # ratio = 0.40, between compress*0.5=0.30 and compress=0.60 → 0
    assert engine.suggest_tier_shift() == 0


def test_suggest_tier_shift_minus1_at_compress_threshold(tracker, engine):
    tracker.record("n1", input_tokens=65_000, output_tokens=0)
    # ratio = 0.65 >= compress=0.60 → -1
    assert engine.suggest_tier_shift() == -1


def test_suggest_tier_shift_minus1_at_restrict_export_threshold(tracker, engine):
    tracker.record("n1", input_tokens=92_000, output_tokens=0)
    # ratio = 0.92 >= restrict_export=0.90 → -1
    assert engine.suggest_tier_shift() == -1


# ── on_threshold_crossed ────────────────────────────────────────────────────

def test_on_threshold_crossed_fires_at_compress(tracker, engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append((name, ratio)))
    tracker.record("n1", input_tokens=62_000, output_tokens=0)

    # Trigger the check by calling on_intent_arrived
    from axor_core.contracts.policy import ExecutionPolicy
    env = pytest.importorskip("axor_core.contracts.envelope")
    # Use policy engine's internal check via on_result_arrived
    from axor_core.budget.policy_engine import BudgetPolicyEngine
    engine.on_result_arrived("n1", 0, ExecutionPolicy())

    assert len(fired) == 1
    assert fired[0][0] == "compress"
    assert fired[0][1] >= 0.60


def test_on_threshold_crossed_fires_once_per_bucket(tracker, engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    from axor_core.contracts.policy import ExecutionPolicy

    tracker.record("n1", input_tokens=62_000, output_tokens=0)
    engine.on_result_arrived("n1", 0, ExecutionPolicy())
    # Second call at same level should NOT re-fire
    engine.on_result_arrived("n1", 0, ExecutionPolicy())
    assert fired.count("compress") == 1


def test_on_threshold_crossed_unsubscribe(tracker, engine):
    fired = []
    unsub = engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    unsub()
    from axor_core.contracts.policy import ExecutionPolicy
    tracker.record("n1", input_tokens=62_000, output_tokens=0)
    engine.on_result_arrived("n1", 0, ExecutionPolicy())
    assert fired == []


def test_on_threshold_crossed_deny_child(tracker, engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    from axor_core.contracts.policy import ExecutionPolicy
    tracker.record("n1", input_tokens=82_000, output_tokens=0)
    engine.on_result_arrived("n1", 0, ExecutionPolicy())
    assert "deny_child" in fired
