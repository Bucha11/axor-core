"""Tests for BudgetPolicyEngine 0.5.0 extensions: suggest_tier_shift, on_threshold_crossed."""
from __future__ import annotations

import pytest

from axor_core.budget import BudgetEstimator, BudgetPolicyEngine, BudgetTracker


@pytest.fixture()
def tracker():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    return tr


@pytest.fixture()
def engine(tracker):
    return BudgetPolicyEngine(
        tracker=tracker,
        estimator=BudgetEstimator(),
        soft_limit=100_000,
    )


# ── suggest_tier_shift ─────────────────────────────────────────────────────

def test_suggest_tier_shift_zero_when_no_limit():
    tr = BudgetTracker()
    engine = BudgetPolicyEngine(
        tracker=tr, estimator=BudgetEstimator(), soft_limit=None
    )
    assert engine.suggest_tier_shift() == 0


def test_suggest_tier_shift_plus1_when_very_low_spend(tracker, engine):
    # No tokens recorded — ratio=0, which is < compress*0.5 (0.30) → return 1
    assert engine.suggest_tier_shift() == 1


def test_suggest_tier_shift_zero_at_moderate_spend(tracker, engine):
    tracker.record("n1", input_tokens=40_000, output_tokens=0)
    # ratio = 0.40: not < 0.30, not >= 0.60 → 0
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
#
# _maybe_notify_threshold is the only path that fires callbacks.
# It is called by on_intent_arrived(); we invoke it directly here
# to keep tests self-contained (avoids building a full ExecutionEnvelope).

def test_on_threshold_crossed_fires_at_compress(engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append((name, ratio)))
    # ratio 0.65 ≥ compress=0.60
    engine._maybe_notify_threshold(0.65)
    assert len(fired) == 1
    assert fired[0][0] == "compress"
    assert fired[0][1] >= 0.60


def test_on_threshold_crossed_fires_once_per_bucket(engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    engine._maybe_notify_threshold(0.65)  # first crossing → fires
    engine._maybe_notify_threshold(0.65)  # same bucket → deduped
    assert fired.count("compress") == 1


def test_on_threshold_crossed_unsubscribe(engine):
    fired = []
    unsub = engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    unsub()
    engine._maybe_notify_threshold(0.65)
    assert fired == []


def test_on_threshold_crossed_deny_child(engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    # ratio 0.82 ≥ deny_child=0.80
    engine._maybe_notify_threshold(0.82)
    assert "deny_child" in fired


def test_on_threshold_crossed_restrict_export(engine):
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    engine._maybe_notify_threshold(0.91)
    assert "restrict_export" in fired


def test_on_threshold_crossed_escalates_to_highest_bucket(engine):
    """Jumping straight to deny_child fires deny_child, not compress."""
    fired = []
    engine.on_threshold_crossed(lambda name, ratio: fired.append(name))
    engine._maybe_notify_threshold(0.82)  # crosses deny_child directly
    assert fired == ["deny_child"]


def test_callback_exception_does_not_break_other_callbacks(engine):
    good = []
    def bad_cb(name, ratio):
        raise RuntimeError("boom")
    engine.on_threshold_crossed(bad_cb)
    engine.on_threshold_crossed(lambda n, r: good.append(n))
    engine._maybe_notify_threshold(0.65)  # bad_cb raises, good_cb must still run
    assert "compress" in good
