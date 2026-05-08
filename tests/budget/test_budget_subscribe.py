"""Tests for BudgetTracker.subscribe() / tier_summary() (0.5.0)."""
from __future__ import annotations

import pytest

from axor_core.budget.tracker import BudgetTracker


# ── subscribe / unsubscribe ─────────────────────────────────────────────────────

def test_subscribe_fires_on_record():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    events = []
    tr.subscribe(events.append)
    tr.record("n1", input_tokens=100, output_tokens=10)
    assert len(events) == 1
    assert events[0]["node_id"] == "n1"
    assert events[0]["input_tokens"] == 100
    assert events[0]["output_tokens"] == 10


def test_subscribe_cumulative_total_in_event():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    events = []
    tr.subscribe(events.append)
    tr.record("n1", input_tokens=50, output_tokens=20)
    assert events[0]["cumulative_total"] == 70


def test_unsubscribe_stops_callbacks():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    events = []
    unsub = tr.subscribe(events.append)
    tr.record("n1", 10, 5)
    unsub()
    tr.record("n1", 20, 10)
    assert len(events) == 1  # only first record


def test_multiple_subscribers_all_receive():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    a, b = [], []
    tr.subscribe(a.append)
    tr.subscribe(b.append)
    tr.record("n1", 100, 50)
    assert len(a) == 1
    assert len(b) == 1


def test_subscriber_exception_does_not_break_record():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    def bad_cb(event):
        raise RuntimeError("subscriber crash")
    tr.subscribe(bad_cb)
    tr.record("n1", 10, 5)  # must not raise
    assert tr.total_tokens() == 15


def test_subscribe_event_contains_tier_field():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)
    events = []
    tr.subscribe(events.append)
    tr.record("n1", 10, 5, tier="tier_0")
    assert events[0]["tier"] == "tier_0"


# ── tier_summary ────────────────────────────────────────────────────────────────

def test_tier_summary_groups_by_tier():
    tr = BudgetTracker()
    tr.register_node("n0", None, 0, tier="tier_0")
    tr.register_node("n1", "n0", 1, tier="tier_1")
    tr.record("n0", 1000, 200, tier="tier_0")
    tr.record("n1", 500, 100, tier="tier_1")
    summary = tr.tier_summary()
    assert "tier_0" in summary
    assert "tier_1" in summary
    assert summary["tier_0"]["input_tokens"] == 1000
    assert summary["tier_1"]["input_tokens"] == 500


def test_tier_summary_unlabeled_nodes_grouped_as_untiered():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0)  # no tier
    tr.record("n1", 100, 50)
    summary = tr.tier_summary()
    assert "untiered" in summary
    assert summary["untiered"]["input_tokens"] == 100


def test_tier_summary_empty_tracker():
    tr = BudgetTracker()
    assert tr.tier_summary() == {}


def test_tier_summary_hit_rate_calculated():
    tr = BudgetTracker()
    tr.register_node("n1", None, 0, tier="tier_0")
    tr.record("n1", 200, 50, cache_read_input_tokens=800)
    summary = tr.tier_summary()
    # hit_rate = 800 / (200 + 800) = 0.8
    assert abs(summary["tier_0"]["hit_rate"] - 0.8) < 1e-9
