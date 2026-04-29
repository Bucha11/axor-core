"""Tests for BudgetTracker cache accounting and BudgetThresholds validation."""
from __future__ import annotations

import pytest

from axor_core.budget import (
    BudgetEstimator,
    BudgetPolicyEngine,
    BudgetThresholds,
    BudgetTracker,
    OptimizationAction,
    TokenCostRates,
)
from axor_core.contracts.policy import ExecutionPolicy


class TestCacheAccounting:

    def test_record_includes_cache_fields(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record(
            node_id="n1",
            input_tokens=200,
            output_tokens=50,
            cache_creation_input_tokens=1000,
            cache_read_input_tokens=8000,
        )
        snap = tr.snapshot()
        assert snap["n1"].cache_creation_input_tokens == 1000
        assert snap["n1"].cache_read_input_tokens == 8000
        assert snap["n1"].total_input_tokens == 9200
        assert snap["n1"].total == 9250

    def test_total_tokens_includes_prompt_cache_counters(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record(
            node_id="n1",
            input_tokens=10_000,
            output_tokens=500,
            cache_creation_input_tokens=25_000,
            cache_read_input_tokens=100_000,
        )

        assert tr.total_tokens() == 135_500
        assert tr.total_billable_tokens() == 135_500

    def test_estimated_cost_uses_prompt_cache_multipliers(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record(
            node_id="n1",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
        )
        rates = TokenCostRates(input_per_m=3.0, output_per_m=15.0)

        summary = tr.cost_summary(rates)

        assert summary["input_cost"] == 3.0
        assert summary["cache_creation_cost"] == 3.75
        assert summary["cache_read_cost"] == pytest.approx(0.3)
        assert summary["output_cost"] == 15.0
        assert summary["total_cost"] == pytest.approx(22.05)
        assert tr.estimated_cost(rates) == pytest.approx(22.05)

    def test_estimated_cost_accepts_explicit_cache_rates(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record(
            node_id="n1",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
        )
        rates = TokenCostRates(
            input_per_m=3.0,
            output_per_m=15.0,
            cache_creation_input_per_m=4.0,
            cache_read_input_per_m=0.5,
        )

        summary = tr.cost_summary(rates)

        assert summary["cache_creation_cost"] == 4.0
        assert summary["cache_read_cost"] == 0.5
        assert summary["total_cost"] == 22.5

    def test_policy_engine_ratio_counts_prompt_cache_volume(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record(
            node_id="n1",
            input_tokens=10_000,
            output_tokens=0,
            cache_read_input_tokens=100_000,
        )
        engine = BudgetPolicyEngine(
            tracker=tr,
            estimator=BudgetEstimator(),
            soft_limit=100_000,
        )

        decision = engine.on_result_arrived(
            node_id="n1",
            result_token_estimate=0,
            policy=ExecutionPolicy(),
        )

        assert decision.action is OptimizationAction.RESTRICT_EXPORT

    def test_custom_thresholds_trigger_before_defaults(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record("n1", input_tokens=55_000, output_tokens=0)

        default_engine = BudgetPolicyEngine(
            tracker=tr,
            estimator=BudgetEstimator(),
            soft_limit=100_000,
        )
        custom_engine = BudgetPolicyEngine(
            tracker=tr,
            estimator=BudgetEstimator(),
            soft_limit=100_000,
            thresholds=BudgetThresholds(
                compress=0.50,
                deny_child=0.70,
                restrict_export=0.85,
                hard_stop=0.95,
            ),
        )

        default_decision = default_engine.on_result_arrived(
            node_id="n1",
            result_token_estimate=0,
            policy=ExecutionPolicy(),
        )
        custom_decision = custom_engine.on_result_arrived(
            node_id="n1",
            result_token_estimate=0,
            policy=ExecutionPolicy(),
        )

        assert default_decision.action is OptimizationAction.PROCEED
        assert custom_decision.action is OptimizationAction.COMPRESS_CONTEXT

    def test_cache_fields_accumulate_across_calls(self):
        tr = BudgetTracker()
        tr.register_node("n1", parent_id=None, depth=0)
        tr.record("n1", input_tokens=100, output_tokens=20,
                  cache_creation_input_tokens=500, cache_read_input_tokens=0)
        tr.record("n1", input_tokens=100, output_tokens=20,
                  cache_creation_input_tokens=0, cache_read_input_tokens=2000)
        snap = tr.snapshot()
        assert snap["n1"].cache_creation_input_tokens == 500
        assert snap["n1"].cache_read_input_tokens == 2000

    def test_cache_summary_aggregates_across_nodes(self):
        tr = BudgetTracker()
        tr.register_node("a", None, 0)
        tr.register_node("b", "a", 1)
        tr.record("a", 100, 50, cache_creation_input_tokens=1000, cache_read_input_tokens=0)
        tr.record("b", 200, 30, cache_creation_input_tokens=0, cache_read_input_tokens=4000)
        s = tr.cache_summary()
        assert s["input_tokens"] == 300
        assert s["output_tokens"] == 80
        assert s["cache_creation_input_tokens"] == 1000
        assert s["cache_read_input_tokens"] == 4000
        assert s["total_input_tokens"] == 5300
        assert s["total_tokens"] == 5380
        # hit_rate = 4000 / (300 + 4000)
        assert abs(s["hit_rate"] - 4000 / 4300) < 1e-9

    def test_cache_summary_empty_session(self):
        tr = BudgetTracker()
        s = tr.cache_summary()
        assert s["hit_rate"] == 0.0
        assert s["cache_read_input_tokens"] == 0

    def test_node_cache_hit_rate(self):
        tr = BudgetTracker()
        tr.register_node("n1", None, 0)
        tr.record("n1", input_tokens=200, output_tokens=10,
                  cache_read_input_tokens=800)
        node = tr.snapshot()["n1"]
        # 800 / (200 + 800) = 0.8
        assert abs(node.cache_hit_rate - 0.8) < 1e-9

    def test_node_cache_hit_rate_zero_when_no_cache(self):
        tr = BudgetTracker()
        tr.register_node("n1", None, 0)
        tr.record("n1", input_tokens=100, output_tokens=10)
        assert tr.snapshot()["n1"].cache_hit_rate == 0.0


class TestBudgetThresholds:

    def test_defaults_validate(self):
        t = BudgetThresholds()
        assert t.compress < t.deny_child < t.restrict_export < t.hard_stop

    def test_custom_valid_thresholds(self):
        t = BudgetThresholds(compress=0.5, deny_child=0.7, restrict_export=0.85, hard_stop=0.95)
        assert t.compress == 0.5

    def test_rejects_out_of_order(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            BudgetThresholds(compress=0.8, deny_child=0.7, restrict_export=0.9, hard_stop=0.95)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            BudgetThresholds(compress=0.0, deny_child=0.5, restrict_export=0.7, hard_stop=0.9)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match="must be in"):
            BudgetThresholds(compress=0.5, deny_child=0.7, restrict_export=0.85, hard_stop=1.5)

    def test_rejects_equal_thresholds(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            BudgetThresholds(compress=0.6, deny_child=0.6, restrict_export=0.9, hard_stop=0.95)
