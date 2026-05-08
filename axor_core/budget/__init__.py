"""
axor_core.budget
────────────────
Token optimization. Fires on every execution event.

    BudgetTracker       — real token accounting across node tree
    BudgetEstimator     — cost estimates and slice sufficiency checks
    BudgetPolicyEngine  — real-time optimizer: minimum sufficient, not hard cap

New in 0.5.0:
    BudgetTracker.subscribe()        — live spend notifications
    BudgetTracker.tier_summary()     — per-tier token breakdown
    BudgetPolicyEngine.on_threshold_crossed() — threshold events for adaptive routing
    BudgetPolicyEngine.suggest_tier_shift()   — advisory tier direction
    NodeBudget.tier                  — optional tier label
    RecordEvent, BudgetSubscriber, Unsubscribe, ThresholdCallback
"""

from axor_core.budget.tracker import (
    BudgetTracker,
    CacheSummary,
    CostSummary,
    NodeBudget,
    TokenCostRates,
    RecordEvent,
    BudgetSubscriber,
    Unsubscribe,
)
from axor_core.budget.estimator import BudgetEstimator
from axor_core.budget.policy_engine import (
    BudgetPolicyEngine,
    BudgetThresholds,
    OptimizationDecision,
    OptimizationAction,
    ThresholdCallback,
)

__all__ = [
    "BudgetTracker",
    "CacheSummary",
    "CostSummary",
    "NodeBudget",
    "TokenCostRates",
    "RecordEvent",
    "BudgetSubscriber",
    "Unsubscribe",
    "BudgetEstimator",
    "BudgetPolicyEngine",
    "BudgetThresholds",
    "OptimizationDecision",
    "OptimizationAction",
    "ThresholdCallback",
]
