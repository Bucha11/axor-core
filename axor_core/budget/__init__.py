"""
axor_core.budget
────────────────
Token optimization. Fires on every execution event.

    BudgetTracker       — real token accounting across node tree
    BudgetEstimator     — cost estimates and slice sufficiency checks
    BudgetPolicyEngine  — real-time optimizer: minimum sufficient, not hard cap

Principle:
    Not "deny if over limit".
    But "tighten what we can, at the moment we can".
"""

from axor_core.budget.tracker import (
    BudgetTracker,
    CacheSummary,
    CostSummary,
    NodeBudget,
    TokenCostRates,
)
from axor_core.budget.estimator import BudgetEstimator
from axor_core.budget.policy_engine import (
    BudgetPolicyEngine,
    BudgetThresholds,
    OptimizationDecision,
    OptimizationAction,
)

__all__ = [
    "BudgetTracker",
    "CacheSummary",
    "CostSummary",
    "NodeBudget",
    "TokenCostRates",
    "BudgetEstimator",
    "BudgetPolicyEngine",
    "BudgetThresholds",
    "OptimizationDecision",
    "OptimizationAction",
]
