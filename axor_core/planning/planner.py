"""TaskSignal → ExecutionPlan.

The planner replaces the plan-producing half of the legacy PolicySelector.
It reads the (untrusted, advisory) task classification and recommends an
execution shape — context breadth, compression, decomposition. It grants
nothing: no branch of any planner may name a tool, a path, a consequence
class or an escalation rule (those are AuthorityPolicy, operator-defined).

HeuristicExecutionPlanner reproduces the plan half of the legacy preset
matrix exactly (complexity × nature → the same context/compression/
fraction/depth the selector presets carried), so swapping the plan source
from split(selector_policy) to the planner is behaviour-preserving. The
scope-only mapping from the RFC (TaskScope local/component/repository)
lands together with the TaskSignal reshape in a later PR.
"""
from __future__ import annotations

import logging
from typing import Protocol

from axor_core.contracts.planning import (
    NEUTRAL_PLAN,
    DecompositionPreference,
    ExecutionPlan,
    RetrievalBreadth,
)
from axor_core.contracts.policy import (
    CompressionMode,
    ContextMode,
    TaskComplexity,
    TaskNature,
    TaskSignal,
)

log = logging.getLogger("axor.planning")


class ExecutionPlanner(Protocol):
    def plan(self, signal: TaskSignal) -> ExecutionPlan:  # pragma: no cover
        ...


def plan_or_neutral(planner: "ExecutionPlanner | None", signal: TaskSignal) -> ExecutionPlan:
    """Invariant I6: planner failure is operational only. An exception (or an
    absent planner) degrades to NEUTRAL_PLAN — it never stops governed
    execution, never changes authority, never produces a capability denial."""
    if planner is None:
        return NEUTRAL_PLAN
    try:
        return planner.plan(signal)
    except Exception:
        log.warning("execution planner raised; using neutral plan", exc_info=True)
        return NEUTRAL_PLAN


class HeuristicExecutionPlanner:
    """Plan-half of the legacy preset matrix. Advisory only."""

    def plan(self, signal: TaskSignal) -> ExecutionPlan:
        complexity, nature = signal.complexity, signal.nature
        source = "task_classifier"

        if complexity == TaskComplexity.EXPANSIVE:
            return ExecutionPlan(
                name="expansive",
                context_mode=ContextMode.BROAD,
                compression_mode=CompressionMode.LIGHT,
                retrieval_breadth=RetrievalBreadth.BROAD,
                decomposition=DecompositionPreference.PREFER,
                suggested_child_depth=3,
                child_context_fraction=0.6,
                expected_scope=999,
                source=source,
            )

        if complexity == TaskComplexity.MODERATE:
            fraction = {
                TaskNature.READONLY: 0.0,
                TaskNature.GENERATIVE: 0.3,
                TaskNature.MUTATIVE: 0.4,
            }[nature]
            decomposition = (
                DecompositionPreference.AVOID
                if nature == TaskNature.READONLY
                else DecompositionPreference.ALLOW
            )
            return ExecutionPlan(
                name=f"moderate_{nature.value}",
                context_mode=ContextMode.MODERATE,
                compression_mode=CompressionMode.BALANCED,
                retrieval_breadth=RetrievalBreadth.MODERATE,
                decomposition=decomposition,
                suggested_child_depth=0 if nature == TaskNature.READONLY else 1,
                child_context_fraction=fraction,
                expected_scope=5,
                source=source,
            )

        # FOCUSED
        compression = (
            CompressionMode.AGGRESSIVE
            if nature == TaskNature.READONLY
            else CompressionMode.BALANCED
        )
        return ExecutionPlan(
            name=f"focused_{nature.value}",
            context_mode=ContextMode.MINIMAL,
            compression_mode=compression,
            retrieval_breadth=RetrievalBreadth.NARROW,
            decomposition=DecompositionPreference.AVOID,
            suggested_child_depth=0,
            child_context_fraction=0.0,
            expected_scope=1,
            source=source,
        )
