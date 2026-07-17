"""Plan presets — named by execution shape, never by task type."""
from __future__ import annotations

from axor_core.contracts.planning import (
    NEUTRAL_PLAN,
    DecompositionPreference,
    ExecutionPlan,
    RetrievalBreadth,
)
from axor_core.contracts.policy import CompressionMode, ContextMode


def neutral() -> ExecutionPlan:
    return NEUTRAL_PLAN


def local() -> ExecutionPlan:
    return ExecutionPlan(
        name="local",
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.BALANCED,
        retrieval_breadth=RetrievalBreadth.NARROW,
        decomposition=DecompositionPreference.AVOID,
        suggested_child_depth=0,
        child_context_fraction=0.0,
        source="preset",
    )


def component() -> ExecutionPlan:
    return ExecutionPlan(
        name="component",
        context_mode=ContextMode.MODERATE,
        compression_mode=CompressionMode.BALANCED,
        retrieval_breadth=RetrievalBreadth.MODERATE,
        decomposition=DecompositionPreference.ALLOW,
        suggested_child_depth=1,
        child_context_fraction=0.3,
        source="preset",
    )


def repository() -> ExecutionPlan:
    return ExecutionPlan(
        name="repository",
        context_mode=ContextMode.BROAD,
        compression_mode=CompressionMode.LIGHT,
        retrieval_breadth=RetrievalBreadth.BROAD,
        decomposition=DecompositionPreference.PREFER,
        suggested_child_depth=2,
        child_context_fraction=0.5,
        source="preset",
    )
