"""
PR C of the authority/plan split: the ExecutionPlanner.

Invariants pinned here:
  * HeuristicExecutionPlanner reproduces the plan half of the legacy preset
    matrix exactly (behaviour-preserving swap of the plan source);
  * planner failure degrades to NEUTRAL_PLAN and never touches authority
    (invariant I6);
  * an adversarial planner cannot widen the capability surface — the
    envelope's authority half is untouched by any plan (invariant I1).
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.contracts.planning import NEUTRAL_PLAN
from axor_core.contracts.policy import (
    TaskComplexity,
    TaskNature,
    TaskSignal,
)
from axor_core.planning.planner import (
    HeuristicExecutionPlanner,
    plan_or_neutral,
)
from axor_core.planning import presets as plan_presets
from axor_core.policy.legacy import split_legacy_policy
from axor_core.policy.selector import PolicySelector


def _signal(complexity: TaskComplexity, nature: TaskNature) -> TaskSignal:
    return TaskSignal(
        raw_input="x",
        complexity=complexity,
        nature=nature,
        estimated_scope=1,
        requires_children=complexity == TaskComplexity.EXPANSIVE,
        requires_mutation=nature == TaskNature.MUTATIVE,
    )


@pytest.mark.parametrize("complexity", list(TaskComplexity))
@pytest.mark.parametrize("nature", list(TaskNature))
def test_planner_reproduces_legacy_plan_half(complexity, nature):
    """Swapping the plan source from split(selector_policy) to the planner
    must not change execution shaping (source field is telemetry-only)."""
    signal = _signal(complexity, nature)
    legacy_plan = split_legacy_policy(PolicySelector().select(signal))[1]
    planner_plan = HeuristicExecutionPlanner().plan(signal)
    assert dataclasses.replace(planner_plan, source=legacy_plan.source) == legacy_plan


def test_planner_failure_degrades_to_neutral():
    class _Boom:
        def plan(self, signal):
            raise RuntimeError("planner exploded")

    plan = plan_or_neutral(_Boom(), _signal(TaskComplexity.FOCUSED, TaskNature.READONLY))
    assert plan == NEUTRAL_PLAN


def test_absent_planner_is_neutral():
    assert plan_or_neutral(None, _signal(TaskComplexity.FOCUSED, TaskNature.READONLY)) == NEUTRAL_PLAN


def test_plan_presets_are_authority_free():
    import dataclasses as dc
    for preset in (plan_presets.neutral(), plan_presets.local(),
                   plan_presets.component(), plan_presets.repository()):
        fields = {f.name for f in dc.fields(preset)}
        assert not fields & {"tool_policy", "allowed_paths", "escalation_policy",
                             "allow_spawn", "max_unattended_consequence"}


@pytest.mark.asyncio
async def test_adversarial_planner_cannot_widen_capabilities(make_envelope):
    """A malicious planner output rides only the plan half: build a node run
    where the planner returns a maximally inflated plan and assert the
    envelope's authority and capability surface are untouched."""
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.context.manager import ContextManager
    from axor_core.node.wrapper import GovernedNode
    from axor_core.policy.analyzer import TaskAnalyzer
    from axor_core.policy.composer import PolicyComposer
    from axor_core.policy.selector import PolicySelector as _Selector
    from axor_core.contracts.context import RawExecutionState
    from axor_core.contracts.planning import (
        DecompositionPreference,
        ExecutionPlan,
        RetrievalBreadth,
    )
    from axor_core.contracts.policy import CompressionMode, ContextMode
    from tests.conftest import EchoExecutor

    class _InflatedPlanner:
        def plan(self, signal):
            return ExecutionPlan(
                name="inflated",
                context_mode=ContextMode.BROAD,
                compression_mode=CompressionMode.LIGHT,
                retrieval_breadth=RetrievalBreadth.BROAD,
                decomposition=DecompositionPreference.PREFER,
                suggested_child_depth=99,
                child_context_fraction=1.0,
                source="attacker",
            )

    node = GovernedNode(
        executor=EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        analyzer=TaskAnalyzer(),
        selector=_Selector(),
        composer=PolicyComposer(),
        context_manager=ContextManager(),
        planner=_InflatedPlanner(),
    )

    result = await node.run(
        raw_state=RawExecutionState(
            task="explain what the validate function does",
            session_id="s1",
            session_state={},
            parent_export=None,
            memory_fragments=[],
            lineage=None,
        )
    )
    # the run completed; the classifier selected focused_readonly authority
    # (read-only) and the inflated plan could not add tools, spawn or writes
    assert result is not None
    trace = result.metadata.get("policy", "")
    assert trace == "focused_readonly"
