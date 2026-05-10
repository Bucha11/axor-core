from __future__ import annotations

from typing import Any

from axor_core.capability.executor import CapabilityExecutor
from axor_core.capability.phantom import PhantomCapabilityExecutor
from axor_core.capability.enforced import EnforcedCapabilityExecutor
from axor_core.contracts.context import ContextFragment
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.plan import AgentPlan
from axor_core.contracts.policy import SignalClassifier
from axor_core.contracts.result import ExecutionResult
from axor_core.contracts.trace import TraceConfig
from axor_core.worker.session import GovernedSession
from axor_core.budget import BudgetThresholds, TokenCostRates
from axor_core.extensions.registry import ExtensionRegistry


class TwoPhaseSession:
    """
    Governed execution with mandatory human approval between plan and action.

    Phase 1 — plan():
        Runs the agent with PhantomCapabilityExecutor. Mutation tools (write,
        bash, delete) are intercepted and recorded as PlanSteps but never touch
        disk. Read tools execute normally. The agent produces a coherent plan
        including multi-step sequences where step N depends on step N-1 output.

    Approval gap:
        .plan() returns an AgentPlan. The caller decides which steps to approve
        (.approve_all(), .approve([0, 2]), or per-step step.approved = False).
        Nothing has changed on disk yet.

    Phase 2 — execute():
        Injects the approved plan as a PINNED context fragment. Runs the agent
        again with EnforcedCapabilityExecutor: mutation tools are gated against
        the approved plan in order. The agent cannot deviate — any mutation not
        in the plan or out of sequence is denied with a structured reason.

    Usage:

        session = TwoPhaseSession(
            executor=ClaudeExecutor(),
            capability_executor=cap,
        )
        plan = await session.plan("refactor auth module")
        print(plan.display())

        # user reviews, possibly skips steps:
        plan.steps[1].approved = False

        result = await session.execute(plan)

    Constructor kwargs are forwarded to both GovernedSession instances so
    classifiers, budget limits, trace config, etc. are shared across phases.
    """

    def __init__(
        self,
        executor: Invokable,
        capability_executor: CapabilityExecutor,
        classifier: SignalClassifier | None = None,
        trace_config: TraceConfig | None = None,
        soft_token_limit: int | None = None,
        budget_thresholds: BudgetThresholds | None = None,
        token_cost_rates: TokenCostRates | None = None,
        child_executor: Invokable | None = None,
        agent_def: Any | None = None,
        memory_provider: Any | None = None,
    ) -> None:
        self._executor = executor
        self._real_cap = capability_executor
        self._session_kwargs: dict[str, Any] = {
            "classifier": classifier,
            "trace_config": trace_config,
            "soft_token_limit": soft_token_limit,
            "budget_thresholds": budget_thresholds,
            "token_cost_rates": token_cost_rates,
            "child_executor": child_executor,
            "agent_def": agent_def,
            "memory_provider": memory_provider,
        }

    async def plan(self, task: str) -> AgentPlan:
        """
        Phase 1: run the agent in phantom mode to produce a mutation plan.

        The agent reads the codebase freely. Every write/bash/delete it attempts
        is recorded as a PlanStep (and a fake success is returned so the agent
        can plan follow-up steps that depend on previous writes). Nothing on disk
        changes. Returns the collected AgentPlan for human review.
        """
        from axor_core.policy import presets

        phantom = PhantomCapabilityExecutor(self._real_cap)
        session = GovernedSession(
            executor=self._executor,
            capability_executor=phantom,
            **self._session_kwargs,
        )

        async with session:
            result = await session.run(task, policy=presets.planning())

        total = (
            result.token_usage.input_tokens
            + result.token_usage.output_tokens
            + result.token_usage.tool_tokens
        )
        plan = phantom.build_plan(
            task=task,
            phase1_tokens=total,
            node_id=result.node_id,
        )
        return plan

    async def execute(self, plan: AgentPlan) -> ExecutionResult:
        """
        Phase 2: execute the approved steps from the plan.

        Injects the plan as a PINNED context fragment so the agent knows
        exactly what to do. Mutations are gated by EnforcedCapabilityExecutor:
        each write/bash/delete must match the next approved step in order.
        Deviations are denied and surfaced as structured tool responses.
        """
        enforced = EnforcedCapabilityExecutor(self._real_cap, plan)
        session = GovernedSession(
            executor=self._executor,
            capability_executor=enforced,
            **self._session_kwargs,
        )

        # Pin the plan as an inviolable fragment — it survives all compression
        # and appears first in every ContextView built during Phase 2.
        plan_text = plan.to_context_fragment()
        session._context_manager.pin_fragment(ContextFragment(
            kind="skill",
            content=plan_text,
            token_estimate=len(plan_text) // 4,
            source="two_phase:plan",
            relevance=1.0,
            value="pinned",
        ))

        async with session:
            result = await session.run(plan.task)

        return result
