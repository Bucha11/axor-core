"""
PR E of the authority/plan split: dynamic replanning within ResourceBudget.

Invariants pinned here:
  * a plan expansion changes envelope.plan, never envelope.authority or
    capabilities (widening a plan is not a capability escalation);
  * ResourceBudget ceilings clamp and are trace-visible;
  * malformed/negative requests fail closed (ignored, never crash);
  * the expansion counter enforces max_plan_expansions.
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.planning import ExecutionPlan, ResourceBudget
from axor_core.contracts.policy import ContextMode
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.planning.composer import PlanComposer


def _loop(budget=None, trace=None):
    return IntentLoop(
        capability_executor=CapabilityExecutor(),
        trace_events=trace if trace is not None else [],
        resource_budget=budget or ResourceBudget(),
    )


def _expansion_event(**args) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "request_plan_expansion", "tool_use_id": "pe-1",
                 "args": {"reason": "need more context", **args}},
        node_id="n1",
    )


def test_expansion_changes_plan_not_authority(make_envelope):
    env = make_envelope()
    authority_before = env.authority
    caps_before = env.capabilities
    trace: list = []
    loop = _loop(trace=trace)

    result = loop._handle_plan_expansion(
        _expansion_event(requested_context_mode="broad"), env
    )
    assert result["granted"] is True
    assert env.plan.context_mode == ContextMode.BROAD
    assert env.plan.source == "plan_expansion"
    assert env.authority == authority_before
    assert env.capabilities == caps_before
    kinds = [e.kind for e in trace]
    assert TraceEventKind.EXECUTION_PLAN_CHANGED in kinds


def test_budget_clamps_reservation_and_reports_constraint(make_envelope):
    env = make_envelope()
    trace: list = []
    loop = _loop(budget=ResourceBudget(max_token_reservation=1000), trace=trace)

    result = loop._handle_plan_expansion(
        _expansion_event(additional_token_reservation=5000), env
    )
    assert result["granted"] is True
    assert env.plan.token_reservation == 1000
    assert any(
        e.kind == TraceEventKind.PLAN_CONSTRAINED_BY_BUDGET for e in trace
    )


def test_max_plan_expansions_enforced(make_envelope):
    env = make_envelope()
    loop = _loop(budget=ResourceBudget(max_plan_expansions=1))

    first = loop._handle_plan_expansion(
        _expansion_event(requested_context_mode="broad"), env
    )
    assert first["granted"] is True
    second = loop._handle_plan_expansion(
        _expansion_event(requested_context_mode="minimal"), env
    )
    assert second["granted"] is False
    assert "exhausted" in second["reason"]


@pytest.mark.parametrize("bad_args", [
    {"requested_child_depth": -5},
    {"requested_child_depth": "lots"},
    {"additional_token_reservation": -1},
    {"requested_context_mode": "omniscient"},
])
def test_malformed_requests_fail_closed(make_envelope, bad_args):
    env = make_envelope()
    plan_before = env.plan
    loop = _loop()
    result = loop._handle_plan_expansion(_expansion_event(**bad_args), env)
    assert result["granted"] is True          # advisory: nothing to grant
    assert env.plan == plan_before            # but nothing changed either


def test_composer_cannot_smuggle_authority_fields():
    """The expansion surface only ever touches planning fields."""
    plan = ExecutionPlan()
    new_plan, _ = PlanComposer().expand(
        plan, ResourceBudget(), requested_context_mode="broad",
    )
    changed = {
        f.name for f in dataclasses.fields(ExecutionPlan)
        if getattr(new_plan, f.name) != getattr(plan, f.name)
    }
    assert changed <= {"context_mode", "source", "name"}
