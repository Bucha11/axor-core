"""Lease/grant consumption is deferred to final approval.

A call authorised by an escalation grant or capability lease but then DENIED by a
later data-flow gate (ssrf / taint / carrier / ...) must not burn a grant op or a
lease use. The consumption commits only once every gate has passed.
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.policy import PolicyDecisionKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.escalation import EscalationManager, _GrantedEscalation
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.engine import TaintEngine


# ── unit: the manager decides without consuming ──────────────────────────────

def test_evaluate_does_not_consume_until_commit():
    mgr = EscalationManager()
    grant = _GrantedEscalation(tool="shutdown", paths=[], ops_remaining=2)
    mgr._granted_escalations["shutdown"] = grant

    decision, pending = mgr.evaluate("shutdown", {})
    assert decision.kind == PolicyDecisionKind.APPROVE
    assert grant.ops_remaining == 2          # not yet consumed
    assert "1 ops remaining" in decision.reason  # message reflects post-commit count

    pending.commit()
    assert grant.ops_remaining == 1          # consumed exactly once on commit


def test_last_op_grant_survives_evaluation_until_commit():
    mgr = EscalationManager()
    mgr._granted_escalations["shutdown"] = _GrantedEscalation(
        tool="shutdown", paths=[], ops_remaining=1
    )
    decision, pending = mgr.evaluate("shutdown", {})
    # The grant is still present through gate evaluation (covers() stays True), so a
    # last-op call does not lose its governance gate mid-cascade.
    assert mgr.covers("shutdown") is True
    pending.commit()
    assert mgr.covers("shutdown") is False   # now exhausted and removed


# ── integration: a grant denied by a later gate burns nothing ────────────────

class _Fetch(ToolHandler):
    @property
    def name(self) -> str:
        return "fetch"

    async def execute(self, args: dict):
        return "ok"


def _loop_env(make_envelope):
    ex = CapabilityExecutor()
    ex.register(_Fetch())
    loop = IntentLoop(ex, [], taint_engine=TaintEngine())
    env = make_envelope()
    caps = dataclasses.replace(env.capabilities, allowed_tools=frozenset({"fetch"}))
    return loop, dataclasses.replace(env, capabilities=caps)


def _fetch(url: str, node_id: str) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": url}},
        node_id=node_id,
    )


@pytest.mark.asyncio
async def test_grant_op_not_consumed_when_a_later_gate_denies(make_envelope):
    loop, env = _loop_env(make_envelope)
    grant = _GrantedEscalation(tool="fetch", paths=[], ops_remaining=3)
    loop._escalation._granted_escalations["fetch"] = grant

    # SSRF gate (always-on, taint-independent) denies an internal-metadata fetch.
    resolved = await loop._resolve_tool_intent(
        _fetch("http://169.254.169.254/latest/meta-data/", env.node_id), env
    )
    assert not resolved.approved
    assert grant.ops_remaining == 3          # denied by a gate → nothing consumed


@pytest.mark.asyncio
async def test_grant_op_consumed_once_when_approved(make_envelope):
    loop, env = _loop_env(make_envelope)
    grant = _GrantedEscalation(tool="fetch", paths=[], ops_remaining=3)
    loop._escalation._granted_escalations["fetch"] = grant

    resolved = await loop._resolve_tool_intent(
        _fetch("http://example.com/page", env.node_id), env
    )
    assert resolved.approved
    assert grant.ops_remaining == 2          # approved → exactly one op consumed
