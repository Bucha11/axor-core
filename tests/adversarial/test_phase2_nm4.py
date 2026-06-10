"""Phase 2 — NM4 (unknown sink posture).

Re-verification outcome: a global deny-by-default (unknown -> CATASTROPHIC) is both
disruptive and wrong (fetch/curl-class custom sinks are not catastrophic). The
kernel's actual posture is: unknown sinks default to CONSEQUENTIAL, which is
fail-closed under the high-assurance ceiling (strict mode lowers it to REVERSIBLE)
and operator-tunable via the `danger` table. These tests pin the fail-closed path so
it cannot silently regress.
"""

from __future__ import annotations

from typing import Any

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial


class _Handler(ToolHandler):
    def __init__(self, name: str) -> None:
        self._n = name

    @property
    def name(self) -> str:
        return self._n

    async def execute(self, args: dict[str, Any]) -> Any:
        return "ok"


def _env(ceiling: ConsequenceClass) -> ExecutionEnvelope:
    policy = ExecutionPolicy(
        name="t", tool_policy=ToolPolicy(allow_bash=True),
        max_unattended_consequence=ceiling,
    )
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({"custom_unknown_sink"}), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=True, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.FULL,
                                       allowed_fields=frozenset(["output"]),
                                       max_export_tokens=1024),
        lineage=lineage, cancel_token=make_token(),
    )


async def _resolve(env):
    ex = CapabilityExecutor()
    ex.register(_Handler("custom_unknown_sink"))
    loop = IntentLoop(capability_executor=ex, trace_events=[], taint_engine=TaintEngine())
    ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                       payload={"tool": "custom_unknown_sink", "args": {"x": 1},
                                "tool_use_id": "u"},
                       node_id=env.node_id)
    return await loop._resolve_tool_intent(ev, env)


@pytest.mark.asyncio
async def test_unknown_sink_gated_under_strict_ceiling():
    # strict posture: ceiling REVERSIBLE -> an unknown CONSEQUENTIAL sink exceeds it
    # and is gated unattended (fail-closed) without an explicit table entry.
    r = await _resolve(_env(ConsequenceClass.REVERSIBLE))
    assert not r.approved
    assert r.result.get("category") == "consequence_gate"


@pytest.mark.asyncio
async def test_unknown_sink_allowed_under_default_ceiling():
    # default posture: ceiling CONSEQUENTIAL -> unknown sink passes unattended (the
    # deliberate usability default; operators tighten via `danger` or strict mode).
    r = await _resolve(_env(ConsequenceClass.CONSEQUENTIAL))
    assert r.approved
