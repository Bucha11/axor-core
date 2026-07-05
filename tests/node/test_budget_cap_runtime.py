"""Runtime budget cap (spec §15): the IntentLoop enforces budget_cap_calls at the
loop boundary, in replay parity — the same ceiling the replay kernel checks, so a
run and its counterfactual agree on when the budget is exhausted. Exhaustion is a
typed denial, never a silent overrun."""
from __future__ import annotations

from typing import Any

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.intent_loop import IntentLoop


class _Handler(ToolHandler):
    def __init__(self, name: str) -> None:
        self._n = name

    @property
    def name(self) -> str:
        return self._n

    async def execute(self, args: dict[str, Any]) -> Any:
        return "ok"


def _executor() -> CapabilityExecutor:
    ex = CapabilityExecutor()
    ex.register(_Handler("read"))
    return ex


def _envelope() -> ExecutionEnvelope:
    policy = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_read=True))
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({"read"}), allow_children=False,
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


async def _drive(loop: IntentLoop, env, n: int) -> list[dict]:
    async def _stream():
        for i in range(n):
            yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                                payload={"tool": "read", "args": {},
                                         "tool_use_id": f"t{i}"},
                                node_id=env.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}},
                            node_id=env.node_id)

    out = []
    async for ev in loop.run(_stream(), env):
        if ev.kind == ExecutorEventKind.TEXT and "approved" in ev.payload:
            out.append(ev.payload)
    return out


@pytest.mark.asyncio
async def test_budget_cap_denies_the_n_plus_first_call() -> None:
    trace: list = []
    loop = IntentLoop(capability_executor=_executor(), trace_events=trace,
                      budget_cap_calls=2)
    results = await _drive(loop, _envelope(), 4)
    approved = [r["approved"] for r in results]
    assert approved == [True, True, False, False]  # first 2 pass, rest denied
    # The denial is typed, not silent: a budget-category denial in the trace.
    denials = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
    assert len(denials) == 2
    assert all("budget" in d.reason and "cap 2 exhausted" in d.reason
               for d in denials)


@pytest.mark.asyncio
async def test_no_cap_means_unlimited() -> None:
    loop = IntentLoop(capability_executor=_executor(), trace_events=[],
                      budget_cap_calls=None)
    results = await _drive(loop, _envelope(), 5)
    assert all(r["approved"] for r in results)


@pytest.mark.asyncio
async def test_gate_denied_calls_do_not_consume_budget() -> None:
    """Budget counts only APPROVED calls, exactly as the replay kernel folds it —
    a call denied by a gate burns nothing. A cap of 1 with the first call denied
    by capability still lets a later allowed call through."""
    ex = CapabilityExecutor()
    ex.register(_Handler("read"))
    loop = IntentLoop(capability_executor=ex, trace_events=[], budget_cap_calls=1)

    async def _stream():
        # 'forbidden' is not in capabilities → capability DENY (no budget spent).
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": "forbidden", "args": {},
                                     "tool_use_id": "a"}, node_id="n1")
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": "read", "args": {},
                                     "tool_use_id": "b"}, node_id="n1")
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": "read", "args": {},
                                     "tool_use_id": "c"}, node_id="n1")
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}},
                            node_id="n1")

    out = []
    async for ev in loop.run(_stream(), _envelope()):
        if ev.kind == ExecutorEventKind.TEXT and "approved" in ev.payload:
            out.append(ev.payload)
    # forbidden → denied (capability, not budget); first read → approved (spends
    # the 1 unit); second read → denied (budget exhausted).
    assert [r["approved"] for r in out] == [False, True, False]
