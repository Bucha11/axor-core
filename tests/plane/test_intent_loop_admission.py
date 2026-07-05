"""End-to-end: a control-plane stop actually halts the IntentLoop, and a
pause holds it between intents — the advisory overlay wired into enforcement.

The plane can only stop or hold; it never widens. A stopped node admits no
further intents (the tools after the stop point never execute), and the boundary
is between intents — the intent in flight is never interrupted mid-effect.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.plane.admission import PlaneAdmission
from axor_core.plane.session import PlaneSession


class _Handler(ToolHandler):
    def __init__(self, name: str, log: list[str]) -> None:
        self._n, self._log = name, log

    @property
    def name(self) -> str:
        return self._n

    async def execute(self, args: dict[str, Any]) -> Any:
        self._log.append(self._n)
        return "ok"


def _executor(log: list[str]) -> CapabilityExecutor:
    ex = CapabilityExecutor()
    for name in ("read", "write"):
        ex.register(_Handler(name, log))
    return ex


def _envelope() -> ExecutionEnvelope:
    policy = ExecutionPolicy(
        name="t", tool_policy=ToolPolicy(allow_read=True, allow_write=True)
    )
    lineage = LineageSummary(
        node_id="n1", parent_id=None, depth=0, ancestry_ids=[],
        inherited_restrictions=[],
    )
    ctx = ContextView(
        node_id="n1", working_summary="t", visible_fragments=[],
        active_constraints=[], lineage=lineage, token_count=0,
        compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset({"read", "write"}), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=True, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.FULL, allowed_fields=frozenset(["output"]),
            max_export_tokens=1024,
        ),
        lineage=lineage, cancel_token=make_token(),
    )


async def _drive(loop: IntentLoop, env: ExecutionEnvelope, calls: list[str]) -> None:
    async def _stream() -> Any:
        for i, tool in enumerate(calls):
            yield ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": tool, "args": {}, "tool_use_id": f"t{i}"},
                node_id=env.node_id,
            )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id=env.node_id
        )

    async for _ in loop.run(_stream(), env):
        pass


async def test_no_admission_runs_everything() -> None:
    log: list[str] = []
    loop = IntentLoop(capability_executor=_executor(log), trace_events=[])
    await _drive(loop, _envelope(), ["read", "write", "read"])
    assert log == ["read", "write", "read"]


async def test_stopped_plane_admits_no_intents() -> None:
    log: list[str] = []
    session = PlaneSession(node_id="n1")
    session.apply_snapshot(1, {"stopped": True})
    loop = IntentLoop(
        capability_executor=_executor(log), trace_events=[],
        admission=PlaneAdmission(session),
    )
    await _drive(loop, _envelope(), ["read", "write"])
    assert log == []  # stop takes effect at the first boundary


async def test_stop_mid_stream_halts_remaining_intents() -> None:
    log: list[str] = []
    session = PlaneSession(node_id="n1")
    adm = PlaneAdmission(session)
    loop = IntentLoop(
        capability_executor=_executor(log), trace_events=[], admission=adm,
    )

    # A handler that stops the node the moment `read` has executed — the
    # intent in flight completes, the next one must never run.
    class _StoppingHandler(ToolHandler):
        @property
        def name(self) -> str:
            return "read"

        async def execute(self, args: dict[str, Any]) -> Any:
            log.append("read")
            session.apply_snapshot(2, {"stopped": True})
            return "ok"

    ex = CapabilityExecutor()
    ex.register(_StoppingHandler())
    ex.register(_Handler("write", log))
    loop = IntentLoop(capability_executor=ex, trace_events=[], admission=adm)

    await _drive(loop, _envelope(), ["read", "write"])
    assert log == ["read"]  # write never admitted; read completed, not interrupted


async def test_pause_holds_then_resume_completes() -> None:
    log: list[str] = []
    session = PlaneSession(node_id="n1")
    session.apply_snapshot(1, {"paused": True})
    adm = PlaneAdmission(session, poll_interval=0.01)
    loop = IntentLoop(
        capability_executor=_executor(log), trace_events=[], admission=adm,
    )

    async def resume_soon() -> None:
        await asyncio.sleep(0.03)
        assert log == []  # nothing ran while paused
        session.apply_snapshot(2, {"paused": False})
        adm.notify()

    resumer = asyncio.create_task(resume_soon())
    await asyncio.wait_for(_drive(loop, _envelope(), ["read", "write"]), timeout=2.0)
    await resumer
    assert log == ["read", "write"]
