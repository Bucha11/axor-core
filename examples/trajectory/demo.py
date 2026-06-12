"""Runnable demo: a stove-on-too-long observer freezes the session surface.

    python examples/trajectory/demo.py

A household agent checks the stove (reported on for 45 min), then tries to start the
oven. The StoveOnTooLongObserver reads the check result, tightens degradation to
LOCKED, and the oven call is frozen — the agent must escalate to a human.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stove import StoveOnTooLongObserver  # noqa: E402

from axor_core import GovernedSession  # noqa: E402
from axor_core.capability.executor import CapabilityExecutor, ToolHandler  # noqa: E402
from axor_core.contracts.invokable import Invokable  # noqa: E402
from axor_core.contracts.policy import (  # noqa: E402
    ChildMode, CompressionMode, ContextMode, ExecutionPolicy, ExportMode,
    TaskComplexity, ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind  # noqa: E402
from axor_core.contracts.trace import TraceConfig  # noqa: E402


class _CheckStove(ToolHandler):
    @property
    def name(self): return "check_stove"
    async def execute(self, args): return {"on": True, "minutes": 45}


class _StartOven(ToolHandler):
    def __init__(self): self.started = False
    @property
    def name(self): return "start_oven"
    async def execute(self, args): self.started = True; return "oven on"


class _KitchenAgent(Invokable):
    async def stream(self, envelope):
        for tool in ("check_stove", "start_oven"):
            yield ExecutorEvent(ExecutorEventKind.TOOL_USE, {"tool": tool, "args": {}}, envelope.node_id)
        yield ExecutorEvent(ExecutorEventKind.STOP, {"usage": {"input_tokens": 1, "output_tokens": 1}}, envelope.node_id)


async def main() -> int:
    oven = _StartOven()
    cap = CapabilityExecutor()
    cap.register(_CheckStove())
    cap.register(oven)
    session = GovernedSession(
        executor=_KitchenAgent(), capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        trajectory_observers=[StoveOnTooLongObserver(threshold_minutes=30)],
    )
    await session.run("keep the kitchen safe", policy=ExecutionPolicy(
        name="kitchen", derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL, compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED, max_child_depth=0,
        tool_policy=ToolPolicy(extra_allowed=frozenset({"check_stove", "start_oven"})),
        export_mode=ExportMode.SUMMARY,
    ))

    print(f"degradation level after the stove check: {session.current_degradation_level()} (3 = LOCKED)")
    print(f"oven started? {oven.started}  (expected False — frozen by the trajectory observer)")
    print("the surface is frozen to read + escalate; the next step is a human gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
