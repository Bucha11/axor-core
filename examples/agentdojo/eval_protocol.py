"""Shared protocol-tool plumbing for the structured-claim properties.

Both Tool (FABRICATED, submit_findings) and Budget (BUDGET_MISREPORT,
report_usage) need the same two pieces over AgentDojo:

- register protocol/meta tools on each task's runtime before the model sees its
  tool list, and
- force the model to actually call the claim tool (AgentDojo's tool loop ends on
  a text answer, so a model that replies in prose never delivers its claim).

These are parameterised here so each property supplies its own tools and its own
"claim delivered" predicate.
"""
from __future__ import annotations

import os
from typing import Callable

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.types import text_content_block_from_string


class ProtocolToolElement(BasePipelineElement):
    """Registers a set of protocol tools on each task's runtime (idempotent)."""

    def __init__(self, tools: list[Callable]) -> None:
        self._tools = tools
        self.name = "axor-protocol-tools"

    def query(self, query, runtime, env=None, messages=[], extra_args={}):
        newly = False
        for fn in self._tools:
            if fn.__name__ not in runtime.functions:
                runtime.register_function(fn)
                newly = True
        if newly and os.environ.get("AXOR_EVAL_DEBUG"):
            print(f"    [protocol] tools now: {sorted(runtime.functions)}")
        return query, runtime, env, messages, extra_args


class ForcingToolsExecutionLoop(ToolsExecutionLoop):
    """ToolsExecutionLoop that, if the model stops (a text answer, no tool call)
    before delivering its structured claim, injects a nudge and continues — up to
    ``max_force`` times. ``is_done`` returns True once the claim is recorded; the
    model still freely chooses the claim value, so this captures it unbiased."""

    def __init__(self, elements, is_done: Callable[[], bool], force_message: str,
                 max_iters: int = 15, max_force: int = 2) -> None:
        super().__init__(elements, max_iters)
        self._is_done = is_done
        self._force_message = force_message
        self._max_force = max_force

    def query(self, query, runtime, env=None, messages=[], extra_args={}):
        dbg = os.environ.get("AXOR_EVAL_DEBUG")
        forces = 0
        for i in range(self.max_iters):
            last = messages[-1]
            stopped = (last["role"] != "assistant" or not last.get("tool_calls"))
            if dbg:
                tcs = [tc.function for tc in (last.get("tool_calls") or [])] if last["role"] == "assistant" else []
                print(f"    [loop {i}] role={last['role']} tool_calls={tcs} stopped={stopped} done={self._is_done()}")
            if stopped:
                if self._is_done() or forces >= self._max_force:
                    break
                forces += 1
                messages = [*messages, {
                    "role": "user",
                    "content": [text_content_block_from_string(self._force_message)],
                }]
            for element in self.elements:
                query, runtime, env, messages, extra_args = element.query(
                    query, runtime, env, messages, extra_args
                )
        return query, runtime, env, messages, extra_args
