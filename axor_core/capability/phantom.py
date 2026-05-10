from __future__ import annotations

from typing import Any, Callable, Awaitable

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.plan import AgentPlan, PlanStep

# Tools intercepted during Phase 1 — recorded as PlanSteps, never touch disk.
_MUTATION_TOOLS = frozenset({"write", "bash", "delete", "patch"})


class PhantomCapabilityExecutor:
    """
    Phase 1 executor: intercepts mutation tools without touching disk.

    Mutation tools (write, bash, delete, patch) are phantom-executed:
    - recorded as PlanStep
    - written content stored in _phantom_fs so the agent can read it back
    - a plausible success result returned to the agent

    This lets the agent plan multi-step sequences correctly. If it phantom-writes
    auth.py and then reads it to write a test, it receives the phantom content —
    the plan stays internally consistent even though nothing hit disk.

    Read and search tools are delegated to the real CapabilityExecutor.

    After Phase 1 finishes, call .build_plan(task) to extract the AgentPlan.
    """

    def __init__(self, real_executor: CapabilityExecutor) -> None:
        self._real = real_executor
        self._phantom_fs: dict[str, str] = {}
        self._steps: list[PlanStep] = []
        self._post_callbacks: list[Callable[[str, dict, Any], Awaitable[None]]] = []

    # ── CapabilityExecutor interface ───────────────────────────────────────────

    def register(self, handler: ToolHandler) -> None:
        self._real.register(handler)

    def register_post_callback(
        self, callback: Callable[[str, dict[str, Any], Any], Awaitable[None]]
    ) -> None:
        self._post_callbacks.append(callback)

    def registered_tools(self) -> frozenset[str]:
        return self._real.registered_tools()

    async def execute(self, intent: Intent, capabilities: Capabilities) -> Any:
        if intent.kind != IntentKind.TOOL_CALL:
            return await self._real.execute(intent, capabilities)

        tool_name: str = intent.payload.get("tool", "")
        args: dict = intent.payload.get("args", {})

        if tool_name in _MUTATION_TOOLS:
            return self._phantom_execute(tool_name, args)

        # Read: serve from phantom_fs if the agent previously phantom-wrote the file.
        if tool_name == "read":
            path = args.get("path", "")
            if path in self._phantom_fs:
                result = self._phantom_fs[path]
                await self._fire_callbacks(tool_name, args, result)
                return result

        # All other tools (read from disk, search, …) — real execution.
        result = await self._real.execute(intent, capabilities)
        await self._fire_callbacks(tool_name, args, result)
        return result

    # ── Plan extraction ────────────────────────────────────────────────────────

    def build_plan(self, task: str, phase1_tokens: int = 0, node_id: str = "") -> AgentPlan:
        return AgentPlan(
            task=task,
            steps=list(self._steps),
            phase1_node_id=node_id,
            phase1_tokens=phase1_tokens,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    def _phantom_execute(self, tool: str, args: dict) -> Any:
        index = len(self._steps)

        if tool == "write":
            path = args.get("path", "")
            content = args.get("content", "")
            self._phantom_fs[path] = content
            result: Any = {"ok": True, "path": path, "bytes_written": len(content)}

        elif tool == "delete":
            path = args.get("path", "")
            self._phantom_fs.pop(path, None)
            result = {"ok": True, "path": path, "deleted": True}

        else:
            # bash, patch — return plausible empty success
            result = {"ok": True, "stdout": "", "stderr": "", "exit_code": 0}

        self._steps.append(PlanStep(
            index=index,
            tool=tool,
            args=args,
            phantom_result=result,
        ))
        return result

    async def _fire_callbacks(self, tool: str, args: dict, result: Any) -> None:
        for cb in self._post_callbacks:
            try:
                await cb(tool, args, result)
            except Exception:
                pass
