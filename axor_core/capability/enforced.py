from __future__ import annotations

from typing import Any, Callable, Awaitable

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.plan import AgentPlan, PlanStep
from axor_core.errors.exceptions import ToolNotAllowedError

_MUTATION_TOOLS = frozenset({"write", "bash", "delete", "patch"})


class EnforcedCapabilityExecutor:
    """
    Phase 2 executor: validates each mutation intent against the approved plan.

    Works as a gate in front of the real CapabilityExecutor:
    - Mutation tools (write, bash, delete, patch): checked against the next
      approved plan step. If the tool name matches, it executes with the
      agent's args and advances the cursor. If it doesn't match, or if the
      agent tries a mutation not in the plan, ToolNotAllowedError is raised —
      IntentLoop catches this and returns a structured denial to the agent.
    - Read and search tools: pass through unconditionally.

    The agent provides args in Phase 2, not the plan. This is intentional:
    the agent may refine content after reading fresh file state in Phase 2.
    What's enforced is the *sequence and type* of mutations, not exact bytes.

    If the agent tries to skip ahead or add extra steps, the cursor mismatch
    causes a denial. The agent can only move linearly through the approved plan.
    """

    def __init__(self, real_executor: CapabilityExecutor, plan: AgentPlan) -> None:
        self._real = real_executor
        self._approved = plan.approved_steps()
        self._cursor = 0
        self._post_callbacks: list[Callable[[str, dict, Any], Awaitable[None]]] = []

    # ── CapabilityExecutor interface ───────────────────────────────────────────

    def register(self, handler: ToolHandler) -> None:
        self._real.register(handler)

    def register_post_callback(
        self, callback: Callable[[str, dict[str, Any], Any], Awaitable[None]]
    ) -> None:
        # Forward to real executor so ContextManager's observer fires normally.
        self._real.register_post_callback(callback)

    def registered_tools(self) -> frozenset[str]:
        return self._real.registered_tools()

    async def execute(self, intent: Intent, capabilities: Capabilities) -> Any:
        if intent.kind != IntentKind.TOOL_CALL:
            return await self._real.execute(intent, capabilities)

        tool_name: str = intent.payload.get("tool", "")

        if tool_name not in _MUTATION_TOOLS:
            # Reads and searches are never restricted.
            return await self._real.execute(intent, capabilities)

        return await self._enforce_mutation(tool_name, intent, capabilities)

    def remaining_steps(self) -> list[PlanStep]:
        return self._approved[self._cursor:]

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _enforce_mutation(
        self,
        tool_name: str,
        intent: Intent,
        capabilities: Capabilities,
    ) -> Any:
        if self._cursor >= len(self._approved):
            raise ToolNotAllowedError(
                tool=tool_name,
                allowed=set(),
            )

        expected = self._approved[self._cursor]

        if tool_name != expected.tool:
            # Return a clear denial reason — IntentLoop converts this exception
            # to a structured string the agent sees as a tool response.
            raise ToolNotAllowedError(
                tool=tool_name,
                allowed={expected.tool},
            )

        self._cursor += 1
        return await self._real.execute(intent, capabilities)
