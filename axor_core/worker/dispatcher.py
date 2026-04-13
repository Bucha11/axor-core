from __future__ import annotations

from axor_core.contracts.context import RawExecutionState
from axor_core.contracts.result import ExecutionResult
from axor_core.node.wrapper import GovernedNode


class Dispatcher:
    """
    Routes normalized input into the governed node flow.

    Worker starts execution.
    Node governs execution.

    Dispatcher is the handoff point between the two.
    It holds no policy logic, no context logic, no capability logic.
    """

    async def dispatch(
        self,
        node: GovernedNode,
        raw_state: RawExecutionState,
        **kwargs,
    ) -> ExecutionResult:
        return await node.run(raw_state, **kwargs)
