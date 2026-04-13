from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from axor_core.contracts.envelope import ExecutionEnvelope
    from axor_core.contracts.events import ExecutorEvent


class Invokable(ABC):
    """
    Universal contract for any raw executor.

    An executor may be:
    - LLM agent (Claude, OpenAI, custom)
    - tool-using executor
    - custom reasoning engine
    - external solver

    The executor receives a governed ExecutionEnvelope.
    It never receives raw ambient system state.

    Execution is stream-based — not return-based.
    This allows the node to intercept tool_use events
    before they execute, converting them into Intents.
    """

    @abstractmethod
    def stream(self, envelope: ExecutionEnvelope) -> AsyncIterator[ExecutorEvent]:
        """
        Stream governed execution.

        The node intercepts each event:
        - tool_use  → becomes Intent → resolved by node
        - text      → passes through export filter
        - stop      → triggers governed result construction

        The executor never knows it is being intercepted.
        It receives tool results as if it called the tools directly.
        """
        ...
