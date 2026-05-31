from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, AsyncIterator, Generator

from axor_core.contracts.invokable import Invokable
from axor_core.contracts.mode import ExecutionMode
from axor_core.errors.exceptions import GovernanceBypassError

if TYPE_CHECKING:
    from axor_core.contracts.envelope import ExecutionEnvelope
    from axor_core.contracts.result import ExecutorEvent


# Set to True only while GovernedNode is executing the governed stream path.
_governance_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_axor_governance_active", default=False
)


@contextmanager
def governance_context() -> Generator[None, None, None]:
    """Context manager used by GovernedNode to authorise executor calls."""
    token = _governance_active.set(True)
    try:
        yield
    finally:
        _governance_active.reset(token)


class LockedExecutor(Invokable):
    """
    Wraps any Invokable and blocks direct calls in PRODUCTION/STRICT mode.

    In PRODUCTION or STRICT mode, stream() may only be called while inside
    a governance_context(). GovernedNode activates this context before
    delegating to the executor. Any direct call from outside raises
    GovernanceBypassError.
    """

    def __init__(self, executor: Invokable, mode: ExecutionMode) -> None:
        self._executor = executor
        self._mode = mode

    def stream(self, envelope: ExecutionEnvelope) -> AsyncIterator[ExecutorEvent]:
        if self._mode in (ExecutionMode.PRODUCTION, ExecutionMode.STRICT):
            if not _governance_active.get():
                raise GovernanceBypassError(
                    "executor.stream() called outside GovernedSession governance context"
                )
        return self._executor.stream(envelope)
