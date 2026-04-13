from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum


class CancelReason(str, Enum):
    USER_ABORT      = "user_abort"       # Ctrl+C or explicit session.cancel()
    BUDGET_EXHAUSTED = "budget_exhausted" # budget engine hit hard stop
    PARENT_CANCELLED = "parent_cancelled" # parent node was cancelled
    POLICY_DENIED   = "policy_denied"    # critical intent denied, no point continuing
    TIMEOUT         = "timeout"          # execution exceeded time limit


@dataclass
class CancelToken:
    """
    Cancellation signal for a single governed node execution.

    Design rules:
        - Parent cancellation does NOT automatically cancel children.
          Parent calls child_token.cancel() explicitly if it wants to.
        - Budget engine cancels via token — not by raising exceptions.
        - User abort cancels the session token — propagates to active node.
        - Cancellation is cooperative — intent_loop checks is_cancelled()
          at each event boundary.
        - Partial results are always returned — never silently dropped.
          A cancelled node returns what it completed before cancellation.

    One token per GovernedNode execution.
    Child nodes receive their own independent token.
    """

    _event:  asyncio.Event = field(default_factory=asyncio.Event)
    _reason: CancelReason | None = field(default=None)
    _detail: str = field(default="")

    def cancel(self, reason: CancelReason, detail: str = "") -> None:
        """Signal cancellation. Idempotent — safe to call multiple times."""
        if not self._event.is_set():
            self._reason = reason
            self._detail = detail
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> CancelReason | None:
        return self._reason

    @property
    def detail(self) -> str:
        return self._detail

    async def wait(self) -> CancelReason:
        """Await cancellation signal. Returns reason when fired."""
        await self._event.wait()
        return self._reason

    def child_token(self) -> "CancelToken":
        """
        Produce an independent token for a child node.

        Child gets its own token — parent cancellation does not
        automatically cancel the child. Parent must explicitly call
        child_token.cancel() if it wants to stop the child.

        This allows:
            - cancelling one child while others continue
            - parent continuing after child times out
            - budget pressure cancelling children before parent
        """
        return CancelToken()


def make_token() -> CancelToken:
    """Convenience constructor."""
    return CancelToken()
