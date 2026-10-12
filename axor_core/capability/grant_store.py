"""The spend side of task-scoped grants (Ring 1): one lock around the kernel store.

``kernel.grants.GrantStore`` is a pure state machine; this module is where it meets
threads and a clock. It belongs in the process that executes effects — the daemon
in the deployed layout — because O5 (atomic spend, at most one execution per spend)
only holds if check-and-decrement and the effect's idempotency key are decided in
one place.

    store = LockedGrantStore()
    ... GrantMinter(store.inner, table, grant_args).from_request(prompt)
    out = execute_granted(store, table, sink, args, handler)

``execute_granted`` is the whole protocol: reserve → run the handler with the
spend's idempotency key → commit; release if the handler reports it did not run;
on an unknown outcome (exception, timeout) reconcile against the handler's own log
instead of retrying.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from axor_core.kernel.grants import Grant, GrantStore, Reservation, SpendState
from axor_core.kernel.handles import HandleTable


class LockedGrantStore:
    """``GrantStore`` behind one lock, with a monotonic clock for expiry."""

    def __init__(self, inner: GrantStore | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.inner = inner if inner is not None else GrantStore()
        self._lock = threading.Lock()
        self._clock = clock

    def find(self, sink: str, args: Mapping[str, Any], table: HandleTable | None = None,
             now: float | None = None) -> Grant | None:
        with self._lock:
            return self.inner.find(sink, args, table, now=self._clock() if now is None else now)

    def has_grants_for(self, sink: str) -> bool:
        with self._lock:
            return self.inner.has_grants_for(sink)

    def reserve(self, sink: str, args: Mapping[str, Any], table: HandleTable | None = None,
                call_key: str | None = None, grant_id: str | None = None) -> Reservation | None:
        with self._lock:
            return self.inner.reserve(sink, args, table, call_key=call_key,
                                      grant_id=grant_id, now=self._clock())

    def commit(self, key: str) -> None:
        with self._lock:
            self.inner.commit(key)

    def release(self, key: str) -> None:
        with self._lock:
            self.inner.release(key)

    def reconcile(self, key: str, executed: Callable[[str], bool | None]) -> SpendState:
        # The handler query runs outside the lock: it may block on I/O. The state
        # transition it leads to is re-checked under the lock by commit/release.
        with self._lock:
            if self.inner.state(key) is not SpendState.RESERVED:
                return self.inner.state(key)
        ran = executed(key)
        with self._lock:
            if self.inner.state(key) is SpendState.RESERVED:
                if ran is True:
                    self.inner.commit(key)
                elif ran is False:
                    self.inner.release(key)
            return self.inner.state(key)

    def stale(self, older_than: float) -> list[str]:
        with self._lock:
            return self.inner.stale(older_than, self._clock())

    def remaining(self, grant_id: str) -> int | None:
        with self._lock:
            return self.inner.remaining(grant_id)

    def snapshot(self) -> tuple:
        with self._lock:
            return self.inner.snapshot()


class IdempotentHandler(Protocol):
    """A tool handler that dedups on the spend key and can answer for it."""

    def run(self, args: Mapping[str, Any], key: str) -> Any: ...

    def executed(self, key: str) -> bool | None: ...


class NotRun(Exception):
    """Raised by a handler that refused before any effect (safe to release)."""


@dataclass
class GrantedOutcome:
    allowed: bool
    key: str | None = None
    state: SpendState | None = None
    result: Any = None
    error: str | None = None


def execute_granted(store: LockedGrantStore, table: HandleTable | None, sink: str,
                    args: Mapping[str, Any], handler: IdempotentHandler,
                    call_key: str | None = None) -> GrantedOutcome:
    r = store.reserve(sink, args, table, call_key=call_key)
    if r is None:
        return GrantedOutcome(False, error="no live grant covers this call")
    if r.state is not SpendState.RESERVED:  # an idempotent retry of a finished call
        return GrantedOutcome(r.state is SpendState.COMMITTED, r.key, r.state)
    try:
        result = handler.run(args, r.key)
    except NotRun as exc:
        store.release(r.key)
        return GrantedOutcome(False, r.key, SpendState.RELEASED, error=str(exc))
    except Exception as exc:  # unknown outcome: never retry, reconcile
        state = store.reconcile(r.key, handler.executed)
        return GrantedOutcome(state is SpendState.COMMITTED, r.key, state, error=repr(exc))
    store.commit(r.key)
    return GrantedOutcome(True, r.key, SpendState.COMMITTED, result)


@dataclass
class DedupHandler:
    """Reference idempotent handler: runs ``effect`` at most once per key."""

    effect: Callable[[Mapping[str, Any]], Any]
    _done: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    effects: int = 0

    def run(self, args: Mapping[str, Any], key: str) -> Any:
        with self._lock:
            if key in self._done:
                return self._done[key]
            out = self.effect(args)
            self._done[key] = out
            self.effects += 1
            return out

    def executed(self, key: str) -> bool | None:
        with self._lock:
            return key in self._done
