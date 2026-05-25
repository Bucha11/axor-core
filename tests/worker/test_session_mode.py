from __future__ import annotations

import logging
import os

import pytest

from axor_core.capability.executor import CapabilityExecutor
from axor_core.capability.locked import LockedExecutor
from axor_core.contracts.mode import ExecutionMode
from axor_core.errors.exceptions import GovernanceBypassError
from axor_core.worker.session import GovernedSession


class _StubInvokable:
    async def stream(self, envelope):  # type: ignore[override]
        return
        yield


def _make_session(mode: ExecutionMode) -> GovernedSession:
    return GovernedSession(
        executor=_StubInvokable(),
        capability_executor=CapabilityExecutor(),
        mode=mode,
    )


def test_production_mode_wraps_executor_in_locked():
    session = _make_session(ExecutionMode.PRODUCTION)
    assert isinstance(session._executor, LockedExecutor)


def test_strict_mode_wraps_executor_in_locked():
    session = _make_session(ExecutionMode.STRICT)
    assert isinstance(session._executor, LockedExecutor)


def test_library_mode_does_not_wrap_executor():
    session = _make_session(ExecutionMode.LIBRARY)
    assert not isinstance(session._executor, LockedExecutor)


def test_production_mode_direct_executor_call_raises():
    session = _make_session(ExecutionMode.PRODUCTION)
    with pytest.raises(GovernanceBypassError):
        session._executor.stream(object())  # type: ignore[arg-type]


def test_library_mode_with_axor_env_production_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("AXOR_ENV", "production")
            _make_session(ExecutionMode.LIBRARY)
    assert "LIBRARY mode" in caplog.text
