from __future__ import annotations

import pytest

from axor_core.capability.locked import LockedExecutor, governance_context
from axor_core.contracts.mode import ExecutionMode
from axor_core.errors.exceptions import GovernanceBypassError


class _FakeInvokable:
    def stream(self, envelope):  # type: ignore[override]
        async def _gen():
            return
            yield  # noqa: unreachable — makes it an async generator

        return _gen()


def test_production_mode_direct_call_raises():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.PRODUCTION)
    with pytest.raises(GovernanceBypassError):
        executor.stream(object())  # type: ignore[arg-type]


def test_strict_mode_direct_call_raises():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.STRICT)
    with pytest.raises(GovernanceBypassError):
        executor.stream(object())  # type: ignore[arg-type]


def test_library_mode_direct_call_succeeds():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.LIBRARY)
    # Should not raise — LIBRARY mode allows direct calls
    result = executor.stream(object())  # type: ignore[arg-type]
    assert result is not None


def test_production_mode_inside_governance_context_succeeds():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.PRODUCTION)
    with governance_context():
        # Should not raise inside the governance context
        result = executor.stream(object())  # type: ignore[arg-type]
        assert result is not None


def test_strict_mode_inside_governance_context_succeeds():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.STRICT)
    with governance_context():
        result = executor.stream(object())  # type: ignore[arg-type]
        assert result is not None


def test_governance_context_does_not_leak():
    executor = LockedExecutor(_FakeInvokable(), ExecutionMode.PRODUCTION)
    with governance_context():
        pass
    # After context exits, direct call should raise again
    with pytest.raises(GovernanceBypassError):
        executor.stream(object())  # type: ignore[arg-type]
