"""Phase 1: process-isolation gate for untrusted agents in PRODUCTION/STRICT."""

from __future__ import annotations

import logging

import pytest
from unittest.mock import MagicMock

from axor_core.capability.daemon_client import DaemonCapabilityClient
from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.agent import AgentDefinition, TrustLevel
from axor_core.contracts.mode import ExecutionMode
from axor_core.errors.exceptions import IsolationRequiredError
from axor_core.worker.session import GovernedSession


def _session(cap, mode, **kwargs):
    return GovernedSession(
        executor=MagicMock(),
        capability_executor=cap,
        mode=mode,
        **kwargs,
    )


def test_require_isolation_flag_raises_for_inprocess_executor():
    with pytest.raises(IsolationRequiredError):
        _session(CapabilityExecutor(), ExecutionMode.PRODUCTION, require_isolation=True)


def test_env_require_isolation_raises(monkeypatch):
    monkeypatch.setenv("AXOR_REQUIRE_ISOLATION", "1")
    with pytest.raises(IsolationRequiredError):
        _session(CapabilityExecutor(), ExecutionMode.STRICT)


def test_isolated_executor_satisfies_requirement():
    daemon = DaemonCapabilityClient(socket_path="/tmp/nonexistent.sock")
    # Does not raise even with require_isolation=True — daemon is out-of-process.
    sess = _session(daemon, ExecutionMode.PRODUCTION, require_isolation=True)
    assert sess._mode == ExecutionMode.PRODUCTION


def test_untrusted_inprocess_warns_by_default(caplog):
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        _session(CapabilityExecutor(), ExecutionMode.PRODUCTION)
    assert any("not isolated" in r.message.lower() or "isolated" in r.message.lower()
               for r in caplog.records)


def test_elevated_trust_inprocess_no_warning(caplog):
    agent = AgentDefinition(name="trusted", trust_level=TrustLevel.ELEVATED)
    with caplog.at_level(logging.WARNING, logger="axor.session"):
        _session(CapabilityExecutor(), ExecutionMode.PRODUCTION, agent_def=agent)
    assert not any("isolated" in r.message.lower() for r in caplog.records)


def test_library_mode_ignores_isolation_gate():
    # Gate only applies to PRODUCTION/STRICT; LIBRARY never raises.
    sess = _session(CapabilityExecutor(), ExecutionMode.LIBRARY, require_isolation=True)
    assert sess._mode == ExecutionMode.LIBRARY
