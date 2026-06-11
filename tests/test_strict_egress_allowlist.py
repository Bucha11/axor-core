"""STRICT mode requires a destination allowlist on every egress sink.

The per-value taint gate on an egress sink is content-derivation — sound in the
deny direction but with a paraphrase residual. An enum allowlist on the
destination is content-blind and provenance-independent, so it closes that
residual. STRICT refuses to ship an egress sink that relies on the leaky gate
alone: construction fails closed.
"""
from __future__ import annotations

import pytest

from axor_core.governor import ToolCallGovernor
from axor_core.policy.value_policy import enum
from axor_core.kernel.registration import validate_egress_allowlists


# ── the validator ───────────────────────────────────────────────────────────────

def test_validator_flags_egress_sink_without_allowlist():
    errs = validate_egress_allowlists({"send_email"}, {})
    assert len(errs) == 1 and "send_email" in errs[0]


def test_validator_accepts_egress_sink_with_enum_allowlist():
    pols = {"send_email": [enum("to", {"alice@corp.com"})]}
    assert validate_egress_allowlists({"send_email"}, pols) == []


def test_validator_rejects_non_enum_predicate():
    from axor_core.policy.value_policy import numeric_range
    # a numeric range is not a destination allowlist
    pols = {"send_email": [numeric_range("amount", 0, 100)]}
    assert validate_egress_allowlists({"send_email"}, pols) != []


def test_validator_no_egress_sinks_is_vacuously_valid():
    assert validate_egress_allowlists(set(), {}) == []


# ── the governor flag ────────────────────────────────────────────────────────────

def test_governor_strict_requires_allowlist():
    with pytest.raises(ValueError, match="strict egress allowlist"):
        ToolCallGovernor(egress_sinks={"send_email"}, require_egress_allowlist=True)


def test_governor_strict_passes_with_allowlist_and_enforces_both():
    gov = ToolCallGovernor(
        egress_sinks={"send_email"},
        value_policies={"send_email": [enum("to", {"alice@corp.com"})]},
        require_egress_allowlist=True,
    )
    # allowlist (sound): a recipient outside the set is denied regardless of taint
    assert not gov.evaluate("send_email", {"to": "attacker@evil.com"}).allowed
    # whitelisted recipient passes
    assert gov.evaluate("send_email", {"to": "alice@corp.com"}).allowed


def test_governor_lax_by_default_no_allowlist_required():
    # default (non-strict): egress sink without allowlist is allowed to exist;
    # the content-derivation gate still applies.
    gov = ToolCallGovernor(egress_sinks={"send_email"})
    assert gov.evaluate("send_email", {"to": "anyone@anywhere.com"}).allowed


# ── the GovernedSession (STRICT mode) ────────────────────────────────────────────

def _session(**kw):
    from axor_core import GovernedSession
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.contracts.mode import ExecutionMode
    from axor_core.contracts.trace import TraceConfig
    return GovernedSession(
        executor=__import__("tests.conftest", fromlist=["EchoExecutor"]).EchoExecutor(),
        capability_executor=CapabilityExecutor(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        mode=ExecutionMode.STRICT,
        **kw,
    )


def test_strict_session_fails_closed_without_allowlist():
    with pytest.raises(ValueError, match="strict egress allowlist"):
        _session(egress_sinks={"send_email"})


def test_strict_session_constructs_with_allowlist():
    s = _session(
        egress_sinks={"send_email"},
        value_policies={"send_email": [enum("to", {"alice@corp.com"})]},
    )
    assert s is not None


def test_non_strict_session_does_not_require_allowlist():
    from axor_core import GovernedSession
    from axor_core.capability.executor import CapabilityExecutor
    from axor_core.contracts.trace import TraceConfig
    from tests.conftest import EchoExecutor
    # LIBRARY mode (default): no allowlist obligation
    s = GovernedSession(
        executor=EchoExecutor(), capability_executor=CapabilityExecutor(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        egress_sinks={"send_email"},
    )
    assert s is not None
