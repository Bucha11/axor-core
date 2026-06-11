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


# ── STRICT role completeness (the symmetric source-side obligation) ──────────────

from axor_core.kernel.registration import validate_role_completeness


def test_role_completeness_flags_unclassified_tool():
    errs = validate_role_completeness(
        {"search_docs", "send_email", "get_time"},
        untrusted_sources={"search_docs"}, egress_sinks={"send_email"},
    )
    assert len(errs) == 1 and "get_time" in errs[0]


def test_role_completeness_satisfied_by_each_role():
    tools = {"a", "b", "c", "d", "e"}
    assert validate_role_completeness(
        tools,
        untrusted_sources={"a"}, sensitive_sources={"b"},
        egress_sinks={"c"}, positional_sinks={"d"}, benign_tools={"e"},
    ) == []


def test_role_completeness_value_policy_counts_as_classified():
    assert validate_role_completeness(
        {"transfer"}, value_policies={"transfer": [enum("x", {"y"})]}
    ) == []


def test_role_completeness_exempts_kernel_intents():
    assert validate_role_completeness({"spawn_child", "escalate_policy"}) == []


def _strict_session_with_tools(tool_names, **kw):
    from axor_core import GovernedSession
    from axor_core.capability.executor import CapabilityExecutor, ToolHandler
    from axor_core.contracts.mode import ExecutionMode
    from axor_core.contracts.trace import TraceConfig
    from tests.conftest import EchoExecutor

    class _H(ToolHandler):
        def __init__(self, n): self._n = n
        @property
        def name(self): return self._n
        async def execute(self, args): return "x"

    cap = CapabilityExecutor()
    for n in tool_names:
        cap.register(_H(n))
    return GovernedSession(
        executor=EchoExecutor(), capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        mode=ExecutionMode.STRICT, **kw,
    )


def test_strict_session_fails_on_unclassified_tool():
    with pytest.raises(ValueError, match="strict role completeness"):
        _strict_session_with_tools(
            ["search_docs", "send_email", "get_time"],
            untrusted_sources={"search_docs"}, egress_sinks={"send_email"},
            value_policies={"send_email": [enum("to", {"a@b.com"})]},
        )  # get_time unclassified


def test_strict_session_constructs_with_full_classification():
    s = _strict_session_with_tools(
        ["search_docs", "send_email", "get_time"],
        untrusted_sources={"search_docs"}, egress_sinks={"send_email"},
        value_policies={"send_email": [enum("to", {"a@b.com"})]},
        benign_tools={"get_time"},
    )
    assert s is not None
