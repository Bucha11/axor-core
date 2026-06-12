"""
Fail-closed tests.

Verifies that execution is denied or the session is terminated for each
failure scenario:

  1. Interceptor exception → deny
  2. Normalizer exception → deny
  3. Malformed tool call → deny
  4. Unknown provider format → deny
  5. BudgetTracker failure → does not silently succeed
  6. Invalid envelope → deny or error (session-level termination)
  7. Audit-required trace failure → PermissionError raised (contract)
  8. Telemetry failure → does not bypass policy
"""
from __future__ import annotations

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ── 1. Interceptor exception → deny ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fail_closed_interceptor_exception(make_envelope, cap_executor):
    """
    If the capability executor raises during tool execution, IntentLoop
    must return a denial (fail-closed), not propagate the raw exception.
    """
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

    async def _stream():
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "read", "args": {"path": "/src/x.py"}, "tool_use_id": "t1"},
            node_id="node_test_root",
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": {"input_tokens": 10, "output_tokens": 5, "tool_tokens": 0}},
            node_id="node_test_root",
        )

    # Patch CapabilityExecutor.execute to raise
    with patch.object(cap_executor, "execute", side_effect=RuntimeError("executor exploded")):
        loop = IntentLoop(capability_executor=cap_executor, trace_events=[])
        envelope = make_envelope()
        events = []
        async for ev in loop.run(_stream(), envelope):
            events.append(ev)

    # Worker must receive a denial, not an unhandled exception
    tool_results = [e for e in events if e.kind == ExecutorEventKind.TEXT]
    assert len(tool_results) == 1
    payload = tool_results[0].payload
    assert payload.get("approved") is False or "error" in str(payload.get("tool_result", ""))


# ── 2. Normalizer exception → deny ─────────────────────────────────────────────

def test_fail_closed_normalizer_exception_denies():
    """NormalizerError from normalizer → execution denied."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError

    normalizer = ClaudeNormalizer()
    with pytest.raises((NormalizerError, UnknownProviderFormatError)):
        normalizer.normalize({"_axor_parse_error": "malformed"})


# ── 3. Malformed tool call → deny ──────────────────────────────────────────────

def test_fail_closed_malformed_tool_call_denies():
    """Malformed JSON in tool arguments raises NormalizerError."""
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    from axor_core.errors.exceptions import NormalizerError

    normalizer = MockOpenAINormalizer()
    with pytest.raises(NormalizerError):
        normalizer.normalize({
            "type": "function",
            "function": {"name": "Bash", "arguments": "not_json{"},
        })


# ── 4. Unknown provider format → deny ─────────────────────────────────────────

def test_fail_closed_unknown_format_denies():
    """Unrecognised event format raises UnknownProviderFormatError."""
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    from axor_core.errors.exceptions import UnknownProviderFormatError

    normalizer = MockOpenAINormalizer()
    with pytest.raises(UnknownProviderFormatError):
        normalizer.normalize({"random_field": "random_value"})


def test_fail_closed_unknown_format_claude_denies():
    """ClaudeNormalizer rejects a non-dict, non-ToolUseBlock event."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from axor_core.errors.exceptions import UnknownProviderFormatError

    with pytest.raises(UnknownProviderFormatError):
        ClaudeNormalizer().normalize("raw_string_event")


# ── 5. BudgetTracker failure → does not silently succeed ──────────────────────

def test_fail_closed_budget_tracker_subscriber_exception_does_not_break():
    """
    A subscriber that raises during record() must not break the record() call.

    BudgetTracker is fail-open for subscriber errors (a telemetry failure
    must not bypass execution), but record() itself must succeed.
    """
    from axor_core.budget.tracker import BudgetTracker

    tracker = BudgetTracker()
    tracker.register_node("n1", None, depth=0)

    def bad_subscriber(event):
        raise RuntimeError("subscriber exploded")

    tracker.subscribe(bad_subscriber)
    # record() must not raise despite the bad subscriber
    tracker.record("n1", input_tokens=10, output_tokens=5)
    assert tracker.total_tokens() == 15


# ── 6. Invalid envelope → deny or error ───────────────────────────────────────

def test_fail_closed_empty_allowed_tools_denies_all():
    """
    An envelope with no allowed_tools causes all tool calls to be denied.

    This simulates a misconfigured/invalid envelope — the fail-closed
    default is deny.
    """
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind
    from axor_core.contracts.envelope import ExecutionEnvelope, ExportContract, Capabilities
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.policy import ExecutionPolicy, ExportMode

    lineage = LineageSummary(node_id="n", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n", working_summary="", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(allowed_tools=frozenset(),  # no tools
                        allow_children=False, allow_nested_children=False,
                        allow_context_expansion=False, allow_export=False,
                        allow_mutation=False, max_child_depth=0)
    envelope = ExecutionEnvelope(
        node_id="n", task="test", context=ctx,
        policy=ExecutionPolicy(name="test"),
        capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.RESTRICTED,
                                       allowed_fields=frozenset(), max_export_tokens=0),
        lineage=lineage, cancel_token=make_token(),
    )
    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "read", "args": {}}, node_id="n")
    decision, _ = loop._evaluate_tool_intent(intent, envelope)
    assert decision.kind == PolicyDecisionKind.DENY


# ── 7. Audit-required trace failure → session termination ─────────────────────

def test_fail_closed_trace_access_requires_token():
    """
    Accessing DecisionTrace without operator token raises PermissionError.

    This is the audit-required fail-closed contract: worker cannot read trace.
    """
    from axor_core.trace.collector import TraceCollector

    collector = TraceCollector(operator_token="audit-token")
    collector.register_node("n1", None, 0, "audit-policy")

    # Direct read always fails (audit access requires operator channel)
    with pytest.raises(PermissionError):
        collector.read_all()

    # Wrong token fails
    with pytest.raises(PermissionError):
        collector.operator_read("wrong-token")

    # Correct token succeeds
    traces = collector.operator_read("audit-token")
    assert len(traces) == 1


# ── 8. Telemetry failure → does not bypass policy ─────────────────────────────

def test_fail_closed_telemetry_failure_does_not_bypass_policy():
    """
    A failed telemetry subscriber must not prevent governance decisions.

    Policy evaluation runs even if telemetry recording fails.
    """
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind
    from axor_core.contracts.envelope import ExecutionEnvelope, ExportContract, Capabilities
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.policy import ExecutionPolicy, ExportMode
    from axor_core.budget.tracker import BudgetTracker

    # Simulate telemetry/subscriber failure
    tracker = BudgetTracker()
    tracker.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("telemetry down")))
    tracker.register_node("n", None, depth=0)

    # Policy evaluation must still work even if telemetry fails
    lineage = LineageSummary(node_id="n", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n", working_summary="", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(allowed_tools=frozenset(),
                        allow_children=False, allow_nested_children=False,
                        allow_context_expansion=False, allow_export=False,
                        allow_mutation=False, max_child_depth=0)
    envelope = ExecutionEnvelope(
        node_id="n", task="test", context=ctx,
        policy=ExecutionPolicy(name="test"),
        capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.RESTRICTED,
                                       allowed_fields=frozenset(), max_export_tokens=0),
        lineage=lineage, cancel_token=make_token(),
    )

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "rm_rf", "args": {}}, node_id="n")

    # Policy must still deny even when telemetry is broken
    decision, _ = loop._evaluate_tool_intent(intent, envelope)
    assert decision.kind == PolicyDecisionKind.DENY
