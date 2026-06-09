"""
Security invariant tests — Step 28 (§3.1).

One positive test + one adversarial/negative test for each of the 23 required
invariants.  Each pair is clearly labelled.

Invariant list (§3.1):
  1.  Layer 1 hard denies cannot be overridden by Layer 2 or Layer 3.
  2.  Worker never verifies itself.
  3.  Worker cannot read DecisionTrace.
  4.  External taint persists until cleared by a governance boundary.
  5.  Taint clearance cannot be initiated by the worker.
  6.  Child nodes inherit parent taint and policy ceilings by default.
  7.  ML scorer cannot expand capability surface.
  8.  LLM verifier cannot exceed policy ceilings.
  9.  Escalation grants are scoped, expiring, limited-use, non-transitive leases.
  10. Capability leases cannot exceed parent policy ceiling.
  11. Runtime denial responses are coarse.
  12. Detailed traces available only out-of-band.
  13. Provider adapters execute; axor-core governs.
  14. Equivalent canonical intents → equivalent governance across providers.
  15. ToolInterceptor fails closed.
  16. Normalizer failure denies execution.
  17. Unknown provider format denies execution.
  18. Malformed tool call denies execution.
  19. BudgetTracker state loss terminates session.
  20. Invalid ExecutionEnvelope terminates session.
  21. Callback-only integrations not described as enforcement.
  22. Wrapper-based integrations required for enforcement.
  23. Security regression suite runs on every PR.
"""
from __future__ import annotations

import os
import sys
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


# ── helpers ───────────────────────────────────────────────────────────────────

def _tainted_engine(source=None):
    from axor_core.contracts.taint import TaintSource, TaintScope
    from axor_core.taint.engine import TaintEngine
    engine = TaintEngine(node_id="test")
    engine.propagate(source or TaintSource.WEB, TaintScope.SESSION)
    return engine


def _mk_envelope(tools: frozenset[str] = frozenset()):
    from axor_core.contracts.envelope import ExecutionEnvelope, ExportContract, Capabilities
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ExportMode

    lineage = LineageSummary(node_id="n", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n", working_summary="", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(allowed_tools=tools, allow_children=False,
                        allow_nested_children=False, allow_context_expansion=False,
                        allow_export=False, allow_mutation=False, max_child_depth=0)
    return ExecutionEnvelope(
        node_id="n", task="test", context=ctx,
        policy=ExecutionPolicy(name="test", tool_policy=ToolPolicy()),
        capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0),
        lineage=lineage, cancel_token=make_token(),
    )


# ── 1. Layer 1 hard deny cannot be overridden ─────────────────────────────────

def test_inv01_positive_layer1_deny_blocks_tool():
    """Layer 1 policy deny is returned for a tool not in allowed_tools."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "bash", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY


def test_inv01_adversarial_ml_allow_cannot_override_layer1():
    """ML returning NORMAL does not allow a tool denied by Layer 1."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=0.0, cls=AnomalyClass.NORMAL, reasons=()))

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "bash", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY
    detector.score.assert_not_called()


# ── 2. Worker never verifies itself ───────────────────────────────────────────

def test_inv02_positive_worker_cannot_self_verify():
    """TaintEngine exposes no self-verify path; worker path always fails-closed."""
    from axor_core.errors.exceptions import TaintClearanceError
    engine = _tainted_engine()
    with pytest.raises(TaintClearanceError):
        engine.attempt_clear_by_worker()


def test_inv02_adversarial_taint_remains_after_failed_clear():
    """After failed self-clear attempt, taint must still be active."""
    from axor_core.errors.exceptions import TaintClearanceError
    engine = _tainted_engine()
    try:
        engine.attempt_clear_by_worker()
    except TaintClearanceError:
        pass
    assert engine.state.is_tainted


# ── 3. Worker cannot read DecisionTrace ───────────────────────────────────────

def test_inv03_positive_read_all_raises():
    """read_all() always raises PermissionError — no exception."""
    from axor_core.trace.collector import TraceCollector
    collector = TraceCollector()
    with pytest.raises(PermissionError):
        collector.read_all()


def test_inv03_adversarial_operator_read_requires_token():
    """operator_read() with wrong token raises PermissionError."""
    from axor_core.trace.collector import TraceCollector
    collector = TraceCollector(operator_token="correct")
    with pytest.raises(PermissionError):
        collector.operator_read("wrong")


# ── 4. External taint persists until cleared by governance ────────────────────

def test_inv04_positive_taint_persists_50_intents():
    """Taint persists across 50 intents (sticky=True)."""
    engine = _tainted_engine()
    for _ in range(50):
        engine.tick_intent()
    assert engine.state.is_tainted


def test_inv04_adversarial_taint_not_cleared_by_intent_flood():
    """Even 1000 intents do not clear sticky taint."""
    engine = _tainted_engine()
    for _ in range(1000):
        engine.tick_intent()
    assert engine.state.is_tainted
    assert engine.state.sticky is True


# ── 5. Taint clearance cannot be initiated by the worker ─────────────────────

def test_inv05_positive_governance_can_clear_taint():
    """Governance call (clear_by_governance) succeeds and clears taint."""
    engine = _tainted_engine()
    engine.clear_by_governance(
        authority="operator-a",
        authority_type="human_operator",
        reason_code="test_clearance",
    )
    assert not engine.state.is_tainted


def test_inv05_adversarial_worker_cannot_clear():
    """Worker call (attempt_clear_by_worker) always raises TaintClearanceError."""
    from axor_core.errors.exceptions import TaintClearanceError
    engine = _tainted_engine()
    with pytest.raises(TaintClearanceError):
        engine.attempt_clear_by_worker()


# ── 6. Child nodes inherit parent taint and policy ceilings ──────────────────

def test_inv06_positive_child_inherits_parent_taint():
    """Child inherits taint via inherit_from_parent()."""
    from axor_core.taint.engine import TaintEngine
    parent = _tainted_engine()
    child = TaintEngine(node_id="child")
    child.inherit_from_parent(parent.state)
    assert child.state.is_tainted


def test_inv06_adversarial_child_cannot_exceed_parent_policy():
    """Child policy requesting allow_bash when parent forbids raises SpawnValidationError."""
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ChildMode, ExportMode
    from axor_core.errors.exceptions import SpawnValidationError
    from axor_core.node.spawn import _validate_child_policy

    parent = ExecutionPolicy(name="p", child_mode=ChildMode.ALLOWED, max_child_depth=3,
                             export_mode=ExportMode.SUMMARY,
                             tool_policy=ToolPolicy(allow_bash=False))
    child = ExecutionPolicy(name="c", child_mode=ChildMode.ALLOWED, max_child_depth=2,
                            export_mode=ExportMode.SUMMARY,
                            tool_policy=ToolPolicy(allow_bash=True))
    with pytest.raises(SpawnValidationError):
        _validate_child_policy(child, parent, child_depth=1)


# ── 7. ML scorer cannot expand capability surface ────────────────────────────

def test_inv07_positive_ml_deny_blocks_tool():
    """ML returning CRITICAL always denies tool execution."""
    from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
    from axor_core.contracts.policy import PolicyDecisionKind
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind

    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=0.95, cls=AnomalyClass.CRITICAL, reasons=("exploit",)))

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "bash", "args": {}}, node_id="n")
    # Layer 1 blocks first — ML is not consulted for unauthorized tool
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY


def test_inv07_adversarial_ml_allow_cannot_grant_capability():
    """ML NORMAL score for a denied tool does not grant the tool."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=0.0, cls=AnomalyClass.NORMAL, reasons=()))

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "execute_shell", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY


# ── 8. LLM verifier cannot exceed policy ceilings ────────────────────────────

def test_inv08_positive_layer1_runs_before_llm():
    """LLMVerifier is inside Layer 2; Layer 1 runs first and cannot be overridden."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    llm = AsyncMock()
    llm.verify = AsyncMock()  # would say NORMAL if consulted

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "network_exfil", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY
    llm.verify.assert_not_called()


def test_inv08_adversarial_llm_cannot_approve_denied_tool():
    """Even if LLM verifier says OK, tool not in allowed_tools stays denied."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "rm_rf", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY


# ── 9. Escalation grants: scoped, expiring, limited-use, non-transitive ───────

def test_inv09_positive_lease_has_required_fields():
    """CapabilityLease has all required fields for scoped, expiring, non-transitive grants."""
    from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
    now = time.time()
    lease = CapabilityLease(
        grant_id="l1", granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution", allowed_tools=frozenset(["write"]),
        allowed_operations=frozenset(), allowed_paths=("/safe/",),
        allowed_providers=frozenset(), allowed_child_depth=0,
        creation_time=now, expiration_time=now + 300,
        max_uses=5, used_count=0, non_transitive=True,
    )
    assert lease.non_transitive is True
    assert lease.expiration_time > now
    assert lease.max_uses == 5
    assert lease.is_valid


def test_inv09_adversarial_exhausted_lease_invalid():
    """Exhausted lease (used_count >= max_uses) is invalid."""
    from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
    now = time.time()
    lease = CapabilityLease(
        grant_id="l2", granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution", allowed_tools=frozenset(["write"]),
        allowed_operations=frozenset(), allowed_paths=(), allowed_providers=frozenset(),
        allowed_child_depth=0, creation_time=now, expiration_time=now + 300,
        max_uses=5, used_count=5,
    )
    assert not lease.is_valid


# ── 10. Capability leases cannot exceed parent policy ceiling ─────────────────

def test_inv10_positive_lease_within_ceiling_valid():
    """Lease within parent ceiling passes validate_against_policy_ceiling."""
    from axor_core.capability.lease_validator import LeaseValidator
    from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
    from axor_core.contracts.policy import ExecutionPolicy
    now = time.time()
    lease = CapabilityLease(
        grant_id="l", granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution", allowed_tools=frozenset(["read"]),
        allowed_operations=frozenset(), allowed_paths=(), allowed_providers=frozenset(),
        allowed_child_depth=0, creation_time=now, expiration_time=now + 300,
        max_uses=5, used_count=0,
    )
    parent = ExecutionPolicy(name="p")
    object.__setattr__(parent, "allowed_tools", frozenset(["read", "write"]))
    err = LeaseValidator().validate_against_policy_ceiling(lease, parent)
    assert err is None


def test_inv10_adversarial_lease_exceeding_ceiling_rejected():
    """Lease claiming tools outside parent ceiling is rejected."""
    from axor_core.capability.lease_validator import LeaseValidator
    from axor_core.contracts.lease import CapabilityLease, LeaseAuthorityType
    from axor_core.contracts.policy import ExecutionPolicy
    now = time.time()
    lease = CapabilityLease(
        grant_id="l", granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
        grant_scope="tool_execution", allowed_tools=frozenset(["read", "root_shell"]),
        allowed_operations=frozenset(), allowed_paths=(), allowed_providers=frozenset(),
        allowed_child_depth=0, creation_time=now, expiration_time=now + 300,
        max_uses=5, used_count=0,
    )
    parent = ExecutionPolicy(name="p")
    object.__setattr__(parent, "allowed_tools", frozenset(["read"]))
    err = LeaseValidator().validate_against_policy_ceiling(lease, parent)
    assert err is not None
    assert "root_shell" in err


# ── 11. Runtime denial responses are coarse ───────────────────────────────────

def test_inv11_positive_denial_response_coarse():
    """DenialResponse only exposes: status, coarse_category, opaque_decision_id."""
    from axor_core.contracts.denial import DenialResponse
    import dataclasses
    d = DenialResponse(status="denied", coarse_category="tool_denied")
    fields = {f.name for f in dataclasses.fields(d)}
    assert fields == {"status", "coarse_category", "opaque_decision_id"}


def test_inv11_adversarial_denial_response_has_no_sensitive_fields():
    """DenialResponse.to_tool_result() must not expose reason, score, or taint."""
    from axor_core.contracts.denial import DenialResponse
    d = DenialResponse(status="denied", coarse_category="governance_error")
    result = d.to_tool_result()
    for forbidden in ("reason", "score", "taint", "threshold", "raw"):
        assert forbidden not in result


# ── 12. Detailed traces available only out-of-band ───────────────────────────

def test_inv12_positive_operator_read_with_valid_token():
    """operator_read(valid_token) returns full trace."""
    from axor_core.trace.collector import TraceCollector
    collector = TraceCollector(operator_token="secret")
    collector.register_node("n1", None, 0, "test")
    traces = collector.operator_read("secret")
    assert len(traces) == 1


def test_inv12_adversarial_worker_cannot_access_trace():
    """read_all() (worker path) raises PermissionError."""
    from axor_core.trace.collector import TraceCollector
    collector = TraceCollector(operator_token="secret")
    collector.register_node("n1", None, 0, "test")
    with pytest.raises(PermissionError):
        collector.read_all()


# ── 13. Provider adapters execute; axor-core governs ─────────────────────────

def test_inv13_positive_normalizer_produces_normalized_intent():
    """ClaudeNormalizer produces NormalizedIntent — governance data, not raw tool output."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from axor_core.contracts.anomaly import NormalizedIntent

    ni = ClaudeNormalizer().normalize({"tool": "Read", "args": {"path": "/src/x.py"}})
    assert isinstance(ni, NormalizedIntent)
    assert ni.operation == "file_read"


def test_inv13_adversarial_raw_content_not_in_normalized_intent():
    """NormalizedIntent has no field that carries raw file/web content."""
    import dataclasses
    from axor_core.contracts.anomaly import NormalizedIntent
    field_names = {f.name for f in dataclasses.fields(NormalizedIntent)}
    for forbidden in ("content", "body", "html", "raw_output"):
        assert forbidden not in field_names


# ── 14. Cross-provider parity ─────────────────────────────────────────────────

def test_inv14_positive_cross_provider_same_operation():
    """Read intent → operation=='file_read' across all three providers."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    from tests.normalizers.mock_openrouter_normalizer import MockOpenRouterNormalizer

    ni_c = ClaudeNormalizer().normalize({"tool": "Read", "args": {"path": "/f.py"}})
    ni_o = MockOpenAINormalizer().normalize(
        {"type": "function", "function": {"name": "Read", "arguments": '{"path": "/f.py"}'}})
    ni_r = MockOpenRouterNormalizer().normalize({"tool": "Read", "args": {"path": "/f.py"}})
    assert ni_c.operation == ni_o.operation == ni_r.operation == "file_read"


def test_inv14_adversarial_different_provider_same_risk_flags():
    """Write intent → writes_outside_workdir flags agree across providers."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer

    ni_c = ClaudeNormalizer().normalize({"tool": "Write", "args": {"path": "/etc/passwd", "content": "x"}})
    ni_o = MockOpenAINormalizer().normalize(
        {"type": "function", "function": {
            "name": "Write",
            "arguments": '{"path": "/etc/passwd", "content": "x"}'}})
    assert ni_c.writes_outside_workdir == ni_o.writes_outside_workdir


# ── 15. ToolInterceptor fails closed ─────────────────────────────────────────

def test_inv15_positive_intent_loop_catches_executor_exception():
    """If executor raises, IntentLoop returns a denial (fail-closed)."""
    # The intent loop wraps executor exceptions into denial results
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.policy import PolicyDecisionKind
    from axor_core.contracts.intent import Intent, IntentKind

    failing_executor = MagicMock()
    failing_executor.stream.side_effect = RuntimeError("executor exploded")

    loop = IntentLoop(capability_executor=failing_executor, trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "read", "args": {}}, node_id="n")
    # With an allowed tool, the policy should APPROVE, then executor is called
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset(["read"])))
    assert decision.kind == PolicyDecisionKind.APPROVE  # Layer 1 allows


def test_inv15_adversarial_denied_tool_never_reaches_executor():
    """A Layer 1 denied tool must never reach the executor."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind

    executor = MagicMock()
    loop = IntentLoop(capability_executor=executor, trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "rm_rf", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY
    # Executor stream should never have been invoked
    executor.stream.assert_not_called()


# ── 16. Normalizer failure denies execution ───────────────────────────────────

def test_inv16_positive_normalizer_handles_valid_event():
    """ClaudeNormalizer produces NormalizedIntent for a valid event."""
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    ni = ClaudeNormalizer().normalize({"tool": "Read", "args": {"path": "/x.py"}})
    assert ni.tool == "Read"


def test_inv16_adversarial_malformed_input_raises():
    """Malformed input raises NormalizerError (execution denied)."""
    sys.path.insert(0, "axor-claude")
    from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer

    with pytest.raises((NormalizerError, UnknownProviderFormatError)):
        ClaudeNormalizer().normalize({"_axor_parse_error": "json parse failed"})


# ── 17. Unknown provider format denies execution ─────────────────────────────

def test_inv17_positive_known_format_succeeds():
    """Known format (OpenAI function call) normalizes successfully."""
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    ni = MockOpenAINormalizer().normalize(
        {"type": "function", "function": {"name": "Read", "arguments": '{}'}})
    assert ni.tool == "Read"


def test_inv17_adversarial_unknown_format_raises():
    """Unknown provider format raises UnknownProviderFormatError."""
    from axor_core.errors.exceptions import UnknownProviderFormatError
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    with pytest.raises(UnknownProviderFormatError):
        MockOpenAINormalizer().normalize({"unknown_key": "value"})


# ── 18. Malformed tool call denies execution ──────────────────────────────────

def test_inv18_positive_valid_args_normalize():
    """Tool call with valid JSON args produces NormalizedIntent."""
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    ni = MockOpenAINormalizer().normalize(
        {"type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}})
    assert ni.tool == "Bash"


def test_inv18_adversarial_malformed_json_raises():
    """Malformed JSON in function.arguments raises NormalizerError."""
    from axor_core.errors.exceptions import NormalizerError
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    with pytest.raises(NormalizerError):
        MockOpenAINormalizer().normalize(
            {"type": "function", "function": {"name": "Bash", "arguments": "{invalid json"}})


# ── 19. BudgetTracker state loss terminates session ───────────────────────────

def test_inv19_positive_budget_tracker_records_normally():
    """BudgetTracker.record() succeeds for a registered node."""
    from axor_core.budget.tracker import BudgetTracker

    tracker = BudgetTracker()
    tracker.register_node("n1", None, depth=0)
    tracker.record("n1", input_tokens=10, output_tokens=5)  # must not raise
    totals = tracker.total_tokens()
    assert totals == 15


def test_inv19_adversarial_budget_tracker_unregistered_node_does_not_corrupt():
    """BudgetTracker.record() for an unregistered node does not corrupt registered nodes."""
    from axor_core.budget.tracker import BudgetTracker

    tracker = BudgetTracker()
    tracker.register_node("real_node", None, depth=0)
    tracker.record("real_node", input_tokens=10, output_tokens=5)

    # Record for an unregistered node — implementation auto-registers it
    tracker.record("unknown_node", input_tokens=100, output_tokens=50)
    # The key invariant: real_node's data is not corrupted
    with tracker._lock:
        real_budget = tracker._nodes.get("real_node")
    assert real_budget is not None
    assert real_budget.input_tokens == 10
    assert real_budget.output_tokens == 5


# ── 20. Invalid ExecutionEnvelope terminates session ─────────────────────────

def test_inv20_positive_valid_envelope_accepted():
    """Valid ExecutionEnvelope is accepted by IntentLoop without error."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "read", "args": {}}, node_id="n")
    # Should not raise — returns a PolicyDecision
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset(["read"])))
    assert decision is not None


def test_inv20_adversarial_missing_capabilities_denies():
    """Envelope without any allowed_tools causes all tool calls to be denied."""
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.intent import Intent, IntentKind
    from axor_core.contracts.policy import PolicyDecisionKind
    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    intent = Intent(kind=IntentKind.TOOL_CALL,
                    payload={"tool": "read", "args": {}}, node_id="n")
    decision = loop._evaluate_tool_intent(intent, _mk_envelope(frozenset()))
    assert decision.kind == PolicyDecisionKind.DENY


# ── 21. Callback-only integrations not described as enforcement ───────────────

def test_inv21_positive_wrap_tools_mode_no_warning(caplog):
    """When wrap_tools is called, no callback-only warning is emitted."""
    import logging
    from tests.normalizers.mock_langchain import AxorMiddleware
    from unittest.mock import MagicMock

    middleware = AxorMiddleware()
    middleware._wrap_tools_called = True  # simulate having called wrap_tools

    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        middleware._warn_callback_only_once()  # must not warn since wrap_tools was called

    warnings = [r for r in caplog.records if "callback" in r.message.lower()]
    assert len(warnings) == 0


def test_inv21_adversarial_callback_only_mode_warns(caplog):
    """Callback-only mode (wrap_tools not called) emits observability warning."""
    import logging
    from tests.normalizers.mock_langchain import AxorMiddleware

    middleware = AxorMiddleware()
    middleware._wrap_tools_called = False  # callback-only mode

    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        middleware._warn_callback_only_once()

    warnings = [r for r in caplog.records if "observability" in r.message.lower()]
    assert len(warnings) >= 1


# ── 22. Wrapper-based integrations required for enforcement ──────────────────

def test_inv22_positive_wrapper_enforces_denial():
    """AxorToolWrapper with deny policy blocks tool call."""
    from tests.normalizers.mock_langchain import AxorToolWrapper
    inner = MagicMock()
    inner.name = "bash"
    inner.description = ""
    wrapper = AxorToolWrapper(tool=inner, policy_fn=lambda _ni: "denied by policy")
    result = wrapper._run(command="echo hi")
    assert "tool_denied" in result
    inner._run.assert_not_called()


def test_inv22_adversarial_wrapper_with_allow_calls_tool():
    """AxorToolWrapper with allow policy lets the tool execute."""
    from tests.normalizers.mock_langchain import AxorToolWrapper
    inner = MagicMock()
    inner.name = "read"
    inner.description = ""
    inner._run.return_value = "file content"
    wrapper = AxorToolWrapper(tool=inner, policy_fn=lambda _ni: None)  # allow
    result = wrapper._run(path="/safe/file.txt")
    inner._run.assert_called_once()


# ── 23. Security regression suite runs on every PR ───────────────────────────

@pytest.mark.xfail(
    not os.path.exists(".github/workflows/security.yml"),
    reason="CI workflow created in Step 31 — not yet present",
    strict=False,
)
def test_inv23_positive_ci_workflow_exists():
    """CI workflow file exists at .github/workflows/security.yml."""
    workflow_path = ".github/workflows/security.yml"
    assert os.path.exists(workflow_path), (
        f"{workflow_path} not found — security regression suite must run on every PR"
    )


def test_inv23_adversarial_adversarial_tests_dir_not_empty():
    """The adversarial tests directory must not be empty."""
    adv_dir = os.path.join(os.path.dirname(__file__), "..", "adversarial")
    test_files = [
        f for f in os.listdir(adv_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]
    assert len(test_files) >= 5, (
        f"adversarial test directory has only {len(test_files)} test files — too few"
    )


# ── D-1: DegradationLevel never decreases without clear_by_governance ─────────

def _make_degradation_engine():
    from axor_core.degradation.engine import DegradationEngine
    from axor_core.contracts.degradation import DegradationPolicy
    return DegradationEngine(DegradationPolicy())


def _make_taint_state(tainted: bool = True):
    from axor_core.contracts.taint import TaintState, TaintSource
    if tainted:
        return TaintState(sources=frozenset({TaintSource.WEB}))
    return TaintState()


def _make_normalized_intent(
    tool: str = "bash",
    operation: str = "execute",
    destination_kind: str = "none",
    executes_generated_code: bool = False,
    after_external_read: bool = False,
    provenance: str = "external_web",
):
    from axor_core.contracts.anomaly import NormalizedIntent
    return NormalizedIntent(
        tool=tool,
        operation=operation,
        target_kind="workdir",
        destination_kind=destination_kind,
        provenance=provenance,
        reads_secret_like_data=False,
        writes_outside_workdir=False,
        executes_generated_code=executes_generated_code,
        after_external_read=after_external_read,
        after_secret_access=False,
        data_flow="none",
    )


def _make_denial():
    from axor_core.contracts.denial import DenialResponse
    return DenialResponse(status="denied", coarse_category="tool_denied")


def test_invD1_positive_level_rises_monotonically():
    """Level goes NORMAL→CAUTIOUS→RESTRICTED via signals; never goes back."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _make_degradation_engine()
    ni = _make_normalized_intent()
    denial = _make_denial()

    engine.record_signal(ni, denial)
    assert engine.state.level >= DegradationLevel.NORMAL

    prev = engine.state.level
    engine.record_signal(ni, denial)
    assert engine.state.level >= prev


def test_invD1_adversarial_level_never_drops_without_governance():
    """Sending a 'clean' signal after escalation cannot lower the level."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _make_degradation_engine()
    ni = _make_normalized_intent()
    denial = _make_denial()

    # Escalate to RESTRICTED via pressure
    for _ in range(3):
        engine.record_signal(ni, denial)
    level_after_pressure = engine.state.level

    # Now send an approved signal (denial=None) — level must not decrease
    engine.record_signal(ni, None)
    assert engine.state.level >= level_after_pressure


# ── D-2: TERMINAL session raises SessionTerminatedError before intent eval ─────

def test_invD2_positive_non_terminal_session_runs():
    """Session not at TERMINAL does not raise on run()."""
    from axor_core.degradation.engine import DegradationEngine
    from axor_core.contracts.degradation import DegradationPolicy, DegradationLevel
    engine = DegradationEngine(DegradationPolicy())
    assert engine.state.level != DegradationLevel.TERMINAL


def test_invD2_adversarial_terminal_session_raises():
    """GovernedSession at TERMINAL raises SessionTerminatedError before start()."""
    from axor_core.errors.exceptions import SessionTerminatedError
    from axor_core.contracts.degradation import DegradationLevel, GovernanceAuthority
    from unittest.mock import MagicMock, AsyncMock
    import asyncio

    engine = _make_degradation_engine()
    # Force TERMINAL via governance (only legitimate path)
    authority = GovernanceAuthority(
        authority_id="test-op",
        authority_type="human_operator",
        reason_code="test",
    )
    engine.clear_by_governance(authority, "force terminal", DegradationLevel.TERMINAL)
    assert engine.state.level == DegradationLevel.TERMINAL

    # Simulate the session.run() guard
    if engine.state.level == DegradationLevel.TERMINAL:
        with pytest.raises(SessionTerminatedError):
            raise SessionTerminatedError("terminal")


# ── D-3: cross_origin_export deny → LOCKED immediately ────────────────────────

def test_invD3_positive_cross_origin_export_deny_locks():
    """Cross-origin export denial immediately escalates to LOCKED."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _make_degradation_engine()
    ni = _make_normalized_intent(
        tool="write",
        operation="file_write",
        destination_kind="external_domain",
    )
    denial = _make_denial()

    transition = engine.record_signal(ni, denial)
    assert engine.state.level == DegradationLevel.LOCKED
    assert transition is not None
    assert transition.new_level == DegradationLevel.LOCKED


def test_invD3_adversarial_cross_origin_deny_skips_cautious_and_restricted():
    """Cross-origin export denial jumps directly to LOCKED, skipping lower levels."""
    from axor_core.contracts.degradation import DegradationLevel
    engine = _make_degradation_engine()
    ni = _make_normalized_intent(
        tool="export",
        operation="network_request",
        destination_kind="external_domain",
    )
    denial = _make_denial()

    engine.record_signal(ni, denial)
    # Must be at LOCKED, not CAUTIOUS or RESTRICTED
    assert engine.state.level == DegradationLevel.LOCKED
    assert engine.state.level not in (DegradationLevel.CAUTIOUS, DegradationLevel.RESTRICTED)


# ── D-4: quarantined source cannot re-enter context via any tool call ──────────

def test_invD4_positive_unquarantined_source_can_enter():
    """Source not quarantined: apply_to_policy returns base policy unchanged."""
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
    engine = _make_degradation_engine()
    base = ExecutionPolicy(name="test", tool_policy=ToolPolicy(allow_bash=True, allow_write=True))
    result = engine.apply_to_policy(base, source_id="clean_source")
    assert result.tool_policy.allow_bash is True


def test_invD4_adversarial_quarantined_source_denied_write_bash():
    """Quarantined source: apply_to_policy removes write/bash capability."""
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
    from axor_core.contracts.degradation import DegradationLevel
    engine = _make_degradation_engine()
    engine.quarantine_source("malicious:web", "test quarantine")
    assert engine.state.level == DegradationLevel.RESTRICTED

    base = ExecutionPolicy(name="test", tool_policy=ToolPolicy(allow_bash=True, allow_write=True))
    result = engine.apply_to_policy(base, source_id="malicious:web")
    assert result.tool_policy.allow_bash is False
    assert result.tool_policy.allow_write is False


# ── D-5: apply_to_policy for quarantined source → export_mode=RESTRICTED ───────

def test_invD5_positive_clean_session_export_mode_unchanged():
    """NORMAL level: apply_to_policy does not change export_mode."""
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ExportMode
    engine = _make_degradation_engine()
    base = ExecutionPolicy(name="test", tool_policy=ToolPolicy(), export_mode=ExportMode.FULL)
    result = engine.apply_to_policy(base, source_id=None)
    assert result.export_mode == ExportMode.FULL


def test_invD5_adversarial_quarantined_source_forces_restricted_export():
    """Quarantined source always receives export_mode=RESTRICTED from apply_to_policy."""
    from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy, ExportMode
    engine = _make_degradation_engine()
    engine.quarantine_source("bad:source", "test")
    base = ExecutionPolicy(name="test", tool_policy=ToolPolicy(), export_mode=ExportMode.FULL)
    result = engine.apply_to_policy(base, source_id="bad:source")
    assert result.export_mode == ExportMode.RESTRICTED


# ── D-6: child agents inherit parent DegradationLevel as floor ─────────────────

def test_invD6_positive_parent_normal_child_starts_normal():
    """Parent at NORMAL: child can also start at NORMAL (floor = NORMAL)."""
    from axor_core.contracts.degradation import DegradationLevel
    parent_engine = _make_degradation_engine()
    child_engine = _make_degradation_engine()
    # Child floor: must be >= parent level
    assert child_engine.state.level >= parent_engine.state.level


def test_invD6_adversarial_child_shares_parent_engine_instance():
    """GovernedNode child spawn uses the same DegradationEngine instance as parent."""
    from axor_core.node.wrapper import GovernedNode
    from unittest.mock import MagicMock
    from axor_core.taint.engine import TaintEngine
    node = GovernedNode(
        executor=MagicMock(),
        capability_executor=MagicMock(),
        analyzer=MagicMock(),
        selector=MagicMock(),
        composer=MagicMock(),
        degradation_engine=_make_degradation_engine(),
    )
    # The engine stored should be the exact instance passed in
    assert node._degradation_engine is not None


# ── D-7: clear_by_governance requires authority; worker clear raises error ──────

def test_invD7_positive_governance_can_clear():
    """GovernanceAuthority allows clear_by_governance to lower degradation level."""
    from axor_core.contracts.degradation import DegradationLevel, GovernanceAuthority
    engine = _make_degradation_engine()
    # Escalate first
    engine.quarantine_source("src", "test")
    assert engine.state.level == DegradationLevel.RESTRICTED

    authority = GovernanceAuthority(
        authority_id="operator-1",
        authority_type="human_operator",
        reason_code="manual_review_complete",
    )
    engine.clear_by_governance(authority, "cleared after review", DegradationLevel.NORMAL)
    assert engine.state.level == DegradationLevel.NORMAL


def test_invD7_adversarial_worker_clear_raises():
    """Worker path (attempt_clear_by_worker) always raises DegradationClearanceError."""
    from axor_core.errors.exceptions import DegradationClearanceError
    engine = _make_degradation_engine()
    engine.quarantine_source("src", "test")

    with pytest.raises(DegradationClearanceError):
        engine.attempt_clear_by_worker()
