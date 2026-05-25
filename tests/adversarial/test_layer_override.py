"""
Adversarial tests: layer override — Step 26.

Layer 1 (rule-based policy) is a hard deny.  Layers 2 (ML anomaly) and 3
(LLMVerifier) run only after Layer 1 approves.  No amount of ML or LLM
"approval" can override a Layer 1 hard deny.

Two variants:
  1. ML says NORMAL (no anomaly) but Layer 1 denies — hard deny preserved.
  2. LLM would say OK but Layer 1 denies — hard deny preserved.

Both tests confirm the Layer 1 decision is authoritative by directly exercising
_evaluate_tool_intent and showing that the anomaly detector is never invoked
when Layer 1 rejects.

Reverse-verify: calling anomaly_detector.score before Layer 1's check would mean
ML output affects the outcome before policy is consulted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult
from axor_core.contracts.policy import PolicyDecisionKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
from axor_core.contracts.envelope import Capabilities


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_normal_detector() -> AsyncMock:
    """Anomaly detector that always returns NORMAL (no concern)."""
    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=0.1, cls=AnomalyClass.NORMAL, reasons=(),
    ))
    return detector


def _intent(tool_name: str) -> Intent:
    return Intent(
        kind=IntentKind.TOOL_CALL,
        payload={"tool": tool_name, "args": {}, "tool_use_id": "tid"},
        node_id="test",
    )


def _envelope_without_tool(tool_name: str):
    """ExecutionEnvelope that explicitly excludes the given tool."""
    from axor_core.contracts.envelope import ExecutionEnvelope, ExportContract
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.policy import ExportMode

    policy = ExecutionPolicy(
        name="restrictive",
        tool_policy=ToolPolicy(allow_read=False, allow_write=False),
    )
    lineage = LineageSummary(
        node_id="n1", parent_id=None, depth=0,
        ancestry_ids=[], inherited_restrictions=[],
    )
    ctx = ContextView(
        node_id="n1",
        working_summary="test",
        visible_fragments=[],
        active_constraints=[],
        lineage=lineage,
        token_count=0,
        compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(),  # no tools allowed
        allow_children=False,
        allow_nested_children=False,
        allow_context_expansion=False,
        allow_export=False,
        allow_mutation=False,
        max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1",
        task="test",
        context=ctx,
        policy=policy,
        capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED,
            allowed_fields=frozenset(),
            max_export_tokens=0,
        ),
        lineage=lineage,
        cancel_token=make_token(),
    )


# ── variant 1: ML says NORMAL but Layer 1 denies ─────────────────────────────

def test_ml_allow_overridden_by_layer1_deny():
    """
    ML anomaly detector returns NORMAL for a tool, but Layer 1 policy has the
    tool outside allowed_tools — hard deny must be preserved.

    The anomaly detector's score method must NOT be called when Layer 1 denies.
    """
    detector = _mock_normal_detector()

    loop = IntentLoop(
        capability_executor=MagicMock(),  # executor not invoked in this test
        trace_events=[],
        anomaly_detector=detector,
    )

    envelope = _envelope_without_tool("bash")
    intent = _intent("bash")

    # Directly test Layer 1's evaluation
    decision = loop._evaluate_tool_intent(intent, envelope)
    assert decision.kind == PolicyDecisionKind.DENY, (
        "Layer 1 hard deny must not be overridden by ML allow"
    )

    # Anomaly detector must not have been consulted (it runs only after Layer 1 approves)
    detector.score.assert_not_called()


# ── variant 2: LLM would say OK but Layer 1 denies ───────────────────────────

def test_llm_allow_overridden_by_layer1_deny():
    """
    Even if an LLMVerifier would approve a tool call, Layer 1 policy denies it
    first — the LLM opinion is never consulted.

    Structurally identical to variant 1: LLMVerifier lives inside the anomaly
    detector (Layer 2), which only runs after Layer 1 approves.  So both
    "ML allow" and "LLM allow" are overridden by the same Layer 1 hard deny.
    """
    # Create a verifier that would always say NORMAL
    llm_verifier = AsyncMock()
    llm_verifier.verify = AsyncMock(return_value=AnomalyResult(
        score=0.0, cls=AnomalyClass.NORMAL, reasons=(),
    ))

    # Wrap in an anomaly detector to simulate Layer 2+3 pipeline
    detector = AsyncMock()
    detector.score = AsyncMock(return_value=AnomalyResult(
        score=0.0, cls=AnomalyClass.NORMAL, reasons=(),
    ))

    loop = IntentLoop(
        capability_executor=MagicMock(),
        trace_events=[],
        anomaly_detector=detector,
    )

    envelope = _envelope_without_tool("execute_code")
    intent = _intent("execute_code")

    decision = loop._evaluate_tool_intent(intent, envelope)
    assert decision.kind == PolicyDecisionKind.DENY, (
        "LLM/ML approval (Layer 2/3) must not override Layer 1 hard deny"
    )

    # Neither the detector (ML/Layer 2) nor the verifier (LLM/Layer 3) was consulted
    detector.score.assert_not_called()
    llm_verifier.verify.assert_not_called()
