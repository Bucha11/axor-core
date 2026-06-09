"""Consequence axis (TM3.1) — content-blind gate on the action class.

Covers: the deterministic lookup (T0), the OpenClaw class (consequential action
under trusted provenance is caught while the provenance axes are silent), the
no-over-gating default, operator ceiling tightening, and the governance gate.
"""

from __future__ import annotations

import pytest

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop, _GrantedEscalation
from axor_core.policy.consequence import consequence_class


# ── lookup (T0) ────────────────────────────────────────────────────────────────

def test_consequence_lookup_is_deterministic_and_content_blind():
    # Same input → same output, every time (T0: no model, no randomness).
    for _ in range(3):
        assert consequence_class("shutdown") == ConsequenceClass.CATASTROPHIC
        assert consequence_class("restart_gateway") == ConsequenceClass.CATASTROPHIC
        assert consequence_class("bash") == ConsequenceClass.CONSEQUENTIAL
        assert consequence_class("write") == ConsequenceClass.REVERSIBLE
        assert consequence_class("read") == ConsequenceClass.BENIGN
    # Unknown sinks → CONSEQUENTIAL (sits at the default ceiling).
    assert consequence_class("totally_unknown_sink") == ConsequenceClass.CONSEQUENTIAL


def test_consequence_ordering():
    assert ConsequenceClass.BENIGN < ConsequenceClass.REVERSIBLE
    assert ConsequenceClass.REVERSIBLE < ConsequenceClass.CONSEQUENTIAL
    assert ConsequenceClass.CONSEQUENTIAL < ConsequenceClass.CATASTROPHIC


# ── gate logic (_check_consequence) ──────────────────────────────────────────────

def _envelope(allowed: set[str], ceiling: ConsequenceClass | None = None) -> ExecutionEnvelope:
    tp = ToolPolicy(allow_read=True, allow_write=True, allow_bash=True)
    kw = {}
    if ceiling is not None:
        kw["max_unattended_consequence"] = ceiling
    policy = ExecutionPolicy(name="t", tool_policy=tp, **kw)
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(
        node_id="n1", working_summary="t", visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(allowed), allow_children=False, allow_nested_children=False,
        allow_context_expansion=False, allow_export=False, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0),
        lineage=lineage, cancel_token=make_token(),
    )


def _loop(cap_executor) -> IntentLoop:
    return IntentLoop(capability_executor=cap_executor, trace_events=[])


def test_catastrophic_gated_under_default_ceiling(cap_executor):
    loop = _loop(cap_executor)
    env = _envelope({"shutdown"})
    reason = loop._check_consequence("shutdown", env)
    assert reason is not None and "CATASTROPHIC" in reason


def test_consequential_allowed_under_default_ceiling(cap_executor):
    # No over-gating: bash (CONSEQUENTIAL) is within the default ceiling.
    loop = _loop(cap_executor)
    assert loop._check_consequence("bash", _envelope({"bash"})) is None


def test_lowered_ceiling_gates_consequential(cap_executor):
    loop = _loop(cap_executor)
    env = _envelope({"bash"}, ceiling=ConsequenceClass.REVERSIBLE)
    assert loop._check_consequence("bash", env) is not None  # now gated


def test_governance_gate_admits_catastrophic(cap_executor):
    # An active escalation grant (human/operator path) satisfies the governance gate.
    loop = _loop(cap_executor)
    env = _envelope({"shutdown"})
    assert loop._check_consequence("shutdown", env) is not None
    loop._granted_escalations["shutdown"] = _GrantedEscalation(tool="shutdown", paths=[], ops_remaining=1)
    assert loop._check_consequence("shutdown", env) is None


# ── OpenClaw (X5): consequential action under trusted provenance ─────────────────

@pytest.mark.asyncio
async def test_openclaw_shutdown_denied_with_provenance_axes_silent(cap_executor):
    """A `shutdown` driven by a trusted user — no taint, no untrusted source — is
    invisible to the provenance axes but CATASTROPHIC by action class, so the
    consequence gate denies it (X5). No taint/degradation engine is wired, proving
    the catch is content-blind and provenance-independent.
    """
    from axor_core.contracts.trace import TraceEventKind

    trace: list = []
    loop = IntentLoop(capability_executor=cap_executor, trace_events=trace)
    env = _envelope({"shutdown"})

    async def _stream():
        yield ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "shutdown", "args": {}, "tool_use_id": "t"},
            node_id=env.node_id,
        )
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id=env.node_id)

    out = [ev async for ev in loop.run(_stream(), env)]
    results = [e for e in out if e.kind == ExecutorEventKind.TEXT]
    assert results[0].payload.get("approved") is False
    denied = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
    assert denied and "consequence gate" in denied[0].reason


@pytest.mark.asyncio
async def test_benign_admin_restart_is_the_accepted_structural_fp(cap_executor):
    """The structural FP (TM3.1): a *benign* admin restart is gated identically to
    a malicious one — the projection is structural and cannot see intent. Accepted
    by K0; the operator relaxes it via the governance gate (escalation/lease), not
    by reading content.
    """
    loop = _loop(cap_executor)
    env = _envelope({"restart_gateway"})
    # Benign or malicious — same structural verdict.
    assert loop._check_consequence("restart_gateway", env) is not None
    # Relief is structural/governed, never content-based:
    loop._granted_escalations["restart_gateway"] = _GrantedEscalation(
        tool="restart_gateway", paths=[], ops_remaining=1
    )
    assert loop._check_consequence("restart_gateway", env) is None
