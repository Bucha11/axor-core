"""
Adversarial tests: trace side channel — Step 26.

Worker code must never be able to recover sensitive trace details:
  1. Direct: read_all() always raises PermissionError
  2. Indirect: ExecutionEnvelope contains no TraceCollector reference
  3. Denial-reason probing: DenialResponse exposes only coarse_category — no
     raw reasons, scores, or taint history

Reverse-verify:
  - Removing TraceAccessGuard (read_all always allowed) breaks variant 1.
  - Adding a TraceCollector reference to ExecutionEnvelope breaks variant 2.
  - Exposing raw reason in DenialResponse breaks variant 3.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.denial import DenialResponse
from axor_core.trace.collector import TraceCollector
from axor_core.contracts.taint import TaintSource


# ── variant 1: direct trace read blocked ─────────────────────────────────────

def test_direct_trace_read_always_blocked():
    """
    read_all() must raise PermissionError — no token, no access.

    A worker that somehow gets a reference to the collector cannot dump the trace.
    """
    collector = TraceCollector()
    with pytest.raises(PermissionError):
        collector.read_all()


def test_direct_trace_read_blocked_even_after_events():
    """
    Even after recording many events, read_all() must still be blocked.
    """
    collector = TraceCollector(operator_token="op-secret")
    collector.register_node(node_id="n1", parent_id=None, depth=0, policy_name="test")
    with pytest.raises(PermissionError):
        collector.read_all()


# ── variant 2: indirect probing via ExecutionEnvelope ─────────────────────────

def test_execution_envelope_has_no_trace_collector_field():
    """
    ExecutionEnvelope must not contain a TraceCollector field or annotation.

    A worker receives the envelope — if the collector were in it, the worker
    could extract the full trace via indirect access.
    """
    from axor_core.contracts.envelope import ExecutionEnvelope
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ExecutionEnvelope)}
    for name in field_names:
        assert "collector" not in name.lower(), (
            f"ExecutionEnvelope.{name} exposes trace collector to worker"
        )
        assert "trace" not in name.lower() or name == "trace_id", (
            f"ExecutionEnvelope.{name} may expose trace access (only trace_id is allowed)"
        )

    # Also verify type annotations carry no TraceCollector
    annotations = getattr(ExecutionEnvelope, "__annotations__", {})
    for name, ann in annotations.items():
        ann_str = str(ann)
        assert "TraceCollector" not in ann_str, (
            f"ExecutionEnvelope.{name}: {ann_str!r} — TraceCollector must not be in worker envelope"
        )


# ── variant 3: denial-reason probing ─────────────────────────────────────────

def test_denial_response_exposes_no_raw_reason():
    """
    DenialResponse must not expose raw denial reasons, scores, taint sources,
    or threshold values.

    The worker receives only: status, coarse_category, opaque_decision_id.
    """
    denial = DenialResponse(
        status="denied",
        coarse_category="governance_error",
    )
    tool_result = denial.to_tool_result()

    # Allowed keys
    assert "error" in tool_result or "status" in tool_result
    assert "category" in tool_result
    assert "decision_id" in tool_result

    # Forbidden keys — raw internal details
    for forbidden in ("reason", "score", "taint", "threshold", "sources", "raw", "trace"):
        assert forbidden not in tool_result, (
            f"DenialResponse.to_tool_result() exposes sensitive field: {forbidden!r}"
        )

    # decision_id must be opaque (hex, not a description)
    decision_id = tool_result["decision_id"]
    assert len(decision_id) > 0
    import re
    assert re.fullmatch(r"[0-9a-f]+", decision_id), (
        f"decision_id {decision_id!r} is not an opaque hex id — may leak reason"
    )


def test_denial_response_coarse_category_only():
    """
    coarse_category must be one of the allowed coarse values, not a detailed reason.
    """
    allowed_categories = {"tool_denied", "spawn_denied", "export_denied", "governance_error"}
    for cat in allowed_categories:
        d = DenialResponse(status="denied", coarse_category=cat)
        result = d.to_tool_result()
        assert result["category"] == cat
