"""
Tests for DenialResponse — Step 20.

Worker receives only: status, coarse_category, opaque_decision_id.
No raw reason, layer scores, thresholds, trigger features, or taint history.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.denial import DenialResponse
from axor_core.node.intent_loop import _classify_denial


# ── DenialResponse dataclass ───────────────────────────────────────────────────

def test_denial_response_fields():
    dr = DenialResponse(status="denied", coarse_category="tool_denied")
    assert dr.status == "denied"
    assert dr.coarse_category == "tool_denied"
    assert dr.opaque_decision_id  # auto-generated


def test_denial_response_to_tool_result():
    dr = DenialResponse(status="denied", coarse_category="tool_denied", opaque_decision_id="abc123")
    result = dr.to_tool_result()
    assert result["error"] == "denied"
    assert result["category"] == "tool_denied"
    assert result["decision_id"] == "abc123"


def test_denial_response_no_raw_reason():
    """to_tool_result must not contain raw denial reason or sensitive fields."""
    dr = DenialResponse(status="denied", coarse_category="governance_error")
    result = dr.to_tool_result()
    forbidden = {"reason", "score", "threshold", "taint", "layer", "features"}
    for field in forbidden:
        assert field not in result, f"sensitive field {field!r} in DenialResponse.to_tool_result()"


def test_denial_response_immutable():
    dr = DenialResponse(status="denied", coarse_category="tool_denied")
    with pytest.raises((AttributeError, TypeError)):
        dr.status = "allowed"  # type: ignore[misc]


# ── _classify_denial ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("reason,expected", [
    ("tool 'write_file' is not in capabilities", "tool_denied"),
    ("spawn_child denied: max depth reached", "spawn_denied"),
    ("child spawn not allowed by policy", "spawn_denied"),
    ("export restriction: filtered mode", "export_denied"),
    ("anomaly detector: CRITICAL score=0.92", "governance_error"),
    ("auto-denied: suspicious escalation", "governance_error"),
    ("escalation rate limit exceeded", "tool_denied"),  # generic fallback
])
def test_classify_denial(reason: str, expected: str):
    assert _classify_denial(reason) == expected


# ── decision_id uniqueness ─────────────────────────────────────────────────────

def test_each_denial_has_unique_decision_id():
    ids = {DenialResponse(status="denied", coarse_category="tool_denied").opaque_decision_id
           for _ in range(20)}
    assert len(ids) == 20, "opaque_decision_id should be unique per denial"
