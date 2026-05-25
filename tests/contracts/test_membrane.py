"""
Tests for SummarizationMembrane contract — Step 22.

Verifies:
- SummarizationMembrane protocol importable
- ClearanceEvent has all §25.6 audit fields
- Default outcome preserves taint
- Worker-initiated summarization cannot produce clearance
"""
from __future__ import annotations

import pytest

from axor_core.contracts.membrane import (
    ClearanceEvent,
    SummarizationMembrane,
    SummarizationOutcome,
    SummarizationResult,
)


# ── SummarizationMembrane protocol ────────────────────────────────────────────

def test_summarization_membrane_importable():
    assert SummarizationMembrane is not None


def test_summarization_membrane_is_protocol():
    from typing import Protocol
    assert issubclass(SummarizationMembrane, Protocol)


# ── ClearanceEvent audit fields ────────────────────────────────────────────────

def test_clearance_event_has_all_required_fields():
    """§25.6: all audit fields must be present."""
    ev = ClearanceEvent(
        membrane_id="mem-1",
        source_taint_set=frozenset(["web"]),
        operator_authority="operator-token",
        resulting_taint_state="clean",
        reason_code="scheduled_clearance",
    )
    assert ev.membrane_id == "mem-1"
    assert ev.source_taint_set == frozenset(["web"])
    assert ev.operator_authority == "operator-token"
    assert ev.timestamp > 0
    assert ev.resulting_taint_state == "clean"
    assert ev.reason_code == "scheduled_clearance"
    assert ev.audit_id  # auto-generated


def test_clearance_event_immutable():
    ev = ClearanceEvent(
        membrane_id="m", source_taint_set=frozenset(), operator_authority="a"
    )
    with pytest.raises((AttributeError, TypeError)):
        ev.operator_authority = "other"  # type: ignore[misc]


def test_clearance_event_audit_id_unique():
    ids = {
        ClearanceEvent(
            membrane_id="m", source_taint_set=frozenset(), operator_authority="a"
        ).audit_id
        for _ in range(10)
    }
    assert len(ids) == 10


# ── SummarizationResult ────────────────────────────────────────────────────────

def test_default_outcome_preserves_taint():
    """Default summarization outcome must preserve taint."""
    result = SummarizationResult(summary="compressed context")
    assert result.outcome == SummarizationOutcome.SUMMARIZE_AND_PRESERVE_TAINT


def test_clear_taint_outcome_requires_clearance_event():
    """CLEAR_TAINT_WITH_AUTHORITY without ClearanceEvent raises ValueError."""
    with pytest.raises(ValueError, match="ClearanceEvent is required"):
        SummarizationResult(
            summary="context",
            outcome=SummarizationOutcome.SUMMARIZE_AND_CLEAR_TAINT_WITH_AUTHORITY,
            clearance_event=None,
        )


def test_clear_taint_outcome_with_clearance_event_valid():
    ev = ClearanceEvent(
        membrane_id="m",
        source_taint_set=frozenset(["web"]),
        operator_authority="op",
    )
    result = SummarizationResult(
        summary="compressed",
        outcome=SummarizationOutcome.SUMMARIZE_AND_CLEAR_TAINT_WITH_AUTHORITY,
        clearance_event=ev,
    )
    assert result.clearance_event is ev


def test_summarize_without_clearance_no_event():
    result = SummarizationResult(
        summary="compressed",
        outcome=SummarizationOutcome.SUMMARIZE_WITHOUT_CLEARANCE,
    )
    assert result.clearance_event is None
    assert result.outcome == SummarizationOutcome.SUMMARIZE_WITHOUT_CLEARANCE


# ── Worker-initiated summarization cannot produce clearance ────────────────────

def test_worker_summary_preserves_taint_by_default():
    """
    A minimal worker-like summarization — with no authority — must use
    the preserve-taint outcome, never the clearance outcome.
    """
    result = SummarizationResult(summary="worker summary")  # no authority, no clearance
    assert result.outcome != SummarizationOutcome.SUMMARIZE_AND_CLEAR_TAINT_WITH_AUTHORITY
    assert result.clearance_event is None
