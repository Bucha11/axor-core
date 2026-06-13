"""
Tests for the DecisionTrace access guard.

(1) collector.read_all() without token raises PermissionError
(2) collector.operator_read(valid_token) returns full trace
(3) ExecutionEnvelope contains no trace reference
"""
from __future__ import annotations

import pytest

from axor_core.trace.collector import TraceCollector
from axor_core.trace.guard import TraceAccessGuard


# ── TraceAccessGuard ───────────────────────────────────────────────────────────

def test_guard_valid_token():
    guard = TraceAccessGuard(operator_token="secret-token")
    assert guard.validate("secret-token")


def test_guard_invalid_token():
    guard = TraceAccessGuard(operator_token="correct-token")
    assert not guard.validate("wrong-token")


def test_guard_require_raises_on_invalid():
    guard = TraceAccessGuard(operator_token="secret")
    with pytest.raises(PermissionError, match="trace access denied"):
        guard.require("wrong")


def test_guard_require_passes_on_valid():
    guard = TraceAccessGuard(operator_token="secret")
    guard.require("secret")  # must not raise


# ── TraceCollector.read_all() ──────────────────────────────────────────────────

def test_read_all_always_raises():
    """read_all() without token always raises PermissionError."""
    collector = TraceCollector()
    with pytest.raises(PermissionError):
        collector.read_all()


def test_read_all_raises_even_with_empty_traces():
    collector = TraceCollector()
    with pytest.raises(PermissionError):
        collector.read_all()


# ── TraceCollector.operator_read() ────────────────────────────────────────────

def test_operator_read_valid_token_returns_traces():
    collector = TraceCollector(operator_token="my-operator-token")
    collector.register_node(node_id="node1", parent_id=None, depth=0, policy_name="test")
    traces = collector.operator_read("my-operator-token")
    assert isinstance(traces, list)
    assert len(traces) == 1


def test_operator_read_invalid_token_raises():
    collector = TraceCollector(operator_token="valid-token")
    with pytest.raises(PermissionError):
        collector.operator_read("wrong-token")


def test_operator_read_empty_returns_empty_list():
    collector = TraceCollector(operator_token="tok")
    traces = collector.operator_read("tok")
    assert traces == []


# ── ExecutionEnvelope has no trace reference ───────────────────────────────────

def test_execution_envelope_has_no_trace_reference():
    """Executor's ExecutionEnvelope must not contain a reference to TraceCollector."""
    from axor_core.contracts.envelope import ExecutionEnvelope

    # Check the fields of ExecutionEnvelope
    fields = {
        f.name: f for f in
        getattr(ExecutionEnvelope, '__dataclass_fields__', {}).values()
    }
    for name, field in fields.items():
        field_type = str(field.type)
        assert "TraceCollector" not in field_type, (
            f"ExecutionEnvelope.{name} exposes TraceCollector — workers should never have trace access"
        )

    # Also check the class annotations
    annotations = getattr(ExecutionEnvelope, '__annotations__', {})
    for name, ann in annotations.items():
        ann_str = str(ann)
        assert "TraceCollector" not in ann_str and "collector" not in name.lower(), (
            f"ExecutionEnvelope.{name} annotation {ann_str!r} may expose trace access"
        )
