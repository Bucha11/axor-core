from __future__ import annotations

import pytest

from axor_core.capability.flood import EscalationFloodGuard


def _guard(**kwargs) -> EscalationFloodGuard:
    defaults = dict(
        max_per_session=5,
        max_per_minute=3,
        max_identical=2,
        cooldown_after_approvals=3,
        cooldown_seconds=60.0,
        auto_deny_after_suspicious=4,
    )
    defaults.update(kwargs)
    return EscalationFloodGuard(**defaults)


def test_first_escalation_allowed():
    guard = _guard()
    result = guard.check("write", [], "need to write", "gray_zone")
    assert result is None


def test_session_limit_triggers():
    guard = _guard(max_per_session=3)
    for i in range(3):
        assert guard.check(f"tool_{i}", [], "reason", "gray_zone") is None
    result = guard.check("tool_extra", [], "reason", "gray_zone")
    assert result is not None
    assert "session limit" in result


def test_rate_limit_triggers():
    guard = _guard(max_per_minute=2, max_per_session=100)
    assert guard.check("write", [], "r1", "gray_zone") is None
    assert guard.check("read", [], "r2", "gray_zone") is None
    result = guard.check("bash", [], "r3", "gray_zone")
    assert result is not None
    assert "rate limit" in result


def test_identical_intent_deduplication():
    guard = _guard(max_identical=2, max_per_session=100, max_per_minute=100)
    assert guard.check("write", ["/tmp"], "need", "gray_zone") is None
    assert guard.check("write", ["/tmp"], "need", "gray_zone") is None
    result = guard.check("write", ["/tmp"], "need", "gray_zone")
    assert result is not None
    assert "deduplicated" in result


def test_escalation_group_deduplication():
    guard = _guard(max_per_session=100, max_per_minute=100, max_identical=1)
    r1 = guard.check("write", [], "need write", "gray_zone")
    r2 = guard.check("write", [], "need write", "gray_zone")
    # First check returns None (proceed); second exceeds max_identical=1 and is deduplicated
    assert r1 is None
    assert r2 is not None and "deduplicated" in r2


def test_repeated_suspicious_auto_deny():
    guard = _guard(
        max_per_session=100,
        max_per_minute=100,
        max_identical=100,
        auto_deny_after_suspicious=3,
    )
    for i in range(3):
        guard.check(f"tool_{i}", [], f"reason_{i}", "suspicious")
    result = guard.check("tool_4", [], "reason_4", "suspicious")
    assert result is not None
    assert "auto-denied" in result


def test_different_tools_not_deduplicated():
    guard = _guard(max_identical=1, max_per_session=100, max_per_minute=100)
    assert guard.check("write", [], "need", "gray_zone") is None
    # Second call for same intent hits dedup limit
    r = guard.check("write", [], "need", "gray_zone")
    assert r is not None
    # Different tool is not deduplicated
    assert guard.check("bash", [], "need", "gray_zone") is None
