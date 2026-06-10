"""
Adversarial tests: escalation flood.

An adversary tries to overwhelm the operator channel via repeated or mutated
gray-zone escalation requests, hoping one slips through or operator fatigue sets
in.

Three variants:
  1. Identical spam — deduplication blocks after max_identical
  2. Mutated spam — the per-minute rate limit kicks in across different intents
  3. Mixed-risk spam — a repeated suspicious pattern triggers auto-deny

Without EscalationFloodGuard (i.e. if check always returned None), all three
tests would fail.
"""
from __future__ import annotations

from axor_core.capability.flood import EscalationFloodGuard


# ── variant 1: identical spam → dedup ─────────────────────────────────────────

def test_flood_identical_spam_deduped():
    """
    Identical tool + paths + reason repeated more than max_identical times
    must be blocked by deduplication — not forwarded to the operator.
    """
    guard = EscalationFloodGuard(max_identical=3, max_per_minute=100, max_per_session=100)

    tool = "Write"
    paths = ["/etc/cron.d/backdoor"]
    reason = "write to cron"

    results = []
    for _ in range(5):
        result = guard.check(tool=tool, paths=paths, reason=reason, risk_class="gray_zone")
        results.append(result)

    # First 3 should be allowed (count 1, 2, 3); count 4 and 5 exceed max_identical
    assert results[0] is None, "first escalation should be allowed"
    assert results[1] is None, "second escalation should be allowed"
    assert results[2] is None, "third escalation should be allowed"
    assert results[3] is not None, "4th identical should be blocked by dedup"
    assert "deduplicated" in results[3]


# ── variant 2: mutated spam → rate limit ──────────────────────────────────────

def test_flood_mutated_spam_rate_limited():
    """
    Different intents submitted in rapid succession must hit the rate limit.
    Adversary mutates paths/reasons slightly to avoid dedup.
    """
    guard = EscalationFloodGuard(
        max_per_minute=3,
        max_identical=100,
        max_per_session=100,
    )

    # Submit 4 distinct escalations
    results = [
        guard.check(tool="Bash", paths=[f"/tmp/cmd_{i}"], reason=f"run cmd {i}", risk_class="gray_zone")
        for i in range(4)
    ]

    # First max_per_minute allowed, beyond that rate-limited
    allowed = [r for r in results if r is None]
    denied = [r for r in results if r is not None]
    assert len(allowed) == 3, f"expected 3 allowed, got {len(allowed)}"
    assert len(denied) == 1, f"expected 1 rate-limited, got {len(denied)}"
    assert "rate limit" in denied[0]


# ── variant 3: mixed-risk spam → auto-deny ────────────────────────────────────

def test_flood_mixed_risk_auto_deny():
    """
    Repeated suspicious-class escalations trigger auto-deny after threshold.
    Adversary mixes requests to avoid dedup but all are suspicious-risk.
    """
    guard = EscalationFloodGuard(
        auto_deny_after_suspicious=3,
        max_per_minute=100,
        max_per_session=100,
        max_identical=100,
    )

    # Submit suspicious escalations until auto-deny kicks in
    results = []
    for i in range(5):
        result = guard.check(
            tool="NetworkRequest",
            paths=[f"http://c2.example.com/beacon/{i}"],
            reason=f"exfil attempt {i}",
            risk_class="suspicious",
        )
        results.append(result)

    # After auto_deny_after_suspicious=3 suspicious escalations, next must be denied
    allowed = [r for r in results if r is None]
    denied = [r for r in results if r is not None and "auto-denied" in r]
    assert len(allowed) == 3, f"expected 3 allowed before auto-deny, got {len(allowed)}"
    assert len(denied) >= 1, "expected at least one auto-denied result"
