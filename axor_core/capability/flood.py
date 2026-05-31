from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass
class EscalationGroup:
    """Deduplicated group of repeated gray-zone escalation intents."""
    representative_intent_hash: str
    count: int
    first_seen: float
    last_seen: float
    risk_class: str
    session_id: str
    node_id: str


class EscalationFloodGuard:
    """
    Protects against ask_human becoming an operator DoS vector.

    Enforces:
    - max total escalations per session
    - max escalations per minute (rate limit)
    - identical-intent deduplication into EscalationGroup
    - cooldown period after repeated approvals of same risk class
    - automatic deny after N repeated suspicious escalations
    """

    def __init__(
        self,
        max_per_session: int = 20,
        max_per_minute: int = 5,
        max_identical: int = 3,
        cooldown_after_approvals: int = 5,
        cooldown_seconds: float = 60.0,
        auto_deny_after_suspicious: int = 10,
    ) -> None:
        self._max_per_session = max_per_session
        self._max_per_minute = max_per_minute
        self._max_identical = max_identical
        self._cooldown_after_approvals = cooldown_after_approvals
        self._cooldown_seconds = cooldown_seconds
        self._auto_deny_after_suspicious = auto_deny_after_suspicious

        self._total_count: int = 0
        self._recent_timestamps: list[float] = []
        self._groups: dict[str, EscalationGroup] = {}
        self._approval_count: int = 0
        self._last_approval_time: float = 0.0
        self._suspicious_count: int = 0

    def check(
        self,
        tool: str,
        paths: list[str],
        reason: str,
        risk_class: str = "gray_zone",
        session_id: str = "",
        node_id: str = "",
    ) -> str | None:
        """
        Check whether this escalation should be allowed to proceed to the operator.

        Returns None if allowed, or a denial reason string if it should be blocked.
        """
        now = time.monotonic()

        # 1. Session limit
        if self._total_count >= self._max_per_session:
            return f"escalation session limit reached ({self._max_per_session})"

        # 2. Rate limit (per minute)
        cutoff = now - 60.0
        self._recent_timestamps = [t for t in self._recent_timestamps if t > cutoff]
        if len(self._recent_timestamps) >= self._max_per_minute:
            return f"escalation rate limit: {self._max_per_minute} per minute exceeded"

        # 3. Cooldown after many approvals
        if (
            self._approval_count >= self._cooldown_after_approvals
            and now - self._last_approval_time < self._cooldown_seconds
        ):
            return f"escalation cooldown active after {self._approval_count} approvals"

        # 4. Auto-deny after repeated suspicious
        if self._suspicious_count >= self._auto_deny_after_suspicious:
            return f"auto-denied: {self._suspicious_count} suspicious escalation patterns detected"

        # 5. Deduplication
        key = self._intent_key(tool, paths, reason)
        group = self._groups.get(key)
        if group is not None:
            group.count += 1
            group.last_seen = time.time()
            if group.count > self._max_identical:
                return (
                    f"escalation deduplicated: identical intent seen {group.count} times "
                    f"(limit {self._max_identical})"
                )
        else:
            self._groups[key] = EscalationGroup(
                representative_intent_hash=key,
                count=1,
                first_seen=time.time(),
                last_seen=time.time(),
                risk_class=risk_class,
                session_id=session_id,
                node_id=node_id,
            )

        self._recent_timestamps.append(now)
        self._total_count += 1
        if risk_class in ("suspicious", "gray_zone"):
            self._suspicious_count += 1
        return None

    def record_approval(self) -> None:
        """Call when an escalation is approved by the operator."""
        self._approval_count += 1
        self._last_approval_time = time.monotonic()

    @staticmethod
    def _intent_key(tool: str, paths: list[str], reason: str) -> str:
        raw = f"{tool}|{sorted(paths)}|{reason}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
