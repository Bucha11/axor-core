from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


class DegradationLevel(IntEnum):
    NORMAL     = 0  # baseline — full policy in effect
    CAUTIOUS   = 1  # one suspicious signal — document soft-blocked
    RESTRICTED = 2  # source in quarantine — tool surface narrowed
    LOCKED     = 3  # tools frozen — read + escalate only
    TERMINAL   = 4  # session stopped


@dataclass
class SourceRecord:
    """Per-source pressure tracking."""
    source_id: str
    taint_source: str                  # TaintSource.value string — avoids circular import
    tool_pressure_count: int = 0       # denials where tool is write/bash/export
    instruction_pressure_count: int = 0  # instruction-takeover signals
    quarantined: bool = False
    first_seen: float = field(default_factory=time.time)
    last_signal: float = field(default_factory=time.time)


@dataclass
class DegradationState:
    """Session-level degradation state owned by DegradationEngine."""
    level: DegradationLevel = DegradationLevel.NORMAL
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    session_deny_count: int = 0
    level_history: list[tuple[float, DegradationLevel, str]] = field(default_factory=list)
    tools_frozen: bool = False
    escalation_triggered: bool = False


@dataclass(frozen=True)
class DegradationPolicy:
    """Configurable thresholds for DegradationEngine."""
    SOURCE_TOOL_PRESSURE_THRESHOLD: int = 2
    SOURCE_INSTR_PRESSURE_THRESHOLD: int = 1
    SESSION_DENY_THRESHOLD: int = 5
    LOCKED_TTL: float = 300.0  # seconds before auto-TERMINAL


@dataclass(frozen=True)
class DegradationTransition:
    """Returned by record_signal when a level change occurs."""
    previous_level: DegradationLevel
    new_level: DegradationLevel
    trigger_source_id: str | None
    trigger_intent: str
    reason: str
    timestamp: float


@dataclass(frozen=True)
class GovernanceAuthority:
    """
    Authority object required for governance-level operations on DegradationEngine.
    Must be supplied explicitly; worker path may not instantiate this.
    """
    authority_id: str
    authority_type: str     # "human_operator" | "automated_policy" | etc.
    reason_code: str
    audit_id: str = ""
