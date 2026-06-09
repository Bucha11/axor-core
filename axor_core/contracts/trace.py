from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from axor_core.contracts.anomaly import AnomalyClass
from axor_core.contracts.policy import TaskSignal


def _default_trace_dir() -> str:
    return os.environ.get("AXOR_TRACE_DIR", "~/.axor/traces")


class TraceEventKind(str, Enum):
    # policy
    SIGNAL_CHOSEN = "signal_chosen"
    POLICY_CHOSEN = "policy_chosen"
    POLICY_ADJUSTED = "policy_adjusted"

    # intents
    INTENT_APPROVED = "intent_approved"
    INTENT_DENIED = "intent_denied"
    INTENT_TRANSFORMED = "intent_transformed"

    # federation
    CHILD_SPAWNED = "child_spawned"
    CHILD_COMPLETED = "child_completed"

    # context
    CONTEXT_COMPRESSED = "context_compressed"

    # budget
    TOKENS_SPENT = "tokens_spent"

    # commands
    COMMAND_ROUTED = "command_routed"

    # extensions
    EXTENSION_LOADED = "extension_loaded"
    PLUGIN_DENIED = "plugin_denied"

    # policy escalation
    ESCALATION_GRANTED = "escalation_granted"
    ESCALATION_DENIED = "escalation_denied"

    # anomaly detection
    ANOMALY_FLAGGED = "anomaly_flagged"  # intent scored SUSPICIOUS or CRITICAL

    # cancellation
    CANCELLED = "cancelled"

    # ── Added in 0.5.0: adapter observability ─────────────────────────────────
    # Emitted by adapters that support prefix-caching (e.g. axor-openrouter).
    CACHE_HIT = "cache_hit"  # tokens served from cache
    CACHE_MISS = "cache_miss"  # cache miss — full prompt processed
    CACHE_WRITE = "cache_write"  # breakpoint written to cache

    # Emitted by adapters when a routing decision is made (model, provider, tier).
    ROUTING_DECISION = "routing_decision"

    # Emitted by BudgetPolicyEngine when spend/cap crosses a threshold.
    COST_THRESHOLD = "cost_threshold"

    # ── Taint events ──────────────────────────────────────────────────────────
    TAINT_PROPAGATED = "taint_propagated"
    TAINT_CLEARANCE_ATTEMPTED = "taint_clearance_attempted"
    TAINT_CLEARED = "taint_cleared"

    # ── Degradation events ────────────────────────────────────────────────────
    DEGRADATION_TRANSITION = "degradation_transition"
    SOURCE_QUARANTINED = "source_quarantined"

    # ── Density (TM3.3) ────────────────────────────────────────────────────────
    # Emitted per high-stakes sink firing: did the driving value carry taint?
    # Aggregated into the per-value density metric (the make-or-break number).
    SINK_DENSITY = "sink_density"

    # ── Detection-register events (TM7) ───────────────────────────────────────
    # Emitted by the detection layer (reputation / anomaly). Detection NEVER
    # gates `allow` directly (would break T1); it records telemetry and may feed
    # degradation as a tightening-only crossing-fact (TM7.1).
    DETECTION_SIGNAL = "detection_signal"


@dataclass(frozen=True)
class TraceEvent:
    """Base for all trace events."""

    kind: TraceEventKind
    node_id: str
    sequence: int
    payload: dict[str, Any] = field(default_factory=dict)


# ── Typed events ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalChosenEvent(TraceEvent):
    raw_input: str = ""
    signal: TaskSignal | None = None
    confidence: float = 0.0
    classifier: str = "heuristic"
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyAdjustedEvent(TraceEvent):
    original_signal: TaskSignal | None = None
    adjusted_signal: TaskSignal | None = None
    reason: str = ""
    tokens_spent_before_adjustment: int = 0


@dataclass(frozen=True)
class IntentDeniedEvent(TraceEvent):
    intent_kind: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ChildSpawnedEvent(TraceEvent):
    child_node_id: str = ""
    child_depth: int = 0
    context_fraction: float = 0.0


@dataclass(frozen=True)
class TokensSpentEvent(TraceEvent):
    input_tokens: int = 0
    output_tokens: int = 0
    tool_tokens: int = 0
    context_tokens: int = 0
    cumulative: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class CommandRoutedEvent(TraceEvent):
    command_name: str = ""
    command_class: str = ""
    allowed: bool = True


@dataclass(frozen=True)
class PluginDeniedEvent(TraceEvent):
    plugin_name: str = ""
    denied_item: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SuspiciousIntentEvent(TraceEvent):
    """
    Emitted when an intent scores SUSPICIOUS or CRITICAL.

    SUSPICIOUS: intent was allowed but flagged.ss
    CRITICAL:   intent was denied by the anomaly detector.
    """

    tool: str = ""
    score: float = 0.0
    anomaly_class: AnomalyClass = AnomalyClass.NORMAL
    reasons: tuple[str, ...] = field(default_factory=tuple)
    policy_action: str = ""  # "flagged" | "denied"
    provenance: str = ""  # NormalizedIntent.provenance value


@dataclass(frozen=True)
class CancelledEvent(TraceEvent):
    reason: str = ""
    detail: str = ""
    completed_intents: int = 0


@dataclass(frozen=True)
class EscalationDeniedEvent(TraceEvent):
    tool: str = ""
    reason: str = ""


@dataclass(frozen=True)
class EscalationGrantedEvent(TraceEvent):
    tool: str = ""
    paths: tuple[str, ...] = field(default_factory=tuple)
    max_ops: int = 0
    reason: str = ""
    auto_approved: bool = True


# ── New in 0.5.0: adapter observability events ─────────────────────────────────


@dataclass(frozen=True)
class CacheEvent(TraceEvent):
    """
    Emitted by caching-aware adapters (e.g. axor-openrouter) to record
    whether a prefix-cache was hit, missed, or written.

    hit=True  → tokens were served from cache (CACHE_HIT kind)
    hit=False → cache miss or write (CACHE_MISS / CACHE_WRITE kind)

    Pairs of CACHE_WRITE followed by CACHE_HIT events across turns allow
    downstream tools to compute cache efficiency and guide TTL tuning.
    """

    hit: bool = True
    tokens: int = 0  # tokens involved in this cache event
    ttl: int = 0  # TTL in seconds (0 = provider default)
    breakpoint_block: str = ""  # "system" | "tools" | "context_top_k"


@dataclass(frozen=True)
class RoutingEvent(TraceEvent):
    """
    Emitted by routing-aware adapters when a model/provider is selected.

    Enables post-session analysis of:
    - which tier handled each depth
    - how often fallbacks were triggered
    - whether sort strategy correlated with cost savings
    """

    provider: str = ""
    model: str = ""
    tier: int = 0  # 0 = root/best, higher = cheaper
    sort_strategy: str = ""  # "price" | "throughput" | "latency" | ""
    fallback_index: int = 0  # 0 = primary, 1+ = Nth fallback used


@dataclass(frozen=True)
class CostThresholdEvent(TraceEvent):
    """
    Emitted by BudgetPolicyEngine when the spend/cap ratio crosses a threshold.

    Enables the adaptive router to shift tier down (or back up) in real time.
    Recorded in trace for post-session budget analysis.
    """

    spent: int = 0
    cap: int = 0
    ratio: float = 0.0
    threshold_name: str = ""  # "compress" | "deny_child" | "restrict_export"
    tier_shift: int = (
        0  # -1 = shifted to cheaper tier, 0 = hold, 1 = shifted up
    )


@dataclass(frozen=True)
class TaintPropagatedEvent(TraceEvent):
    """Emitted when taint is propagated from an external read."""
    taint_source: str = ""
    taint_scope: str = ""


@dataclass(frozen=True)
class TaintClearanceAttemptedEvent(TraceEvent):
    """Emitted when worker attempts to clear taint (always denied)."""
    attempted_by: str = "worker"


@dataclass(frozen=True)
class TaintClearedEvent(TraceEvent):
    """Emitted when governance clears taint."""
    cleared_by: str = ""
    authority_type: str = ""
    reason_code: str = ""
    audit_id: str = ""


@dataclass(frozen=True)
class DegradationTransitionEvent(TraceEvent):
    """Emitted when DegradationEngine changes the session level."""
    previous_level: str = ""
    new_level: str = ""
    trigger_source_id: str = ""
    trigger_intent: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SourceQuarantinedEvent(TraceEvent):
    """Emitted when a source is quarantined by DegradationEngine."""
    source_id: str = ""
    quarantined_at: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class SinkDensityEvent(TraceEvent):
    """Emitted when a high-stakes sink fires — records, per axis, whether the
    driving value (per-value model) and the session (session-sticky shadow) were
    tainted (TM3.3). Aggregated into per-value vs session-sticky density.

    `tainted` is the per-value INTEGRITY label (kept as the primary field name for
    back-compat); `sensitive` is the per-value CONFIDENTIALITY label. The
    `session_*` fields are the observe-only session-sticky shadow — never an
    enforcement input.
    """
    operation: str = ""
    tainted: bool = False
    sensitive: bool = False
    session_tainted: bool = False
    session_sensitive: bool = False


@dataclass(frozen=True)
class DetectionSignalEvent(TraceEvent):
    """Emitted by the detection layer (reputation / anomaly).

    Detection is out-of-band from `allow` (TM7): it never returns a decision.
    A "crossing" verdict is a decidable threshold-crossing fact that may feed
    degradation tightening-only (TM7.1); a "flagged"/"error" verdict is telemetry
    only. `fed_degradation` records whether this signal tightened degradation.
    """
    detector: str = ""        # "reputation" | "anomaly"
    verdict: str = ""         # "crossing" | "flagged" | "error"
    score: float = 0.0
    tool: str = ""
    fed_degradation: bool = False
    reason: str = ""


# ── Telemetry contracts ───────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    @property
    def kind(self) -> str: ...
    def embed(self, text: str) -> list[float]: ...


class TelemetrySink(ABC):
    @abstractmethod
    async def send(self, records: list["AnonymizedTraceRecord"]) -> None: ...
    @abstractmethod
    async def flush(self) -> None: ...
    async def aclose(self) -> None:
        await self.flush()


@dataclass(frozen=True)
class AnonymizedTraceRecord:
    signal_chosen: TaskSignal
    classifier_used: str
    confidence: float
    tokens_spent: int
    policy_adjusted: bool
    input_embedding: list[float] | None = None
    fingerprint_kind: str = ""


@dataclass(frozen=True)
class TraceConfig:
    local_only: bool = True
    persist_inputs: bool = False
    persist_to_disk: bool = True
    training_opt_in: bool = False
    trace_dir: str = field(default_factory=_default_trace_dir)
    retention_days: int = 30
    audit_required: bool = False  # if True, trace write failure terminates session


@dataclass
class DecisionTrace:
    node_id: str
    parent_id: str | None
    depth: int
    policy_name: str
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        spent = [e for e in self.events if isinstance(e, TokensSpentEvent)]
        return sum(
            e.input_tokens
            + e.cache_creation_input_tokens
            + e.cache_read_input_tokens
            + e.output_tokens
            for e in spent
        )

    @property
    def had_policy_adjustment(self) -> bool:
        return any(
            e.kind == TraceEventKind.POLICY_ADJUSTED for e in self.events
        )
