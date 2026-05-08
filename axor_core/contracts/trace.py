from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from axor_core.contracts.policy import TaskSignal, PolicyDecisionKind


def _default_trace_dir() -> str:
    return os.environ.get("AXOR_TRACE_DIR", "~/.axor/traces")


class TraceEventKind(str, Enum):
    # policy
    SIGNAL_CHOSEN      = "signal_chosen"
    POLICY_CHOSEN      = "policy_chosen"
    POLICY_ADJUSTED    = "policy_adjusted"

    # intents
    INTENT_APPROVED    = "intent_approved"
    INTENT_DENIED      = "intent_denied"
    INTENT_TRANSFORMED = "intent_transformed"

    # federation
    CHILD_SPAWNED      = "child_spawned"
    CHILD_COMPLETED    = "child_completed"

    # context
    CONTEXT_COMPRESSED    = "context_compressed"
    CONTEXT_SLICE_DERIVED = "context_slice_derived"

    # budget
    TOKENS_SPENT    = "tokens_spent"
    BUDGET_WARNING  = "budget_warning"

    # commands
    COMMAND_ROUTED = "command_routed"

    # extensions
    EXTENSION_LOADED = "extension_loaded"
    PLUGIN_DENIED    = "plugin_denied"
    SKILL_ACTIVATED  = "skill_activated"

    # cancellation
    CANCELLED = "cancelled"

    # ── Added in 0.5.0: adapter observability ─────────────────────────────────
    # Emitted by adapters that support prefix-caching (e.g. axor-openrouter).
    CACHE_HIT   = "cache_hit"    # tokens served from cache
    CACHE_MISS  = "cache_miss"   # cache miss — full prompt processed
    CACHE_WRITE = "cache_write"  # breakpoint written to cache

    # Emitted by adapters when a routing decision is made (model, provider, tier).
    ROUTING_DECISION = "routing_decision"

    # Emitted by BudgetPolicyEngine when spend/cap crosses a threshold.
    COST_THRESHOLD = "cost_threshold"


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
    raw_input: str            = ""
    signal: TaskSignal | None = None
    confidence: float         = 0.0
    classifier: str           = "heuristic"
    scores: dict[str, float]  = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyAdjustedEvent(TraceEvent):
    original_signal: TaskSignal | None  = None
    adjusted_signal: TaskSignal | None  = None
    reason: str                         = ""
    tokens_spent_before_adjustment: int = 0


@dataclass(frozen=True)
class IntentDeniedEvent(TraceEvent):
    intent_kind: str = ""
    reason: str      = ""


@dataclass(frozen=True)
class ChildSpawnedEvent(TraceEvent):
    child_node_id: str      = ""
    child_depth: int        = 0
    context_fraction: float = 0.0


@dataclass(frozen=True)
class TokensSpentEvent(TraceEvent):
    input_tokens: int   = 0
    output_tokens: int  = 0
    tool_tokens: int    = 0
    context_tokens: int = 0
    cumulative: int     = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int     = 0


@dataclass(frozen=True)
class CommandRoutedEvent(TraceEvent):
    command_name: str   = ""
    command_class: str  = ""
    allowed: bool       = True


@dataclass(frozen=True)
class PluginDeniedEvent(TraceEvent):
    plugin_name: str    = ""
    denied_item: str    = ""
    reason: str         = ""


@dataclass(frozen=True)
class CancelledEvent(TraceEvent):
    reason: str           = ""
    detail: str           = ""
    completed_intents: int = 0


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
    tokens: int = 0                # tokens involved in this cache event
    ttl: int = 0                   # TTL in seconds (0 = provider default)
    breakpoint_block: str = ""     # "system" | "tools" | "context_top_k"


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
    tier: int = 0                  # 0 = root/best, higher = cheaper
    sort_strategy: str = ""        # "price" | "throughput" | "latency" | ""
    fallback_index: int = 0        # 0 = primary, 1+ = Nth fallback used


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
    threshold_name: str = ""       # "compress" | "deny_child" | "restrict_export"
    tier_shift: int = 0            # -1 = shifted to cheaper tier, 0 = hold, 1 = shifted up


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
    local_only: bool       = True
    persist_inputs: bool   = False
    persist_to_disk: bool  = True
    training_opt_in: bool  = False
    trace_dir: str         = field(default_factory=_default_trace_dir)
    retention_days: int    = 30


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
        return any(e.kind == TraceEventKind.POLICY_ADJUSTED for e in self.events)
