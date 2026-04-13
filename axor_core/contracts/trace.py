from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from axor_core.contracts.policy import TaskSignal, PolicyDecisionKind


class TraceEventKind(str, Enum):
    # policy
    SIGNAL_CHOSEN      = "signal_chosen"
    POLICY_CHOSEN      = "policy_chosen"
    POLICY_ADJUSTED    = "policy_adjusted"      # initial signal was wrong — valuable training data

    # intents
    INTENT_APPROVED    = "intent_approved"
    INTENT_DENIED      = "intent_denied"
    INTENT_TRANSFORMED = "intent_transformed"

    # federation
    CHILD_SPAWNED      = "child_spawned"
    CHILD_COMPLETED    = "child_completed"

    # context
    CONTEXT_COMPRESSED = "context_compressed"
    CONTEXT_SLICE_DERIVED = "context_slice_derived"

    # budget
    TOKENS_SPENT       = "tokens_spent"
    BUDGET_WARNING     = "budget_warning"

    # commands
    COMMAND_ROUTED     = "command_routed"

    # extensions
    EXTENSION_LOADED   = "extension_loaded"
    PLUGIN_DENIED      = "plugin_denied"        # plugin tried to register something policy blocked
    SKILL_ACTIVATED    = "skill_activated"

    # cancellation
    CANCELLED          = "cancelled"            # node execution was cancelled


@dataclass(frozen=True)
class TraceEvent:
    """Base for all trace events."""
    kind: TraceEventKind
    node_id: str
    sequence: int            # global sequence number for ordering
    payload: dict[str, Any] = field(default_factory=dict)


# ── Typed events ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalChosenEvent(TraceEvent):
    """
    Records which classifier produced the TaskSignal and with what confidence.
    Used for classifier performance tracking.
    """
    raw_input: str           = ""
    signal: TaskSignal | None = None
    confidence: float        = 0.0
    classifier: str          = "heuristic"   # "heuristic" | "local" | "cloud"


@dataclass(frozen=True)
class PolicyAdjustedEvent(TraceEvent):
    """
    Policy was corrected mid-execution because the initial signal was wrong.

    This is the most valuable training signal for classifiers:
    - what was the input
    - what signal was chosen
    - what signal was actually needed
    - how many tokens were wasted before correction
    """
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
    context_fraction: float = 0.0    # how much of parent context the child received


@dataclass(frozen=True)
class TokensSpentEvent(TraceEvent):
    input_tokens: int   = 0
    output_tokens: int  = 0
    tool_tokens: int    = 0
    context_tokens: int = 0
    cumulative: int     = 0          # total for this lineage tree so far


@dataclass(frozen=True)
class CommandRoutedEvent(TraceEvent):
    command_name: str   = ""
    command_class: str  = ""         # governance | context | passthrough
    allowed: bool       = True


@dataclass(frozen=True)
class PluginDeniedEvent(TraceEvent):
    plugin_name: str    = ""
    denied_item: str    = ""         # tool name, command name, etc.
    reason: str         = ""


@dataclass(frozen=True)
class CancelledEvent(TraceEvent):
    """
    Node execution was cancelled before completion.
    Partial result was returned.
    """
    reason: str          = ""   # CancelReason value
    detail: str          = ""
    completed_intents: int = 0  # how many intents completed before cancel


# ── AnonymizedTraceRecord — for classifier training ───────────────────────────

@dataclass(frozen=True)
class AnonymizedTraceRecord:
    """
    What leaves the machine for cloud classifier training.

    Critically: no raw_input, no user code, no file contents.
    Only embeddings and governance metadata.

    Users must explicitly opt in to training_opt_in=True in TraceConfig.
    """
    input_embedding: list[float]     # embedding of raw_input — never the text itself
    signal_chosen: TaskSignal
    classifier_used: str
    confidence: float
    tokens_spent: int
    policy_adjusted: bool            # was the initial signal wrong?
    # deliberately no: raw_input, session_id, user_id, code content


# ── TraceConfig ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraceConfig:
    """
    Controls what trace data is persisted and where.

    Defaults are maximally private:
    - local_only=True      traces never leave the machine
    - persist_inputs=False raw inputs are not written even locally
    - training_opt_in=False anonymized records not sent for cloud training
    """
    local_only: bool       = True
    persist_inputs: bool   = False
    training_opt_in: bool  = False
    trace_dir: str         = "~/.axor/traces"


# ── DecisionTrace ──────────────────────────────────────────────────────────────

@dataclass
class DecisionTrace:
    """
    Full trace for one node execution.
    Belongs to lineage — not flat logging.
    """
    node_id: str
    parent_id: str | None
    depth: int
    policy_name: str
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        spent = [e for e in self.events if isinstance(e, TokensSpentEvent)]
        return sum(e.input_tokens + e.output_tokens for e in spent)

    @property
    def had_policy_adjustment(self) -> bool:
        return any(e.kind == TraceEventKind.POLICY_ADJUSTED for e in self.events)
