from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


# ── Task Classification ────────────────────────────────────────────────────────

class TaskComplexity(str, Enum):
    FOCUSED   = "focused"    # write a test, fix a bug, explain a function
    MODERATE  = "moderate"   # add a feature, refactor a module
    EXPANSIVE = "expansive"  # rewrite repo, migrate stack, architectural refactor


class TaskNature(str, Enum):
    READONLY   = "readonly"    # analyze, explain, search
    GENERATIVE = "generative"  # write test, create file, generate code
    MUTATIVE   = "mutative"    # refactor, migrate, rewrite existing code


@dataclass(frozen=True)
class TaskSignal:
    """
    Normalized signal derived from raw input.
    This is what policy selection operates on — never raw text.
    """
    raw_input: str
    complexity: TaskComplexity
    nature: TaskNature
    estimated_scope: int           # approximate number of files/modules affected
    requires_children: bool        # agent will likely want to spawn child nodes
    requires_mutation: bool        # agent will write or modify files
    domain: str = "general"        # AgentDomain.value — str to avoid circular import


# ── Classifier Contract ────────────────────────────────────────────────────────

class SignalClassifier(ABC):
    """
    Pluggable classifier contract.

    Core ships with HeuristicClassifier (rule-based, free, in core).
    External packages implement this to provide ML-based classification:
        axor-classifier-local   — trained on user's own traces
        axor-classifier-cloud   — trained on anonymized traces from all users

    Core never imports external classifiers.
    They are injected at GovernedSession construction time.
    """

    @abstractmethod
    async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
        """
        Returns (TaskSignal, confidence).
        confidence is 0.0–1.0.
        Analyzer uses confidence to decide whether to escalate to next classifier.
        """
        ...


# ── Policy Modes ───────────────────────────────────────────────────────────────

class ContextMode(str, Enum):
    MINIMAL  = "minimal"   # only directly relevant fragments
    MODERATE = "moderate"  # relevant + immediate surroundings
    BROAD    = "broad"     # wide context — for expansive tasks


class CompressionMode(str, Enum):
    AGGRESSIVE = "aggressive"  # maximum compression, facts preserved
    BALANCED   = "balanced"    # facts + necessary context
    LIGHT      = "light"       # minimal compression


class ExportMode(str, Enum):
    FULL       = "full"        # full executor output leaves node
    SUMMARY    = "summary"     # summarized output only
    FILTERED   = "filtered"    # output passes through export filter
    RESTRICTED = "restricted"  # no intermediate export


class ChildMode(str, Enum):
    DENIED  = "denied"   # spawn_child always denied
    SHALLOW = "shallow"  # allowed but max_depth=1
    ALLOWED = "allowed"  # allowed up to max_child_depth


# ── Tool Policy ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolPolicy:
    """
    Defines which tools are available in the execution envelope.

    Capabilities are derived from this — agents never see tools
    outside what ToolPolicy allows.
    """
    allow_read:    bool = True
    allow_write:   bool = False
    allow_bash:    bool = False
    allow_search:  bool = False
    allow_spawn:   bool = False

    extra_allowed: tuple[str, ...] = field(default_factory=tuple)  # additional tool names
    extra_denied:  tuple[str, ...] = field(default_factory=tuple)  # override to deny specific tools


# ── Execution Policy ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionPolicy:
    """
    Single semantic authority for one execution.

    Policy meaning lives only in core.
    Adapters never define or modify policy semantics.

    Selected by policy/selector.py based on TaskSignal.
    May be composed with skill overrides and parent restrictions
    by policy/composer.py.
    """
    name: str                  = "default"
    derived_from: TaskComplexity = TaskComplexity.FOCUSED

    # context
    context_mode:     ContextMode     = ContextMode.MINIMAL
    compression_mode: CompressionMode = CompressionMode.AGGRESSIVE

    # children
    child_mode:      ChildMode = ChildMode.DENIED
    max_child_depth: int       = 0

    # tools
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)

    # export
    export_mode: ExportMode = ExportMode.SUMMARY

    # child context — how much of parent context flows to children
    child_context_fraction: float = 0.0   # 0.0 = no inheritance, 1.0 = full

    # pass-through slash commands allowed beyond governance class
    allowed_passthrough_commands: tuple[str, ...] = field(default_factory=tuple)
    allow_model_switch: bool = False


# ── Policy Decision ────────────────────────────────────────────────────────────

class PolicyDecisionKind(str, Enum):
    APPROVE          = "approve"
    DENY             = "deny"
    TRANSFORM        = "transform"   # approved but with modified payload
    PARTIALLY_APPROVE = "partially_approve"


@dataclass(frozen=True)
class PolicyDecision:
    """
    Result of resolving an Intent against policy.
    Produced by node/intent_loop.py for each intercepted intent.
    """
    kind: PolicyDecisionKind
    reason: str
    transformed_payload: dict | None = None  # populated when kind=TRANSFORM
