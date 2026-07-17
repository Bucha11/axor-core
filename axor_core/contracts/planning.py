"""ExecutionPlan — the advisory, optimization-only half of governance.

A plan answers one question: how to execute the task efficiently — how much
context to load, how aggressively to compress it, whether decomposition is
worth it, what resources to reserve. A plan grants nothing and overrides no
authority check: every field here can be derived from an untrusted task
classification, because the worst a wrong plan can do is waste resources
(bounded by :class:`ResourceBudget`), never produce an effect the operator
did not authorize.

The trusted half — what the agent may do at all — lives in
:mod:`axor_core.contracts.authority` (``AuthorityPolicy``). This module must
never import authority constructors; the ``planning-non-authoritative``
import-linter contract pins that direction, and ``authority-plan-separation``
pins the reverse (enforcement surfaces never read planning types).

ExecutionPlan deliberately has NO: ToolPolicy, allowed_paths, consequence
ceiling, escalation rules, allow_spawn, human-approval policy, model-switch
authority, or passthrough authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from axor_core.contracts.policy import CompressionMode, ContextMode


class RetrievalBreadth(str, Enum):
    NARROW = "narrow"
    MODERATE = "moderate"
    BROAD = "broad"


class DecompositionPreference(str, Enum):
    """Whether the planner thinks child decomposition is worth it.

    A preference, not a permission: PREFER does not grant spawn (that is
    ``ChildAuthorityPolicy.allow_spawn``), and AVOID does not deny it — an
    agent that spawns anyway is judged by the authority gates alone.
    """

    AVOID = "avoid"
    ALLOW = "allow"
    PREFER = "prefer"


@dataclass(frozen=True)
class ExecutionPlan:
    """Advisory execution shape for one task.

    ``source`` records where the plan came from ("default", "operator",
    "task_classifier", "legacy_split", ...) and ``confidence`` the
    classifier's confidence when applicable — both are telemetry, not
    decision inputs for any enforcement gate.
    """

    name: str = "default"
    context_mode: ContextMode = ContextMode.MODERATE
    compression_mode: CompressionMode = CompressionMode.BALANCED
    retrieval_breadth: RetrievalBreadth = RetrievalBreadth.MODERATE
    decomposition: DecompositionPreference = DecompositionPreference.ALLOW
    # Planning hint for child topology — clamped by the authority's
    # child_authority.max_depth at spawn time, never a grant.
    suggested_child_depth: int = 0
    child_context_fraction: float = 0.0   # 0.0 = no inheritance, 1.0 = full
    expected_scope: int | None = None     # approximate files/modules affected
    token_reservation: int | None = None
    source: str = "default"
    confidence: float | None = None


NEUTRAL_PLAN = ExecutionPlan(name="neutral", source="default")
"""Applied when no classifier/planner is configured or classification fails.

Classifier failure is operational only (invariant I6): it degrades to this
plan — it never stops governed execution, never changes authority, and never
produces a capability denial.
"""


@dataclass(frozen=True)
class ResourceBudget:
    """Hard ceilings on what a plan may consume. ``None`` = unlimited.

    Plans may expand and shrink dynamically (planning is not monotonic), but
    only within these ceilings — they are what bounds the availability blast
    radius of an adversarially inflated plan (a prompt-injected "this is a
    repository-wide task" can at most spend up to the budget, never widen
    the capability surface).
    """

    max_context_tokens: int | None = None
    max_retrieval_items: int | None = None
    max_children: int | None = None
    max_plan_expansions: int | None = None
    max_token_reservation: int | None = None
    max_concurrency: int | None = None
