"""AuthorityPolicy — the trusted, operator-defined half of governance.

Authority answers one question: which effects may this agent produce AT ALL.
It must originate only from a trusted source (operator configuration,
application code, an authenticated control plane, a validated parent-node
ceiling, an explicit per-run override, or an approved capability lease) —
never from an interpretation of the task text.

The advisory half — how to execute efficiently — lives in
:mod:`axor_core.contracts.planning` (``ExecutionPlan``). The two are
deliberately separate types so that the task classifier can influence
planning without any code path existing through which it could influence
authority. Import-linter contracts (``authority-plan-separation``,
``planning-non-authoritative`` in ``.importlinter``) pin the boundary.

During the migration window the legacy :class:`ExecutionPolicy` (which mixes
both concerns) remains the runtime object; :mod:`axor_core.policy.legacy`
converts between the models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExportMode,
    ToolPolicy,
)


@dataclass(frozen=True)
class ChildAuthorityPolicy:
    """The authority portion of child-node topology.

    ``allow_spawn`` grants only the RIGHT to issue ``spawn_child`` — whether
    decomposition is a good idea for a given task is a planning concern
    (``ExecutionPlan.decomposition``) and carries no permission.

    The legacy ``ChildMode`` mixed both: DENIED/SHALLOW/ALLOWED was
    simultaneously a permission and a planning shape. Here SHALLOW is simply
    ``allow_spawn=True, max_depth=1``; nesting is a consequence of depth,
    not a separate mode.
    """

    allow_spawn: bool = False
    max_depth: int = 0
    max_children_per_node: int | None = None


@dataclass(frozen=True)
class ExportAuthorityPolicy:
    """The security-sensitive ceiling on export.

    ``ExportMode`` forms a total order in the current implementation —
    allowed-field sets chain RESTRICTED {} ⊂ SUMMARY {output} ⊂ FILTERED
    {output, metadata} ⊂ FULL, and the token caps are monotone (see
    ``axor_core.node.envelope``); ``tests/invariants`` pins the chain. A
    planner may pick a mode at or below ``max_mode`` (formatting is a
    planning choice); it can never pick above it.
    """

    max_mode: ExportMode = ExportMode.SUMMARY


@dataclass(frozen=True)
class AuthorityPolicy:
    """Trusted, operator-defined authority for one execution (or session).

    Every field here is security-sensitive: tools, filesystem ceiling,
    consequence ceiling, child topology rights, escalation rules, export
    ceiling, passthrough commands and model-switch rights. None of them may
    ever be derived from task classification (architectural invariant I1 —
    classification is non-authoritative).
    """

    name: str = "standard"
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    # Filesystem ceiling — empty tuple means "no path restriction"; when set,
    # every path-bearing tool call must resolve within one of these roots.
    allowed_paths: tuple[str, ...] = field(default_factory=tuple)
    # The most irreversible action class permitted unattended (without a
    # governance/human gate).
    max_unattended_consequence: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL
    child_authority: ChildAuthorityPolicy = field(default_factory=ChildAuthorityPolicy)
    escalation_policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    export_policy: ExportAuthorityPolicy = field(default_factory=ExportAuthorityPolicy)
    allowed_passthrough_commands: tuple[str, ...] = field(default_factory=tuple)
    allow_model_switch: bool = False
