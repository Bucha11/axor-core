from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.policy import ExecutionPolicy, ToolPolicy
from axor_core.contracts.cancel import CancelToken, make_token


@dataclass(frozen=True)
class Capabilities:
    """
    Derived from ExecutionPolicy by capability/resolver.py.

    Agents never self-assign capabilities.
    These are computed — not configured by the executor.

    This is what actually appears in the envelope.
    Not the raw ToolPolicy — the resolved capability set
    for this specific execution context.
    """
    allowed_tools: frozenset[str]
    allow_children: bool
    allow_nested_children: bool
    allow_context_expansion: bool
    allow_export: bool
    allow_mutation: bool
    max_child_depth: int


@dataclass(frozen=True)
class ExportContract:
    """
    Defines what may leave this node.
    Applied by node/export.py to executor output.
    """
    mode: str                          # full | summary | filtered | restricted
    allowed_fields: frozenset[str]     # for filtered mode
    max_export_tokens: int | None      # None = no limit


@dataclass
class ExecutionEnvelope:
    """
    Complete governed execution state delivered to a node.

    This is the central execution object — not task text.

    The executor receives this. It never receives raw context,
    raw session state, or unfiltered tool lists.

    Built by node/envelope.py from:
    - ContextView (from context subsystem)
    - ExecutionPolicy (from policy subsystem)
    - Capabilities (derived from policy)
    - ExportContract (derived from policy)
    - Lineage (from context/lineage.py)
    """
    node_id: str
    task: str
    context: ContextView
    policy: ExecutionPolicy
    capabilities: Capabilities
    export_contract: ExportContract
    lineage: LineageSummary
    cancel_token: CancelToken = field(default_factory=make_token)
    parent_metadata: dict[str, Any] = field(default_factory=dict)
