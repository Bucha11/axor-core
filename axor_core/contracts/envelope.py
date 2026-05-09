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


@dataclass(frozen=True)
class CacheHints:
    """Typed cache directives for adapters that support prefix-caching.

    Adapters that do not support caching silently ignore this object.
    Core never populates it — adapters read it, callers set it.
    """
    ttl: str | None = None
    cacheable_blocks: tuple[str, ...] = ()
    response_cache_allowed: bool = False
    relevance_k: int | None = None


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

    # ── Added in 0.5.0: adapter-facing optimisation hints ─────────────────────
    cache_hints: CacheHints | None = None

    # deterministic: True when the executor is expected to produce reproducible
    # output (temperature=0 or equivalent). Adapters may enable response-level
    # caching when this flag is set.
    deterministic: bool = False

    # depth: depth of this node in the execution tree (structural fact).
    # Mirror of lineage.depth — provided as a direct field to avoid lineage
    # traversal in hot paths.
    depth: int = 0

    # parent_node_id: shortcut to lineage.parent_id.
    parent_node_id: str | None = None

    # routing_tier: explicit tier override for adapters that support cascade routing.
    # When set, the adapter uses this tier index directly instead of deriving one
    # from depth. Keeps depth as a structural fact and tier as a routing decision.
    #
    # Example: a leaf node generating a security patch may set routing_tier=0 to
    # force the most capable model regardless of its position in the tree.
    routing_tier: int | None = None
