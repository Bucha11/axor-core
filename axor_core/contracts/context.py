from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LineageSummary:
    """
    Ancestry visible to a single node.
    Never full parent context — only what governance allows.
    """
    node_id: str
    parent_id: str | None
    depth: int
    ancestry_ids: list[str]           # ordered root → parent
    inherited_restrictions: list[str] # restriction names carried from ancestors


@dataclass(frozen=True)
class ContextFragment:
    """
    A single piece of context — fact, observation, tool result, memory.
    Always shaped before reaching a node. Never raw.

    value controls compression priority (FragmentValue semantics, stored as str
    to avoid circular import):
      "pinned"    → never touched by compressor, survives all turns
      "knowledge" → compressed carefully, retained across turns
      "working"   → normal compression rules (default)
      "ephemeral" → evicted first, often after one turn
    """
    kind: str                        # "fact" | "tool_result" | "memory" | "parent_export" | "skill"
    content: str
    token_estimate: int
    source: str                      # where this fragment came from
    relevance: float = 1.0           # 0.0–1.0, used by selector
    value: str = "working"           # FragmentValue.value as str


@dataclass(frozen=True)
class ContextView:
    """
    Operational context visible to a single node.

    This is never raw history.
    It is always shaped by the context subsystem.

    Rules:
    - working_summary replaces raw turn history
    - visible_fragments are selected and scored by relevance
    - child nodes never receive this directly — they receive a derived slice
    """
    node_id: str
    working_summary: str
    visible_fragments: list[ContextFragment]
    active_constraints: list[str]    # human-readable constraint names
    lineage: LineageSummary
    token_count: int                 # total tokens in this view
    compression_ratio: float         # how much was compressed (1.0 = no compression)


@dataclass
class RawExecutionState:
    """
    Raw state before context subsystem processes it.
    Never passed to executors directly.
    """
    task: str
    session_id: str
    parent_export: str | None        # what parent node exported
    session_state: dict[str, Any]    # session-level memory
    memory_fragments: list[str]      # retrieved memory (plain strings for backward compat)
    lineage: LineageSummary | None   # None for root nodes
    prior_turns: list[dict[str, Any]] = field(default_factory=list)
