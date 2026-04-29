from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LineageSummary:
    """
    Ancestry visible to a single node.
    Never full parent context — only what governance allows.

    `ancestry_ids` and `inherited_restrictions` are stored as tuples so a
    consumer holding the dataclass cannot retroactively `.append()` to
    them — the frozen=True guarantee otherwise leaks via mutable list fields.
    Lists supplied at construction time are coerced to tuples in __post_init__.
    """
    node_id: str
    parent_id: str | None
    depth: int
    ancestry_ids: tuple[str, ...]           # ordered root → parent
    inherited_restrictions: tuple[str, ...] # restriction names carried from ancestors

    def __post_init__(self) -> None:
        if not isinstance(self.ancestry_ids, tuple):
            object.__setattr__(self, "ancestry_ids", tuple(self.ancestry_ids))
        if not isinstance(self.inherited_restrictions, tuple):
            object.__setattr__(self, "inherited_restrictions", tuple(self.inherited_restrictions))


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
    turn: int = 0                    # which turn created this fragment (0 = unknown)


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

    Sequence-typed fields are coerced to tuples so the frozen=True guarantee
    isn't undermined by `view.visible_fragments.append(...)` after the fact.
    """
    node_id: str
    working_summary: str
    visible_fragments: tuple[ContextFragment, ...]
    active_constraints: tuple[str, ...]    # human-readable constraint names
    lineage: LineageSummary
    token_count: int                 # total tokens in this view
    compression_ratio: float         # how much was compressed (1.0 = no compression)

    def __post_init__(self) -> None:
        if not isinstance(self.visible_fragments, tuple):
            object.__setattr__(self, "visible_fragments", tuple(self.visible_fragments))
        if not isinstance(self.active_constraints, tuple):
            object.__setattr__(self, "active_constraints", tuple(self.active_constraints))


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
