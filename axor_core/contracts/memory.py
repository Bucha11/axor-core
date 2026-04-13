from __future__ import annotations

"""
Memory contracts — MemoryFragment, FragmentValue, MemoryProvider.

FragmentValue defines the semantic lifecycle of a context fragment:
  PINNED     — never compressed, never evicted, survives all turns
  KNOWLEDGE  — compressed carefully, retained across turns
  WORKING    — normal compression rules apply, session-scoped
  EPHEMERAL  — evicted first, often after one turn

MemoryProvider is the pluggable persistence interface.
Core defines the protocol. Implementations live in separate packages:
  axor-memory-sqlite  — SQLite, local, zero dependencies
  axor-memory-redis   — Redis, shared, multi-session

Core never imports MemoryProvider implementations.
They are injected at GovernedSession construction time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── Fragment Value ─────────────────────────────────────────────────────────────

class FragmentValue(str, Enum):
    """
    Semantic lifecycle of a context fragment.

    Priority (evicted first → last):
      EPHEMERAL → WORKING → KNOWLEDGE → PINNED

    Compressor rules:
      PINNED    → never touched, regardless of token pressure
      KNOWLEDGE → may be summarized at AGGRESSIVE mode, never dropped
      WORKING   → normal compression, may be dropped when stale
      EPHEMERAL → compressed or dropped after current turn
    """
    PINNED    = "pinned"
    KNOWLEDGE = "knowledge"
    WORKING   = "working"
    EPHEMERAL = "ephemeral"


# ── Memory Fragment ────────────────────────────────────────────────────────────

@dataclass
class MemoryFragment:
    """
    A piece of memory retrieved from MemoryProvider or created during execution.

    Memory fragments become ContextFragments with appropriate FragmentValue
    when injected into ContextView by ContextManager.
    """
    key:         str
    namespace:   str
    content:     str
    value:       FragmentValue   = FragmentValue.WORKING
    token_count: int             = 0
    tags:        list[str]       = field(default_factory=list)
    created_at:  datetime        = field(default_factory=datetime.utcnow)
    accessed_at: datetime        = field(default_factory=datetime.utcnow)
    metadata:    dict[str, Any]  = field(default_factory=dict)

    def touch(self) -> MemoryFragment:
        self.accessed_at = datetime.utcnow()
        return self

    def as_context_fragment_kwargs(self) -> dict[str, Any]:
        return {
            "kind":           "memory",
            "content":        self.content,
            "token_estimate": self.token_count or len(self.content) // 4,
            "source":         f"memory:{self.namespace}:{self.key}",
            "relevance":      1.0,
            "value":          self.value.value,
        }


# ── Memory Query ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryQuery:
    """Parameters for retrieving memory fragments."""
    namespaces:  tuple[str, ...]         = field(default_factory=tuple)
    tags:        tuple[str, ...]         = field(default_factory=tuple)
    values:      tuple[FragmentValue, ...] = field(default_factory=tuple)
    max_results: int    = 20
    max_tokens:  int    = 4000
    query_text:  str    = ""


# ── Memory Provider Protocol ───────────────────────────────────────────────────

class MemoryProvider(ABC):
    """
    Pluggable memory persistence interface.

    Core defines this contract.
    Implementations live in separate packages (axor-memory-sqlite, etc.)
    Core never imports implementations — they are injected at session time.
    """

    @abstractmethod
    async def load(self, query: MemoryQuery) -> list[MemoryFragment]:
        """Retrieve fragments matching query. Must not raise — return [] on error."""
        ...

    @abstractmethod
    async def save(self, fragments: list[MemoryFragment]) -> None:
        """Persist or update fragments. Upsert by (namespace, key)."""
        ...

    @abstractmethod
    async def delete(self, namespace: str, keys: list[str]) -> int:
        """Delete fragments by namespace + keys. Returns count deleted."""
        ...

    @abstractmethod
    async def evict(
        self,
        namespace: str,
        values: tuple[FragmentValue, ...] = (FragmentValue.EPHEMERAL,),
        max_age_seconds: int | None = None,
    ) -> int:
        """Evict stale fragments. Returns count evicted."""
        ...

    @abstractmethod
    async def namespaces(self) -> list[str]:
        """List all known namespaces."""
        ...

    async def search(self, query_text: str, namespace: str | None = None,
                     max_results: int = 10) -> list[MemoryFragment]:
        """Semantic similarity search. Default falls back to load()."""
        q = MemoryQuery(
            namespaces=(namespace,) if namespace else (),
            query_text=query_text,
            max_results=max_results,
        )
        return await self.load(q)

    async def close(self) -> None:
        """Clean up connections."""
        pass


# ── Null Provider ──────────────────────────────────────────────────────────────

class NullMemoryProvider(MemoryProvider):
    """No-op implementation. Used when no memory provider is configured."""

    async def load(self, query: MemoryQuery) -> list[MemoryFragment]:
        return []

    async def save(self, fragments: list[MemoryFragment]) -> None:
        pass

    async def delete(self, namespace: str, keys: list[str]) -> int:
        return 0

    async def evict(self, namespace: str, values=(), max_age_seconds=None) -> int:
        return 0

    async def namespaces(self) -> list[str]:
        return []
