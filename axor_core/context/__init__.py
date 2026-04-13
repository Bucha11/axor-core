"""
axor_core.context
─────────────────
Spine of core. Controls everything visible to an executor.

    ContextManager   — orchestrates all stages: ingest → invalidate →
                       compress → select → scope → ContextView
    ContextCache     — file content cache + tool result memoization
    ContextCompressor — reduces size without losing facts
    ContextSelector  — relevance scoring and working set management
    ContextInvalidator — stale detection: git state, symbol drift, TTL
    SymbolTable      — live symbol registry, drift detection, pending intents
    LineageManager   — child context slice derivation

Waste eliminated:
    verbose prose        → compressor (prose → key decisions)
    oversized outputs    → compressor (smart truncation: head + tail)
    stale branch history → invalidator (git TTL)
    repeated validation  → cache (file hash + tool memoization)
    symbol drift         → symbol_table + invalidator (deprecation + penalty)
    file rediscovery     → cache (cached_paths registry)
    unnecessary rereads  → cache (get_file before tool executes)
    turn accumulation    → compressor (rolling summary after N turns)
    error repetition     → compressor (collapse to single entry)
    working set drift    → invalidator (relevance decay by turn distance)
    path explosion       → compressor (absolute → relative normalization)
"""

from axor_core.context.manager import ContextManager
from axor_core.context.cache import ContextCache
from axor_core.context.compressor import ContextCompressor
from axor_core.context.selector import ContextSelector
from axor_core.context.invalidator import ContextInvalidator
from axor_core.context.symbol_table import SymbolTable, SymbolStatus, PendingIntent
from axor_core.context.lineage import LineageManager

__all__ = [
    "ContextManager",
    "ContextCache",
    "ContextCompressor",
    "ContextSelector",
    "ContextInvalidator",
    "SymbolTable",
    "SymbolStatus",
    "PendingIntent",
    "LineageManager",
]
