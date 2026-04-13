from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from axor_core.context.cache import ContextCache
from axor_core.context.symbol_table import SymbolTable


class InvalidationReason(str, Enum):
    FILE_MODIFIED    = "file_modified"     # content hash changed
    SYMBOL_RENAMED   = "symbol_renamed"    # fragment references deprecated symbol
    TTL_EXPIRED      = "ttl_expired"       # time-based expiry
    GIT_STATE_STALE  = "git_state_stale"   # git-derived content is stale
    WORKING_SET_DRIFT = "working_set_drift" # file no longer in active working set


@dataclass
class InvalidationResult:
    invalidated_paths: list[str]
    invalidated_tool_keys: list[str]
    reasons: dict[str, InvalidationReason]   # path/key → reason
    fragments_to_penalise: list[str]          # fragment sources to deprioritise (not remove)


# Git-related tool calls that go stale quickly
_GIT_TOOLS = {"bash"}
_GIT_COMMANDS = {"git log", "git diff", "git status", "git branch"}
_GIT_TTL = 30.0  # seconds

# Files not touched in this many turns are considered drifted from working set
_WORKING_SET_DRIFT_TURNS = 8


class ContextInvalidator:
    """
    Detects stale context and triggers cache invalidation.

    Stale signals:
        1. File content hash changed       → invalidate file cache entry
        2. Deprecated symbol in fragment   → penalise relevance (don't remove)
        3. Git-derived content TTL expired → invalidate bash cache for git commands
        4. Working set drift               → penalise old file fragments
        5. Error repetition detected       → collapse repeated error fragments

    Does not remove fragments directly — it signals to selector.py
    what to penalise and to cache.py what to invalidate.

    This separation means:
        cache.py    — knows about storage
        invalidator — knows about staleness signals
        selector.py — knows about relevance scoring
    """

    def __init__(
        self,
        cache: ContextCache,
        symbol_table: SymbolTable,
    ) -> None:
        self._cache  = cache
        self._symbols = symbol_table
        self._last_git_check: float = 0.0

    def run(
        self,
        current_turn: int,
        active_paths: set[str],          # files currently in working set
        seen_errors: list[str],          # recent error messages
    ) -> InvalidationResult:
        """
        Run all invalidation checks and return what needs to change.
        """
        invalidated_paths: list[str]    = []
        invalidated_tool_keys: list[str] = []
        reasons: dict[str, str]          = {}
        fragments_to_penalise: list[str] = []

        # 1. working set drift — files not in active_paths but still cached
        cached_paths = self._cache.cached_paths()
        drifted = cached_paths - active_paths
        for path in drifted:
            cached = self._cache.get_file(path)
            if cached and (current_turn - cached.turn) >= _WORKING_SET_DRIFT_TURNS:
                fragments_to_penalise.append(path)
                reasons[path] = InvalidationReason.WORKING_SET_DRIFT

        # 2. git staleness — invalidate bash tool results for git commands
        now = time.time()
        if now - self._last_git_check > _GIT_TTL:
            self._cache.invalidate_tool_results(tool="bash")
            self._last_git_check = now
            invalidated_tool_keys.append("bash:git_*")
            reasons["bash:git"] = InvalidationReason.GIT_STATE_STALE

        # 3. deprecated symbols — penalise fragments containing old names
        deprecated = self._symbols.deprecated_names()
        if deprecated:
            for path, cached_file in self._cache.snapshot_files().items():
                if any(name in cached_file.content for name in deprecated):
                    fragments_to_penalise.append(path)
                    if path not in reasons:
                        reasons[path] = InvalidationReason.SYMBOL_RENAMED

        return InvalidationResult(
            invalidated_paths=invalidated_paths,
            invalidated_tool_keys=invalidated_tool_keys,
            reasons=reasons,
            fragments_to_penalise=fragments_to_penalise,
        )

    def detect_error_repetition(self, recent_outputs: list[str]) -> list[tuple[str, int]]:
        """
        Detect the same error appearing multiple times in recent outputs.
        Returns list of (error_signature, count) for collapsing.
        """
        signatures: dict[str, int] = {}
        for output in recent_outputs:
            for line in output.splitlines():
                line = line.strip()
                if any(kw in line.lower() for kw in ("error:", "exception:", "traceback", "failed:")):
                    sig = line[:80]  # first 80 chars as signature
                    signatures[sig] = signatures.get(sig, 0) + 1

        return [(sig, count) for sig, count in signatures.items() if count > 1]
