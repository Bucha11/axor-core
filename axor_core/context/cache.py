from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CachedFile:
    path: str
    content: str
    content_hash: str
    read_at: float          # unix timestamp
    turn: int               # which turn this was read
    token_estimate: int


@dataclass
class CachedToolResult:
    tool: str
    args_hash: str
    result: Any
    cached_at: float
    ttl_seconds: float      # how long this result is valid
    # Original args, kept so invalidator can selectively drop entries
    # by content (e.g. only `git`-class bash commands, not all bash output).
    args: dict = field(default_factory=dict)


# TTL per tool type — some results go stale faster
_DEFAULT_TTL: dict[str, float] = {
    "bash":   30.0,         # bash output — stale quickly (git log, ls, etc.)
    "search": 120.0,        # search results — moderately stable
    "read":   300.0,        # file reads — valid until mtime changes
}
_FALLBACK_TTL = 60.0


class ContextCache:
    """
    Two-level cache for context subsystem.

    Level 1 — file content cache:
        Stores content + hash per path.
        Prevents file rediscovery and unnecessary rereads.
        Invalidated when content hash changes (file was modified).

    Level 2 — tool result memoization:
        Memoizes (tool, args) → result with TTL.
        Prevents repeated bash/search calls returning same output.
        TTL is tool-specific — git log goes stale faster than file reads.

    Cache is session-scoped. One cache per GovernedSession.
    Child nodes share parent cache via lineage (read-only access).

    Both layers are bounded — long sessions reading many files used to grow
    the dicts without limit. `max_files` and `max_tool_results` cap each
    layer at LRU; the oldest entry is evicted when the cap is exceeded.
    """

    DEFAULT_MAX_FILES = 256
    DEFAULT_MAX_TOOL_RESULTS = 512

    def __init__(
        self,
        max_files: int = DEFAULT_MAX_FILES,
        max_tool_results: int = DEFAULT_MAX_TOOL_RESULTS,
    ) -> None:
        if max_files <= 0 or max_tool_results <= 0:
            raise ValueError("cache caps must be positive")
        self._max_files = max_files
        self._max_tool_results = max_tool_results
        # OrderedDict preserves insertion order; we move-to-end on read so
        # eviction picks the genuinely-coldest entry, not just the oldest.
        self._files: OrderedDict[str, CachedFile] = OrderedDict()
        self._tool_results: OrderedDict[str, CachedToolResult] = OrderedDict()
        self._current_turn: int = 0

    def advance_turn(self) -> None:
        self._current_turn += 1

    # ── File cache ─────────────────────────────────────────────────────────────

    def get_file(self, path: str) -> CachedFile | None:
        cached = self._files.get(path)
        if cached is not None:
            self._files.move_to_end(path)
        return cached

    def put_file(self, path: str, content: str) -> CachedFile:
        content_hash = _hash(content)
        cached = CachedFile(
            path=path,
            content=content,
            content_hash=content_hash,
            read_at=time.time(),
            turn=self._current_turn,
            token_estimate=len(content) // 4,
        )
        self._files[path] = cached
        self._files.move_to_end(path)
        # LRU eviction — drop oldest entries when over cap.
        while len(self._files) > self._max_files:
            self._files.popitem(last=False)
        return cached

    def file_changed(self, path: str, new_content: str) -> bool:
        """Check if file content has changed since last read."""
        cached = self._files.get(path)
        if not cached:
            return True
        return cached.content_hash != _hash(new_content)

    def invalidate_file(self, path: str) -> None:
        self._files.pop(path, None)

    def cached_paths(self) -> set[str]:
        return set(self._files.keys())

    def stale_files(self, current_contents: dict[str, str]) -> list[str]:
        """Return paths where content has changed since caching."""
        return [
            path for path, content in current_contents.items()
            if self.file_changed(path, content)
        ]

    # ── Tool result cache ──────────────────────────────────────────────────────

    def get_tool_result(self, tool: str, args: dict) -> Any | None:
        key = _tool_key(tool, args)
        cached = self._tool_results.get(key)
        if not cached:
            return None
        if time.time() - cached.cached_at > cached.ttl_seconds:
            del self._tool_results[key]
            return None
        self._tool_results.move_to_end(key)
        return cached.result

    def put_tool_result(self, tool: str, args: dict, result: Any) -> None:
        key = _tool_key(tool, args)
        ttl = _DEFAULT_TTL.get(tool, _FALLBACK_TTL)
        self._tool_results[key] = CachedToolResult(
            tool=tool,
            args_hash=key,
            result=result,
            cached_at=time.time(),
            ttl_seconds=ttl,
            # Store a shallow copy of args so callers cannot mutate the
            # cached version after the fact.
            args=dict(args),
        )
        self._tool_results.move_to_end(key)
        while len(self._tool_results) > self._max_tool_results:
            self._tool_results.popitem(last=False)

    def invalidate_tool_results(self, tool: str | None = None) -> None:
        """Invalidate all results for a tool, or all results if tool=None."""
        if tool is None:
            self._tool_results.clear()
        else:
            keys = [k for k, v in self._tool_results.items() if v.tool == tool]
            for k in keys:
                del self._tool_results[k]

    def invalidate_tool_results_where(
        self,
        predicate,  # Callable[[CachedToolResult], bool]
    ) -> int:
        """Invalidate cached tool results matching `predicate`.

        Use this when you need finer granularity than tool-name (e.g. drop
        only `bash` results whose args look like git commands while keeping
        `bash` results from other commands cached).

        Returns the number of entries invalidated.
        """
        keys = [k for k, v in self._tool_results.items() if predicate(v)]
        for k in keys:
            del self._tool_results[k]
        return len(keys)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def snapshot_files(self) -> dict[str, "CachedFile"]:
        """Return a snapshot of the file cache for invalidation checks."""
        return dict(self._files)

    def stats(self) -> dict:
        now = time.time()
        live_tool_results = sum(
            1 for v in self._tool_results.values()
            if now - v.cached_at <= v.ttl_seconds
        )
        return {
            "cached_files":       len(self._files),
            "cached_tool_results": live_tool_results,
            "total_file_tokens":  sum(f.token_estimate for f in self._files.values()),
            "current_turn":       self._current_turn,
        }


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _tool_key(tool: str, args: dict) -> str:
    args_str = str(sorted(args.items()))
    return f"{tool}:{hashlib.sha256(args_str.encode()).hexdigest()[:12]}"
