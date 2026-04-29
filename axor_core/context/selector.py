from __future__ import annotations

from axor_core.contracts.context import ContextFragment
from axor_core.contracts.policy import ContextMode
from axor_core.context.symbol_table import SymbolTable


# Max fragments returned per mode
_MAX_FRAGMENTS: dict[ContextMode, int] = {
    ContextMode.MINIMAL:  8,
    ContextMode.MODERATE: 20,
    ContextMode.BROAD:    50,
}

# Max total tokens per mode
_MAX_TOKENS: dict[ContextMode, int] = {
    ContextMode.MINIMAL:  1_500,
    ContextMode.MODERATE: 4_000,
    ContextMode.BROAD:    12_000,
}


class ContextSelector:
    """
    Selects which context fragments are visible for a given node.

    Takes all available fragments and returns only what fits
    within the policy's context mode limits.

    Selection is relevance-based — not recency-based.
    A file read 5 turns ago is still included if it's relevant.
    A file read last turn is excluded if it's not.

    Scoring factors:
        base relevance          (fragment.relevance, set by ingestion)
        task keyword match      (fragment content ∩ task keywords)
        symbol table penalty    (deprecated symbols in fragment)
        recency boost           (recent fragments score slightly higher)
        kind weight             (facts > tool_results > prose)

    Working set:
        Tracks which files are actively referenced.
        Used by invalidator.py to detect drift.
    """

    _KIND_WEIGHTS = {
        "fact":           1.2,
        "tool_result":    1.0,
        "parent_export":  1.1,
        "skill":          0.9,
        "memory":         0.8,
        "assistant_prose": 0.6,
        "reasoning":      0.5,
    }

    # Long sessions accumulate every file ever seen via a fragment.source.
    # Cap the set; oldest entries (FIFO) are dropped when over.
    DEFAULT_MAX_ACTIVE_PATHS = 512

    def __init__(
        self,
        symbol_table: SymbolTable,
        max_active_paths: int = DEFAULT_MAX_ACTIVE_PATHS,
    ) -> None:
        if max_active_paths <= 0:
            raise ValueError("max_active_paths must be positive")
        self._symbols = symbol_table
        self._max_active_paths = max_active_paths
        # Insertion-ordered for predictable FIFO eviction.
        self._active_paths: dict[str, None] = {}

    def select(
        self,
        fragments: list[ContextFragment],
        task: str,
        mode: ContextMode,
        current_turn: int,
    ) -> list[ContextFragment]:
        """
        Score and select fragments for the given task and context mode.
        """
        max_fragments = _MAX_FRAGMENTS[mode]
        max_tokens    = _MAX_TOKENS[mode]

        task_keywords = self._extract_keywords(task)
        scored = [
            (f, self._score(f, task_keywords, current_turn))
            for f in fragments
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected: list[ContextFragment] = []
        total_tokens = 0

        for fragment, score in scored:
            if len(selected) >= max_fragments:
                break
            if total_tokens + fragment.token_estimate > max_tokens:
                continue    # skip this fragment, try smaller ones
            selected.append(fragment)
            total_tokens += fragment.token_estimate
            # track active paths for drift detection (any file with extension)
            if fragment.source and "." in fragment.source.rsplit("/", 1)[-1]:
                if fragment.source in self._active_paths:
                    # bump to "most recent" so it survives eviction
                    del self._active_paths[fragment.source]
                self._active_paths[fragment.source] = None
                # Cap to prevent unbounded growth across long sessions.
                while len(self._active_paths) > self._max_active_paths:
                    oldest = next(iter(self._active_paths))
                    self._active_paths.pop(oldest)

        return selected

    def active_paths(self) -> set[str]:
        return set(self._active_paths.keys())

    def select_child_slice(
        self,
        fragments: list[ContextFragment],
        child_task: str,
        fraction: float,
    ) -> list[ContextFragment]:
        """
        Derive a minimal context slice for a child node.

        fraction = policy.child_context_fraction (0.0 → 1.0)
        Selects the most relevant fragments up to fraction of total tokens.
        """
        if fraction == 0.0:
            return []

        total_tokens = sum(f.token_estimate for f in fragments)
        token_budget = int(total_tokens * fraction)

        task_keywords = self._extract_keywords(child_task)
        scored = sorted(
            [(f, self._score(f, task_keywords, current_turn=0)) for f in fragments],
            key=lambda x: x[1],
            reverse=True,
        )

        selected: list[ContextFragment] = []
        used_tokens = 0
        for fragment, _ in scored:
            if used_tokens + fragment.token_estimate > token_budget:
                continue
            selected.append(fragment)
            used_tokens += fragment.token_estimate

        return selected

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score(
        self,
        fragment: ContextFragment,
        task_keywords: set[str],
        current_turn: int,
    ) -> float:
        score = fragment.relevance

        # kind weight
        score *= self._KIND_WEIGHTS.get(fragment.kind, 1.0)

        # keyword match boost
        content_lower = fragment.content.lower()
        matched = sum(1 for kw in task_keywords if kw in content_lower)
        if task_keywords:
            score += 0.3 * (matched / len(task_keywords))

        # symbol drift penalty
        penalty = self._symbols.relevance_penalty(fragment.content)
        score -= penalty

        # recency boost (small — relevance matters more than recency)
        if fragment.turn > 0 and current_turn > 0:
            age = current_turn - fragment.turn
            score += max(0.0, 0.1 - age * 0.01)

        return max(0.0, score)

    def _extract_keywords(self, task: str) -> set[str]:
        """Extract meaningful keywords from task string."""
        stop_words = {
            "a", "an", "the", "and", "or", "for", "to", "in", "of",
            "is", "it", "this", "that", "with", "from", "by", "on",
            "write", "create", "make", "add", "do", "please", "can", "you",
        }
        words = task.lower().split()
        return {w.strip(".,!?") for w in words if w not in stop_words and len(w) > 2}
