from __future__ import annotations

from axor_core.contracts.context import (
    ContextView,
    ContextFragment,
    LineageSummary,
    RawExecutionState,
)
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.context.cache import ContextCache
from axor_core.context.compressor import ContextCompressor
from axor_core.context.invalidator import ContextInvalidator
from axor_core.context.selector import ContextSelector
from axor_core.context.symbol_table import SymbolTable


class ContextManager:
    """
    Spine of core. Orchestrates all context subsystem stages.

    Pipeline per execution:
        ingest     → collect raw fragments from state
        invalidate → detect and remove stale fragments
        compress   → reduce size without losing facts
        select     → score and filter to policy limits
        scope      → enforce visibility boundaries
        → ContextView

    Post-execution:
        update     → persist result-derived state back into context

    One ContextManager per GovernedSession.
    Shared across all nodes via session — child nodes call
    lineage.py to derive their slice, not this manager directly.

    All seven waste categories are addressed here:
        verbose prose        → compressor._compress_prose()
        oversized outputs    → compressor._truncate_tool_outputs()
        stale branch history → invalidator (git TTL)
        repeated validation  → cache (file hash + tool memoization)
        symbol drift         → symbol_table + invalidator
        file rediscovery     → cache.cached_paths()
        unnecessary rereads  → cache.get_file() before tool execution
    """

    _MAX_FRAGMENTS = 200

    def __init__(self, max_fragments: int = _MAX_FRAGMENTS) -> None:
        self._cache       = ContextCache()
        self._symbols     = SymbolTable()
        self._compressor  = ContextCompressor()
        self._selector    = ContextSelector(self._symbols)
        self._invalidator = ContextInvalidator(self._cache, self._symbols)
        self._turn        = 0
        self._max_fragments = max_fragments
        self._all_fragments: list[ContextFragment] = []
        self._pinned_fragments: list[ContextFragment] = []  # never compressed/evicted
        self._recent_outputs: list[str] = []

    def pin_fragment(self, fragment: ContextFragment) -> None:
        """
        Add a pinned fragment that survives all compression and selection.

        Pinned fragments always appear in ContextView regardless of policy,
        token pressure, or compression mode.

        Used for: agent personality, critical system instructions,
        user-defined rules that must never be forgotten.

        Deduplicates by source — calling pin_fragment with the same source
        replaces the existing fragment.
        """
        # ensure value is set to pinned
        if fragment.value != "pinned":
            fragment = ContextFragment(
                kind=fragment.kind,
                content=fragment.content,
                token_estimate=fragment.token_estimate,
                source=fragment.source,
                relevance=fragment.relevance,
                value="pinned",
            )
        # deduplicate by source
        self._pinned_fragments = [
            f for f in self._pinned_fragments if f.source != fragment.source
        ]
        self._pinned_fragments.append(fragment)

    def add_knowledge(self, content: str, source: str, token_estimate: int = 0) -> None:
        """
        Add a knowledge fragment — compressed carefully, retained across turns.

        Used for: RAG results, domain documentation, reference material
        that should survive compression but may be summarized.
        """
        fragment = ContextFragment(
            kind="memory",
            content=content,
            token_estimate=token_estimate or len(content) // 4,
            source=source,
            relevance=0.85,
            value="knowledge",
            turn=self._turn,
        )
        # replace existing knowledge from same source
        self._all_fragments = [
            f for f in self._all_fragments if f.source != source
        ]
        self._all_fragments.append(fragment)
        self._evict_if_needed()

    def ingest_fragments(self, fragments: list[ContextFragment]) -> None:
        """
        Add fragments from external sources (e.g. adapters).

        Deduplicates by source — if a fragment with the same source already
        exists, it is replaced. Evicts lowest-relevance fragments when the
        collection exceeds max_fragments.
        """
        existing_sources = {f.source for f in self._all_fragments}
        for frag in fragments:
            if frag.source in existing_sources:
                self._all_fragments = [
                    f for f in self._all_fragments if f.source != frag.source
                ]
            self._all_fragments.append(frag)
            existing_sources.add(frag.source)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop lowest-relevance fragments when collection exceeds limit."""
        if len(self._all_fragments) <= self._max_fragments:
            return
        self._all_fragments.sort(key=lambda f: f.relevance, reverse=True)
        self._all_fragments = self._all_fragments[:self._max_fragments]

    def build(
        self,
        raw_state: RawExecutionState,
        lineage: LineageSummary,
        policy: ExecutionPolicy | None = None,
    ) -> ContextView:
        """
        Full pipeline: raw state → shaped ContextView.
        Policy is passed per-call — always reflects the current execution's
        policy, not a stale value from session initialization.

        Pinned fragments are always included first — they bypass all
        compression, selection, and scope rules.
        """
        from axor_core.contracts.policy import ContextMode, CompressionMode
        context_mode     = policy.context_mode     if policy else ContextMode.MINIMAL
        compression_mode = policy.compression_mode if policy else CompressionMode.BALANCED

        self._turn += 1
        self._cache.advance_turn()
        self._symbols.advance_turn()

        # ── ingest ────────────────────────────────────────────────────────────
        fragments = self._ingest(raw_state)

        # ── invalidate ────────────────────────────────────────────────────────
        invalidation = self._invalidator.run(
            current_turn=self._turn,
            active_paths=self._selector.active_paths(),
            seen_errors=self._recent_outputs[-5:],
        )
        fragments = self._apply_penalties(fragments, invalidation.fragments_to_penalise)

        # ── compress ──────────────────────────────────────────────────────────
        # pinned fragments are prepended AFTER compression so compressor
        # never sees them — they are invisible to all compression strategies
        compression_result = self._compressor.compress(
            fragments=fragments,
            mode=compression_mode,
            current_turn=self._turn,
        )
        fragments = compression_result.fragments

        # ── select ────────────────────────────────────────────────────────────
        selected = self._selector.select(
            fragments=fragments,
            task=raw_state.task,
            mode=context_mode,
            current_turn=self._turn,
        )

        # ── scope ─────────────────────────────────────────────────────────────
        if not raw_state.parent_export:
            selected = [f for f in selected if f.source != "parent_export"]

        # ── prepend pinned fragments ───────────────────────────────────────────
        # pinned always first — before all other fragments
        final = list(self._pinned_fragments) + selected
        total_tokens = sum(f.token_estimate for f in final)

        return ContextView(
            node_id=lineage.node_id,
            working_summary=self._build_summary(raw_state, final, policy),
            visible_fragments=final,
            active_constraints=self._active_constraints(policy),
            lineage=lineage,
            token_count=total_tokens,
            compression_ratio=compression_result.compression_ratio,
        )

    def update(self, result_output: str, node_id: str) -> None:
        """
        Post-execution: update context from result.
        Called after node execution completes.
        """
        self._recent_outputs.append(result_output)
        if len(self._recent_outputs) > 10:
            self._recent_outputs = self._recent_outputs[-10:]

        # extract decisions from assistant output for next turn context
        if result_output:
            self._symbols.ingest_assistant_text(result_output)
            fragment = ContextFragment(
                kind="assistant_prose",
                content=result_output[:2000],  # cap before storing
                token_estimate=min(len(result_output) // 4, 500),
                source=f"node:{node_id}",
                relevance=0.7,
                turn=self._turn,
            )
            self._all_fragments.append(fragment)
            self._evict_if_needed()

    def record_file_read(self, path: str, content: str) -> None:
        """
        Called by CapabilityExecutor after a read tool executes.
        Updates cache and symbol table.
        """
        cached = self._cache.put_file(path, content)
        self._symbols.ingest_file(path, content)

        fragment = ContextFragment(
            kind="fact",
            content=f"[{path}]\n{content}",
            token_estimate=cached.token_estimate,
            source=path,
            relevance=1.0,
            turn=self._turn,
        )
        # replace existing fragment for this path or append
        self._all_fragments = [
            f for f in self._all_fragments if f.source != path
        ]
        self._all_fragments.append(fragment)

    def is_cached(self, path: str) -> bool:
        """Check if file is already cached — prevents unnecessary rereads."""
        return self._cache.get_file(path) is not None

    def get_cached_tool_result(self, tool: str, args: dict):
        return self._cache.get_tool_result(tool, args)

    def cache_tool_result(self, tool: str, args: dict, result) -> None:
        self._cache.put_tool_result(tool, args, result)

    # ── Private ────────────────────────────────────────────────────────────────

    def _ingest(self, raw_state: RawExecutionState) -> list[ContextFragment]:
        fragments = list(self._all_fragments)

        # task as a high-relevance fragment
        fragments.append(ContextFragment(
            kind="fact",
            content=raw_state.task,
            token_estimate=len(raw_state.task) // 4,
            source="task",
            relevance=1.0,
            turn=self._turn,
        ))

        # parent export
        if raw_state.parent_export:
            fragments.append(ContextFragment(
                kind="parent_export",
                content=raw_state.parent_export,
                token_estimate=len(raw_state.parent_export) // 4,
                source="parent_export",
                relevance=0.9,
                turn=self._turn,
            ))

        # memory fragments
        for mem in raw_state.memory_fragments:
            fragments.append(ContextFragment(
                kind="memory",
                content=mem,
                token_estimate=len(mem) // 4,
                source="memory",
                relevance=0.6,
                turn=self._turn,
            ))

        return fragments

    def _apply_penalties(
        self,
        fragments: list[ContextFragment],
        penalised_sources: list[str],
    ) -> list[ContextFragment]:
        penalised_set = set(penalised_sources)
        result = []
        for f in fragments:
            if f.source in penalised_set:
                result.append(ContextFragment(
                    kind=f.kind,
                    content=f.content,
                    token_estimate=f.token_estimate,
                    source=f.source,
                    relevance=f.relevance * 0.4,   # heavy penalty
                ))
            else:
                result.append(f)
        return result

    def _build_summary(
        self,
        raw_state: RawExecutionState,
        selected: list[ContextFragment],
        policy: ExecutionPolicy | None = None,
    ) -> str:
        facts = [f for f in selected if f.kind == "fact"]
        policy_name = policy.name if policy else "unknown"
        return (
            f"Task: {raw_state.task[:80]} | "
            f"Turn: {self._turn} | "
            f"Policy: {policy_name} | "
            f"Facts: {len(facts)} | "
            f"Files cached: {len(self._cache.cached_paths())}"
        )

    def _active_constraints(self, policy: ExecutionPolicy | None = None) -> list[str]:
        constraints = [f"turn={self._turn}"]
        if policy:
            constraints.insert(0, policy.context_mode.value)
            constraints.insert(1, policy.compression_mode.value)
        return constraints
