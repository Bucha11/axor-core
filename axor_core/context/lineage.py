from __future__ import annotations

from axor_core.contracts.context import (
    ContextView,
    ContextFragment,
    LineageSummary,
)
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.context.selector import ContextSelector
from axor_core.context.symbol_table import SymbolTable


class LineageManager:
    """
    Manages parent → child context inheritance.

    Child never receives full parent context.
    Child receives only a derived slice — minimum sufficient
    for its subtask.

    Slice composition:
        1. Task-relevant fragments (selected by ContextSelector)
        2. Pending intents summary (from SymbolTable)
        3. Parent node metadata (node_id, depth, policy name)

    The slice is built by ContextSelector using child_context_fraction
    as the token budget. Core never passes full parent context to child.
    """

    def __init__(self, symbol_table: SymbolTable) -> None:
        self._symbols = symbol_table
        self._selector = ContextSelector(symbol_table)

    def derive_child_context(
        self,
        parent_context: ContextView,
        child_task: str,
        child_lineage: LineageSummary,
        policy: ExecutionPolicy,
    ) -> ContextView:
        """
        Derive a governed context slice for a child node.

        Returns a new ContextView containing only what the child needs.
        Never the full parent context.
        """
        # select relevant fragments within token budget
        selected_fragments = self._selector.select_child_slice(
            fragments=parent_context.visible_fragments,
            child_task=child_task,
            fraction=policy.child_context_fraction,
        )

        # add pending intents summary if there are unresolved intents
        pending = self._symbols.pending_summary()
        if pending:
            selected_fragments.append(ContextFragment(
                kind="fact",
                content=pending,
                token_estimate=len(pending) // 4,
                source="lineage:pending_intents",
                relevance=0.8,
            ))

        # add parent summary as a lightweight breadcrumb
        parent_breadcrumb = (
            f"Parent node: {parent_context.node_id} | "
            f"Depth: {child_lineage.depth} | "
            f"Summary: {parent_context.working_summary[:120]}"
        )
        selected_fragments.append(ContextFragment(
            kind="fact",
            content=parent_breadcrumb,
            token_estimate=len(parent_breadcrumb) // 4,
            source="lineage:parent_breadcrumb",
            relevance=0.9,
        ))

        total_tokens = sum(f.token_estimate for f in selected_fragments)

        return ContextView(
            node_id=child_lineage.node_id,
            working_summary=f"[child of {parent_context.node_id}] {child_task}",
            visible_fragments=selected_fragments,
            active_constraints=parent_context.active_constraints,
            lineage=child_lineage,
            token_count=total_tokens,
            compression_ratio=total_tokens / max(1, parent_context.token_count),
        )