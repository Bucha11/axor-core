from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field


@dataclass
class NodeBudget:
    """Token usage record for a single node."""
    node_id: str
    parent_id: str | None
    depth: int
    input_tokens: int  = 0
    output_tokens: int = 0
    tool_tokens: int   = 0
    context_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class BudgetTracker:
    """
    Tracks real token consumption across the entire node tree.

    Thread-safe — nodes at the same depth may run concurrently.

    Does not enforce limits.
    Provides data to policy_engine.py which makes decisions.

    One tracker per GovernedSession — shared across all child nodes.
    """

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._nodes: dict[str, NodeBudget] = {}

    def register_node(
        self,
        node_id: str,
        parent_id: str | None,
        depth: int,
    ) -> None:
        with self._lock:
            self._nodes[node_id] = NodeBudget(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
            )

    def record(
        self,
        node_id: str,
        input_tokens: int,
        output_tokens: int,
        tool_tokens: int   = 0,
        context_tokens: int = 0,
    ) -> None:
        with self._lock:
            if node_id not in self._nodes:
                # node registered late — create on the fly
                self._nodes[node_id] = NodeBudget(
                    node_id=node_id,
                    parent_id=None,
                    depth=0,
                )
            budget = self._nodes[node_id]
            self._nodes[node_id] = NodeBudget(
                node_id=budget.node_id,
                parent_id=budget.parent_id,
                depth=budget.depth,
                input_tokens=budget.input_tokens   + input_tokens,
                output_tokens=budget.output_tokens + output_tokens,
                tool_tokens=budget.tool_tokens     + tool_tokens,
                context_tokens=budget.context_tokens + context_tokens,
            )

    # ── Queries ────────────────────────────────────────────────────────────────

    def total_tokens(self) -> int:
        """Total tokens spent across all nodes in this session."""
        with self._lock:
            return sum(n.total for n in self._nodes.values())

    def node_tokens(self, node_id: str) -> int:
        with self._lock:
            node = self._nodes.get(node_id)
            return node.total if node else 0

    def subtree_tokens(self, node_id: str) -> int:
        """Total tokens for a node and all its descendants."""
        with self._lock:
            ids = self._subtree_ids(node_id)
            return sum(
                self._nodes[nid].total
                for nid in ids
                if nid in self._nodes
            )

    def depth_tokens(self, depth: int) -> int:
        """Total tokens spent at a specific depth level."""
        with self._lock:
            return sum(
                n.total for n in self._nodes.values()
                if n.depth == depth
            )

    def snapshot(self) -> dict[str, NodeBudget]:
        with self._lock:
            return dict(self._nodes)

    # ── Private ────────────────────────────────────────────────────────────────

    def _subtree_ids(self, root_id: str) -> set[str]:
        """Collect node_id + all descendant ids via BFS."""
        # build parent→children index for O(n) traversal
        children: dict[str, list[str]] = {}
        for node in self._nodes.values():
            if node.parent_id is not None:
                children.setdefault(node.parent_id, []).append(node.node_id)

        result: set[str] = set()
        queue = deque([root_id])
        while queue:
            nid = queue.popleft()
            if nid in result:
                continue
            result.add(nid)
            queue.extend(children.get(nid, []))
        return result
