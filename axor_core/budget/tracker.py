from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TypedDict


class CacheSummary(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    total_input_tokens: int
    total_tokens: int
    hit_rate: float


class CostSummary(CacheSummary):
    currency: str
    input_cost: float
    cache_creation_cost: float
    cache_read_cost: float
    output_cost: float
    total_cost: float
    input_per_m: float
    cache_creation_input_per_m: float
    cache_read_input_per_m: float
    output_per_m: float


@dataclass(frozen=True)
class TokenCostRates:
    """
    Provider/model pricing in currency units per million tokens.

    `input_per_m` applies to uncached input. When prompt-cache rates are not
    supplied explicitly, Anthropic-style multipliers are used:
    cache creation = 1.25x input, cache read = 0.1x input.
    """
    input_per_m: float
    output_per_m: float
    cache_creation_input_per_m: float | None = None
    cache_read_input_per_m: float | None = None
    currency: str = "USD"

    @property
    def effective_cache_creation_input_per_m(self) -> float:
        if self.cache_creation_input_per_m is not None:
            return self.cache_creation_input_per_m
        return self.input_per_m * 1.25

    @property
    def effective_cache_read_input_per_m(self) -> float:
        if self.cache_read_input_per_m is not None:
            return self.cache_read_input_per_m
        return self.input_per_m * 0.1


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
    # Anthropic prompt-cache accounting. These counters are separate from
    # input_tokens as returned by the Messages API. Keep them split so cost
    # calculation can apply per-class multipliers (1.25x creation, 0.1x read).
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int     = 0

    @property
    def total_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total(self) -> int:
        return self.total_input_tokens + self.output_tokens

    def estimated_cost(self, rates: TokenCostRates) -> float:
        return (
            self.input_tokens / 1_000_000 * rates.input_per_m
            + self.cache_creation_input_tokens / 1_000_000
            * rates.effective_cache_creation_input_per_m
            + self.cache_read_input_tokens / 1_000_000
            * rates.effective_cache_read_input_per_m
            + self.output_tokens / 1_000_000 * rates.output_per_m
        )

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of full input volume that was served from cache."""
        cached = self.cache_read_input_tokens
        denom  = self.input_tokens + self.cache_read_input_tokens
        return cached / denom if denom > 0 else 0.0


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
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int     = 0,
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
                cache_creation_input_tokens=(
                    budget.cache_creation_input_tokens + cache_creation_input_tokens
                ),
                cache_read_input_tokens=(
                    budget.cache_read_input_tokens + cache_read_input_tokens
                ),
            )

    # ── Queries ────────────────────────────────────────────────────────────────

    def total_tokens(self) -> int:
        """Total processed token volume across all nodes in this session."""
        with self._lock:
            return sum(n.total for n in self._nodes.values())

    def total_billable_tokens(self) -> int:
        """
        Total billable token volume across all nodes.

        This is an unweighted token count: input + cache_creation + cache_read
        + output. Provider-specific dollar cost can apply different
        multipliers to each input class.
        """
        return self.total_tokens()

    def estimated_cost(self, rates: TokenCostRates) -> float:
        """Estimated money cost across all nodes using the supplied rates."""
        with self._lock:
            return sum(n.estimated_cost(rates) for n in self._nodes.values())

    def cache_summary(self) -> CacheSummary:
        """
        Aggregate cache accounting across the whole session.

        Returns:
            input_tokens:                non-cached input tokens
            cache_creation_input_tokens: tokens billed at 1.25x to populate cache
            cache_read_input_tokens:     tokens billed at 0.1x served from cache
            total_input_tokens:          input + cache_creation + cache_read
            output_tokens:               output tokens
            total_tokens:                total_input_tokens + output
            hit_rate:                    cache_read / (input + cache_read)
        """
        with self._lock:
            inp = sum(n.input_tokens for n in self._nodes.values())
            out = sum(n.output_tokens for n in self._nodes.values())
            cw  = sum(n.cache_creation_input_tokens for n in self._nodes.values())
            cr  = sum(n.cache_read_input_tokens for n in self._nodes.values())
            denom = inp + cr
            return {
                "input_tokens":                inp,
                "output_tokens":               out,
                "cache_creation_input_tokens": cw,
                "cache_read_input_tokens":     cr,
                "total_input_tokens":          inp + cw + cr,
                "total_tokens":                inp + cw + cr + out,
                "hit_rate":                    (cr / denom) if denom > 0 else 0.0,
            }

    def cost_summary(self, rates: TokenCostRates) -> CostSummary:
        """
        Aggregate token and money accounting across the whole session.

        Money fields use the supplied provider/model rates per million tokens.
        """
        with self._lock:
            inp = sum(n.input_tokens for n in self._nodes.values())
            out = sum(n.output_tokens for n in self._nodes.values())
            cw  = sum(n.cache_creation_input_tokens for n in self._nodes.values())
            cr  = sum(n.cache_read_input_tokens for n in self._nodes.values())
            uncached_input_cost = inp / 1_000_000 * rates.input_per_m
            cache_write_cost = (
                cw / 1_000_000 * rates.effective_cache_creation_input_per_m
            )
            cache_read_cost = (
                cr / 1_000_000 * rates.effective_cache_read_input_per_m
            )
            output_cost = out / 1_000_000 * rates.output_per_m
            return {
                "currency": rates.currency,
                "input_tokens": inp,
                "output_tokens": out,
                "cache_creation_input_tokens": cw,
                "cache_read_input_tokens": cr,
                "total_input_tokens": inp + cw + cr,
                "total_tokens": inp + cw + cr + out,
                "input_cost": uncached_input_cost,
                "cache_creation_cost": cache_write_cost,
                "cache_read_cost": cache_read_cost,
                "output_cost": output_cost,
                "total_cost": (
                    uncached_input_cost
                    + cache_write_cost
                    + cache_read_cost
                    + output_cost
                ),
                "input_per_m": rates.input_per_m,
                "cache_creation_input_per_m": (
                    rates.effective_cache_creation_input_per_m
                ),
                "cache_read_input_per_m": rates.effective_cache_read_input_per_m,
                "output_per_m": rates.output_per_m,
            }

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
