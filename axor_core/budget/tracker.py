from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, TypedDict

log = logging.getLogger(__name__)


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
    input_tokens: int   = 0
    output_tokens: int  = 0
    tool_tokens: int    = 0
    context_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int     = 0
    # Optional tier label (e.g. "tier_0", "tier_1") set by adapter cascade
    tier: str | None = None

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
        cached = self.cache_read_input_tokens
        denom  = self.input_tokens + self.cache_read_input_tokens
        return cached / denom if denom > 0 else 0.0


# ── Subscription types (new in 0.5.0) ─────────────────────────────────────────────

class RecordEvent(TypedDict):
    """Payload passed to BudgetTracker subscribers on every record() call."""
    node_id: str
    input_tokens: int
    output_tokens: int
    tool_tokens: int
    context_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    tier: str | None
    cumulative_total: int    # session-wide total after this record


# Callable that receives a RecordEvent. Return value is ignored.
BudgetSubscriber = Callable[[RecordEvent], None]
# Returned by subscribe() — call it to unsubscribe.
Unsubscribe = Callable[[], None]


class BudgetTracker:
    """
    Tracks real token consumption across the entire node tree.

    Thread-safe — nodes at the same depth may run concurrently.

    Does not enforce limits.
    Provides data to policy_engine.py which makes decisions.

    One tracker per GovernedSession — shared across all child nodes.

    Subscriptions (new in 0.5.0)
    ────────────────────────────
    Subscribers (e.g. the adaptive router in axor-openrouter) can register
    callbacks to react to real-time token spend without polling:

        unsub = tracker.subscribe(lambda e: print(e["cumulative_total"]))
        # … later …
        unsub()  # remove subscription

    Callbacks are called synchronously inside record() while holding the lock.
    Keep them short; do not call tracker methods from inside a callback.
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._nodes: dict[str, NodeBudget] = {}
        self._subscribers: list[BudgetSubscriber] = []

    # ── Subscription API ──────────────────────────────────────────────────────────

    def subscribe(self, callback: BudgetSubscriber) -> Unsubscribe:
        """
        Register a callback fired after every record() call.

        Returns a no-arg callable that removes the subscription.
        """
        with self._lock:
            self._subscribers.append(callback)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe

    # ── Core API ───────────────────────────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        parent_id: str | None,
        depth: int,
        tier: str | None = None,
    ) -> None:
        """Register a node's lineage metadata (depth / parent) for depth- and
        subtree-aware accounting. Idempotent: re-registering an existing node
        updates its metadata but PRESERVES accumulated token counters (so it is
        safe to call before every record())."""
        with self._lock:
            existing = self._nodes.get(node_id)
            if existing is None:
                self._nodes[node_id] = NodeBudget(
                    node_id=node_id,
                    parent_id=parent_id,
                    depth=depth,
                    tier=tier,
                )
            else:
                existing.parent_id = parent_id
                existing.depth = depth
                if tier is not None:
                    existing.tier = tier

    def record(
        self,
        node_id: str,
        input_tokens: int,
        output_tokens: int,
        tool_tokens: int    = 0,
        context_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int     = 0,
        tier: str | None = None,
    ) -> None:
        """Record token usage for a node. Notifies subscribers after updating."""
        with self._lock:
            if node_id not in self._nodes:
                log.warning(
                    "BudgetTracker.record() called for unregistered node %r — "
                    "call register_node() first; defaulting to depth=0, parent=None",
                    node_id,
                )
                self._nodes[node_id] = NodeBudget(
                    node_id=node_id,
                    parent_id=None,
                    depth=0,
                    tier=tier,
                )
            budget = self._nodes[node_id]
            effective_tier = tier if tier is not None else budget.tier
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
                tier=effective_tier,
            )
            cumulative = sum(n.total for n in self._nodes.values())
            event: RecordEvent = {
                "node_id":                      node_id,
                "input_tokens":                 input_tokens,
                "output_tokens":                output_tokens,
                "tool_tokens":                  tool_tokens,
                "context_tokens":               context_tokens,
                "cache_creation_input_tokens":  cache_creation_input_tokens,
                "cache_read_input_tokens":      cache_read_input_tokens,
                "tier":                         effective_tier,
                "cumulative_total":              cumulative,
            }
            # notify inside the lock so subscribers see a consistent state
            for cb in self._subscribers:
                try:
                    cb(event)
                except Exception:
                    pass  # subscriber errors must never break execution

    # ── Queries ───────────────────────────────────────────────────────────────────

    def total_tokens(self) -> int:
        with self._lock:
            return sum(n.total for n in self._nodes.values())

    def total_billable_tokens(self) -> int:
        return self.total_tokens()

    def estimated_cost(self, rates: TokenCostRates) -> float:
        with self._lock:
            return sum(n.estimated_cost(rates) for n in self._nodes.values())

    def cache_summary(self) -> CacheSummary:
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
        with self._lock:
            inp = sum(n.input_tokens for n in self._nodes.values())
            out = sum(n.output_tokens for n in self._nodes.values())
            cw  = sum(n.cache_creation_input_tokens for n in self._nodes.values())
            cr  = sum(n.cache_read_input_tokens for n in self._nodes.values())
            uncached_input_cost = inp / 1_000_000 * rates.input_per_m
            cache_write_cost    = cw  / 1_000_000 * rates.effective_cache_creation_input_per_m
            cache_read_cost     = cr  / 1_000_000 * rates.effective_cache_read_input_per_m
            output_cost         = out / 1_000_000 * rates.output_per_m
            return {
                "currency":                    rates.currency,
                "input_tokens":                inp,
                "output_tokens":               out,
                "cache_creation_input_tokens": cw,
                "cache_read_input_tokens":     cr,
                "total_input_tokens":          inp + cw + cr,
                "total_tokens":                inp + cw + cr + out,
                "input_cost":                  uncached_input_cost,
                "cache_creation_cost":         cache_write_cost,
                "cache_read_cost":             cache_read_cost,
                "output_cost":                 output_cost,
                "total_cost":                  (
                    uncached_input_cost + cache_write_cost
                    + cache_read_cost   + output_cost
                ),
                "input_per_m":                 rates.input_per_m,
                "cache_creation_input_per_m":  rates.effective_cache_creation_input_per_m,
                "cache_read_input_per_m":       rates.effective_cache_read_input_per_m,
                "output_per_m":                rates.output_per_m,
            }

    def tier_summary(self) -> dict[str, CacheSummary]:
        """
        Per-tier token summary (new in 0.5.0).

        Returns a dict keyed by tier label (e.g. "tier_0", "tier_1").
        Nodes without a tier label are grouped under the key "untiered".
        Useful for analysing cost distribution across cascade tiers.
        """
        with self._lock:
            tiers: dict[str, list[NodeBudget]] = {}
            for n in self._nodes.values():
                key = n.tier if n.tier is not None else "untiered"
                tiers.setdefault(key, []).append(n)

        result: dict[str, CacheSummary] = {}
        for key, nodes in tiers.items():
            inp = sum(n.input_tokens for n in nodes)
            out = sum(n.output_tokens for n in nodes)
            cw  = sum(n.cache_creation_input_tokens for n in nodes)
            cr  = sum(n.cache_read_input_tokens for n in nodes)
            denom = inp + cr
            result[key] = {
                "input_tokens":                inp,
                "output_tokens":               out,
                "cache_creation_input_tokens": cw,
                "cache_read_input_tokens":     cr,
                "total_input_tokens":          inp + cw + cr,
                "total_tokens":                inp + cw + cr + out,
                "hit_rate":                    (cr / denom) if denom > 0 else 0.0,
            }
        return result

    def node_tokens(self, node_id: str) -> int:
        with self._lock:
            node = self._nodes.get(node_id)
            return node.total if node else 0

    def subtree_tokens(self, node_id: str) -> int:
        with self._lock:
            ids = self._subtree_ids(node_id)
            return sum(
                self._nodes[nid].total
                for nid in ids
                if nid in self._nodes
            )

    def depth_tokens(self, depth: int) -> int:
        with self._lock:
            return sum(
                n.total for n in self._nodes.values()
                if n.depth == depth
            )

    def snapshot(self) -> dict[str, NodeBudget]:
        with self._lock:
            return dict(self._nodes)

    def _subtree_ids(self, root_id: str) -> set[str]:
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
