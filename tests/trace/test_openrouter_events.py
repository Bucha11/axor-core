"""Tests for the 0.5.0 trace event factories: cache_event, routing_decision, cost_threshold."""
from __future__ import annotations


from axor_core.contracts.trace import (
    CacheEvent,
    CostThresholdEvent,
    RoutingEvent,
    TraceEventKind,
)
from axor_core.trace.events import cache_event, cost_threshold, routing_decision


# ── cache_event ────────────────────────────────────────────────────────────────

def test_cache_hit_event():
    ev = cache_event("n1", TraceEventKind.CACHE_HIT, tokens=500, ttl=3600, breakpoint_block="system")
    assert isinstance(ev, CacheEvent)
    assert ev.kind == TraceEventKind.CACHE_HIT
    assert ev.hit is True
    assert ev.tokens == 500
    assert ev.ttl == 3600
    assert ev.breakpoint_block == "system"
    assert ev.node_id == "n1"


def test_cache_miss_event():
    ev = cache_event("n2", TraceEventKind.CACHE_MISS, tokens=1000)
    assert ev.hit is False
    assert ev.kind == TraceEventKind.CACHE_MISS


def test_cache_write_event():
    ev = cache_event("n3", TraceEventKind.CACHE_WRITE, tokens=2000, ttl=300)
    assert ev.hit is False
    assert ev.kind == TraceEventKind.CACHE_WRITE
    assert ev.ttl == 300


# ── routing_decision ─────────────────────────────────────────────────────────

def test_routing_decision_event():
    ev = routing_decision(
        "n1",
        provider="anthropic",
        model="anthropic/claude-opus-4-7",
        tier=0,
        sort_strategy="quality",
        fallback_index=0,
    )
    assert isinstance(ev, RoutingEvent)
    assert ev.kind == TraceEventKind.ROUTING_DECISION
    assert ev.provider == "anthropic"
    assert ev.model == "anthropic/claude-opus-4-7"
    assert ev.tier == 0
    assert ev.sort_strategy == "quality"
    assert ev.fallback_index == 0


def test_routing_decision_defaults():
    ev = routing_decision("n1", provider="openai", model="openai/gpt-4o")
    assert ev.tier == 0
    assert ev.sort_strategy == ""
    assert ev.fallback_index == 0


def test_routing_decision_fallback_index():
    ev = routing_decision("n1", provider="openai", model="openai/gpt-4o", fallback_index=2)
    assert ev.fallback_index == 2


# ── cost_threshold ──────────────────────────────────────────────────────────

def test_cost_threshold_event():
    ev = cost_threshold(
        "n1",
        spent=60_000,
        cap=100_000,
        ratio=0.6,
        threshold_name="compress",
        tier_shift=-1,
    )
    assert isinstance(ev, CostThresholdEvent)
    assert ev.kind == TraceEventKind.COST_THRESHOLD
    assert ev.spent == 60_000
    assert ev.cap == 100_000
    assert abs(ev.ratio - 0.6) < 1e-9
    assert ev.threshold_name == "compress"
    assert ev.tier_shift == -1


def test_cost_threshold_defaults():
    ev = cost_threshold("n1", spent=0, cap=0, ratio=0.0)
    assert ev.threshold_name == ""
    assert ev.tier_shift == 0
