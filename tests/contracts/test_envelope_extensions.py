"""Tests for ExecutionEnvelope 0.5.0 fields: cache_hints, deterministic, depth, parent_node_id."""
from __future__ import annotations

import pytest

from axor_core.contracts.envelope import CacheHints


def test_envelope_default_fields(make_envelope):
    env = make_envelope()
    assert env.cache_hints is None
    assert env.deterministic is False
    assert env.depth == 0
    assert env.parent_node_id is None


def test_envelope_cache_hints_set(make_envelope):
    hints = CacheHints(cacheable_blocks=("system",), ttl="5m")
    env = make_envelope()
    env.cache_hints = hints
    assert env.cache_hints.ttl == "5m"
    assert "system" in env.cache_hints.cacheable_blocks


def test_cache_hints_defaults():
    hints = CacheHints()
    assert hints.ttl is None
    assert hints.cacheable_blocks == ()
    assert hints.response_cache_allowed is False
    assert hints.relevance_k is None


def test_cache_hints_response_cache_allowed():
    hints = CacheHints(response_cache_allowed=True)
    assert hints.response_cache_allowed is True


def test_cache_hints_relevance_k():
    hints = CacheHints(relevance_k=5)
    assert hints.relevance_k == 5


def test_envelope_deterministic_set(make_envelope):
    env = make_envelope()
    env.deterministic = True
    assert env.deterministic is True


def test_envelope_depth_set(make_envelope):
    env = make_envelope()
    env.depth = 3
    assert env.depth == 3


def test_envelope_parent_node_id_set(make_envelope):
    env = make_envelope()
    env.parent_node_id = "node_parent_xyz"
    assert env.parent_node_id == "node_parent_xyz"


def test_envelope_depth_independent_of_lineage(make_envelope):
    """depth field is a direct shortcut, independent of lineage.depth."""
    env = make_envelope()
    env.depth = 7
    assert env.depth == 7
    assert env.lineage.depth == 0  # lineage unchanged
