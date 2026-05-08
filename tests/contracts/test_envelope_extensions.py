"""Tests for ExecutionEnvelope 0.5.0 fields: cache_hints, deterministic, depth, parent_node_id."""
from __future__ import annotations

import pytest


def test_envelope_default_fields(make_envelope):
    env = make_envelope()
    assert env.cache_hints is None
    assert env.deterministic is False
    assert env.depth == 0
    assert env.parent_node_id is None


def test_envelope_cache_hints_set(make_envelope):
    env = make_envelope()
    # ExecutionEnvelope is a regular (non-frozen) dataclass
    env.cache_hints = {"blocks": ["system"], "ttl": "5m"}
    assert env.cache_hints["ttl"] == "5m"


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
