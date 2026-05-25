"""
Tests for ChildSpawner spawn validation hardening — Step 18.

Each validation failure must raise SpawnValidationError (not AssertionError).
Validation must work even with python -O (no assert removal).
"""
from __future__ import annotations

import pytest
from dataclasses import replace

from axor_core.contracts.policy import (
    ChildMode,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.errors.exceptions import SpawnValidationError
from axor_core.node.spawn import _validate_child_policy


def _policy(**kw) -> ExecutionPolicy:
    defaults = dict(
        name="test",
        child_mode=ChildMode.ALLOWED,
        max_child_depth=3,
        export_mode=ExportMode.SUMMARY,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=False,
            allow_bash=False,
            allow_spawn=False,
        ),
    )
    defaults.update(kw)
    return ExecutionPolicy(**defaults)


# ── allow_write escalation ─────────────────────────────────────────────────────

def test_child_allow_write_denied_when_parent_forbids():
    parent = _policy(tool_policy=ToolPolicy(allow_write=False))
    child = _policy(tool_policy=ToolPolicy(allow_write=True))
    with pytest.raises(SpawnValidationError, match="allow_write"):
        _validate_child_policy(child, parent, child_depth=1)


def test_child_allow_write_permitted_when_parent_allows():
    parent = _policy(tool_policy=ToolPolicy(allow_write=True))
    child = _policy(tool_policy=ToolPolicy(allow_write=True))
    _validate_child_policy(child, parent, child_depth=1)  # no exception


# ── allow_bash escalation ──────────────────────────────────────────────────────

def test_child_allow_bash_denied_when_parent_forbids():
    parent = _policy(tool_policy=ToolPolicy(allow_bash=False))
    child = _policy(tool_policy=ToolPolicy(allow_bash=True))
    with pytest.raises(SpawnValidationError, match="allow_bash"):
        _validate_child_policy(child, parent, child_depth=1)


# ── allow_spawn escalation ─────────────────────────────────────────────────────

def test_child_allow_spawn_denied_when_parent_forbids():
    parent = _policy(tool_policy=ToolPolicy(allow_spawn=False))
    child = _policy(tool_policy=ToolPolicy(allow_spawn=True))
    with pytest.raises(SpawnValidationError, match="allow_spawn"):
        _validate_child_policy(child, parent, child_depth=1)


# ── extra_allowed tool ceiling ─────────────────────────────────────────────────

def test_child_extra_allowed_subset_of_parent():
    parent = _policy(tool_policy=ToolPolicy(extra_allowed=("write_file",)))
    child = _policy(tool_policy=ToolPolicy(extra_allowed=("write_file",)))
    _validate_child_policy(child, parent, child_depth=1)  # no exception


def test_child_extra_allowed_exceeds_parent():
    parent = _policy(tool_policy=ToolPolicy(extra_allowed=("write_file",)))
    child = _policy(tool_policy=ToolPolicy(extra_allowed=("write_file", "delete_file")))
    with pytest.raises(SpawnValidationError, match="extra_allowed"):
        _validate_child_policy(child, parent, child_depth=1)


# ── export_mode ceiling ────────────────────────────────────────────────────────

def test_child_less_restrictive_export_mode_denied():
    parent = _policy(export_mode=ExportMode.RESTRICTED)
    child = _policy(export_mode=ExportMode.FULL)
    with pytest.raises(SpawnValidationError, match="export_mode"):
        _validate_child_policy(child, parent, child_depth=1)


def test_child_same_export_mode_allowed():
    parent = _policy(export_mode=ExportMode.FILTERED)
    child = _policy(export_mode=ExportMode.FILTERED)
    _validate_child_policy(child, parent, child_depth=1)  # no exception


def test_child_more_restrictive_export_mode_allowed():
    parent = _policy(export_mode=ExportMode.SUMMARY)
    child = _policy(export_mode=ExportMode.RESTRICTED)
    _validate_child_policy(child, parent, child_depth=1)  # no exception


# ── child depth ceiling ────────────────────────────────────────────────────────

def test_child_depth_within_limit():
    parent = _policy(max_child_depth=3)
    child = _policy(max_child_depth=2)
    _validate_child_policy(child, parent, child_depth=1)  # no exception


def test_child_depth_exceeds_parent_max():
    parent = _policy(max_child_depth=2)
    child = _policy(max_child_depth=1)
    with pytest.raises(SpawnValidationError, match="depth"):
        _validate_child_policy(child, parent, child_depth=3)


# ── SpawnValidationError is NOT AssertionError ─────────────────────────────────

def test_spawn_validation_error_not_assert_error():
    """SpawnValidationError must work even with python -O (assert removal)."""
    parent = _policy(tool_policy=ToolPolicy(allow_write=False))
    child = _policy(tool_policy=ToolPolicy(allow_write=True))
    exc = None
    try:
        _validate_child_policy(child, parent, child_depth=1)
    except SpawnValidationError as e:
        exc = e
    assert exc is not None, "Expected SpawnValidationError"
    assert not isinstance(exc, AssertionError), "Must not be AssertionError"
    assert exc.reason
