"""
Adversarial tests for federation lateral movement.

A multi-agent federation must preserve taint and policy ceilings across
parent->child boundaries.  An adversary cannot use a child to escape taint
that was present in the parent, or to claim capabilities the parent lacks.

Covered scenarios:
  1. Tainted parent -> child inherits taint (cannot escape via spawning)
  2. Clean parent -> tainted child is not gifted extra capabilities
  3. Isolated child cannot clear inherited taint via worker path
  4. Child export back to parent is bounded by parent's export_mode ceiling
"""
from __future__ import annotations

import pytest

from axor_core.contracts.policy import (
    ChildMode,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.taint import TaintSource
from axor_core.errors.exceptions import SpawnValidationError, TaintClearanceError
from axor_core.node.spawn import _validate_child_policy
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

_WEBVAL = "WEB_FRAGMENT_aabbccddeeff00"


# ── helpers ───────────────────────────────────────────────────────────────────

def _policy(**kw) -> ExecutionPolicy:
    defaults = dict(
        name="test",
        child_mode=ChildMode.ALLOWED,
        max_child_depth=3,
        export_mode=ExportMode.SUMMARY,
        tool_policy=ToolPolicy(allow_read=True),
    )
    defaults.update(kw)
    return ExecutionPolicy(**defaults)


# ── variant 1: tainted parent → child inherits taint ──────────────────────────

def test_tainted_parent_child_inherits_taint():
    """
    A child inherits the parent's per-value provenance ledger on spawn.

    Values the parent marked tainted remain tainted when derived in the child.
    The adversary cannot use spawning to wash a value's provenance.
    """
    parent = TaintEngine(node_id="parent")
    parent.register_value(_WEBVAL, CausalRoot.external_read(TaintSource.WEB))
    assert parent.derive_value(_WEBVAL).is_tainted

    child = TaintEngine(node_id="child")
    assert child.derive_value(_WEBVAL).is_tainted is False  # clean before inheriting
    child.inherit_value_ledger(parent)

    assert child.derive_value(_WEBVAL).is_tainted, "child must inherit parent provenance"
    assert TaintSource.WEB in child.derive_value(_WEBVAL).sources


# ── variant 2: clean parent → tainted child is not a privilege escalation ─────

def test_clean_parent_tainted_child_cannot_exceed_parent_policy():
    """
    A clean parent spawning a child that then becomes tainted cannot claim
    capabilities beyond the parent's policy ceiling.

    Even if child acquires taint independently, its policy ceiling is the parent's.
    SpawnValidationError is raised if child policy exceeds parent.
    """
    parent_policy = _policy(
        tool_policy=ToolPolicy(allow_write=False, allow_bash=False),
        export_mode=ExportMode.SUMMARY,
    )
    # Child tries to claim write capability that parent doesn't have
    child_policy = _policy(
        tool_policy=ToolPolicy(allow_write=True, allow_bash=False),
        export_mode=ExportMode.SUMMARY,
    )
    with pytest.raises(SpawnValidationError, match="allow_write"):
        _validate_child_policy(child_policy, parent_policy, child_depth=1)


# ── variant 3: isolated child cannot clear inherited taint ────────────────────

def test_isolated_child_cannot_clear_inherited_taint():
    """
    A child spawned with an "isolated" intent still inherits parent taint.
    The worker within the child cannot clear that taint via the worker path.

    Isolation boundary does not imply taint clearance.
    """
    parent = TaintEngine(node_id="parent")
    parent.register_value(_WEBVAL, CausalRoot.external_read(TaintSource.WEB))

    child = TaintEngine(node_id="isolated-child")
    child.inherit_value_ledger(parent)

    # Worker within isolated child cannot clear taint
    with pytest.raises(TaintClearanceError):
        child.attempt_clear_by_worker()

    assert child.derive_value(_WEBVAL).is_tainted, "provenance must remain after failed clear"


# ── variant 4: child export is bounded by parent export ceiling ───────────────

def test_child_export_mode_cannot_exceed_parent():
    """
    A child cannot export more freely than its parent allows.

    Restrictiveness order (most → least permissive):
        FULL(0) < FILTERED(1) < SUMMARY(2) < RESTRICTED(3)
    FILTERED allows output+metadata (4096 tokens).
    SUMMARY allows output only (1024 tokens) — more restrictive than FILTERED.

    If parent uses ExportMode.SUMMARY, child cannot use FILTERED or FULL.
    """
    parent_policy = _policy(export_mode=ExportMode.SUMMARY)

    # Child attempts less restrictive modes — must be rejected
    child_policy_filtered = _policy(export_mode=ExportMode.FILTERED)
    child_policy_full = _policy(export_mode=ExportMode.FULL)

    with pytest.raises(SpawnValidationError, match="export_mode"):
        _validate_child_policy(child_policy_filtered, parent_policy, child_depth=1)

    with pytest.raises(SpawnValidationError, match="export_mode"):
        _validate_child_policy(child_policy_full, parent_policy, child_depth=1)

    # Child with same or more restrictive mode is fine
    child_policy_summary = _policy(export_mode=ExportMode.SUMMARY)
    _validate_child_policy(child_policy_summary, parent_policy, child_depth=1)  # OK

    child_policy_restricted = _policy(export_mode=ExportMode.RESTRICTED)
    _validate_child_policy(child_policy_restricted, parent_policy, child_depth=1)  # OK
