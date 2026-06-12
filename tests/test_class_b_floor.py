"""Class B — confidentiality-floor soundness redesign.

The floor was a flat integer decremented on `derive(content).sensitive`. That
desynchronised from the refcounted ledger in two directions; these tests pin both
closed, plus the governance-authority capability tightening.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.degradation import GovernanceAuthority
from axor_core.contracts.taint import TaintSource
from axor_core.errors.exceptions import TaintClearanceError
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

_GA = GovernanceAuthority("operator", "human_operator", "reviewed")


def _read_secret(eng: TaintEngine, secret: str) -> None:
    eng.register_value(secret, CausalRoot.external_read(TaintSource.FILE, sensitive=True))


# ── Trouble 1 (under-block): endorsing a DIFFERENT value cannot lift another
#    secret's floor, even when the two share a ledger fragment ────────────────────

def test_near_collision_endorsement_does_not_lift_another_secrets_floor():
    eng = TaintEngine()
    # Space-separated so each word is a registered ≥12-char segment.
    secret = "correcthorsebattery staplexyzzy1234"
    _read_secret(eng, secret)
    assert eng.confidentiality_floor_active() is True
    # A benign value that merely SHARES one of the secret's segments (so it derives
    # as sensitive under the leaky ledger) — endorsing it must NOT lift the secret's
    # floor, because it is a different value (different fingerprint).
    benign_overlap = "the word correcthorsebattery appears in this log line"
    # Precondition: this value DOES derive as sensitive via the shared segment, so
    # the old derive()-keyed floor WOULD have been (wrongly) lifted by endorsing it.
    assert eng.derive_value(benign_overlap).sensitive is True
    eng.endorse_value(benign_overlap, _GA)
    assert eng.confidentiality_floor_active() is True       # secret still outstanding


# ── Trouble 2 (over-block): a sub-threshold secret can be released by endorsement,
#    not stuck until a wholesale clear ─────────────────────────────────────────────

def test_sub_threshold_secret_is_releasable_by_endorsement():
    eng = TaintEngine()
    _read_secret(eng, "pin42")                              # shorter than segment min
    assert eng.confidentiality_floor_active() is True       # armed on the read fact
    eng.endorse_value("pin42", _GA)
    assert eng.confidentiality_floor_active() is False       # and releasable by id


def test_endorsing_exact_secret_lifts_only_that_one():
    eng = TaintEngine()
    _read_secret(eng, "SECRET_ONE_aaaaaaaaaaaa")
    _read_secret(eng, "SECRET_TWO_bbbbbbbbbbbb")
    eng.endorse_value("SECRET_ONE_aaaaaaaaaaaa", _GA)
    assert eng.confidentiality_floor_active() is True        # the other is still loose
    eng.endorse_value("SECRET_TWO_bbbbbbbbbbbb", _GA)
    assert eng.confidentiality_floor_active() is False


def test_floor_and_ledger_agree_after_endorsement():
    """After releasing the only secret, the floor and the per-value gate agree —
    no 'floor active while nothing derives sensitive' (or vice versa) desync."""
    eng = TaintEngine()
    secret = "SENSITIVE_VALUE_zzz12345"
    _read_secret(eng, secret)
    eng.endorse_value(secret, _GA)
    assert eng.confidentiality_floor_active() is False
    assert eng.derive_value(secret).sensitive is False


def test_child_inherits_floor_without_double_counting():
    parent = TaintEngine()
    _read_secret(parent, "SECRET_INHERIT_aaaa1234")
    child = TaintEngine(node_id="c")
    child.inherit_value_ledger(parent)
    child.inherit_value_ledger(parent)                      # idempotent re-merge
    assert child.confidentiality_floor_active() is True
    # one endorsement of the secret releases it (no inflated count to drain)
    child.endorse_value("SECRET_INHERIT_aaaa1234", _GA)
    assert child.confidentiality_floor_active() is False


# ── B2/A: governance authority is an unforgeable capability, not three strings ────

def test_endorse_requires_governance_authority_object():
    eng = TaintEngine()
    _read_secret(eng, "SECRET_VALUE_abcd1234")
    # A worker-fabricated dict / strings are NOT a GovernanceAuthority capability.
    with pytest.raises(TaintClearanceError):
        eng.endorse_value("SECRET_VALUE_abcd1234", {"authority_type": "automated_policy"})
    with pytest.raises(TaintClearanceError):
        eng.endorse_value("SECRET_VALUE_abcd1234", "automated_policy")


def test_clear_rejects_worker_labelled_authority():
    eng = TaintEngine()
    _read_secret(eng, "SECRET_VALUE_abcd1234")
    worker = GovernanceAuthority(authority_id="", authority_type="worker", reason_code="")
    with pytest.raises(TaintClearanceError):
        eng.clear_by_governance(worker)
    assert eng.confidentiality_floor_active() is True        # not lifted


def test_valid_authority_clears():
    eng = TaintEngine()
    _read_secret(eng, "SECRET_VALUE_abcd1234")
    eng.clear_by_governance(_GA)
    assert eng.confidentiality_floor_active() is False
