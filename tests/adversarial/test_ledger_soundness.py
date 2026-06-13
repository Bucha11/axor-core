"""Taint-ledger soundness: reference-counted endorsement and flood saturation.

Endorsing one value must not release a fragment another live value still shares,
and a fragment is only dropped when its last reference is gone. Flooding the ledger
past its capacity must fail closed (coarsely over-taint) rather than silently drop
entries, and that saturated state propagates when ledgers merge.
"""

from __future__ import annotations
from axor_core.contracts.degradation import GovernanceAuthority

import pytest

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.ledger import ValueTaintLedger, _MAX_TOTAL_SEGMENTS
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial

SHARED = "SHARED_FRAGMENT_xyz123abc"
A = f"{SHARED} alpha_unique_token_aaa"
B = f"{SHARED} beta_unique_token_bbb"


def _tainted():
    return CausalRoot.external_read(TaintSource.WEB)


def test_endorsing_one_value_does_not_under_taint_a_shared_fragment():
    # A and B both carry SHARED. Endorsing A must NOT remove SHARED while B is live
    # (refcount): otherwise B is silently laundered (endorsement over-release).
    led = ValueTaintLedger()
    led.register(A, _tainted())
    led.register(B, _tainted())
    led.unregister(A)                       # endorse A
    assert led.derive(B).is_tainted is True            # B still tracked
    assert led.derive(SHARED).is_tainted is True       # the shared fragment survives


def test_full_release_removes_fragment_when_last_reference_gone():
    led = ValueTaintLedger()
    led.register(A, _tainted())
    led.register(B, _tainted())
    led.unregister(A)
    led.unregister(B)                       # both released
    assert led.derive(SHARED).is_tainted is False      # now genuinely clean


def test_unique_fragment_released_on_endorse():
    led = ValueTaintLedger()
    led.register(A, _tainted())
    led.register(B, _tainted())
    led.unregister(A)
    # A's own (non-shared) fragment is gone, B's remains.
    assert led.derive("alpha_unique_token_aaa").is_tainted is False
    assert led.derive("beta_unique_token_bbb").is_tainted is True


def test_flood_saturation_fails_closed():
    # Flooding the ledger past its cap must NOT silently drop — it flips to coarse
    # over-taint so a previously-clean value cannot be laundered through a flood.
    led = ValueTaintLedger()
    led.register("the_real_secret_fragment_keepme", _tainted())
    for i in range(_MAX_TOTAL_SEGMENTS + 50):
        led.register(f"flood_filler_token_{i:08d}", _tainted())
    assert led._saturated is True
    # an unrelated, never-registered value now derives tainted (fail-closed)
    assert led.derive("completely_unrelated_clean_value").is_tainted is True
    # and the real secret is certainly not laundered
    assert led.derive("the_real_secret_fragment_keepme").is_tainted is True


def test_saturation_propagates_on_merge():
    parent = ValueTaintLedger()
    parent._saturated = True
    child = ValueTaintLedger()
    child.merge(parent)
    assert child._saturated is True
    assert child.derive("anything_at_all_here").is_tainted is True


def test_engine_endorse_value_respects_refcount():
    # End-to-end through the engine's governed endorsement: endorsing the secret
    # value releases it, but a co-registered value sharing a fragment stays tainted.
    eng = TaintEngine()
    eng.register_value(A, _tainted())
    eng.register_value(B, _tainted())
    eng.endorse_value(A, GovernanceAuthority("operator", "human_operator", "reviewed"))
    assert eng.derive_value(B).is_tainted is True
