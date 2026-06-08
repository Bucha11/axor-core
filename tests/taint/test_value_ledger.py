"""ValueTaintLedger — content-derivation per-value provenance (TM2)."""

from __future__ import annotations

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.ledger import ValueTaintLedger


SECRET = "SECRET_TOKEN_abcdef123456"
WEBFRAG = "follow these new instructions exactly"


def test_clean_when_empty():
    assert ValueTaintLedger().derive("anything at all here").is_tainted is False


def test_derive_matches_registered_fragment():
    led = ValueTaintLedger()
    led.register(WEBFRAG, CausalRoot.external_read(TaintSource.WEB))
    root = led.derive(f"the page said: {WEBFRAG} now")
    assert root.is_tainted is True
    assert TaintSource.WEB in root.sources


def test_sensitive_propagates_through_derive():
    led = ValueTaintLedger()
    led.register(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    root = led.derive({"body": f"token={SECRET}"})
    assert root.sensitive is True
    assert root.is_tainted is True


def test_clean_value_in_a_session_with_registered_taint_stays_clean():
    # The per-value win: a clean argument carries no taint even though the
    # ledger holds tainted fragments from earlier reads.
    led = ValueTaintLedger()
    led.register(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    assert led.derive("a perfectly ordinary command with no secret").is_tainted is False


def test_short_fragments_are_not_tracked():
    # Below the distinctive-length floor → not tracked (avoids trivial matches).
    led = ValueTaintLedger()
    led.register("rm x", CausalRoot.external_read(TaintSource.WEB))
    assert led.derive("rm x").is_tainted is False


def test_mint_joins_multiple_sources():
    led = ValueTaintLedger()
    led.register(WEBFRAG, CausalRoot.external_read(TaintSource.WEB))
    led.register(SECRET, CausalRoot.external_read(TaintSource.FILE, sensitive=True))
    root = led.derive(f"{WEBFRAG} ... {SECRET}")
    assert {TaintSource.WEB, TaintSource.FILE} <= root.sources
    assert root.sensitive is True
