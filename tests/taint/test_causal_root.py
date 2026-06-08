"""TM2 constructor semantics for the per-value CausalRoot shadow model."""

from __future__ import annotations

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot


def test_constant_is_clean():
    r = CausalRoot.constant()
    assert not r.is_tainted
    assert r.sources == frozenset()
    assert r.sensitive is False


def test_external_read_taints_with_its_source():
    r = CausalRoot.external_read(TaintSource.WEB)
    assert r.is_tainted
    assert r.sources == frozenset({TaintSource.WEB})
    assert r.sensitive is False


def test_external_read_can_be_sensitive():
    r = CausalRoot.external_read(TaintSource.FILE, sensitive=True)
    assert r.is_tainted
    assert r.sensitive is True


def test_mint_unions_sources_and_ors_sensitivity():
    a = CausalRoot.external_read(TaintSource.WEB)
    b = CausalRoot.external_read(TaintSource.FILE, sensitive=True)
    c = CausalRoot.constant()
    minted = CausalRoot.mint(a, b, c)
    assert minted.sources == frozenset({TaintSource.WEB, TaintSource.FILE})
    assert minted.sensitive is True  # over-taint: any sensitive input -> sensitive


def test_mint_of_constants_is_clean():
    minted = CausalRoot.mint(CausalRoot.constant(), CausalRoot.constant())
    assert not minted.is_tainted


def test_mint_of_nothing_is_constant():
    assert CausalRoot.mint() == CausalRoot.constant()


def test_parse_is_passthrough():
    r = CausalRoot.external_read(TaintSource.MCP, sensitive=True)
    assert CausalRoot.parse(r) == r


def test_cross_process_in_remints_untrusted_nonsensitive():
    # TM4.1: re-mint to maximal integrity taint, explicitly non-sensitive.
    r = CausalRoot.cross_process_in()
    assert r.is_tainted
    assert r.sensitive is False
    assert TaintSource.UNKNOWN_EXTERNAL in r.sources


def test_causal_root_is_hashable_and_frozen():
    # K3.5 wants projections finite/hashable (TM3.4 memoization).
    r = CausalRoot.external_read(TaintSource.WEB)
    assert len({r, r}) == 1
