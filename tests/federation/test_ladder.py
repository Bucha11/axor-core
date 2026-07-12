"""Inter-federation trust ladder (spec v2 Ch.1 §2): declaration buys discount,
never label authority; forgery falls to L0 evidenced; critical sinks ignore
discounts."""
from __future__ import annotations

import pytest
from axor_core.contracts.taint import TaintSource
from axor_core.federation.ladder import (
    L0,
    L1,
    L2,
    LabelAssertion,
    PeerDeclaration,
    effective_root_for_sink,
    receive_foreign,
)
from axor_core.federation.signing import HmacSigner

KEY = b"peer-shared-key-32-bytes-xxxxxxx" + b"x"
SIGNER = HmacSigner(shared_key=KEY)


def _assert_env(message_class: str = "research", sources: tuple = (),
                sensitive: bool = False, forge: bool = False) -> LabelAssertion:
    payload = f"{message_class}:{sources}:{sensitive}".encode()
    sig = HmacSigner(shared_key=b"WRONG-key-32-bytes-xxxxxxxxxxxxxx").sign(payload) \
        if forge else SIGNER.sign(payload)
    return LabelAssertion(
        peer_id="partner", message_class=message_class,
        sources=sources, sensitive=sensitive, payload=payload, signature=sig,
    )


def _decl(level: str = L2, classes: tuple = ("research",), attested: bool = False) -> PeerDeclaration:
    return PeerDeclaration(
        peer_id="partner", level=level,
        verifier=SIGNER if level != L0 else None,
        discount_classes=frozenset(classes),
        governance_attested=attested,
    )


def test_undeclared_peer_is_l0_full_taint() -> None:
    v = receive_foreign(False, None)
    assert v.level == L0 and v.root.is_tainted and not v.discounted
    assert "undeclared_peer_l0" in v.evidence


def test_l1_buys_attribution_not_trust() -> None:
    v = receive_foreign(False, _decl(L1, ()), _assert_env(sources=()))
    assert v.level == L1 and v.root.is_tainted and not v.discounted


def test_l2_discount_is_bounded_never_clean() -> None:
    # peer asserts "clean" — the discount can NEVER reach clean
    v = receive_foreign(False, _decl(), _assert_env(sources=()))
    assert v.discounted and v.root.is_tainted
    assert TaintSource.UNKNOWN_EXTERNAL in v.root.sources


def test_l2_discount_accepts_asserted_sources_as_evidence() -> None:
    v = receive_foreign(False, _decl(), _assert_env(sources=("web",)))
    assert v.discounted
    assert v.root.sources == frozenset({TaintSource.WEB})
    assert v.root.is_tainted  # narrower label, still tainted


def test_discount_only_for_declared_message_classes() -> None:
    v = receive_foreign(False, _decl(classes=("billing",)), _assert_env("research"))
    assert not v.discounted and v.root.is_tainted
    assert any("class_not_discounted" in e for e in v.evidence)


def test_forged_assertion_falls_to_l0_evidenced() -> None:
    v = receive_foreign(False, _decl(), _assert_env(forge=True))
    assert v.level == L0 and not v.discounted and v.root.is_tainted
    assert any("assertion_forged_fell_to_l0" in e for e in v.evidence)


def test_peer_cannot_clear_our_sensitivity_hint() -> None:
    # never affecting the confidentiality floor: sensitive=False from the peer
    # does not clear OUR channel-level sensitivity knowledge
    v = receive_foreign(True, _decl(), _assert_env(sources=("web",), sensitive=False))
    assert v.root.sensitive is True and v.full_root.sensitive is True


def test_critical_sink_ignores_discount() -> None:
    v = receive_foreign(False, _decl(), _assert_env(sources=("web",)))
    assert v.discounted
    critical = effective_root_for_sink(v, sink_is_critical=True)
    standard = effective_root_for_sink(v, sink_is_critical=False)
    assert critical == v.full_root  # undiscounted — criticality non-negotiable
    assert standard == v.root


def test_declared_levels_validate() -> None:
    with pytest.raises(ValueError):
        PeerDeclaration(peer_id="p", level="l3")
    with pytest.raises(ValueError):
        PeerDeclaration(peer_id="p", level=L2, verifier=None)


def test_attested_marks_evidence() -> None:
    v = receive_foreign(False, _decl(attested=True), _assert_env(sources=("web",)))
    assert "governance_attested" in v.evidence
