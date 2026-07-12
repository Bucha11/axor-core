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


# ── channel establishment (protocol v0.2 §6a; decisions v2-4/v2-6) ────────────

from axor_core.federation.ladder import (  # noqa: E402
    ChannelGrant,
    GovernanceAttestation,
    establish_channel,
)


def _attestation(forge: bool = False) -> GovernanceAttestation:
    att = GovernanceAttestation(
        peer_id="partner", kernel_version="0.9.2",
        config_hash="sha256:abc", signature=b"",
    )
    signer = HmacSigner(shared_key=b"WRONG-key-32-bytes-xxxxxxxxxxxxxx") if forge else SIGNER
    return GovernanceAttestation(
        peer_id=att.peer_id, kernel_version=att.kernel_version,
        config_hash=att.config_hash, signature=signer.sign(att.payload()),
    )


def test_undeclared_channel_pins_l0() -> None:
    grant = establish_channel(None)
    assert grant.level == L0 and "undeclared_peer_l0" in grant.evidence


def test_mcp_transport_pins_l2_down_to_l1() -> None:
    grant = establish_channel(_decl(L2), transport="mcp")
    assert grant.level == L1
    assert "mcp_transport_pinned_l1" in grant.evidence


def test_native_transport_keeps_l2() -> None:
    assert establish_channel(_decl(L2), transport="native").level == L2


def test_valid_attestation_grants_attested_standing() -> None:
    grant = establish_channel(_decl(L2, attested=True), attestation=_attestation())
    assert grant.governance_attested is True
    assert grant.config_hash == "sha256:abc"
    assert "governance_attestation_verified" in grant.evidence


def test_forged_attestation_is_evidenced_not_attested() -> None:
    grant = establish_channel(
        _decl(L2, attested=True), attestation=_attestation(forge=True)
    )
    assert grant.governance_attested is False and grant.config_hash is None
    assert "governance_attestation_failed" in grant.evidence


def test_missing_attestation_for_attested_declaration_is_evidenced() -> None:
    grant = establish_channel(_decl(L2, attested=True))
    assert grant.governance_attested is False
    assert "governance_attestation_failed" in grant.evidence
