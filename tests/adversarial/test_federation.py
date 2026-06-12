"""Federation trust levels L1/L2: authenticated peers exchange provenance receipts.
A receipt from a compatible, federated peer restores its provenance labels at L2;
an authentic-but-incompatible or unattested value degrades to untrusted L1; and a
forged, tampered, or unknown-peer receipt is denied outright. Crypto is pluggable:
the default symmetric HMAC is covered here, plus an asymmetric ed25519 path (skipped
when the optional backend is unavailable)."""

from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintSource
from axor_core.federation import (
    FederationError,
    FederationGateway,
    FederationLevel,
    FederationPeer,
    HmacSigner,
    LocalIdentity,
    mint_receipt,
)
from axor_core.taint.causal_root import CausalRoot

pytestmark = pytest.mark.adversarial

KERNEL = "axor-core/4.12"
DOMAIN = "trusted.example"


def _identity(peer_id="peerA", key=b"shared-secret-key".ljust(32, b"x"), kernel=KERNEL, domain=DOMAIN):
    """The peer's OWN identity (signs its receipts)."""
    return LocalIdentity(peer_id=peer_id, kernel_version=kernel, domain=domain,
                         signer=HmacSigner(key))


def _peer(peer_id="peerA", key=b"shared-secret-key".ljust(32, b"x"), kernel=KERNEL, domain=DOMAIN):
    """Local view of a trusted peer (verifies its receipts)."""
    return FederationPeer(peer_id=peer_id, verifier=HmacSigner(key),
                          kernel_version=kernel, domain=domain)


def _gateway(peer, *, kernels=(KERNEL,), domains=(DOMAIN,)):
    return FederationGateway(
        peers={peer.peer_id: peer},
        compatible_kernels=set(kernels),
        federated_domains=set(domains),
    )


# ── L2: provenance restored ───────────────────────────────────────────────────

def test_l2_restores_clean_provenance():
    val = "a value the peer computed cleanly"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity())
    root, level = _gateway(_peer()).receive(val, receipt)
    assert level == FederationLevel.L2
    assert root.is_tainted is False        # trusted — NOT re-minted untrusted


def test_l2_restores_tainted_sensitive_provenance():
    val = "a secret the peer read"
    receipt = mint_receipt(
        val, CausalRoot.external_read(TaintSource.FILE, sensitive=True), _identity())
    root, level = _gateway(_peer()).receive(val, receipt)
    assert level == FederationLevel.L2
    assert root.is_tainted is True and root.sensitive is True   # labels preserved


# ── L1: untrusted re-mint ─────────────────────────────────────────────────────

def test_l1_default_when_no_receipt():
    root, level = _gateway(_peer()).receive("unattested bytes", None)
    assert level == FederationLevel.L1
    assert root.is_tainted is True and root.sensitive is False


def test_l1_degrade_on_incompatible_kernel():
    val = "clean federated value"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity())
    gw = _gateway(_peer(), kernels=("axor-core/9.9",))   # local accepts only 9.9
    root, level = gw.receive(val, receipt)
    assert level == FederationLevel.L1               # authentic but not compatible
    assert root.is_tainted is True


def test_l1_degrade_on_non_federated_domain():
    val = "clean federated value"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity(domain="stranger.example"))
    gw = _gateway(_peer(domain="stranger.example"), domains=("trusted.example",))
    root, level = gw.receive(val, receipt)
    assert level == FederationLevel.L1
    assert root.is_tainted is True


# ── DENY: forgery / tampering / unknown peer ──────────────────────────────────

def test_forged_receipt_is_denied():
    val = "attacker data dressed as clean"
    forged = mint_receipt(val, CausalRoot.constant(),
                          _identity(key=b"the-wrong-key".ljust(32, b"x")))     # signed with wrong key
    with pytest.raises(FederationError, match="forged"):
        _gateway(_peer(key=b"the-real-key".ljust(32, b"x"))).receive(val, forged)


def test_receipt_cannot_be_detached_to_another_value():
    # A valid receipt for value A must not validate value B (the value hash is signed).
    receipt_for_a = mint_receipt("value A (clean)", CausalRoot.constant(), _identity())
    with pytest.raises(FederationError, match="value-mismatched"):
        _gateway(_peer()).receive("value B (attacker)", receipt_for_a)


def test_unknown_peer_is_denied():
    receipt = mint_receipt("x", CausalRoot.constant(), _identity())
    gw = FederationGateway(peers={}, compatible_kernels={KERNEL}, federated_domains={DOMAIN})
    with pytest.raises(FederationError, match="unknown peer"):
        gw.receive("x", receipt)


def test_lateral_peer_cannot_launder_attacker_data_as_clean():
    # The whole point: a peer (or MITM) cannot get attacker-controlled data accepted
    # as clean without the signing key. Without it, the receipt fails verification →
    # deny; so the only accepted clean values are those a key-holding peer attested.
    payload = "rm -rf / disguised as a clean config value"
    mitm_receipt = mint_receipt(payload, CausalRoot.constant(), _identity(key=b"mitm-key".ljust(32, b"x")))
    with pytest.raises(FederationError):
        _gateway(_peer(key=b"real-key".ljust(32, b"x"))).receive(payload, mitm_receipt)


# ── asymmetric ed25519 path (optional backend) ────────────────────────────────

def test_ed25519_l2_and_forgery():
    pytest.importorskip("cryptography")
    from axor_core.federation import Ed25519Signer, Ed25519Verifier, generate_ed25519_keypair
    try:
        priv, pub = generate_ed25519_keypair()
    except RuntimeError:
        pytest.skip("ed25519 backend unavailable")

    identity = LocalIdentity("peerB", KERNEL, DOMAIN, Ed25519Signer(priv))
    peer = FederationPeer("peerB", Ed25519Verifier(pub), KERNEL, DOMAIN)
    gw = FederationGateway(peers={"peerB": peer},
                           compatible_kernels={KERNEL}, federated_domains={DOMAIN})

    val = "value signed by peer B's private key"
    receipt = mint_receipt(val, CausalRoot.constant(), identity)
    root, level = gw.receive(val, receipt)
    assert level == FederationLevel.L2 and root.is_tainted is False

    # a different keypair cannot forge for peerB
    priv2, _ = generate_ed25519_keypair()
    forged = mint_receipt(val, CausalRoot.constant(), LocalIdentity("peerB", KERNEL, DOMAIN, Ed25519Signer(priv2)))
    with pytest.raises(FederationError):
        gw.receive(val, forged)


# ── replay defence (nonce cache) ──────────────────────────────────────────────

def test_replay_of_consumed_receipt_is_denied():
    val = "attested value"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity())
    gw = _gateway(_peer())
    gw.receive(val, receipt)                         # first use: accepted
    with pytest.raises(FederationError, match="replayed"):
        gw.receive(val, receipt)                     # replay: denied


def test_one_peer_cannot_burn_another_peers_nonce():
    # The nonce cache is keyed by (peer_id, nonce), so peerB reusing a nonce value
    # peerA already spent does not get peerA's distinct receipt rejected.
    val = "v"
    rA = mint_receipt(val, CausalRoot.constant(), _identity(peer_id="peerA"))
    # peerB signs a DIFFERENT receipt that happens to collide on the nonce string.
    import dataclasses
    rB_src = mint_receipt(val, CausalRoot.constant(),
                          _identity(peer_id="peerB", key=b"peerB-secret-key".ljust(32, b"x")))
    rB = dataclasses.replace(rB_src, nonce=rA.nonce)  # collide the nonce string
    # re-sign rB over its (collided) nonce so it still verifies for peerB
    from axor_core.federation.receipt import _payload
    from axor_core.federation import HmacSigner
    signer = HmacSigner(b"peerB-secret-key".ljust(32, b"x"))
    payload = _payload(rB.peer_id, rB.kernel_version, rB.domain, rB.algorithm,
                       rB.value_hash, tuple(rB.sources), rB.sensitive, rB.nonce, rB.expires_at)
    rB = dataclasses.replace(rB, signature=signer.sign(payload))

    gw = FederationGateway(
        peers={"peerA": _peer(peer_id="peerA"),
               "peerB": _peer(peer_id="peerB", key=b"peerB-secret-key".ljust(32, b"x"))},
        compatible_kernels={KERNEL}, federated_domains={DOMAIN},
    )
    gw.receive(val, rA, claimed_peer_id="peerA")     # peerA spends the nonce
    # peerB's distinct receipt with the same nonce string is still accepted.
    _, level = gw.receive(val, rB, claimed_peer_id="peerB")
    assert level == FederationLevel.L2


def test_receiver_clamps_an_over_long_ttl():
    from axor_core.federation.receipt import _MAX_RECEIPT_LIFETIME_SECONDS, verify_receipt
    val = "v"
    # A trusted-keyed but greedy peer mints a receipt valid for a year.
    receipt = mint_receipt(val, CausalRoot.constant(), _identity(),
                           ttl_seconds=_MAX_RECEIPT_LIFETIME_SECONDS * 1000)
    assert verify_receipt(val, receipt, _peer()) is False
    with pytest.raises(FederationError):
        _gateway(_peer()).receive(val, receipt)


def test_legacy_nonceless_receipt_is_rejected():
    import dataclasses
    from axor_core.federation.receipt import verify_receipt
    val = "v"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity())
    legacy = dataclasses.replace(receipt, nonce="", expires_at=0.0)
    # Not re-signed, but the nonce/expiry checks fail closed before signature anyway.
    assert verify_receipt(val, legacy, _peer()) is False


def test_unknown_source_label_does_not_restore_clean():
    # A peer attests a source label this kernel does not know. It must NOT collapse
    # to an empty (clean) root — that would be an under-taint in the trusted path.
    import dataclasses
    from axor_core.federation.receipt import _payload, restore_root
    from axor_core.federation import HmacSigner
    val = "v"
    receipt = mint_receipt(val, CausalRoot.constant(), _identity())
    tampered = dataclasses.replace(receipt, sources=("some_future_source",))
    # restore_root on an unknown-only source set degrades to untrusted, not clean.
    root = restore_root(tampered)
    assert root.is_tainted is True
