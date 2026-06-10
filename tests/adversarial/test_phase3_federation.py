"""Phase 3 — TM4.2 federation L1/L2: authenticated peers, provenance receipts,
restore-on-L2, degrade-to-L1, deny-on-forgery."""

from __future__ import annotations

import pytest

from axor_core.contracts.taint import TaintSource
from axor_core.federation import (
    FederationError,
    FederationGateway,
    FederationLevel,
    FederationPeer,
    mint_receipt,
)
from axor_core.taint.causal_root import CausalRoot

pytestmark = pytest.mark.adversarial

KERNEL = "axor-core/4.12"
DOMAIN = "trusted.example"


def _peer(peer_id="peerA", key=b"shared-secret-key", kernel=KERNEL, domain=DOMAIN):
    return FederationPeer(peer_id=peer_id, shared_key=key, kernel_version=kernel, domain=domain)


def _gateway(peer, *, kernels=(KERNEL,), domains=(DOMAIN,)):
    return FederationGateway(
        peers={peer.peer_id: peer},
        compatible_kernels=set(kernels),
        federated_domains=set(domains),
    )


# ── L2: provenance restored ───────────────────────────────────────────────────

def test_l2_restores_clean_provenance():
    peer = _peer()
    val = "a value the peer computed cleanly"
    receipt = mint_receipt(val, CausalRoot.constant(), peer)
    root, level = _gateway(peer).receive(val, receipt)
    assert level == FederationLevel.L2
    assert root.is_tainted is False        # trusted — NOT re-minted untrusted


def test_l2_restores_tainted_sensitive_provenance():
    peer = _peer()
    val = "a secret the peer read"
    receipt = mint_receipt(val, CausalRoot.external_read(TaintSource.FILE, sensitive=True), peer)
    root, level = _gateway(peer).receive(val, receipt)
    assert level == FederationLevel.L2
    assert root.is_tainted is True and root.sensitive is True   # labels preserved


# ── L1: untrusted re-mint ─────────────────────────────────────────────────────

def test_l1_default_when_no_receipt():
    peer = _peer()
    root, level = _gateway(peer).receive("unattested bytes", None)
    assert level == FederationLevel.L1
    assert root.is_tainted is True and root.sensitive is False


def test_l1_degrade_on_incompatible_kernel():
    peer = _peer(kernel="axor-core/4.12")
    val = "clean federated value"
    receipt = mint_receipt(val, CausalRoot.constant(), peer)
    gw = _gateway(peer, kernels=("axor-core/9.9",))   # local accepts only 9.9
    root, level = gw.receive(val, receipt)
    assert level == FederationLevel.L1               # authentic but not compatible
    assert root.is_tainted is True


def test_l1_degrade_on_non_federated_domain():
    peer = _peer(domain="stranger.example")
    val = "clean federated value"
    receipt = mint_receipt(val, CausalRoot.constant(), peer)
    gw = _gateway(peer, domains=("trusted.example",))
    root, level = gw.receive(val, receipt)
    assert level == FederationLevel.L1
    assert root.is_tainted is True


# ── DENY: forgery / tampering / unknown peer ──────────────────────────────────

def test_forged_receipt_is_denied():
    peer = _peer(key=b"the-real-key")
    forger = _peer(key=b"the-wrong-key")          # same id, wrong secret
    val = "attacker data dressed as clean"
    forged = mint_receipt(val, CausalRoot.constant(), forger)
    with pytest.raises(FederationError, match="forged"):
        _gateway(peer).receive(val, forged)


def test_receipt_cannot_be_detached_to_another_value():
    # A valid receipt for value A must not validate value B (the value hash is MAC'd).
    peer = _peer()
    receipt_for_a = mint_receipt("value A (clean)", CausalRoot.constant(), peer)
    with pytest.raises(FederationError, match="forged or value-mismatched"):
        _gateway(peer).receive("value B (attacker)", receipt_for_a)


def test_unknown_peer_is_denied():
    peer = _peer("peerA")
    val = "x"
    receipt = mint_receipt(val, CausalRoot.constant(), peer)
    gw = FederationGateway(peers={}, compatible_kernels={KERNEL}, federated_domains={DOMAIN})
    with pytest.raises(FederationError, match="unknown peer"):
        gw.receive(val, receipt)


def test_lateral_peer_cannot_launder_attacker_data_as_clean():
    # The whole point: a peer (or MITM) cannot get attacker-controlled data accepted
    # as clean without the key. Without it, the receipt fails verification → deny;
    # so the only accepted clean values are those a key-holding peer truly attested.
    real = _peer(key=b"real-key")
    mitm = _peer(key=b"mitm-key")
    payload = "rm -rf / disguised as a clean config value"
    mitm_receipt = mint_receipt(payload, CausalRoot.constant(), mitm)
    with pytest.raises(FederationError):
        _gateway(real).receive(payload, mitm_receipt)
