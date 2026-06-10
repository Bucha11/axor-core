"""Provenance receipts for federation.

A receipt is a peer's signed attestation of a value's causal_root: "I, peer P
running kernel version V, assert that this value (by content hash) has provenance
{sources, sensitive}." It is authenticated with an HMAC over a canonical byte
string keyed on the peer's shared secret, so a receipt cannot be forged by anyone
without the key, and it cannot be detached from its value (the value hash is in the
MAC'd payload).

Symmetric HMAC keeps core dependency-free; an asymmetric backend can replace
`_mac` without touching the gateway logic.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot


@dataclass(frozen=True)
class FederationPeer:
    """An authenticated federation peer (local view): identity, shared secret,
    kernel version, and the domain it belongs to."""
    peer_id: str
    shared_key: bytes
    kernel_version: str
    domain: str


@dataclass(frozen=True)
class FederationReceipt:
    """A peer's signed provenance attestation for one value."""
    peer_id: str
    kernel_version: str
    domain: str
    value_hash: str                       # sha256 of the value's canonical bytes
    sources: tuple[str, ...] = field(default_factory=tuple)  # TaintSource values
    sensitive: bool = False
    mac: str = ""                         # hex HMAC-SHA256 over the payload


def value_hash(value: object) -> str:
    """Stable content hash of a value (the receipt binds to this)."""
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _payload(
    peer_id: str, kernel_version: str, domain: str, value_hash: str,
    sources: tuple[str, ...], sensitive: bool,
) -> bytes:
    # Canonical, order-stable byte string. Sources are sorted so the MAC is
    # independent of set iteration order.
    src = ",".join(sorted(sources))
    return f"{peer_id}\x1f{kernel_version}\x1f{domain}\x1f{value_hash}\x1f{src}\x1f{int(sensitive)}".encode()


def _mac(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def mint_receipt(
    value: object, root: CausalRoot, peer: FederationPeer,
) -> FederationReceipt:
    """Peer-side: produce a signed receipt attesting `value`'s provenance."""
    vh = value_hash(value)
    sources = tuple(sorted(s.value for s in root.sources))
    payload = _payload(peer.peer_id, peer.kernel_version, peer.domain, vh,
                       sources, root.sensitive)
    return FederationReceipt(
        peer_id=peer.peer_id,
        kernel_version=peer.kernel_version,
        domain=peer.domain,
        value_hash=vh,
        sources=sources,
        sensitive=root.sensitive,
        mac=_mac(peer.shared_key, payload),
    )


def verify_receipt(
    value: object, receipt: FederationReceipt, peer: FederationPeer,
) -> bool:
    """Local-side: True iff the receipt is authentic for `peer` AND binds to
    `value`. Constant-time MAC comparison; rejects a value/receipt mismatch."""
    if receipt.peer_id != peer.peer_id:
        return False
    if value_hash(value) != receipt.value_hash:
        return False  # receipt detached from / not bound to this value
    payload = _payload(
        receipt.peer_id, receipt.kernel_version, receipt.domain,
        receipt.value_hash, tuple(receipt.sources), receipt.sensitive,
    )
    expected = _mac(peer.shared_key, payload)
    return hmac.compare_digest(expected, receipt.mac)


def restore_root(receipt: FederationReceipt) -> CausalRoot:
    """Reconstruct the attested causal_root from a (verified) receipt."""
    sources = frozenset(
        TaintSource(s) for s in receipt.sources if s in TaintSource._value2member_map_
    )
    return CausalRoot(sources=sources, sensitive=receipt.sensitive)
