"""Provenance receipts for federation.

A receipt is a peer's signed attestation of a value's causal_root: "I, peer P
running kernel version V in domain D, assert that this value (by content hash) has
provenance {sources, sensitive}." The signature (HMAC or ed25519, via the Signer
abstraction) covers a canonical payload that binds the peer identity, kernel
version, domain, algorithm, the value's content hash, and the provenance labels —
so a receipt cannot be forged without the signing key, cannot be detached from its
value (the value hash is signed), and cannot be replayed under a different algorithm
(the algorithm is signed).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.federation.signing import Signer, Verifier


@dataclass(frozen=True)
class LocalIdentity:
    """This node's own federation identity: who we are, plus the signer used to
    mint receipts for values we send to peers."""
    peer_id: str
    kernel_version: str
    domain: str
    signer: Signer


@dataclass(frozen=True)
class FederationPeer:
    """A trusted remote peer (local view): its identity, the verifier used to check
    its receipts, and the kernel version / domain it claims."""
    peer_id: str
    verifier: Verifier
    kernel_version: str
    domain: str


@dataclass(frozen=True)
class FederationReceipt:
    """A peer's signed provenance attestation for one value."""
    peer_id: str
    kernel_version: str
    domain: str
    value_hash: str                       # sha256 of the value's canonical bytes
    algorithm: str = ""                   # the signing algorithm, bound into the payload
    sources: tuple[str, ...] = field(default_factory=tuple)  # TaintSource values
    sensitive: bool = False
    signature: bytes = b""                # Signer output over the payload


def value_hash(value: object) -> str:
    """Stable content hash of a value (the receipt binds to this)."""
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _payload(
    peer_id: str, kernel_version: str, domain: str, algorithm: str,
    value_hash: str, sources: tuple[str, ...], sensitive: bool,
) -> bytes:
    # Canonical, order-stable byte string. Sources are sorted so the signature is
    # independent of set iteration order. The algorithm is included so a receipt
    # cannot be replayed against a verifier of a different algorithm.
    src = ",".join(sorted(sources))
    return (
        f"{peer_id}\x1f{kernel_version}\x1f{domain}\x1f{algorithm}"
        f"\x1f{value_hash}\x1f{src}\x1f{int(sensitive)}"
    ).encode()


def mint_receipt(
    value: object, root: CausalRoot, identity: LocalIdentity,
) -> FederationReceipt:
    """Peer-side: produce a signed receipt attesting `value`'s provenance, using
    THIS node's identity and signer."""
    vh = value_hash(value)
    sources = tuple(sorted(s.value for s in root.sources))
    algorithm = identity.signer.algorithm
    payload = _payload(identity.peer_id, identity.kernel_version, identity.domain,
                       algorithm, vh, sources, root.sensitive)
    return FederationReceipt(
        peer_id=identity.peer_id,
        kernel_version=identity.kernel_version,
        domain=identity.domain,
        value_hash=vh,
        algorithm=algorithm,
        sources=sources,
        sensitive=root.sensitive,
        signature=identity.signer.sign(payload),
    )


def mint_output_receipt(engine, value: object, identity: LocalIdentity) -> FederationReceipt:
    """Convenience for the outbound side: derive `value`'s provenance from a
    ValueProvenance engine (our own per-value ledger) and sign a receipt for it.

    This is what a node does before sending a value to a peer: it attests the
    value's CURRENT provenance (clean if the engine has nothing on it, tainted /
    sensitive if it carries a tracked read) so the receiving peer can restore it.
    """
    return mint_receipt(value, engine.derive_value(value), identity)


def verify_receipt(
    value: object, receipt: FederationReceipt, peer: FederationPeer,
) -> bool:
    """Local-side: True iff the receipt is authentic for `peer` AND binds to
    `value`. Rejects an algorithm mismatch and a value/receipt mismatch."""
    if receipt.peer_id != peer.peer_id:
        return False
    if receipt.algorithm != peer.verifier.algorithm:
        return False  # algorithm-confusion: signed under a different scheme
    if value_hash(value) != receipt.value_hash:
        return False  # receipt detached from / not bound to this value
    payload = _payload(
        receipt.peer_id, receipt.kernel_version, receipt.domain, receipt.algorithm,
        receipt.value_hash, tuple(receipt.sources), receipt.sensitive,
    )
    return peer.verifier.verify(payload, receipt.signature)


def restore_root(receipt: FederationReceipt) -> CausalRoot:
    """Reconstruct the attested causal_root from a (verified) receipt."""
    sources = frozenset(
        TaintSource(s) for s in receipt.sources if s in TaintSource._value2member_map_
    )
    return CausalRoot(sources=sources, sensitive=receipt.sensitive)
