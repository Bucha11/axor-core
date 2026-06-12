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
import json
import secrets
import time
from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.federation.signing import Signer, Verifier

# Default receipt lifetime. A receipt is a one-shot attestation, not a bearer
# token; bounding its validity caps the replay window even if a peer's nonce
# cache is lost (e.g. a restarted gateway).
_DEFAULT_TTL_SECONDS = 300.0


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
    nonce: str = ""                       # per-receipt unique token (replay defence)
    expires_at: float = 0.0               # unix epoch; 0 == no expiry (legacy)
    signature: bytes = b""                # Signer output over the payload


def value_hash(value: object) -> str:
    """Canonical content hash of a value (the receipt binds to this).

    Avoids ``repr`` — which is order-unstable for dicts and non-injective for
    objects with a custom/constant ``__repr__`` (two distinct values could share a
    receipt). Strings/bytes hash directly; everything else goes through canonical,
    sorted JSON, falling back to a type-tagged repr only for the genuinely
    unserialisable (where the type tag at least prevents cross-type collision)."""
    if isinstance(value, bytes):
        material = b"b:" + value
    elif isinstance(value, str):
        material = b"s:" + value.encode()
    else:
        try:
            material = b"j:" + json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        except (TypeError, ValueError):
            material = f"r:{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode()
    return hashlib.sha256(material).hexdigest()


def _payload(
    peer_id: str, kernel_version: str, domain: str, algorithm: str,
    value_hash: str, sources: tuple[str, ...], sensitive: bool,
    nonce: str, expires_at: float,
) -> bytes:
    # Canonical, order-stable byte string. Sources are sorted so the signature is
    # independent of set iteration order. The algorithm is included so a receipt
    # cannot be replayed against a verifier of a different algorithm; the nonce and
    # expiry are included so neither can be stripped or altered without breaking
    # the signature.
    src = ",".join(sorted(sources))
    return (
        f"{peer_id}\x1f{kernel_version}\x1f{domain}\x1f{algorithm}"
        f"\x1f{value_hash}\x1f{src}\x1f{int(sensitive)}"
        f"\x1f{nonce}\x1f{expires_at!r}"
    ).encode()


def mint_receipt(
    value: object, root: CausalRoot, identity: LocalIdentity,
    *, ttl_seconds: float = _DEFAULT_TTL_SECONDS, now: float | None = None,
) -> FederationReceipt:
    """Peer-side: produce a signed receipt attesting `value`'s provenance, using
    THIS node's identity and signer. Each receipt carries a fresh nonce and an
    expiry so it cannot be replayed indefinitely."""
    vh = value_hash(value)
    sources = tuple(sorted(s.value for s in root.sources))
    algorithm = identity.signer.algorithm
    nonce = secrets.token_hex(16)
    expires_at = (time.time() if now is None else now) + ttl_seconds
    payload = _payload(identity.peer_id, identity.kernel_version, identity.domain,
                       algorithm, vh, sources, root.sensitive, nonce, expires_at)
    return FederationReceipt(
        peer_id=identity.peer_id,
        kernel_version=identity.kernel_version,
        domain=identity.domain,
        value_hash=vh,
        algorithm=algorithm,
        sources=sources,
        sensitive=root.sensitive,
        nonce=nonce,
        expires_at=expires_at,
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
    *, now: float | None = None,
) -> bool:
    """Local-side: True iff the receipt is authentic for `peer`, binds to `value`,
    and has not expired. Rejects an algorithm mismatch, a value/receipt mismatch,
    and a stale receipt. Replay of an unexpired receipt is caught by the gateway's
    nonce cache, not here (verification is stateless)."""
    if receipt.peer_id != peer.peer_id:
        return False
    if receipt.algorithm != peer.verifier.algorithm:
        return False  # algorithm-confusion: signed under a different scheme
    if value_hash(value) != receipt.value_hash:
        return False  # receipt detached from / not bound to this value
    if receipt.expires_at and (time.time() if now is None else now) > receipt.expires_at:
        return False  # stale: outside its validity window
    payload = _payload(
        receipt.peer_id, receipt.kernel_version, receipt.domain, receipt.algorithm,
        receipt.value_hash, tuple(receipt.sources), receipt.sensitive,
        receipt.nonce, receipt.expires_at,
    )
    return peer.verifier.verify(payload, receipt.signature)


def restore_root(receipt: FederationReceipt) -> CausalRoot:
    """Reconstruct the attested causal_root from a (verified) receipt."""
    sources = frozenset(
        TaintSource(s) for s in receipt.sources if s in TaintSource._value2member_map_
    )
    return CausalRoot(sources=sources, sensitive=receipt.sensitive)
