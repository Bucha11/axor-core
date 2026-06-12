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

import secrets
import time
from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.fingerprint import content_fingerprint
from axor_core.federation.signing import Signer, Verifier

# Default receipt lifetime. A receipt is a one-shot attestation, not a bearer
# token; bounding its validity caps the replay window even if a peer's nonce
# cache is lost (e.g. a restarted gateway).
_DEFAULT_TTL_SECONDS = 300.0
# Receiver-side ceiling on a receipt's remaining validity. The TTL is chosen by
# the SENDER, so a sloppy or malicious (but trusted-keyed) peer could mint a
# receipt valid for years — and since the nonce cache is lost on restart, that
# receipt would be indefinitely replayable. The verifier refuses to honour any
# receipt whose validity window extends further than this from now, clamping the
# replay exposure to a bound WE control regardless of what the sender asked for.
_MAX_RECEIPT_LIFETIME_SECONDS = 3600.0


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


# The receipt binds to the value's canonical fingerprint (shared Ring-0 definition).
value_hash = content_fingerprint


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
    # Fail closed on a legacy unbound receipt: a modern receipt always carries a
    # nonce (replay defence) and an expiry (window bound). One missing either is
    # not replay-protected, so we refuse it rather than accept it as "unreplayable".
    if not receipt.nonce or not receipt.expires_at:
        return False
    _now = time.time() if now is None else now
    if _now > receipt.expires_at:
        return False  # stale: outside its validity window
    if receipt.expires_at - _now > _MAX_RECEIPT_LIFETIME_SECONDS:
        return False  # sender asked for a window longer than we will honour
    payload = _payload(
        receipt.peer_id, receipt.kernel_version, receipt.domain, receipt.algorithm,
        receipt.value_hash, tuple(receipt.sources), receipt.sensitive,
        receipt.nonce, receipt.expires_at,
    )
    return peer.verifier.verify(payload, receipt.signature)


def restore_root(receipt: FederationReceipt) -> CausalRoot:
    """Reconstruct the attested causal_root from a (verified) receipt.

    Fail safe on label skew: if the receipt names a source this kernel does not
    know (a newer peer attesting a source label added after our version), we do not
    silently drop it — dropping every unknown source would restore an *empty*
    (clean) root, an under-taint in the trusted L2 path. Instead any unknown label
    degrades the value to an untrusted cross-process re-mint, preserving sensitivity
    if the receipt asserted it."""
    known = frozenset(
        TaintSource(s) for s in receipt.sources if s in TaintSource._value2member_map_
    )
    dropped_unknown = len(known) != len(set(receipt.sources))
    if dropped_unknown:
        base = CausalRoot.cross_process_in()
        return CausalRoot(
            sources=base.sources | known,
            sensitive=base.sensitive or receipt.sensitive,
        )
    return CausalRoot(sources=known, sensitive=receipt.sensitive)
