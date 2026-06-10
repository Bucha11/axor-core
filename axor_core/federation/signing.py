"""Pluggable signing for federation receipts.

The receipt MAC/signature is abstracted behind a Signer / Verifier pair so the
crypto is swappable without touching the gateway logic, and so the kernel keeps
ZERO required dependencies:

  • HmacSigner    — symmetric HMAC-SHA256, standard-library only, the default. Both
                    sides share one secret. Fine for a few pre-arranged peers.
  • Ed25519Signer / Ed25519Verifier — asymmetric signatures (the peer signs with its
                    private key; everyone verifies with its public key). Scales to
                    many peers without distributing shared secrets. Requires the
                    optional `cryptography` backend (`pip install axor-core[federation]`),
                    imported lazily so core stays dependency-free.

The algorithm label is part of the signed payload and is checked on verify, so a
receipt produced under one algorithm cannot be replayed against a verifier of
another (algorithm-confusion is rejected).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    """Produces a signature over a payload. Held by a peer for its OWN receipts."""
    algorithm: str

    def sign(self, payload: bytes) -> bytes:
        ...


@runtime_checkable
class Verifier(Protocol):
    """Verifies a signature. Held locally, one per trusted peer, to check THEIR
    receipts."""
    algorithm: str

    def verify(self, payload: bytes, signature: bytes) -> bool:
        ...


# ── Symmetric default: HMAC-SHA256 (stdlib, zero-dependency) ───────────────────

class HmacSigner:
    """Symmetric signer/verifier — the same shared key signs and verifies."""
    algorithm = "hmac-sha256"

    def __init__(self, shared_key: bytes) -> None:
        self._key = shared_key

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)


# ── Asymmetric: Ed25519 (optional `cryptography` backend, lazy) ────────────────

def _ed25519():
    # Catch BaseException, not just Exception: a broken backend (e.g. a missing
    # _cffi_backend, or a Rust binding that raises pyo3 PanicException, which is a
    # BaseException) must surface as our clean RuntimeError, not crash the caller.
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        return ed25519
    except BaseException as exc:  # ImportError, or a broken/unavailable backend
        raise RuntimeError(
            "ed25519 federation signing requires the optional 'cryptography' "
            "backend: pip install 'axor-core[federation]'"
        ) from exc


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Return (private_bytes, public_bytes) raw 32-byte ed25519 key material."""
    ed = _ed25519()
    from cryptography.hazmat.primitives import serialization
    sk = ed.Ed25519PrivateKey.generate()
    priv = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, pub


class Ed25519Signer:
    """Asymmetric signer — the peer holds the private key and signs its receipts."""
    algorithm = "ed25519"

    def __init__(self, private_bytes: bytes) -> None:
        ed = _ed25519()
        self._sk = ed.Ed25519PrivateKey.from_private_bytes(private_bytes)

    def sign(self, payload: bytes) -> bytes:
        return self._sk.sign(payload)


class Ed25519Verifier:
    """Asymmetric verifier — anyone holds the peer's public key to check receipts."""
    algorithm = "ed25519"

    def __init__(self, public_bytes: bytes) -> None:
        ed = _ed25519()
        self._pk = ed.Ed25519PublicKey.from_public_bytes(public_bytes)

    def verify(self, payload: bytes, signature: bytes) -> bool:
        try:
            self._pk.verify(signature, payload)
            return True
        except Exception:
            return False
