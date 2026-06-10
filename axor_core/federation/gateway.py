"""Federation gateway (TM4.2) — decide the provenance of an incoming federated value.

receive() returns the CausalRoot to register locally for a value arriving from a
peer, applying the L1/L2 ladder:

  • no receipt                       → L1: re-mint untrusted (cross_process_in)
  • receipt fails verification       → DENY (FederationError) — a forged or tampered
                                        receipt is an attack, not a downgrade
  • valid receipt, but peer's kernel
    is incompatible or its domain is
    not in our federated set         → L1: degrade to untrusted re-mint
  • valid receipt, compatible kernel,
    federated domain                 → L2: RESTORE the attested provenance
"""

from __future__ import annotations

from enum import Enum

from axor_core.taint.causal_root import CausalRoot
from axor_core.federation.receipt import (
    FederationPeer,
    FederationReceipt,
    restore_root,
    verify_receipt,
)


class FederationError(Exception):
    """A federated value must be rejected (forged/tampered receipt, unknown peer)."""


class FederationLevel(str, Enum):
    L1 = "l1"   # untrusted re-mint
    L2 = "l2"   # provenance restored


class FederationGateway:
    """Local federation policy: which peers are trusted, which kernel versions are
    compatible, and which domains are federated. Each session/tenant owns one."""

    def __init__(
        self,
        peers: dict[str, FederationPeer] | None = None,
        compatible_kernels: frozenset[str] | set[str] | None = None,
        federated_domains: frozenset[str] | set[str] | None = None,
    ) -> None:
        self._peers = dict(peers or {})
        self._compatible_kernels = frozenset(compatible_kernels or ())
        self._federated_domains = frozenset(federated_domains or ())

    def receive(
        self,
        value: object,
        receipt: FederationReceipt | None,
        claimed_peer_id: str | None = None,
    ) -> tuple[CausalRoot, FederationLevel]:
        """Return (causal_root, level) for an incoming federated value.

        Raises FederationError when the value must be rejected outright (a receipt
        that fails authentication — never silently downgraded, since a forged
        receipt is an active attack to launder provenance)."""
        # L1 default: no receipt → the peer is unauthenticated for this value.
        if receipt is None:
            return CausalRoot.cross_process_in(), FederationLevel.L1

        peer_id = claimed_peer_id or receipt.peer_id
        peer = self._peers.get(peer_id)
        if peer is None:
            raise FederationError(f"receipt from unknown peer {peer_id!r}")

        if not verify_receipt(value, receipt, peer):
            raise FederationError(
                f"forged or value-mismatched receipt from peer {peer_id!r}"
            )

        # Authentic receipt. L2 requires BOTH a compatible kernel AND a federated
        # domain; otherwise we trust the peer's identity but not its labels → L1.
        kernel_ok = receipt.kernel_version in self._compatible_kernels
        domain_ok = receipt.domain in self._federated_domains
        if kernel_ok and domain_ok:
            return restore_root(receipt), FederationLevel.L2
        return CausalRoot.cross_process_in(), FederationLevel.L1
