"""Federation — provenance across authenticated peers.

Two levels govern a value arriving from another peer:

L1 — any value crossing a process/peer boundary is re-minted untrusted (the safe
     default; no trust in the peer's labels).
L2 — a value from an authenticated peer in a federated domain running a compatible
     kernel, carrying a valid provenance receipt, has its provenance restored (we
     trust the peer's labels instead of re-minting). A forged or tampered receipt is
     denied; an incompatible kernel or non-federated domain degrades to L1.
"""

from axor_core.federation.receipt import (
    FederationPeer,
    FederationReceipt,
    mint_receipt,
    verify_receipt,
)
from axor_core.federation.gateway import (
    FederationError,
    FederationGateway,
    FederationLevel,
)

__all__ = [
    "FederationPeer", "FederationReceipt", "mint_receipt", "verify_receipt",
    "FederationGateway", "FederationError", "FederationLevel",
]
