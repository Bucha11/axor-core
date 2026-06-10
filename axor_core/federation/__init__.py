"""Federation (TM4.2) — provenance across authenticated peers.

L1 — any value crossing a process/peer boundary is re-minted untrusted (the safe
     default; no trust in the peer's labels).
L2 — a value from an AUTHENTICATED peer in a FEDERATED domain running a COMPATIBLE
     kernel, carrying a VALID provenance RECEIPT, has its provenance RESTORED (we
     trust the peer's labels instead of re-minting). A forged/tampered receipt is
     DENIED; an incompatible kernel or non-federated domain DEGRADES to L1.
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
